"""bench.py (4-prompt decode) with host-side timers around the expert path of single-token (decode) MoE calls.

Per MoE call, host wall time is split into:
- sync: from entry to `release_host_buffers` (the routing `.cpu()` copy: waits for all queued device work);
- read_wait: blocked in `_take_host` for this layer's own experts (misses whose read had not finished);
- install_ahead: installing next-layer guesses (excluding any read wait inside it);
- rest: bookkeeping, slot-table copies and kernel launches.
`outside` = decode wall time - MoE time (host launching dense layers, sampling, generate overhead).
"""

import importlib.util
import json
import os
import sys
import time
from collections import defaultdict

from transformers.integrations import experts_offload as ssd_moe


T = defaultdict(float)
C = defaultdict(int)
S = {"decode": False, "in_ahead": False, "entry": None}
orig_fwd = ssd_moe.experts_offload_forward


def fwd(module, hidden_states, top_k_index, top_k_weights):
    decode = top_k_index.shape[0] == 1
    S["decode"], S["entry"] = decode, time.perf_counter()
    try:
        return orig_fwd(module, hidden_states, top_k_index, top_k_weights)
    finally:
        if decode:
            T["moe"] += time.perf_counter() - S["entry"]
            C["calls"] += 1
        S["decode"] = False


ssd_moe.experts_offload_forward = fwd

ES = ssd_moe.ExpertStore
orig_release = ES.release_host_buffers


def release(self):
    if S["decode"] and S["entry"] is not None:
        T["sync"] += time.perf_counter() - S["entry"]
        S["entry_done"] = True
    return orig_release(self)


ES.release_host_buffers = release
orig_take = ES._take_host


def take(self, prefix, expert):
    t = time.perf_counter()
    pending = (prefix, expert) in self.pending
    try:
        return orig_take(self, prefix, expert)
    finally:
        if S["decode"]:
            dt = time.perf_counter() - t
            key = "take_in_ahead" if S["in_ahead"] else "read_wait"
            T[key] += dt
            C[key] += 1
            C[key + ("_prefetched" if pending else "_unstarted")] += 1


ES._take_host = take
orig_ahead = ES.install_ahead


def ahead(self, prefix, experts):
    t = time.perf_counter()
    S["in_ahead"] = True
    try:
        return orig_ahead(self, prefix, experts)
    finally:
        S["in_ahead"] = False
        if S["decode"]:
            T["install_ahead"] += time.perf_counter() - t


ES.install_ahead = ahead


# Finer split of "rest": host time of individual pieces during decode MoE calls.
def timed(owner, name, key=None):
    orig = getattr(owner, name)
    key = key or name

    def wrapper(*args, **kwargs):
        if not S["decode"]:
            return orig(*args, **kwargs)
        t = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            T["f_" + key] += time.perf_counter() - t
            C["f_" + key] += 1

    setattr(owner, name, wrapper)


timed(ssd_moe, "_quantized_pool_forward", "pool_forward_launch")
timed(ES, "device_slots")
timed(ES, "acquire")  # includes read waits of misses (read_wait)
timed(ES, "prefetch")
timed(ES, "discard")
timed(ES, "_install")

spec = importlib.util.spec_from_file_location(
    "bench", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
_gen = bench.generate
PROFILES = []


def generate(model, tokenizer, prompt, new_tokens, prefill_chunk_size=None):
    T.clear()
    C.clear()
    out = _gen(model, tokenizer, prompt, new_tokens, prefill_chunk_size)
    tokens = C["calls"] / 48
    decode_s = (out["new_tokens"] - 1) / out["decode_tokens_per_s"]
    ms = lambda v: round(1000 * v / tokens, 2)
    prof = {
        "prompt": prompt[:40],
        "decode_tok_s": round(out["decode_tokens_per_s"], 2),
        "tokens": tokens,
        "ms_per_token": round(1000 * decode_s / tokens, 1),
        "moe_ms": ms(T["moe"]),
        "sync_ms": ms(T["sync"]),
        "read_wait_ms": ms(T["read_wait"]),
        "install_ahead_ms": ms(T["install_ahead"] - T["take_in_ahead"]),
        "rest_ms": ms(T["moe"] - T["sync"] - T["read_wait"] - T["install_ahead"]),
        "outside_moe_ms": ms(decode_s - T["moe"]),
        "fine_ms": {k[2:]: ms(v) for k, v in T.items() if k.startswith("f_")},
        "counts": dict(C),
    }
    PROFILES.append(prof)
    print("PROFILE", json.dumps(prof), flush=True)
    return out


bench.generate = generate
sys.argv = ["bench.py"] + sys.argv[1:]
try:
    bench.main()
finally:
    json.dump(PROFILES, open(os.environ["PROFILE_OUT"], "w"), indent=1)
