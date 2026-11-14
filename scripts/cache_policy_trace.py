"""Run bench.py (same arguments) and log each MoE call's unique experts (demand trace) to $TRACE_OUT, for
cache_policy_sim.py. Example: TRACE_OUT=trace.json python cache_policy_trace.py CONVERTED --cache-gib 12 --dense-bits 8
--io-threads 8 --skip-prefill --output report.json"""

import importlib.util
import json
import os
import sys

from transformers.integrations import experts_offload as ssd_moe


TRACE = []  # (generation index, layer prefix, [experts])
STATE = {"gen": -1}
_orig = ssd_moe._offloaded_experts


def traced(module, store, prefix, hidden_states, top_k_index, top_k_weights, routing, experts):
    TRACE.append((STATE["gen"], prefix, list(experts)))
    return _orig(module, store, prefix, hidden_states, top_k_index, top_k_weights, routing, experts)


ssd_moe._offloaded_experts = traced

spec = importlib.util.spec_from_file_location(
    "bench", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
PROMPTS = []  # filled as bench.py runs its decode prompts
_gen = bench.generate


def generate(model, tokenizer, prompt, new_tokens, prefill_chunk_size=None):
    STATE["gen"] += 1
    PROMPTS.append(prompt)
    out = _gen(model, tokenizer, prompt, new_tokens, prefill_chunk_size)
    STATE["slots"] = model.experts_offload_store.num_slots
    return out


bench.generate = generate
try:
    sys.argv = ["bench.py"] + sys.argv[1:]
    bench.main()
finally:
    out = os.environ["TRACE_OUT"]
    layers = sorted({p for _, p, _ in TRACE}, key=lambda s: int(s.split("layers.")[1].split(".")[0]))
    json.dump(
        {
            "slots": STATE.get("slots"),
            "prompts": PROMPTS,
            "layers": layers,
            "calls": [(g, layers.index(p), e) for g, p, e in TRACE],
        },
        open(out, "w"),
    )
    print("trace calls", len(TRACE), "slots", STATE.get("slots"))
