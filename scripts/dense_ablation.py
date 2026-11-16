"""Dense-path decode cost per module category, by ablation: experts return zeros (no SSD), and an ablated category
replays the output of its first decode call instead of computing (zero device and host cost). The decode time saved
by ablating a category is what making it free would buy, measured under normal pipelining.

Usage: dense_ablation.py CONVERTED_DIR [GGUF_SHARD]  (GGUF: dense layers from it, see gguf_dense)
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.integrations import experts_offload

from moe_expert_streaming import load_qwen38


experts_offload.experts_offload_forward = lambda module, hidden_states, top_k_index, top_k_weights: torch.zeros_like(
    hidden_states
)
root = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(root / "dense")
model = load_qwen38(root, cache_gib=1, io_threads=0, gguf_dense=sys.argv[2] if len(sys.argv) > 2 else None)
model.eval()
layers = model.model.layers
CATS = {
    "linear_attn (GDN, 36 layers)": [l.linear_attn for l in layers if hasattr(l, "linear_attn")],
    "self_attn (12 layers, incl. indexer)": [l.self_attn for l in layers if hasattr(l, "self_attn")],
    "hyper-connections (97)": [m for l in layers for m in (l.attn_hyper_connection, l.mlp_hyper_connection)]
    + [model.model.hyper_connection_mixer],
    "shared expert + gate (48)": [m for l in layers for m in (l.mlp.shared_expert, l.mlp.shared_expert_gate)],
    "router (48)": [l.mlp.gate for l in layers],
    "ple (n-gram embedding)": [l.ple for l in layers if l.ple is not None],
    "lm_head": [model.lm_head],
}
ABLATED = set()
PHASE = {"decode": False}


def wrap(module, cat):
    orig = module.forward
    state = {}

    def forward(*args, **kwargs):
        if PHASE["decode"] and (cat in ABLATED or "ALL" in ABLATED):
            if "out" in state:
                return state["out"]
            out = orig(*args, **kwargs)
            state["out"] = out
            return out
        state.clear()
        return orig(*args, **kwargs)

    module.forward = forward


for cat, mods in CATS.items():
    for m in mods:
        wrap(m, cat)

prompt = "Write a short story about a lighthouse keeper who finds a message in a bottle."
ids = tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    enable_thinking=False,
)["input_ids"].to("mps")
N = 120


def run():
    # prefill (not ablated: the replay records the first decode call's output)
    with torch.inference_mode():
        out = model(ids, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        seq = [ids, nxt]
        torch.mps.synchronize()
        t = time.perf_counter()
        PHASE["decode"] = True
        for _ in range(N):
            out = model(nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1:].argmax(-1)
            seq.append(nxt)
        nxt.cpu()
        dt = time.perf_counter() - t
        PHASE["decode"] = False
    return 1000 * dt / N


def ablate(cats):
    ABLATED.clear()
    ABLATED.update(cats)


configs = [("baseline (experts zero)", [])] + [(c, [c]) for c in CATS] + [("ALL of the above", ["ALL"])]
run()  # warmup
res = {name: [] for name, _ in configs}
for rep in range(3):
    for name, cats in configs:
        ablate(cats)
        res[name].append(run())
base = statistics.median(res["baseline (experts zero)"])
print(f"decode ms/token, {N} tokens, median of 3 (M5, 8-bit dense, experts = zeros)")
for name, _ in configs:
    m = statistics.median(res[name])
    print(f"  {name:40s} {m:6.1f} ms   saves {base - m:5.1f} ms   runs {[round(x, 1) for x in res[name]]}")
json.dump(res, open(os.environ.get("OUT", "dense_ablation.json"), "w"), indent=1)
