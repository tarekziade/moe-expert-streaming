"""Run bench.py (same arguments) and log, for each MoE call, the experts' input hidden states and routing, for
evaluating next-layer expert predictors offline (predict_sim.py).

Saved to $TRACE_OUT (torch.save): per call (generation index, layer index, hidden fp16 (rows, hidden), top_k_index
int16 (rows, k), unique experts). Calls with more than one row (prefill) keep every $TRACE_PREFILL_STRIDE-th row
(default 4) of hidden/top_k_index; their unique experts cover all rows.

Example: TRACE_OUT=trace.pt python predict_trace.py CONVERTED --cache-gib 10 --dense-bits 8 --io-threads 8
--prefill-chunk-size 1024 --gguf-dense GGUF --output report.json
"""

import importlib.util
import os
import sys

import torch
from transformers.integrations import experts_offload


STRIDE = int(os.environ.get("TRACE_PREFILL_STRIDE", "4"))
CALLS = []
STATE = {"gen": -1}
_orig = experts_offload._offloaded_experts


def traced(module, store, prefix, hidden_states, top_k_index, top_k_weights, routing, experts):
    rows = slice(None) if hidden_states.shape[0] == 1 else slice(None, None, STRIDE)
    layer = int(prefix.split("layers.")[1].split(".")[0])
    CALLS.append(
        (
            STATE["gen"],
            layer,
            hidden_states[rows].to(torch.float16).cpu(),
            top_k_index[rows].to(torch.int16).cpu(),
            list(experts),
        )
    )
    return _orig(module, store, prefix, hidden_states, top_k_index, top_k_weights, routing, experts)


experts_offload._offloaded_experts = traced

spec = importlib.util.spec_from_file_location(
    "bench", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
PROMPTS = []
_gen = bench.generate


def generate(model, tokenizer, prompt, new_tokens, prefill_chunk_size=None):
    STATE["gen"] += 1
    PROMPTS.append(prompt[:200])
    store = model.experts_offload_store
    STATE["slot_bytes"] = store.slot_bytes
    out = _gen(model, tokenizer, prompt, new_tokens, prefill_chunk_size)
    save()  # after each generation, so a run killed later keeps what it has
    return out


def save():
    torch.save({"prompts": PROMPTS, "slot_bytes": STATE.get("slot_bytes"), "calls": CALLS}, os.environ["TRACE_OUT"])


bench.generate = generate
try:
    sys.argv = ["bench.py"] + sys.argv[1:]
    bench.main()
finally:
    save()
    print("trace calls", len(CALLS), flush=True)
