"""Qwen3.8-Flash-Next on a 32 GB Mac with Transformers expert streaming, following Strata's Mac protocol.

Strata's Mac guide reports decode tokens/s on a 200-token story answer and prefill tokens/s on a 4K-token prompt,
with MTP off on 32 GB Macs. This script runs both on the checkpoint written by ``scripts/convert_fp8.py``:
- story: a short story request, exactly 200 new tokens (``min_new_tokens``), greedy; decode tok/s excludes the first token.
  The story runs first on a cold cache, then three other prompts follow (``--runs`` repeats this list); the report's
  ``decode_median`` is over all of them;
- prefill: a ~4,000-token prompt, time to first token and prefill tok/s.
Normal use (OS page cache on) unless ``--bypass-page-cache``. Writes one JSON report.

Usage: bench.py CONVERTED_DIR --cache-gib 6 --output report.json [--bits 3] [--runs 1]
"""

import argparse
import json
import platform
import resource
import threading
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from moe_expert_streaming import load_qwen38


STORY = "Write a short story about a lighthouse keeper who finds a message in a bottle."
# Decode is measured on different prompts, each once: the expert cache carries over from one generation to the next, so
# repeating a prompt replays the same routing and overstates the hit rate (~91% instead of ~89% for the story).
DECODE_PROMPTS = [
    STORY,
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Explain the causes of the French Revolution to a high school student.",
    "Translate into French and explain the grammar: 'The children would have gone if it had not rained.'",
]


DEVICE = "cuda" if torch.cuda.is_available() else "mps"


def device_memory():
    """Device memory in use: the MPS driver's allocation, or what PyTorch reserved on CUDA."""
    return torch.mps.driver_allocated_memory() if DEVICE == "mps" else torch.cuda.memory_reserved()


def synchronize():
    torch.mps.synchronize() if DEVICE == "mps" else torch.cuda.synchronize()


class DevicePeak:
    """Peak MPS driver memory, sampled every 20 ms (the allocator's own peak is not exposed on MPS)."""

    def __init__(self):
        self.peak, self.running = 0, True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while self.running:
            self.peak = max(self.peak, device_memory())
            time.sleep(0.02)

    def stop(self):
        self.running = False
        self.thread.join()
        return self.peak


def long_prompt(tokenizer, target=4000):
    notes = [
        f"Note {i}: the harbour town kept records of ships, storms, cargo and the keepers who tended its light, "
        f"and entry {i} describes one season in detail, with names, dates and the weather."
        for i in range(1, 2000)
    ]
    text = []
    for note in notes:
        text.append(note)
        if len(tokenizer("\n".join(text)).input_ids) >= target:
            break
    return "Read these records.\n" + "\n".join(text) + "\nSummarize the history of the harbour town."


def generate(model, tokenizer, prompt, new_tokens, prefill_chunk_size=None):
    messages = [{"role": "user", "content": prompt}]
    try:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False
        )
    except TypeError:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    ids = inputs["input_ids"].to(DEVICE)
    first = {"calls": 0}

    class FirstToken:
        """`generate` calls `put` once with the prompt, then once per new token: the second call is the first token."""

        def put(self, value):
            first["calls"] += 1
            if first["calls"] == 2:
                first["t"] = time.perf_counter()

        def end(self):
            pass

    stats_before = model.experts_offload_store.stats()
    synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            streamer=FirstToken(),
            prefill_chunk_size=prefill_chunk_size,
        )
    synchronize()
    end = time.perf_counter()
    stats = model.experts_offload_store.stats()
    generated = out.shape[1] - ids.shape[1]
    ttft = first.get("t", end) - start
    delta = {k: v - stats_before[k] for k, v in stats.items() if isinstance(v, (int, float)) and k in stats_before}
    return {
        "prompt_tokens": ids.shape[1],
        "new_tokens": generated,
        "time_to_first_token_s": ttft,
        "prefill_tokens_per_s": ids.shape[1] / ttft,
        "decode_tokens_per_s": (generated - 1) / (end - start - ttft),
        "total_s": end - start,
        "store": delta,
        "text": tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("converted", type=Path)
    parser.add_argument("--cache-gib", type=float, required=True)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--io-threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--bypass-page-cache", action="store_true")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-story", action="store_true")
    parser.add_argument("--cache-policy", default="lru", choices=("lru", "lfu"))
    parser.add_argument("--prefill-chunk-size", type=int, default=None, help="Prefill long prompts in chunks")
    parser.add_argument(
        "--dense-bits", type=int, default=None, help="Quantize the dense nn.Linear weights (MetalConfig)"
    )
    parser.add_argument("--no-mps-commit", action="store_true", help="A/B: no early MPS command-buffer commits")
    parser.add_argument("--device", default=None, help="mps or cuda (default: the available one)")
    parser.add_argument("--host-cache-gib", type=float, default=0.0, help="Tier of experts in pinned host memory")
    parser.add_argument("--host-preload", action="store_true", help="Read every expert into the host tier at load")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device or DEVICE
    dense = args.converted / "dense"
    tokenizer = AutoTokenizer.from_pretrained(dense)
    peak = DevicePeak()
    start = time.perf_counter()
    model = load_qwen38(
        args.converted,
        cache_gib=args.cache_gib,
        bits=args.bits,
        dense_bits=args.dense_bits if args.dense_bits is not None else "auto",
        io_threads=args.io_threads,
        cache_policy=args.cache_policy,
        bypass_page_cache=args.bypass_page_cache,
        mps_commit=not args.no_mps_commit,
        device=DEVICE,
        host_cache_gib=args.host_cache_gib,
        host_preload=args.host_preload,
    )
    load_s = time.perf_counter() - start
    after_load = device_memory()
    report = {
        "model": "Qwen3.8-Flash-Next (from Qwen/Qwen3.8-Flash-Next-FP8)",
        "experts": f"{args.bits}-bit MLX affine, group 64",
        "cache_gib": args.cache_gib,
        "io_threads": args.io_threads,
        "page_cache": "bypassed" if args.bypass_page_cache else "on",
        "dense": "8-bit MetalConfig" if (args.dense_bits or (args.dense_bits is None and DEVICE == "mps")) else "bf16",
        "device": DEVICE,
        "host_cache_gib": args.host_cache_gib,
        "host_preload": args.host_preload,
        "prefill_chunk_size": args.prefill_chunk_size,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "load_s": load_s,
        "device_after_load_bytes": after_load,
        "runs": {"story": [], "prefill_4k": []},
    }
    for _ in range(0 if args.skip_story else args.runs):
        for prompt in DECODE_PROMPTS:
            report["runs"]["story"].append({"prompt": prompt, **generate(model, tokenizer, prompt, 200)})
            print(json.dumps({k: v for k, v in report["runs"]["story"][-1].items() if k != "text"}), flush=True)
    if report["runs"]["story"]:
        rates = sorted(run["decode_tokens_per_s"] for run in report["runs"]["story"])
        report["decode_median"] = (rates[(len(rates) - 1) // 2] + rates[len(rates) // 2]) / 2
    if not args.skip_prefill:
        prompt = long_prompt(tokenizer)
        for _ in range(args.runs):
            report["runs"]["prefill_4k"].append(generate(model, tokenizer, prompt, 32, args.prefill_chunk_size))
            print(json.dumps({k: v for k, v in report["runs"]["prefill_4k"][-1].items() if k != "text"}), flush=True)
    report["device_peak_sampled_bytes"] = peak.stop()
    report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.output.write_text(json.dumps(report, indent=2))
    if report["runs"]["story"]:
        print(report["runs"]["story"][0]["text"][:400])
        print("decode median", report["decode_median"])


if __name__ == "__main__":
    main()
