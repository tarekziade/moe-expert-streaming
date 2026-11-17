"""Write a routing profile (lookups per expert) for ``load_qwen38(preload_profile=...)`` / ``bench.py
--preload-profile``: generates on prompts that are NOT the benchmark's, so a benchmark with the profile does not replay
its own routing.

Usage: routing_profile.py CONVERTED_DIR OUT.json [load_qwen38 options as bench.py: --cache-gib, --gguf-dense, ...]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402

from moe_expert_streaming import load_qwen38  # noqa: E402


PROMPTS = [
    "Write an email to a colleague proposing to move our weekly meeting to Thursday afternoon.",
    "Give me a recipe for a vegetarian lasagna, with quantities and steps.",
    "Explain how a hash table works and when its lookups become slow.",
    "Write a JavaScript function that debounces another function, and explain how to use it.",
    "Summarize the main ideas of stoicism in a few paragraphs.",
    "Write a short poem about a train crossing a snowy plain at night.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("converted", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-gib", type=float, default=12)
    parser.add_argument("--io-threads", type=int, default=8)
    parser.add_argument("--dense-bits", type=int, default=8)
    parser.add_argument("--gguf-dense", type=Path, nargs="*", default=None)
    parser.add_argument("--gguf-experts", action="store_true")
    parser.add_argument("--tokens", type=int, default=200)
    args = parser.parse_args()
    model = load_qwen38(
        args.converted, cache_gib=args.cache_gib, io_threads=args.io_threads, dense_bits=args.dense_bits,
        gguf_dense=args.gguf_dense, gguf_experts=args.gguf_experts,
    )  # fmt: skip
    tokenizer = AutoTokenizer.from_pretrained(args.converted / "dense")
    for prompt in PROMPTS:
        result = bench.generate(model, tokenizer, prompt, args.tokens)
        print(f"{result['decode_tokens_per_s']:.2f} tok/s: {prompt[:50]}", flush=True)
    profile = model.experts_offload_store.routing_profile()
    args.output.write_text(json.dumps(profile))
    print(f"wrote {args.output}: {sum(len(v) for v in profile.values())} experts over {len(profile)} layers")


if __name__ == "__main__":
    with torch.no_grad():
        main()
