"""Teacher-forced perplexity through the decode path (one token per forward), so decode-only kernels are exercised.

Usage: decode_ppl.py CONVERTED_DIR TEXT_FILE TAG [TOKENS=512] [DENSE_BITS=8, 16 for BF16] [CACHE_GIB=12]
Prints the perplexity and saves each step's top-1 token to top1_TAG.pt (compare runs for agreement).
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from moe_expert_streaming import load_qwen38


root, text_file, tag = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 512
BITS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
CACHE = float(sys.argv[6]) if len(sys.argv) > 6 else 12
tok = AutoTokenizer.from_pretrained(root / "dense")
model = load_qwen38(root, cache_gib=CACHE, dense_bits=BITS if BITS < 16 else None)
text = open(text_file).read()
ids = tok(text, return_tensors="pt").input_ids[:, :N].to("mps")
nll, top1 = [], []
t0 = time.perf_counter()
with torch.inference_mode():
    out = model(ids[:, :1], use_cache=True)
    cache = out.past_key_values
    for i in range(1, N):
        logits = out.logits[0, -1].float()
        nll.append(torch.nn.functional.cross_entropy(logits[None], ids[0, i : i + 1]))
        top1.append(logits.argmax())
        out = model(ids[:, i : i + 1], past_key_values=cache, use_cache=True)
nll = torch.stack(nll)
top1 = torch.stack(top1).cpu()
torch.save(top1, f"top1_{tag}.pt")
print(tag, "decode-path tokens", N, "ppl", round(nll.mean().exp().item(), 4), f"({time.perf_counter() - t0:.0f} s)")
