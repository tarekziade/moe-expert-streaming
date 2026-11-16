"""Teacher-forced perplexity through the prefill path (the whole text in chunks of 1,024 tokens, one forward each).

Usage: prefill_ppl.py CONVERTED_DIR TEXT_FILE TAG [TOKENS=1024] [DENSE_BITS=8, 16 for BF16] [CACHE_GIB=10] [GGUF...]
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from moe_expert_streaming import load_qwen38


root, text_file, tag = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
BITS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
CACHE = float(sys.argv[6]) if len(sys.argv) > 6 else 10
GGUF = sys.argv[7:] or None
tok = AutoTokenizer.from_pretrained(root / "dense")
device = "cuda" if torch.cuda.is_available() else "mps"
model = load_qwen38(root, cache_gib=CACHE, dense_bits=BITS if BITS < 16 else None, device=device, gguf_dense=GGUF)
ids = tok(open(text_file).read(), return_tensors="pt").input_ids[:, :N].to(device)
t0 = time.perf_counter()
with torch.inference_mode():
    logits = model(ids, use_cache=False).logits[0, :-1].float()
nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:])
print(
    tag, "prefill-path tokens", ids.shape[1], "ppl", round(nll.exp().item(), 4), f"({time.perf_counter() - t0:.0f} s)"
)
