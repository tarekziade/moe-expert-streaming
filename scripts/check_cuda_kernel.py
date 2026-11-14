"""Check the Triton gather_qmv kernel against the PyTorch dequantize reference, and time it (CUDA).

Exits non-zero on a mismatch. Shapes: Qwen3.8-Flash-Next experts (gate_up 1280 x 2560, down 2560 x 640), 10 rows.
"""

import sys
import time

import torch

from moe_expert_streaming import mlx_affine
from moe_expert_streaming.kernels.triton_gather_qmv import gather_qmv


def bank(experts, n, k, bits, device):
    quantized = [
        mlx_affine.quantize(torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02, 64, bits)
        for _ in range(experts)
    ]
    return [torch.stack(t) for t in zip(*quantized)]


def main():
    device = "cuda"
    torch.manual_seed(0)
    failures = 0
    for bits in (2, 3, 4, 5, 6, 8):
        for n, k in ((1280, 2560), (2560, 640)):
            w, s, b = bank(16, n, k, bits, device)
            x = torch.randn(10, k, device=device, dtype=torch.bfloat16)
            rhs = torch.randint(0, 16, (10,), device=device)
            got = gather_qmv(x, w, s, b, rhs, 64, bits).float()
            dense = mlx_affine.dequantize(w.view(torch.int32)[rhs], s[rhs], b[rhs], 64, bits, dtype=torch.float32)
            want = torch.bmm(x.float().unsqueeze(1), dense.transpose(1, 2)).squeeze(1)
            rel = ((got - want).norm() / want.norm()).item()
            ok = rel < 1e-2
            failures += not ok
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(100):
                gather_qmv(x, w, s, b, rhs, 64, bits)
            torch.cuda.synchronize()
            us = (time.perf_counter() - t) / 100 * 1e6
            packed_bytes = 10 * (w[0].numel() * 4 + s[0].numel() * 2 * 2)
            print(
                f"bits {bits} [{n}x{k}] rel err {rel:.2e} {'ok' if ok else 'MISMATCH'}  {us:7.1f} us  {packed_bytes / us / 1e3:6.1f} GB/s",
                flush=True,
            )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
