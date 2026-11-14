import torch
from kernels import get_kernel


k = get_kernel("kernels-community/mlx-quantization-metal-kernels", version=2)
torch.manual_seed(0)
for N, K in ((2560, 2560), (320, 10240), (10240, 320), (4, 10240), (48, 2560), (248320, 2560), (12288, 2560)):
    w = torch.randn(N, K, device="mps", dtype=torch.bfloat16)
    wq, s, b = k.quantize(w, group_size=64, bits=4)
    ref_w = k.dequantize(wq, s, b, group_size=64, bits=4).float()
    row = []
    for M in (1, 29, 65, 96, 128, 129, 192, 255):
        x = torch.randn(1, M, K, dtype=torch.bfloat16, device="mps")
        out = k.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
        ref = x.float() @ ref_w.T
        row.append(f"M{M}:{((out.float() - ref).norm() / ref.norm()).item():.3f}")
    print(f"N{N} K{K}", " ".join(row))
