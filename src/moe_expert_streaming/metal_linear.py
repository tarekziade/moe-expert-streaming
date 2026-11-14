"""MetalConfig dense layers on the v2 MLX quantization kernels.

A subset of huggingface/transformers#49467 ("Update metal quant kernels"), carried here until it is merged:

- ``MetalLinear.forward`` through ``quantized_matmul`` of ``kernels-community/mlx-quantization-metal-kernels`` v2 (the
  kernel the ``experts_offload`` PR already uses for the experts), with the M5 workaround below;
- quantization at load time with MLX's own rounding (``mx.quantize`` on MPS, an exact torch reference elsewhere);
- ggml's layer kernels (norms, gated delta net, top-k) applied after loading, as the GGUF path does.

Drop this module once #49467 is in the transformers this project depends on.
"""

import functools

import torch


KERNEL_REPO, KERNEL_VERSION = "kernels-community/mlx-quantization-metal-kernels", 2


@functools.cache
def load_metal_kernel():
    from kernels import get_kernel

    return get_kernel(KERNEL_REPO, version=KERNEL_VERSION)


def _metal_linear_forward(self, input):
    if self.weight.dtype != torch.uint32:
        return torch.nn.functional.linear(input, self.weight, self.bias)
    # On M5 GPUs the v2 NAX matmul returns wrong results for bfloat16/float16 inputs once a call has enough rows
    # (~30-128 depending on the shape; kernels-community#1332); float32 is correct. Single-row calls (decode) take the
    # matrix-vector kernel, which is correct in every dtype.
    # TODO(portability): M5 NAX workaround, specific to Apple GPUs; select it by device/kernel version.
    compute_dtype = input.dtype if input.numel() == input.shape[-1] else torch.float32
    output = (
        load_metal_kernel()
        .quantized_matmul(
            input.to(compute_dtype),
            self.weight,
            self.scales.to(compute_dtype),
            self.qbiases.to(compute_dtype),
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        .to(input.dtype)
    )
    if self.bias is not None:
        output = output + self.bias
    return output


def _round_half_away(t):
    return t.sign() * (t.abs() + 0.5).floor()


def affine_quantize(weight, group_size, bits):
    """``mx.quantize``: on MPS through the kernel, elsewhere an exact torch reference (same codes, same scales).

    Returns ``(w_packed uint32 [N, K * bits / 32], scales, biases)``, scales and biases in ``weight``'s dtype.
    """
    if weight.device.type == "mps":
        return load_metal_kernel().quantize(weight.contiguous(), group_size=group_size, bits=bits)
    n, k = weight.shape
    per_int, max_val, groups = 32 // bits, (1 << bits) - 1, k // group_size
    grouped = weight.float().reshape(n, groups, group_size)
    w_min = grouped.amin(dim=-1)
    w_max = grouped.amax(dim=-1).clamp(min=0)  # MLX starts its running max at 0
    scales = ((w_max - w_min) / max_val).clamp(min=1e-7)
    # Snap the larger-magnitude edge onto the grid so it is represented exactly, as MLX does.
    side = w_min.abs() > w_max.abs()
    scales = torch.where(side, scales, -scales)
    edge = torch.where(side, w_min, w_max)
    q0 = _round_half_away(edge / scales)
    at_zero = q0 == 0
    scales = torch.where(at_zero, scales, edge / q0)
    biases = torch.where(at_zero, torch.zeros_like(edge), edge)
    codes = _round_half_away((grouped - biases.unsqueeze(-1)) / scales.unsqueeze(-1)).clamp(0, max_val)
    codes = codes.to(torch.int32).reshape(n, k)
    packed = torch.zeros(n, k // per_int, dtype=torch.int32, device=weight.device)
    for i in range(per_int):
        packed |= codes[:, i::per_int] << (bits * i)
    return packed.to(torch.uint32), scales.to(weight.dtype), biases.to(weight.dtype)


def _quantize_tensor(weight, group_size, bits):
    # transformers' MetalQuantize casts the scales and biases back to the weight's dtype itself.
    return affine_quantize(weight, group_size, bits)


def apply():
    """Patch transformers' MetalLinear and MetalConfig quantization (idempotent)."""
    from transformers.integrations import metal_quantization

    metal_quantization.MetalLinear.forward = _metal_linear_forward
    metal_quantization._affine_quantize_tensor = _quantize_tensor


def kernelize_layers(model):
    """ggml's layer kernels on the loaded model, as transformers' GGUF path does (MetalConfig in #49467 does the same)."""
    from transformers.integrations.gguf.kernels import kernelize_ggml_layers

    kernelize_ggml_layers(model)
