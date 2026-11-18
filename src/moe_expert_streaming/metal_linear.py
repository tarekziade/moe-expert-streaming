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


KERNEL_REPO = "kernels-community/mlx-quantization-metal-kernels"


@functools.cache
def load_metal_kernel():
    from .hub_kernels import get_locked

    return get_locked(KERNEL_REPO)


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


def affine_quantize(weight, group_size, bits):
    """``mx.quantize`` (Metal kernel on MPS, a bit-exact torch version elsewhere)."""
    from . import mlx_affine

    return mlx_affine.quantize(weight, group_size, bits)


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
