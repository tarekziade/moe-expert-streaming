"""MLX affine quantization in plain PyTorch, bit-exact with ``mx.quantize`` / ``mx.dequantize``, any device.

Layout (MLX): a row of ``K`` codes of ``bits`` bits is one LSB-first bitstream stored in little-endian ``uint32`` words
(``K * bits / 32`` of them), so 3-, 5- and 6-bit codes may straddle two words. Each group of ``group_size`` codes has a
scale and a bias: ``value = code * scale + bias``.

Used on CUDA/CPU, where the Metal kernels are unavailable: by the converter (to write the experts) and by
:func:`gather_qmm` (to run them).
"""

import functools

import torch


BITS = (2, 3, 4, 5, 6, 8)


def _round_half_away(t):
    return t.sign() * (t.abs() + 0.5).floor()


def quantize_codes(weight, group_size, bits):
    """MLX's affine quantization of ``weight [N, K]``: ``(codes int32 [N, K], scales, biases)``, scales and biases in
    ``weight``'s dtype."""
    n, k = weight.shape
    max_val = (1 << bits) - 1
    grouped = weight.float().reshape(n, k // group_size, group_size)
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
    return codes.to(torch.int32).reshape(n, k), scales.to(weight.dtype), biases.to(weight.dtype)


def pack(codes, bits):
    """``codes int [..., K]`` -> ``uint32 [..., K * bits / 32]`` (LSB-first bitstream)."""
    *lead, k = codes.shape
    words = k * bits // 32
    codes = codes.to(torch.int64).reshape(-1, k)
    out = torch.zeros(codes.shape[0], words + 1, dtype=torch.int64, device=codes.device)
    offsets = torch.arange(k, device=codes.device) * bits
    word, shift = offsets // 32, offsets % 32
    lo = (codes << shift) & 0xFFFFFFFF
    hi = codes >> (32 - shift)  # bits that spill into the next word (0 when they all fit)
    out.scatter_add_(1, word.expand_as(lo), lo)
    out.scatter_add_(1, (word + 1).expand_as(hi), hi)
    return out[:, :words].to(torch.int32).view(torch.uint32).reshape(*lead, words) if lead else out[0, :words]


@functools.cache
def _unpack_plan(k, bits, device):
    offsets = torch.arange(k, device=device) * bits
    word, shift = offsets // 32, offsets % 32
    return word, (word + 1).clamp(max=k * bits // 32 - 1), shift


def unpack(packed, k, bits):
    """``uint32 [..., K * bits / 32]`` -> ``int64 codes [..., K]``."""
    word, next_word, shift = _unpack_plan(k, bits, packed.device)
    w = packed.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    lo = w.index_select(-1, word) >> shift
    hi = (w.index_select(-1, next_word) << (32 - shift)) & 0xFFFFFFFF
    return (lo | hi) & ((1 << bits) - 1)


def dequantize(packed, scales, biases, group_size, bits, dtype=None):
    """``mx.dequantize``: ``[..., N, K*bits/32]`` packed with ``[..., N, K/group_size]`` scales/biases -> ``[..., N, K]``."""
    groups = scales.shape[-1]
    k = groups * group_size
    codes = unpack(packed, k, bits).to(torch.float32).unflatten(-1, (groups, group_size))
    # In float32, rounded once to the output dtype, as MLX does.
    values = codes * scales.unsqueeze(-1).float() + biases.unsqueeze(-1).float()
    return values.flatten(-2).to(scales.dtype if dtype is None else dtype)


def quantize(weight, group_size, bits):
    """``mx.quantize`` (Metal kernel on MPS, this module elsewhere): ``(uint32 packed, scales, biases)``."""
    if weight.device.type == "mps":
        from .metal_linear import load_metal_kernel

        return load_metal_kernel().quantize(weight.contiguous(), group_size=group_size, bits=bits)
    codes, scales, biases = quantize_codes(weight, group_size, bits)
    return pack(codes, bits), scales, biases


@functools.cache
def _triton_available():
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def gather_qmm(x, w, scales, biases, lhs_indices=None, rhs_indices=None, transpose=True, group_size=64, bits=4, **_):
    """``mx.gather_qmm(x, w, ..., transpose=True)`` for CUDA/CPU: row ``i`` of ``x [S, 1, K]`` times expert
    ``rhs_indices[i]`` of the quantized bank ``w [E, N, K*bits/32]`` -> ``[S, 1, N]``.

    Small calls (decode) on CUDA: the fused Triton kernel (:mod:`.kernels.triton_gather_qmv`), which reads only the
    packed experts. Otherwise the experts are dequantized to ``x``'s dtype and run as dense matmuls: one batched
    matmul over the gathered experts for small calls, one matmul per distinct expert for large ones (prefill).
    """
    if not transpose:
        raise NotImplementedError("gather_qmm: only transpose=True (nn.Linear layout) is supported")
    rows = x.shape[0]
    inputs = x.reshape(rows, -1)
    if lhs_indices is not None:
        inputs = inputs.index_select(0, lhs_indices.reshape(-1))
        rows = inputs.shape[0]
    rhs = rhs_indices.reshape(-1)
    w = w.view(torch.int32)  # CUDA cannot index uint32 tensors; same bits
    if rows <= 64 and inputs.is_cuda and _triton_available():
        from .kernels.triton_gather_qmv import gather_qmv

        return gather_qmv(inputs, w, scales, biases, rhs, group_size, bits).unsqueeze(1)
    if rows <= 64:
        weights = dequantize(w[rhs], scales[rhs], biases[rhs], group_size, bits, dtype=inputs.dtype)  # [S, N, K]
        return torch.bmm(inputs.unsqueeze(1), weights.transpose(1, 2))
    output = inputs.new_empty(rows, w.shape[1])
    order = torch.argsort(rhs, stable=True)
    experts, counts = torch.unique_consecutive(rhs[order], return_counts=True)
    start = 0
    for expert, count in zip(experts.tolist(), counts.tolist()):
        index = order[start : start + count]
        weight = dequantize(w[expert], scales[expert], biases[expert], group_size, bits, dtype=inputs.dtype)
        output[index] = inputs[index] @ weight.T
        start += count
    return output.unsqueeze(1)
