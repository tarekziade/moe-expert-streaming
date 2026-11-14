"""Triton kernel: ``out[s] = x[s] @ dequantize(w[rhs[s]]).T`` for MLX affine quantized expert banks (CUDA).

Fused gather + dequantize + matrix-vector product for decode-sized calls: each program handles one row of ``x`` and a
block of output features, reads the packed words of its expert (``bits`` per code, LSB-first bitstream in 32-bit words,
so 3-, 5- and 6-bit codes may straddle two words), unpacks them in registers and accumulates in float32. Only the
packed bytes are read: 2.15 MB per 3-bit Qwen3.8 expert instead of a dequantized copy.

TODO(portability): CUDA (Triton) only; a fused prefill (many rows per expert) kernel is still missing.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_qmv_kernel(
    x_ptr,  # [S, K] activations
    w_ptr,  # [E, N, KW] int32 (uint32 bits)
    s_ptr,  # [E, N, G] scales
    b_ptr,  # [E, N, G] biases
    rhs_ptr,  # [S] expert of each row
    out_ptr,  # [S, N] float32
    N,
    K,
    KW,
    w_se,  # strides (elements) of the expert and row dimensions; the last dimension is contiguous
    w_sn,
    s_se,
    s_sn,
    b_se,
    b_sn,
    BITS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < N
    expert = tl.load(rhs_ptr + row).to(tl.int64)
    rows64 = n[:, None].to(tl.int64)
    w_rows = w_ptr + expert * w_se + rows64 * w_sn
    s_rows = s_ptr + expert * s_se + rows64 * s_sn
    b_rows = b_ptr + expert * b_se + rows64 * b_sn
    code_mask = (1 << BITS) - 1
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        mask = n_mask[:, None] & k_mask[None, :]
        offset = k * BITS
        word = offset // 32
        shift = (offset % 32).to(tl.uint32)
        lo = tl.load(w_rows + word[None, :], mask=mask, other=0).to(tl.uint32, bitcast=True)
        straddles = (shift + BITS) > 32
        next_word = tl.minimum(word + 1, KW - 1)
        hi = tl.load(w_rows + next_word[None, :], mask=mask & straddles[None, :], other=0).to(tl.uint32, bitcast=True)
        # A shift by 32 is undefined: shift by (32 - shift) & 31 and keep it only where the code straddles two words.
        hi = tl.where(straddles[None, :], hi << ((32 - shift) & 31)[None, :], 0)
        code = ((lo >> shift[None, :]) | hi) & code_mask
        group = k // GROUP
        scale = tl.load(s_rows + group[None, :], mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(b_rows + group[None, :], mask=mask, other=0.0).to(tl.float32)
        xk = tl.load(x_ptr + row.to(tl.int64) * K + k, mask=k_mask, other=0.0).to(tl.float32)
        acc += tl.sum((code.to(tl.float32) * scale + bias) * xk[None, :], axis=1)
    tl.store(out_ptr + row.to(tl.int64) * N + n, acc, mask=n_mask)


def gather_qmv(x, w, scales, biases, rhs, group_size, bits, block_n=32, block_k=128):
    """``x [S, K]`` (any float dtype), ``w [E, N, K*bits/32]`` uint32/int32, ``scales``/``biases [E, N, K/group]``,
    ``rhs [S]`` -> ``[S, N]`` in ``x``'s dtype."""
    rows, k = x.shape
    _, n, kw = w.shape
    # The banks are usually strided views into the expert cache's slot buffer: never copy them.
    w = w.view(torch.int32)
    for bank in (w, scales, biases):
        if bank.stride(-1) != 1:
            raise ValueError("gather_qmv: the last dimension of the banks must be contiguous")
    x = x.contiguous()
    rhs = rhs.reshape(-1).contiguous()
    out = torch.empty(rows, n, dtype=torch.float32, device=x.device)
    grid = (rows, triton.cdiv(n, block_n))
    _gather_qmv_kernel[grid](
        x, w, scales, biases, rhs, out, n, k, kw,
        w.stride(0), w.stride(1), scales.stride(0), scales.stride(1), biases.stride(0), biases.stride(1),
        BITS=bits, GROUP=group_size, BLOCK_N=block_n, BLOCK_K=block_k,
    )  # fmt: skip
    return out.to(x.dtype)
