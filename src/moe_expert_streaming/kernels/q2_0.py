"""Q2_0 (Strata's ggml fork, type 42) for routed experts: a gather-gemv Metal kernel and a PyTorch dequantize.

A Q2_0 block is 64 values in 18 bytes: an fp16 scale ``d``, then 16 bytes of 2-bit codes (value ``j`` in byte ``j / 4``,
bits ``2 * (j % 4)``); value = ``d * (q - 1)``. The ggml Hub kernels have no Metal gemv for it, and Strata stores 30 of
Qwen3.8-Flash-Next's 48 expert down projections in it.
"""

import functools

import torch


GGML_TYPE = 42
BLOCK_VALUES, BLOCK_BYTES = 64, 18

_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// out[r, n] = sum_k x[r, k] * W[ids[r], n, k], W in Q2_0 rows of `row_bytes`.
// A threadgroup of 8 SIMD groups covers 32 output rows of one route; a SIMD group 4 rows; a lane 8 values at a time.
template <typename T>
kernel void q2_0_mul_mat_id(
    device const uchar* blocks [[buffer(0)]],
    device const T* x [[buffer(1)]],
    device const int* ids [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const uint* shape [[buffer(4)]],  // N (rows per expert), K (inputs), row_bytes, tiles (= ceil(N / 32))
    uint group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint N = shape[0], K = shape[1], row_bytes = shape[2], tiles = shape[3];
  const uint route = group / tiles;
  const uint first = (group % tiles) * 32 + simd_group * 4;
  device const T* xr = x + ulong(route) * K;
  device const uchar* expert = blocks + ulong(ids[route]) * N * row_bytes;
  const uint units = K / 8;  // 8 values (one ushort of codes) per unit
  for (uint row = first; row < min(first + 4, N); ++row) {
    device const uchar* w = expert + ulong(row) * row_bytes;
    float sum = 0.0f;
    for (uint i = lane; i < units; i += 32) {
      const uint block = i >> 3, part = i & 7;
      device const uchar* b = w + block * 18;
      const float d = float(*(device const half*)b);
      const uint q = *(device const ushort*)(b + 2 + 2 * part);
      device const T* xs = xr + block * 64 + part * 8;
      float dot = 0.0f, total = 0.0f;
      for (uint j = 0; j < 8; ++j) {
        const float v = float(xs[j]);
        dot += float((q >> (2 * j)) & 3) * v;
        total += v;
      }
      sum += d * (dot - total);
    }
    sum = simd_sum(sum);
    if (lane == 0) out[ulong(route) * N + row] = sum;
  }
}

// One thread per block (64 values): out[block * 64 + j] = d * (q_j - 1); blocks are contiguous 18-byte records.
template <typename T>
kernel void q2_0_dequantize(
    device const uchar* blocks [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint block [[thread_position_in_grid]]) {
  device const uchar* b = blocks + ulong(block) * 18;
  const float d = float(*(device const half*)b);
  device T* o = out + ulong(block) * 64;
  for (uint i = 0; i < 16; ++i) {
    const uint q = b[2 + i];
    for (uint j = 0; j < 4; ++j) o[4 * i + j] = T(d * (float((q >> (2 * j)) & 3) - 1.0f));
  }
}

#define INSTANTIATE(T, NAME) \\
  template [[host_name("q2_0_mul_mat_id_" NAME)]] kernel void q2_0_mul_mat_id<T>( \\
      device const uchar*, device const T*, device const int*, device float*, device const uint*, uint, uint, uint); \\
  template [[host_name("q2_0_dequantize_" NAME)]] kernel void q2_0_dequantize<T>(device const uchar*, device T*, uint);
INSTANTIATE(float, "float")
INSTANTIATE(bfloat, "bfloat")
INSTANTIATE(half, "half")
"""

_SUFFIX = {torch.bfloat16: "bfloat", torch.float16: "half", torch.float32: "float"}


@functools.cache
def _library():
    return torch.mps.compile_shader(_SOURCE)


@functools.cache
def _shape(out_features, in_features, row_bytes, tiles, device):
    return torch.tensor([out_features, in_features, row_bytes, tiles], dtype=torch.int32, device=device)


def mul_mat_id(blocks, x, ids, out_features):
    """``out[r] = x[r] @ W[ids[r]].T`` for ``blocks`` ``(experts, out_features, row_bytes)`` uint8 Q2_0 rows, ``x``
    ``(routes, in_features)``, ``ids`` ``(routes,)``. Returns ``(routes, out_features)`` float32."""
    routes, in_features = x.shape
    if blocks.device.type != "mps":
        weights = dequantize(blocks[ids], in_features, torch.float32)
        return torch.bmm(weights, x.float()[:, :, None]).squeeze(-1)
    tiles = (out_features + 31) // 32
    out = torch.empty(routes, out_features, dtype=torch.float32, device=x.device)
    shape = _shape(out_features, in_features, blocks.shape[-1], tiles, x.device)
    getattr(_library(), f"q2_0_mul_mat_id_{_SUFFIX[x.dtype]}")(
        blocks,
        x.contiguous(),
        ids.to(torch.int32).contiguous(),
        out,
        shape,
        threads=routes * tiles * 256,
        group_size=256,
    )
    return out


def dequantize(blocks, in_features, dtype):
    """``(..., rows, row_bytes)`` uint8 Q2_0 rows -> ``(..., rows, in_features)`` of ``dtype``."""
    if blocks.device.type == "mps":
        out = torch.empty(*blocks.shape[:-1], in_features, dtype=dtype, device=blocks.device)
        count = out.numel() // BLOCK_VALUES
        getattr(_library(), f"q2_0_dequantize_{_SUFFIX[dtype]}")(blocks.contiguous(), out, threads=count)
        return out
    raw = blocks.reshape(*blocks.shape[:-1], in_features // BLOCK_VALUES, BLOCK_BYTES)
    scale = raw[..., :2].contiguous().view(torch.float16).to(dtype)  # (..., blocks, 1)
    codes = raw[..., 2:].to(torch.int16)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int16, device=blocks.device)
    values = ((codes[..., None] >> shifts) & 3).to(dtype) - 1  # (..., blocks, 16, 4)
    return (values.flatten(-2) * scale).flatten(-2)
