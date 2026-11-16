"""Gated RMSNorm (``Qwen4ExpTextRMSNormGated``, the Gated DeltaNet output norm) as one Metal dispatch on MPS.

The reference runs ~8 PyTorch ops over the ``(tokens * value_heads, head_dim)`` output (9 ms for 2,048 tokens of
Qwen3.8-Flash-Next on an M5); here one SIMD group normalizes one row, with the reference's roundings: the normalized row
is rounded to the input dtype, multiplied by the weight in that dtype, then by ``silu(gate)`` in float32 and rounded.
"""

import functools

import torch


_SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void gated_rms_norm(
    device const T* x [[buffer(0)]],
    device const T* gate [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device T* out [[buffer(3)]],
    device const float* params [[buffer(4)]],  // rows, width, eps
    uint row [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]]) {
  const uint r = row * simd_groups + simd_group;
  if (r >= uint(params[0])) return;
  const uint width = uint(params[1]);
  const ulong base = ulong(r) * width;
  float sum = 0.0f;
  for (uint i = lane; i < width; i += 32) {
    const float v = float(x[base + i]);
    sum += v * v;
  }
  const float scale = rsqrt(simd_sum(sum) / float(width) + params[2]);
  for (uint i = lane; i < width; i += 32) {
    const T normed = T(float(x[base + i]) * scale);
    const T weighted = T(float(weight[i]) * float(normed));
    const float z = float(gate[base + i]);
    out[base + i] = T(float(weighted) * (z / (1.0f + exp(-z))));
  }
}

template [[host_name("gated_rms_norm_bfloat")]] kernel void gated_rms_norm<bfloat>(
    device const bfloat*, device const bfloat*, device const bfloat*, device bfloat*, device const float*, uint, uint,
    uint, uint);
template [[host_name("gated_rms_norm_half")]] kernel void gated_rms_norm<half>(
    device const half*, device const half*, device const half*, device half*, device const float*, uint, uint, uint,
    uint);
"""

_SUFFIX = {torch.bfloat16: "bfloat", torch.float16: "half"}
ROWS_PER_GROUP = 8


@functools.cache
def _library():
    return torch.mps.compile_shader(_SOURCE)


@functools.cache
def _params(rows, width, eps, device):
    return torch.tensor([rows, width, eps], dtype=torch.float32, device=device)


def supports(module, hidden_states, gate):
    return (
        hidden_states.device.type == "mps"
        and hidden_states.dtype in _SUFFIX
        and gate.dtype == hidden_states.dtype
        and module.weight.dtype == hidden_states.dtype
        and getattr(module, "activation", "silu") == "silu"
        and gate.shape == hidden_states.shape
        and not torch.is_grad_enabled()
    )


def gated_rms_norm(module, hidden_states, gate):
    width = hidden_states.shape[-1]
    x = hidden_states.contiguous()
    rows = x.numel() // width
    out = torch.empty_like(x)
    getattr(_library(), f"gated_rms_norm_{_SUFFIX[x.dtype]}")(
        x,
        gate.contiguous(),
        module.weight,
        out,
        _params(rows, width, module.variance_epsilon, x.device),
        threads=-(-rows // ROWS_PER_GROUP) * ROWS_PER_GROUP * 32,
        group_size=ROWS_PER_GROUP * 32,
    )
    return out
