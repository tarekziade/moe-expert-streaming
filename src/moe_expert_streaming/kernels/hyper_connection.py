# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Metal kernels for one decode step of a gated hyper-connection (Qwen3.8's `GatedResidual`) on MPS.

The reference runs ~25 small ops per call around its three projections; here the group RMSNorm, the SiLU of the
low-rank code and the stream mixing (with the injection weights) are one dispatch each, and the projections keep
their own (quantized) matmuls. Compiled at first use with `torch.mps.compile_shader`.
"""

import functools

import torch


_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// One threadgroup per (token, stream) row: normed[r, h] = x[r, h] * rsqrt(mean(x[r, :]^2) + eps) * (1 + weight[r % count, h]).
template <typename T>
kernel void hc_norm(
    device const T* x [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* normed [[buffer(2)]],
    device const float* params [[buffer(3)]],  // hidden, eps, count
    uint stream [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint hidden = uint(params[0]);
  const uint base = stream * hidden, weight_base = (stream % uint(params[2])) * hidden;
  threadgroup float partial[32];
  float sum = 0.0f;
  for (uint h = tid; h < hidden; h += threads) {
    const float v = float(x[base + h]);
    sum += v * v;
  }
  sum = simd_sum(sum);
  if (lane == 0) partial[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum = 0.0f;
  for (uint s = 0; s < (threads + 31) / 32; ++s) sum += partial[s];
  const float scale = rsqrt(sum / float(hidden) + params[1]);
  for (uint h = tid; h < hidden; h += threads) {
    normed[base + h] = T(float(x[base + h]) * scale * (1.0f + float(weight[weight_base + h])));
  }
}

// a = silu(T(d / count)), rounded like the reference (bf16 division, then SiLU).
template <typename T>
kernel void hc_act(
    device const T* d [[buffer(0)]],
    device T* a [[buffer(1)]],
    device const float* params [[buffer(2)]],  // count
    uint i [[thread_position_in_grid]]) {
  const float v = float(T(float(d[i]) / params[0]));
  a[i] = T(v / (1.0f + exp(-v)));
}

// Per token t: mixed[t, h] = mean_s sigmoid(u[t, s, h]) * normed[t, s, h]; injection[t, s] = 2 * sigmoid(inject[t, s] / count).
// One thread per (t, h).
template <typename T>
kernel void hc_mix(
    device const T* u [[buffer(0)]],
    device const T* normed [[buffer(1)]],
    device const T* inject [[buffer(2)]],
    device T* mixed [[buffer(3)]],
    device T* injection [[buffer(4)]],
    device const float* params [[buffer(5)]],  // hidden, count
    uint gid [[thread_position_in_grid]]) {
  const uint hidden = uint(params[0]), count = uint(params[1]);
  const uint t = gid / hidden, h = gid % hidden;
  if (h < count) {
    const float z = float(T(float(inject[t * count + h]) / params[1]));
    injection[t * count + h] = T(2.0f * float(T(1.0f / (1.0f + exp(-z)))));
  }
  const ulong base = ulong(t) * count * hidden;
  float acc = 0.0f;
  for (uint s = 0; s < count; ++s) {
    const T gate = T(1.0f / (1.0f + exp(-float(u[base + s * hidden + h]))));
    acc += float(T(float(gate) * float(normed[base + s * hidden + h])));
  }
  mixed[ulong(t) * hidden + h] = T(acc / float(count));
}

#define INSTANTIATE(T, NAME)                                                                                    \\
  template [[host_name("hc_norm_" NAME)]] kernel void hc_norm<T>(                                               \\
      device const T*, device const T*, device T*, device const float*, uint, uint, uint, uint, uint);          \\
  template [[host_name("hc_act_" NAME)]] kernel void hc_act<T>(device const T*, device T*, device const float*, \\
                                                               uint);                                           \\
  template [[host_name("hc_mix_" NAME)]] kernel void hc_mix<T>(device const T*, device const T*, device const T*, \\
                                                               device T*, device T*, device const float*, uint);
INSTANTIATE(bfloat, "bfloat")
INSTANTIATE(half, "half")
INSTANTIATE(float, "float")
"""

_SUFFIX = {torch.bfloat16: "bfloat", torch.float16: "half", torch.float32: "float"}


@functools.cache
def _library():
    return torch.mps.compile_shader(_SOURCE)


@functools.cache
def _params(values, device):
    return torch.tensor(values, dtype=torch.float32, device=device)


# TODO(portability): Metal-only decode step (MPS), including the fused down/inject projection; other backends use
# the PyTorch path.
def supports(module, hyper_input):
    """Any number of tokens on MPS, with the combine projection (`use_combine`)."""
    return (
        hyper_input.device.type == "mps"
        and hyper_input.numel() % (module.hc_count * module.hidden_size) == 0
        and hyper_input.dtype in _SUFFIX
        and module.block_inject_weight is not None
        and module.hc_norm.group_size == module.hidden_size
        and not torch.is_grad_enabled()
    )


def _fused_projections(module):
    """``input_mix_weight_down`` and ``block_inject_weight`` as one quantized matmul: both read the normalized input,
    and on its own the 4-row ``block_inject_weight`` costs twice the 320-row projection (49 vs 25 us on the M5), while
    the fused 324-row call costs as much as the 320-row one. Each output row is quantized on its own, so the results are
    bit-exact. The two layers' tensors become row views of the fused ones: no memory is added.

    Returns ``((weight, scales, biases), bits, group_size, rows of the first)``, or ``None`` when the two are not
    MetalLinear layers with the same quantization."""
    if "_hc_fused" not in module.__dict__:
        from transformers.integrations.metal_quantization import MetalLinear

        down, inject, fused = module.input_mix_weight_down, module.block_inject_weight, None
        if (
            isinstance(down, MetalLinear)
            and isinstance(inject, MetalLinear)
            and down.weight.dtype == inject.weight.dtype == torch.uint32
            and (down.bits, down.group_size) == (inject.bits, inject.group_size)
            and down.bias is None
            and inject.bias is None
        ):
            tensors = []
            for name in ("weight", "scales", "qbiases"):
                first, second = getattr(down, name), getattr(inject, name)
                both = torch.cat([first.data, second.data])
                first.data, second.data = both[: first.shape[0]], both[first.shape[0] :]
                tensors.append(both)
            fused = (tuple(tensors), down.bits, down.group_size, down.out_features)
        module.__dict__["_hc_fused"] = fused
    return module.__dict__["_hc_fused"]


def gated_residual_step(module, hyper_input):
    """`GatedResidual.forward` for a single token: returns ``(mixed_input, hyper_input, injection_weights)``."""
    library, suffix = _library(), _SUFFIX[hyper_input.dtype]
    count, hidden = module.hc_count, module.hidden_size
    device = hyper_input.device
    x = hyper_input.contiguous()
    tokens = x.numel() // (count * hidden)
    normed = torch.empty_like(x)
    getattr(library, f"hc_norm_{suffix}")(
        x,
        module.hc_norm.weight.to(x.dtype),
        normed,
        _params((float(hidden), module.hc_norm.eps, float(count)), device),
        threads=tokens * count * 256,
        group_size=256,
    )
    # The fused call uses quantized_matmul in the input dtype: correct for one row, but wrong on M5 GPUs for bf16/f16
    # inputs with many rows (kernels-community#1332); MetalLinear.forward computes those in float32.
    fused = _fused_projections(module) if tokens == 1 else None
    if fused is not None:
        from ..metal_linear import load_metal_kernel

        (weight, scales, biases), bits, group_size, rows = fused
        both = load_metal_kernel().quantized_matmul(
            normed, weight, scales.to(x.dtype), biases.to(x.dtype), transpose=True, group_size=group_size, bits=bits
        )
        down, inject = both[..., :rows].contiguous(), both[..., rows:].contiguous()
    else:
        down, inject = module.input_mix_weight_down(normed), module.block_inject_weight(normed)
    code = torch.empty_like(down)
    getattr(library, f"hc_act_{suffix}")(down, code, _params((float(count),), device), threads=down.numel())
    up = module.input_mix_weight_up(code)
    mixed = torch.empty(*x.shape[:-1], hidden, dtype=x.dtype, device=device)
    injection = torch.empty_like(inject)
    getattr(library, f"hc_mix_{suffix}")(
        up, normed, inject, mixed, injection, _params((float(hidden), float(count)), device), threads=tokens * hidden
    )
    return mixed, hyper_input, injection
