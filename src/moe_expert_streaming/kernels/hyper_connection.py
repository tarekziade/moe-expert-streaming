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
"""Metal kernels for a gated hyper-connection (Qwen3.8's `GatedResidual`) on MPS.

The reference runs ~25 small ops per call around its three projections; here the group RMSNorm, the SiLU of the
low-rank code and the stream mixing (with the injection weights) are one dispatch each, and the projections keep
their own (quantized) matmuls. Compiled at first use with `torch.mps.compile_shader`.
"""

import functools
import os

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

// Decode (one token), 8-bit MLX affine projections with groups of 64 (weights [rows, cols / 4] uint32, float32 scales
// and biases [rows, cols / 64], rounded to T as MetalLinear does). K = count * hidden inputs.
//
// hc_down_q8: one simdgroup per row of the fused down (lowrank rows) + inject (count rows) projection. Each threadgroup
// computes the group RMSNorm of x into threadgroup memory (normed rounded to T, as hc_norm), then its rows' dot products.
// Inject rows also give injection = 2 * sigmoid(T(inject / count)). Threadgroup 0 stores the per-stream norm scales.
template <typename T>
kernel void hc_down_q8(
    device const T* x [[buffer(0)]],
    device const T* norm_weight [[buffer(1)]],
    device const uint* w [[buffer(2)]],
    device const T* scales [[buffer(3)]],
    device const T* biases [[buffer(4)]],
    device T* down [[buffer(5)]],
    device T* injection [[buffer(6)]],
    device float* stream_scale [[buffer(7)]],
    device const float* params [[buffer(8)]],  // hidden, eps, count, lowrank
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint hidden = uint(params[0]), count = uint(params[2]), lowrank = uint(params[3]);
  const uint K = count * hidden, rows = lowrank + count;
  threadgroup T normed[4 * 2560];
  threadgroup float partial[4][8];
  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint k = tid; k < K; k += 256) {
    const float v = float(x[k]);
    sums[k / hidden] += v * v;
  }
  for (uint s = 0; s < count; ++s) {
    const float total = simd_sum(sums[s]);
    if (lane == 0) partial[s][simd_group] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale[4];
  for (uint s = 0; s < count; ++s) {
    float total = 0.0f;
    for (uint g = 0; g < 8; ++g) total += partial[s][g];
    scale[s] = rsqrt(total / float(hidden) + params[1]);
  }
  if (group == 0 && tid < count) stream_scale[tid] = scale[tid];
  for (uint k = tid; k < K; k += 256) {
    normed[k] = T(float(x[k]) * scale[k / hidden] * (1.0f + float(norm_weight[k])));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row = group * 8 + simd_group;
  if (row >= rows) return;
  const uint words = K / 4, groups = K / 64;
  device const uint* wr = w + ulong(row) * words;
  device const T* sr = scales + ulong(row) * groups;
  device const T* br = biases + ulong(row) * groups;
  float acc = 0.0f;
  // Consecutive words across lanes (conflict-free threadgroup reads, coalesced weight reads).
  for (uint word_index = lane; word_index < words; word_index += 32) {
    const uint g = word_index / 16;
    const float sc = float(sr[g]), bi = float(br[g]);
    const uint word = wr[word_index];
    const uint k = word_index * 4;
    float dot = 0.0f, total = 0.0f;
    for (uint b = 0; b < 4; ++b) {
      const float n = float(normed[k + b]);
      dot += float((word >> (8 * b)) & 0xffu) * n;
      total += n;
    }
    acc += sc * dot + bi * total;
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    if (row < lowrank) {
      down[row] = T(acc);
    } else {
      const float z = float(T(float(T(acc)) / float(count)));
      injection[row - lowrank] = T(2.0f * float(T(1.0f / (1.0f + exp(-z)))));
    }
  }
}

// hc_up_mix_q8: code = silu(T(down / count)) (as hc_act), u = up(code) (rows s * hidden + h), and
// mixed[h] = mean_s T(sigmoid(u[s, h]) * normed[s, h]) with normed recomputed from x and the stream scales.
// One simdgroup per h (its count rows), 8 per threadgroup.
template <typename T>
kernel void hc_up_mix_q8(
    device const T* down [[buffer(0)]],
    device const T* x [[buffer(1)]],
    device const T* norm_weight [[buffer(2)]],
    device const float* stream_scale [[buffer(3)]],
    device const uint* w [[buffer(4)]],
    device const T* scales [[buffer(5)]],
    device const T* biases [[buffer(6)]],
    device T* mixed [[buffer(7)]],
    device const float* params [[buffer(8)]],  // hidden, eps, count, lowrank
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint hidden = uint(params[0]), count = uint(params[2]), lowrank = uint(params[3]);
  threadgroup float code[512];
  for (uint i = tid; i < lowrank; i += 256) {
    const float v = float(T(float(down[i]) / float(count)));
    code[i] = float(T(v / (1.0f + exp(-v))));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint h = group * 8 + simd_group;
  if (h >= hidden) return;
  const uint words = lowrank / 4, groups = lowrank / 64;
  float acc_mix = 0.0f;
  for (uint s = 0; s < count; ++s) {
    const uint row = s * hidden + h;
    device const uint* wr = w + ulong(row) * words;
    float acc = 0.0f;
    for (uint word_index = lane; word_index < words; word_index += 32) {
      const uint g = word_index / 16;
      const float sc = float(scales[ulong(row) * groups + g]), bi = float(biases[ulong(row) * groups + g]);
      const uint word = wr[word_index];
      const uint k = word_index * 4;
      for (uint b = 0; b < 4; ++b) acc += (sc * float((word >> (8 * b)) & 0xffu) + bi) * code[k + b];
    }
    acc = simd_sum(acc);
    const T u = T(acc);
    const T gate = T(1.0f / (1.0f + exp(-float(u))));
    const ulong k = ulong(s) * hidden + h;
    const T n = T(float(x[k]) * stream_scale[s] * (1.0f + float(norm_weight[k])));
    acc_mix += float(T(float(gate) * float(n)));
  }
  if (lane == 0) mixed[h] = T(acc_mix / float(count));
}

// hc_down_q8_split: the fused down + inject matvec over normed (from hc_norm, in device memory); 4 simdgroups per row
// (K split in 4), 2 rows per threadgroup. Inject rows give injection as in hc_down_q8.
template <typename T>
kernel void hc_down_q8_split(
    device const T* normed [[buffer(0)]],
    device const uint* w [[buffer(1)]],
    device const T* scales [[buffer(2)]],
    device const T* biases [[buffer(3)]],
    device T* down [[buffer(4)]],
    device T* injection [[buffer(5)]],
    device const float* params [[buffer(6)]],  // hidden, eps, count, lowrank
    uint group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint hidden = uint(params[0]), count = uint(params[2]), lowrank = uint(params[3]);
  const uint K = count * hidden, rows = lowrank + count, words = K / 4, groups = K / 64;
  const uint row = group * 2 + simd_group / 4, part = simd_group % 4;
  threadgroup float partial[8];
  float acc = 0.0f;
  if (row < rows) {
    device const uint* wr = w + ulong(row) * words;
    device const T* sr = scales + ulong(row) * groups;
    device const T* br = biases + ulong(row) * groups;
    const uint begin = part * (words / 4), end = begin + words / 4;
    for (uint word_index = begin + lane; word_index < end; word_index += 32) {
      const uint g = word_index / 16;
      const float sc = float(sr[g]), bi = float(br[g]);
      const uint word = wr[word_index];
      const uint k = word_index * 4;
      float dot = 0.0f, total = 0.0f;
      for (uint b = 0; b < 4; ++b) {
        const float n = float(normed[k + b]);
        dot += float((word >> (8 * b)) & 0xffu) * n;
        total += n;
      }
      acc += sc * dot + bi * total;
    }
  }
  acc = simd_sum(acc);
  if (lane == 0) partial[simd_group] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (part != 0 || lane != 0 || row >= rows) return;
  const uint first = (simd_group / 4) * 4;
  acc = partial[first] + partial[first + 1] + partial[first + 2] + partial[first + 3];
  if (row < lowrank) {
    down[row] = T(acc);
  } else {
    const float z = float(T(float(T(acc)) / float(count)));
    injection[row - lowrank] = T(2.0f * float(T(1.0f / (1.0f + exp(-z)))));
  }
}

// hc_up_mix_q8n: hc_up_mix_q8 reading normed from device memory (from hc_norm) instead of recomputing it.
template <typename T>
kernel void hc_up_mix_q8n(
    device const T* down [[buffer(0)]],
    device const T* normed [[buffer(1)]],
    device const uint* w [[buffer(2)]],
    device const T* scales [[buffer(3)]],
    device const T* biases [[buffer(4)]],
    device T* mixed [[buffer(5)]],
    device const float* params [[buffer(6)]],  // hidden, eps, count, lowrank
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint hidden = uint(params[0]), count = uint(params[2]), lowrank = uint(params[3]);
  threadgroup float code[512];
  for (uint i = tid; i < lowrank; i += 256) {
    const float v = float(T(float(down[i]) / float(count)));
    code[i] = float(T(v / (1.0f + exp(-v))));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint h = group * 8 + simd_group;
  if (h >= hidden) return;
  const uint words = lowrank / 4, groups = lowrank / 64;
  float acc_mix = 0.0f;
  for (uint s = 0; s < count; ++s) {
    const uint row = s * hidden + h;
    device const uint* wr = w + ulong(row) * words;
    float acc = 0.0f;
    for (uint word_index = lane; word_index < words; word_index += 32) {
      const uint g = word_index / 16;
      const float sc = float(scales[ulong(row) * groups + g]), bi = float(biases[ulong(row) * groups + g]);
      const uint word = wr[word_index];
      const uint k = word_index * 4;
      for (uint b = 0; b < 4; ++b) acc += (sc * float((word >> (8 * b)) & 0xffu) + bi) * code[k + b];
    }
    acc = simd_sum(acc);
    const T u = T(acc);
    const T gate = T(1.0f / (1.0f + exp(-float(u))));
    acc_mix += float(T(float(gate) * float(normed[ulong(s) * hidden + h])));
  }
  if (lane == 0) mixed[h] = T(acc_mix / float(count));
}

#define INSTANTIATE(T, NAME)                                                                                    \\
  template [[host_name("hc_norm_" NAME)]] kernel void hc_norm<T>(                                               \\
      device const T*, device const T*, device T*, device const float*, uint, uint, uint, uint, uint);          \\
  template [[host_name("hc_act_" NAME)]] kernel void hc_act<T>(device const T*, device T*, device const float*, \\
                                                               uint);                                           \\
  template [[host_name("hc_mix_" NAME)]] kernel void hc_mix<T>(device const T*, device const T*, device const T*, \\
                                                               device T*, device T*, device const float*, uint); \
  template [[host_name("hc_down_q8_" NAME)]] kernel void hc_down_q8<T>(                                          \
      device const T*, device const T*, device const uint*, device const T*, device const T*, device T*, \
      device T*, device float*, device const float*, uint, uint, uint, uint);                                   \
  template [[host_name("hc_up_mix_q8_" NAME)]] kernel void hc_up_mix_q8<T>(                                      \
      device const T*, device const T*, device const T*, device const float*, device const uint*,               \
      device const T*, device const T*, device T*, device const float*, uint, uint, uint, uint);
#define INSTANTIATE_SPLIT(T, NAME)                                                                              \
  template [[host_name("hc_down_q8_split_" NAME)]] kernel void hc_down_q8_split<T>(                              \
      device const T*, device const uint*, device const T*, device const T*, device T*, device T*,              \
      device const float*, uint, uint, uint);                                                                   \
  template [[host_name("hc_up_mix_q8n_" NAME)]] kernel void hc_up_mix_q8n<T>(                                    \
      device const T*, device const T*, device const uint*, device const T*, device const T*, device T*,         \
      device const float*, uint, uint, uint, uint);
INSTANTIATE_SPLIT(bfloat, "bfloat")
INSTANTIATE_SPLIT(half, "half")
INSTANTIATE_SPLIT(float, "float")
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


def _q8(layer, dtype):
    """``(weight, scales, biases)`` of an 8-bit, group-64 MetalLinear without bias, scales and biases in ``dtype``
    (converted once and cached when stored in float32, as MetalLinear rounds them per call), else None."""
    from transformers.integrations.metal_quantization import MetalLinear

    if not (
        isinstance(layer, MetalLinear)
        and layer.bits == 8
        and layer.group_size == 64
        and layer.bias is None
        and layer.weight.dtype == torch.uint32
        and layer.scales.dtype in (torch.float32, dtype)
    ):
        return None
    if layer.scales.dtype == dtype:
        return layer.weight, layer.scales, layer.qbiases
    cached = layer.__dict__.get("_q8_rounded")
    if cached is None or cached[0].dtype != dtype:
        cached = layer.__dict__["_q8_rounded"] = (layer.scales.to(dtype), layer.qbiases.to(dtype))
    return (layer.weight, *cached)


def fused_step_supported(module, hyper_input):
    """The two-kernel decode step: one token, count <= 4, hidden <= 2560, lowrank <= 512 and a multiple of 64, 8-bit
    group-64 projections (the fused down + inject one included)."""
    count, hidden = module.hc_count, module.hidden_size
    lowrank = module.input_mix_weight_down.out_features
    return (
        os.environ.get("MOE_STREAMING_HC_FUSED", "1") == "1"
        and hyper_input.numel() == count * hidden
        and count <= 4
        and hidden <= 2560
        and hidden % 64 == 0
        and lowrank <= 512
        and lowrank % 64 == 0
        and _fused_projections(module) is not None
        and _q8(module.input_mix_weight_up, hyper_input.dtype) is not None
        and _fused_projections(module)[1:3] == (8, 64)
        and module.hc_norm.weight.dtype == hyper_input.dtype
    )


def fused_step(module, hyper_input):
    """`GatedResidual.forward` for one token in two dispatches (see hc_down_q8 / hc_up_mix_q8)."""
    library, suffix = _library(), _SUFFIX[hyper_input.dtype]
    count, hidden = module.hc_count, module.hidden_size
    device, dtype = hyper_input.device, hyper_input.dtype
    (weight, scales, biases), _, _, lowrank = _fused_projections(module)
    if scales.dtype != dtype:
        cached = module.__dict__.get("_hc_fused_rounded")
        if cached is None or cached[0].dtype != dtype:
            cached = module.__dict__["_hc_fused_rounded"] = (scales.to(dtype), biases.to(dtype))
        scales, biases = cached
    x = hyper_input.contiguous()
    params = _params((float(hidden), module.hc_norm.eps, float(count), float(lowrank)), device)
    down = torch.empty(lowrank, dtype=dtype, device=device)
    injection = torch.empty(*x.shape[:-1], count, dtype=dtype, device=device)
    stream_scale = torch.empty(count, dtype=torch.float32, device=device)
    rows = lowrank + count
    up_weight, up_scales, up_biases = _q8(module.input_mix_weight_up, dtype)
    mixed = torch.empty(*x.shape[:-1], hidden, dtype=dtype, device=device)
    if os.environ.get("MOE_STREAMING_HC_FUSED_SPLIT", "1") == "1":
        normed = torch.empty_like(x)
        getattr(library, f"hc_norm_{suffix}")(
            x,
            module.hc_norm.weight,
            normed,
            _params((float(hidden), module.hc_norm.eps, float(count)), device),
            threads=count * 256,
            group_size=256,
        )
        getattr(library, f"hc_down_q8_split_{suffix}")(
            normed, weight, scales, biases, down, injection, params, threads=-(-rows // 2) * 256, group_size=256
        )
        getattr(library, f"hc_up_mix_q8n_{suffix}")(
            down,
            normed,
            up_weight,
            up_scales,
            up_biases,
            mixed,
            params,
            threads=-(-hidden // 8) * 256,
            group_size=256,
        )
        return mixed, hyper_input, injection
    getattr(library, f"hc_down_q8_{suffix}")(
        x,
        module.hc_norm.weight,
        weight,
        scales,
        biases,
        down,
        injection,
        stream_scale,
        params,
        threads=-(-rows // 8) * 256,
        group_size=256,
    )
    getattr(library, f"hc_up_mix_q8_{suffix}")(
        down,
        x,
        module.hc_norm.weight,
        stream_scale,
        up_weight,
        up_scales,
        up_biases,
        mixed,
        params,
        threads=-(-hidden // 8) * 256,
        group_size=256,
    )
    return mixed, hyper_input, injection


def gated_residual_step(module, hyper_input):
    """`GatedResidual.forward` for a single token: returns ``(mixed_input, hyper_input, injection_weights)``."""
    if fused_step_supported(module, hyper_input):
        return fused_step(module, hyper_input)
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
