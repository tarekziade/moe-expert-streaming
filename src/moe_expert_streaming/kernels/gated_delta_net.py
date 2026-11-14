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
"""One decode step of a Gated DeltaNet layer (Qwen3-Next / Qwen3.5 / Qwen3.8 linear attention) on MPS, fused.

The PyTorch path runs ~34 ops per layer and token. Here:

1. the four input projections (qkv, z, b, a) are one matmul over their concatenated weights (the layers keep views
   into the merged tensors, so no memory is duplicated);
2. one kernel updates the convolution state, applies the depthwise causal conv + SiLU, and computes the gates
   beta = sigmoid(b) and g = -exp(A_log) * softplus(a + dt_bias);
3. one kernel runs the gated delta rule (q/k L2 norm, decay, delta update, output) on the float32 recurrent state in
   place, then the gated RMSNorm of each head;
4. `out_proj`.

Rounding follows the PyTorch path (bfloat16 between the same steps). Compiled at first use with
`torch.mps.compile_shader`.
"""

import functools

import torch
import torch.nn.functional as F


_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Conv state (C, W) shifts left and takes x; out = silu(T(sum_j state[j] * w[j])). Threads c < H also compute the
// gates of value head c. params: C, W, H.
template <typename T>
kernel void gdn_conv_gates(
    device const T* x [[buffer(0)]],          // projections: qkv (C) then z, b (H), a (H)
    device T* conv_state [[buffer(1)]],       // (C, W), updated in place
    device const T* conv_weight [[buffer(2)]],// (C, W)
    device const float* a_log [[buffer(3)]],  // (H)
    device const float* dt_bias [[buffer(4)]],// (H)
    device T* qkv [[buffer(5)]],              // (C)
    device float* g [[buffer(6)]],            // (H)
    device T* beta [[buffer(7)]],             // (H)
    device const int* params [[buffer(8)]],   // C, W, H, offset of b, offset of a
    uint c [[thread_position_in_grid]]) {
  const uint C = params[0], W = params[1], H = params[2];
  if (c < H) {
    const float b = float(x[params[3] + c]);
    beta[c] = T(1.0f / (1.0f + exp(-b)));
    const float a = float(x[params[4] + c]) + dt_bias[c];
    const float softplus = a > 20.0f ? a : log(1.0f + exp(a));
    g[c] = -exp(a_log[c]) * softplus;
  }
  if (c >= C) return;
  device T* state = conv_state + c * W;
  float acc = 0.0f;
  for (uint j = 0; j + 1 < W; ++j) {
    state[j] = state[j + 1];
    acc += float(state[j]) * float(conv_weight[c * W + j]);
  }
  state[W - 1] = x[c];
  acc += float(x[c]) * float(conv_weight[c * W + W - 1]);
  const float v = float(T(acc));
  qkv[c] = T(v / (1.0f + exp(-v)));
}

// One threadgroup per value head, one thread per value column. dims: Hk, Hv, K, V, key_dim, gate_is_sigmoid,
// offset of z in `projections`. q = qkv[0:key_dim], k = qkv[key_dim:2 key_dim], v = qkv[2 key_dim:]; z: (Hv, V).
template <typename T>
kernel void gdn_delta_norm(
    device const T* qkv [[buffer(0)]],
    device const float* g [[buffer(1)]],
    device const T* beta [[buffer(2)]],
    device float* state [[buffer(3)]],        // (Hv, K, V) float32, in place
    device const T* projections [[buffer(4)]],
    device const T* norm_weight [[buffer(5)]],// (V)
    device T* out [[buffer(6)]],              // (Hv, V)
    device const int* dims [[buffer(7)]],
    device const float* eps [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint Hk = dims[0], Hv = dims[1], K = dims[2], V = dims[3], key_dim = dims[4];
  const bool sigmoid_gate = dims[5] != 0;
  device const T* z = projections + dims[6];
  const uint k_head = head / (Hv / Hk);
  device const T* q = qkv + k_head * K;
  device const T* k = qkv + key_dim + k_head * K;
  device const T* v = qkv + 2 * key_dim + head * V;

  threadgroup float qs[256];
  threadgroup float ks[256];
  threadgroup float partial[3][32];
  float q_sq = 0.0f, k_sq = 0.0f;
  for (uint i = tid; i < K; i += threads) {
    const float qi = float(q[i]), ki = float(k[i]);
    qs[i] = qi;
    ks[i] = ki;
    q_sq += qi * qi;
    k_sq += ki * ki;
  }
  q_sq = simd_sum(q_sq);
  k_sq = simd_sum(k_sq);
  if (lane == 0) {
    partial[0][simd_group] = q_sq;
    partial[1][simd_group] = k_sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  q_sq = 0.0f;
  k_sq = 0.0f;
  const uint groups = (threads + 31) / 32;
  for (uint s = 0; s < groups; ++s) {
    q_sq += partial[0][s];
    k_sq += partial[1][s];
  }
  const float q_scale = rsqrt(q_sq + 1e-6f) * rsqrt(float(K));
  const float k_scale = rsqrt(k_sq + 1e-6f);
  const float decay = exp(g[head]);
  const float beta_t = float(beta[head]);

  // The kernel is launched with V threads: thread `tid` owns value column `tid`.
  const uint j = tid;
  device float* column = state + head * K * V + j;
  float memory = 0.0f;
  for (uint i = 0; i < K; ++i) {
    memory += column[i * V] * decay * ks[i] * k_scale;
  }
  const float delta = (float(v[j]) - memory) * beta_t;
  float o = 0.0f;
  for (uint i = 0; i < K; ++i) {
    const float updated = column[i * V] * decay + ks[i] * k_scale * delta;
    column[i * V] = updated;
    o += updated * qs[i] * q_scale;
  }
  // The reference rounds the attention output to T, then normalizes in float32.
  o = float(T(o));
  float sq = simd_sum(o * o);
  if (lane == 0) partial[2][simd_group] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sq = 0.0f;
  for (uint s = 0; s < groups; ++s) sq += partial[2][s];
  const T normed = T(o * rsqrt(sq / float(V) + eps[0]));
  const T weighted = T(float(norm_weight[j]) * float(normed));
  const float zf = float(z[head * V + j]);
  const float gate = sigmoid_gate ? 1.0f / (1.0f + exp(-zf)) : zf / (1.0f + exp(-zf));
  out[head * V + j] = T(float(weighted) * gate);
}

#define INSTANTIATE(T, NAME)                                                                                       \\
  template [[host_name("gdn_conv_gates_" NAME)]] kernel void gdn_conv_gates<T>(                                    \\
      device const T*, device T*, device const T*, device const float*, device const float*, device T*,            \\
      device float*, device T*, device const int*, uint);                                                          \\
  template [[host_name("gdn_delta_norm_" NAME)]] kernel void gdn_delta_norm<T>(                                    \\
      device const T*, device const float*, device const T*, device float*, device const T*, device const T*,      \\
      device T*, device const int*, device const float*, uint, uint, uint, uint, uint);
INSTANTIATE(bfloat, "bfloat")
INSTANTIATE(half, "half")
INSTANTIATE(float, "float")
"""

_SUFFIX = {torch.bfloat16: "bfloat", torch.float16: "half", torch.float32: "float"}


@functools.cache
def _library():
    return torch.mps.compile_shader(_SOURCE)


def _is_metal_linear(linear):
    return linear.weight.dtype == torch.uint32 and hasattr(linear, "scales")


def _merge_projections(module):
    """Concatenate in_proj_qkv / z / b / a along the output dimension, once; the layers keep views into the result."""
    linears = [module.in_proj_qkv, module.in_proj_z, module.in_proj_b, module.in_proj_a]
    names = ("weight", "scales", "qbiases") if _is_metal_linear(linears[0]) else ("weight",)
    merged = {name: torch.cat([getattr(linear, name).data for linear in linears]) for name in names}
    start = 0
    for linear in linears:
        rows = linear.weight.shape[0]
        for name in names:
            setattr(linear, name, torch.nn.Parameter(merged[name][start : start + rows], requires_grad=False))
        start += rows
    quantized = None
    if len(names) == 3:
        quantized = (linears[0].group_size, linears[0].bits)
    module._metal_in_proj = (merged, quantized)
    module._metal_gate_params = (module.A_log.detach().float(), module.dt_bias.detach().float())


# TODO(portability): Metal-only decode step (MPS); other backends use the PyTorch / hub-kernel path.
def supports(module, hidden_states, cache_params, attention_mask):
    if hidden_states.device.type != "mps" or hidden_states.dtype not in _SUFFIX or torch.is_grad_enabled():
        return False
    if hidden_states.shape[0] != 1 or hidden_states.shape[1] != 1 or attention_mask is not None:
        return False
    if cache_params is None or cache_params.is_recording_past(module.layer_idx):
        return False
    conv_state = cache_params.get_conv_state(module.layer_idx, state_idx=0)
    state = cache_params.get_recurrent_state(module.layer_idx, state_idx=0)
    if conv_state is None or state is None or state.dtype != torch.float32 or not state.is_contiguous():
        return False
    linears = [module.in_proj_qkv, module.in_proj_z, module.in_proj_b, module.in_proj_a]
    if len({_is_metal_linear(linear) for linear in linears}) != 1 or any(
        linear.bias is not None for linear in linears
    ):
        return False
    if _is_metal_linear(linears[0]) and len({(linear.group_size, linear.bits) for linear in linears}) != 1:
        return False
    return (
        conv_state.is_contiguous()
        and conv_state.dtype == hidden_states.dtype
        and module.conv1d.bias is None
        and module.activation in ("silu", "swish")
        and module.norm.activation in ("silu", "swish", "sigmoid")
        and module.head_k_dim <= 256
        and module.head_v_dim <= 1024
        and module.num_v_heads % module.num_k_heads == 0
    )


@functools.cache
def _int_params(values, device):
    return torch.tensor(values, dtype=torch.int32, device=device)


@functools.cache
def _float_params(values, device):
    return torch.tensor(values, dtype=torch.float32, device=device)


def gated_delta_net_step(module, hidden_states, cache_params):
    """`GatedDeltaNet.forward` for one cached decode token; updates the cache states in place."""
    if not hasattr(module, "_metal_in_proj"):
        _merge_projections(module)
    merged, quantized = module._metal_in_proj
    library, suffix = _library(), _SUFFIX[hidden_states.dtype]
    device = hidden_states.device
    x = hidden_states.reshape(1, -1)
    if quantized is not None:
        from ..metal_linear import load_metal_kernel

        group_size, bits = quantized
        projections = load_metal_kernel().quantized_matmul(
            x, merged["weight"], merged["scales"], merged["qbiases"], transpose=True, group_size=group_size, bits=bits
        )
    else:
        projections = F.linear(x, merged["weight"])
    projections = projections.reshape(-1)
    conv_dim, value_dim, heads = module.conv_dim, module.value_dim, module.num_v_heads
    conv_state = cache_params.get_conv_state(module.layer_idx, state_idx=0)
    state = cache_params.get_recurrent_state(module.layer_idx, state_idx=0)
    a_log, dt_bias = module._metal_gate_params
    qkv = torch.empty(conv_dim, dtype=x.dtype, device=device)
    g = torch.empty(heads, dtype=torch.float32, device=device)
    beta = torch.empty(heads, dtype=x.dtype, device=device)
    width = module.conv_kernel_size
    getattr(library, f"gdn_conv_gates_{suffix}")(
        projections,
        conv_state,
        module.conv1d.weight.reshape(conv_dim, width),
        a_log,
        dt_bias,
        qkv,
        g,
        beta,
        _int_params((conv_dim, width, heads, conv_dim + value_dim, conv_dim + value_dim + heads), device),
        threads=conv_dim,
    )
    out = torch.empty(value_dim, dtype=x.dtype, device=device)
    sigmoid_gate = int(module.norm.activation == "sigmoid")
    getattr(library, f"gdn_delta_norm_{suffix}")(
        qkv,
        g,
        beta,
        state,
        projections,
        module.norm.weight,
        out,
        _int_params(
            (module.num_k_heads, heads, module.head_k_dim, module.head_v_dim, module.key_dim, sigmoid_gate, conv_dim),
            device,
        ),
        _float_params((module.norm.variance_epsilon,), device),
        threads=heads * module.head_v_dim,
        group_size=module.head_v_dim,
    )
    return module.out_proj(out.reshape(1, 1, value_dim))
