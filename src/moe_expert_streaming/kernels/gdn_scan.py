"""The gated delta rule over a prompt chunk as one Metal scan (MPS), after Strata's ``gdn_scan_r`` (MIT).

Only the state update is sequential, and the columns of a value head's ``(key, value)`` state are independent in it:
column ``j`` needs ``k``, ``q``, ``v_j``, the gate and beta. So a lane owns a column, the 128 key rows of the column are
split over ``IB`` simdgroups (the state lives in registers), the partial dots are summed through threadgroup memory once
per token, and ``q``/``k``/``v`` are staged ``TB`` tokens at a time. Per token, with ``S`` the old state::

    a = S^T k, b = S^T q, delta = (v - exp(g) a) beta, o = exp(g) b + (k . q) delta, S = exp(g) S + k delta^T

the reference's recurrence (``o = S_new^T q``) with different rounding. :func:`prep_qk` L2-normalizes q/k (q scaled by
``1/sqrt(d)``) from rows of the conv output, and :func:`conv_silu` writes that output token-major; value head ``h`` reads key head ``h // (Hv / Hk)`` as the reference's
``repeat_interleave``.
"""

import functools

import torch


_SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <int IB, int TB, typename VT>
kernel void gdn_scan(
    device float* state [[buffer(0)]],          // [B, Hv, 128 (key i), 128 (value j)], updated in place
    device const float4* qn [[buffer(1)]],      // [B, T, Hk, 128], normalized and scaled
    device const float4* kn [[buffer(2)]],      // [B, T, Hk, 128], normalized
    device const VT* vc [[buffer(3)]],          // [B, T, Hv * 128] rows of dims[3] elements
    device const float2* gb [[buffer(4)]],      // [B, T, Hv]: exp(g), beta
    device const float* kq [[buffer(5)]],       // [B, T, Hk]: k . q
    device float* o [[buffer(6)]],              // [B, T, Hv, 128]
    device const uint* dims [[buffer(7)]],      // T, Hk, Hv, v row stride
    uint tgi [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int S = 128, R = S / IB, R4 = R / 4;
  threadgroup float4 qs[TB][32], ks[TB][32];
  threadgroup float2 red[2][IB][32];
  threadgroup float vs[TB][32];
  threadgroup float4 gk[TB];
  const int T = int(dims[0]), hkn = int(dims[1]), hv = int(dims[2]);
  const ulong v_row = dims[3];
  const int bh = int(tgi) / 4, b = bh / hv, h = bh % hv, hk = h / (hv / hkn);
  const int j = (int(tgi) % 4) * 32 + lane, i0 = sg * R;
  device float* st_base = state + ulong(bh) * S * S;
  const ulong tok0 = ulong(b) * T;
  float st[R];
#pragma clang loop unroll(full)
  for (int i = 0; i < R; ++i) st[i] = st_base[(i0 + i) * S + j];
  for (int t0 = 0; t0 < T; t0 += TB) {
    const int nb = min(TB, T - t0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < nb * 32; idx += IB * 32) {
      const int tt = idx >> 5, w = idx & 31;
      const ulong t = tok0 + t0 + tt;
      qs[tt][w] = qn[(t * hkn + hk) * 32 + w];
      ks[tt][w] = kn[(t * hkn + hk) * 32 + w];
      vs[tt][w] = float(vc[t * v_row + h * S + (j - lane) + w]);
      if (w == 0) {
        const float2 g_b = gb[t * hv + h];
        gk[tt] = float4(g_b.x, g_b.y, kq[t * hkn + hk], 0.0f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int tt = 0; tt < nb; ++tt) {
      const int slot = (t0 + tt) & 1;
      float ka = 0.0f, qa = 0.0f;
#pragma clang loop unroll(full)
      for (int w = 0; w < R4; ++w) {
        const float4 k4 = ks[tt][i0 / 4 + w], q4 = qs[tt][i0 / 4 + w];
        ka = fma(st[4 * w], k4.x, ka); ka = fma(st[4 * w + 1], k4.y, ka);
        ka = fma(st[4 * w + 2], k4.z, ka); ka = fma(st[4 * w + 3], k4.w, ka);
        qa = fma(st[4 * w], q4.x, qa); qa = fma(st[4 * w + 1], q4.y, qa);
        qa = fma(st[4 * w + 2], q4.z, qa); qa = fma(st[4 * w + 3], q4.w, qa);
      }
      red[slot][sg][lane] = float2(ka, qa);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float2 s = red[slot][0][lane];
#pragma clang loop unroll(full)
      for (int r = 1; r < IB; ++r) s += red[slot][r][lane];
      const float4 g_b = gk[tt];
      const float g = g_b.x;
      const float delta = (vs[tt][lane] - g * s.x) * g_b.y;
      if (sg == 0) o[((tok0 + t0 + tt) * hv + h) * S + j] = fma(g, s.y, delta * g_b.z);
#pragma clang loop unroll(full)
      for (int w = 0; w < R4; ++w) {
        const float4 k4 = ks[tt][i0 / 4 + w];
        st[4 * w] = fma(g, st[4 * w], k4.x * delta);
        st[4 * w + 1] = fma(g, st[4 * w + 1], k4.y * delta);
        st[4 * w + 2] = fma(g, st[4 * w + 2], k4.z * delta);
        st[4 * w + 3] = fma(g, st[4 * w + 3], k4.w * delta);
      }
    }
  }
#pragma clang loop unroll(full)
  for (int i = 0; i < R; ++i) st_base[(i0 + i) * S + j] = st[i];
}

#define INST(IB, TB, VT) template [[host_name("gdn_scan_" #IB "_" #TB "_" #VT)]] kernel decltype(gdn_scan<IB, TB, VT>) gdn_scan<IB, TB, VT>;
INST(4, 16, float) INST(4, 16, bfloat) INST(4, 16, half) INST(4, 8, float) INST(4, 4, float) INST(8, 8, float) INST(2, 16, float)

// q / k of one (token, key head) per simdgroup: L2 norm (x * rsqrt(sum x^2 + 1e-6), float32), q times 1/sqrt(128),
// k . q; q and k are rows of dims[2] elements (views of the conv output)
template <typename T>
kernel void gdn_prep_qk(
    device const T* q [[buffer(0)]], device const T* k [[buffer(1)]],
    device float4* qn [[buffer(2)]], device float4* kn [[buffer(3)]], device float* kq [[buffer(4)]],
    device const uint* dims [[buffer(5)]],   // rows (B * T * Hk), Hk, row stride
    uint tgi [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sgs [[simdgroups_per_threadgroup]]) {
  const uint r = tgi * sgs + sg;
  if (r >= dims[0]) return;
  const uint tok = r / dims[1], hk = r % dims[1];
  const ulong base = ulong(tok) * dims[2] + hk * 128 + lane * 4;
  const float4 qv = float4(float(q[base]), float(q[base + 1]), float(q[base + 2]), float(q[base + 3]));
  const float4 kv = float4(float(k[base]), float(k[base + 1]), float(k[base + 2]), float(k[base + 3]));
  const float4 qo = qv * rsqrt(simd_sum(dot(qv, qv)) + 1e-6f) * rsqrt(128.0f);
  const float4 ko = kv * rsqrt(simd_sum(dot(kv, kv)) + 1e-6f);
  qn[r * 32 + lane] = qo;
  kn[r * 32 + lane] = ko;
  const float d = simd_sum(dot(qo, ko));
  if (lane == 0) kq[r] = d;
}
template [[host_name("gdn_prep_qk_bfloat")]] kernel decltype(gdn_prep_qk<bfloat>) gdn_prep_qk<bfloat>;
template [[host_name("gdn_prep_qk_half")]] kernel decltype(gdn_prep_qk<half>) gdn_prep_qk<half>;
template [[host_name("gdn_prep_qk_float")]] kernel decltype(gdn_prep_qk<float>) gdn_prep_qk<float>;

// depthwise causal conv (kernel 4, no bias) + SiLU, token-major output out[b, t, c]; x[b, c, t] at strides
// dims[3..5]; a thread makes CONV_TOKENS consecutive tokens of a channel with a sliding window; the conv in float32
// rounded to T, SiLU of that rounded (the reference's conv1d then activation)
constant constexpr uint CONV_TOKENS = 8;
template <typename T>
kernel void gdn_conv_silu(
    device const T* x [[buffer(0)]], device const T* w [[buffer(1)]], device T* out [[buffer(2)]],
    device const uint* dims [[buffer(3)]],   // B, C, T, stride b, stride c, stride t
    uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, C = dims[1], Tn = dims[2];
  const uint blocks = (Tn + CONV_TOKENS - 1) / CONV_TOKENS;
  if (c >= C || gid.y >= dims[0] * blocks) return;
  const uint b = gid.y / blocks, t0 = (gid.y % blocks) * CONV_TOKENS;
  device const T* xc = x + ulong(b) * dims[3] + ulong(c) * dims[4];
  const ulong st = dims[5];
  const float w0 = float(w[c * 4]), w1 = float(w[c * 4 + 1]), w2 = float(w[c * 4 + 2]), w3 = float(w[c * 4 + 3]);
  float a0 = t0 >= 3 ? float(xc[ulong(t0 - 3) * st]) : 0.0f;
  float a1 = t0 >= 2 ? float(xc[ulong(t0 - 2) * st]) : 0.0f;
  float a2 = t0 >= 1 ? float(xc[ulong(t0 - 1) * st]) : 0.0f;
  const uint t1 = min(t0 + CONV_TOKENS, Tn);
  for (uint t = t0; t < t1; ++t) {
    const float a3 = float(xc[ulong(t) * st]);
    const float y = float(T(fma(a3, w3, fma(a2, w2, fma(a1, w1, a0 * w0)))));
    out[(ulong(b) * Tn + t) * C + c] = T(y / (1.0f + exp(-y)));
    a0 = a1; a1 = a2; a2 = a3;
  }
}
template [[host_name("gdn_conv_silu_bfloat")]] kernel decltype(gdn_conv_silu<bfloat>) gdn_conv_silu<bfloat>;
template [[host_name("gdn_conv_silu_half")]] kernel decltype(gdn_conv_silu<half>) gdn_conv_silu<half>;
"""

HEAD_DIM = 128


@functools.cache
def _library():
    return torch.mps.compile_shader(_SOURCE)


_SUFFIX = {torch.bfloat16: "bfloat", torch.float16: "half", torch.float32: "float"}
ROWS_PER_GROUP = 8
CONV_TOKENS = 8


def _u32(*values, device):
    return torch.tensor(values, dtype=torch.int32, device=device)


def _row_view(x):
    """``x`` [B, T, H, 128] as rows of the underlying buffer: (row stride) when its last two dims are dense and its
    tokens evenly spaced (a view of a token-major projection or conv output), else None."""
    b, t, h, d = x.shape
    if x.stride(-1) != 1 or x.stride(-2) != d or (b > 1 and x.stride(0) != t * x.stride(1)):
        return None
    return x.stride(1)


def conv_silu(hidden_states, weight):
    """Depthwise causal conv (no bias) + SiLU of ``hidden_states`` [B, C, T] (any strides) with ``weight`` [C, K]; the
    result is token-major ([B, T, C] memory) returned as a [B, C, T] view."""
    batch, channels, tokens = hidden_states.shape
    out = torch.empty(batch, tokens, channels, dtype=hidden_states.dtype, device=hidden_states.device)
    weight = weight.contiguous()
    assert weight.shape[-1] == 4
    getattr(_library(), f"gdn_conv_silu_{_SUFFIX[hidden_states.dtype]}")(
        hidden_states,
        weight,
        out,
        _u32(batch, channels, tokens, *hidden_states.stride(), device=out.device),
        threads=(channels, batch * -(-tokens // CONV_TOKENS)),
        group_size=(256, 1),
    )
    return out.transpose(1, 2)


def prep_qk(query, key):
    """L2-normalized query (times 1/sqrt(128)) and key in float32 [B, T, Hk, 128] plus k . q [B, T, Hk]."""
    batch, tokens, k_heads, _ = query.shape
    stride = _row_view(query)
    if stride is None or _row_view(key) != stride or key.dtype != query.dtype:
        query, key = query.contiguous(), key.to(query.dtype).contiguous()
        stride = k_heads * HEAD_DIM
    rows = batch * tokens * k_heads
    qn = torch.empty(batch, tokens, k_heads, HEAD_DIM, dtype=torch.float32, device=query.device)
    kn, kq = torch.empty_like(qn), torch.empty(batch, tokens, k_heads, dtype=torch.float32, device=query.device)
    getattr(_library(), f"gdn_prep_qk_{_SUFFIX[query.dtype]}")(
        query,
        key,
        qn,
        kn,
        kq,
        _u32(rows, k_heads, stride, device=query.device),
        threads=-(-rows // ROWS_PER_GROUP) * ROWS_PER_GROUP * 32,
        group_size=ROWS_PER_GROUP * 32,
    )
    return qn, kn, kq


def scan(qn, kn, kq, value, g, beta, state, ib=4, tb=16):
    """``qn``/``kn``/``kq`` from :func:`prep_qk`, value [B, T, Hv, 128] (a row view is read in place), g (log decay) and
    beta [B, T, Hv]. ``state`` [B, Hv, 128, 128] float32 is updated in place; returns the output [B, T, Hv, 128] float32."""
    batch, tokens, k_heads, _ = qn.shape
    v_heads = value.shape[2]
    v_stride = _row_view(value)
    if v_stride is None or value.dtype not in _SUFFIX:
        value = value.float().contiguous()
        v_stride = v_heads * HEAD_DIM
    vt = _SUFFIX[value.dtype] if (ib, tb) == (4, 16) else "float"
    if vt == "float" and value.dtype != torch.float32:
        value, v_stride = value.float().contiguous(), v_heads * HEAD_DIM
    gate_beta = torch.stack((g.float().exp(), beta.float()), dim=-1)
    out = torch.empty(batch, tokens, v_heads, HEAD_DIM, dtype=torch.float32, device=qn.device)
    getattr(_library(), f"gdn_scan_{ib}_{tb}_{vt}")(
        state,
        qn,
        kn,
        value,
        gate_beta,
        kq,
        out,
        _u32(tokens, k_heads, v_heads, v_stride, device=qn.device),
        threads=batch * v_heads * 4 * ib * 32,
        group_size=ib * 32,
    )
    return out
