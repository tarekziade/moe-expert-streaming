"""The gated delta rule over a prompt chunk as one Metal scan (MPS), after Strata's ``gdn_scan_r`` (MIT).

Only the state update is sequential, and the columns of a value head's ``(key, value)`` state are independent in it:
column ``j`` needs ``k``, ``q``, ``v_j``, the gate and beta. So a lane owns a column, the 128 key rows of the column are
split over ``IB`` simdgroups (the state lives in registers), the partial dots are summed through threadgroup memory once
per token, and ``q``/``k``/``v`` are staged ``TB`` tokens at a time. Per token, with ``S`` the old state::

    a = S^T k, b = S^T q, delta = (v - exp(g) a) beta, o = exp(g) b + (k . q) delta, S = exp(g) S + k delta^T

the reference's recurrence (``o = S_new^T q``) with different rounding. Inputs are prepared in PyTorch (float32, q/k
L2-normalized, q scaled by ``1/sqrt(d)``); value head ``h`` reads key head ``h // (Hv / Hk)`` as the reference's
``repeat_interleave``.
"""

import functools

import torch


_SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <int IB, int TB>
kernel void gdn_scan(
    device float* state [[buffer(0)]],          // [B, Hv, 128 (key i), 128 (value j)], updated in place
    device const float4* qn [[buffer(1)]],      // [B, T, Hk, 128], normalized and scaled
    device const float4* kn [[buffer(2)]],      // [B, T, Hk, 128], normalized
    device const float* vc [[buffer(3)]],       // [B, T, Hv, 128]
    device const float2* gb [[buffer(4)]],      // [B, T, Hv]: exp(g), beta
    device const float* kq [[buffer(5)]],       // [B, T, Hk]: k . q
    device float* o [[buffer(6)]],              // [B, T, Hv, 128]
    device const uint* dims [[buffer(7)]],      // T, Hk, Hv
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
      vs[tt][w] = vc[(t * hv + h) * S + (j - lane) + w];
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

#define INST(IB, TB) template [[host_name("gdn_scan_" #IB "_" #TB)]] kernel decltype(gdn_scan<IB, TB>) gdn_scan<IB, TB>;
INST(4, 16) INST(4, 8) INST(4, 4) INST(8, 8) INST(2, 16)
"""

HEAD_DIM = 128


@functools.cache
def _library():
    return torch.mps.compile_shader(_SOURCE)


@functools.cache
def _dims(tokens, k_heads, v_heads, device):
    return torch.tensor([tokens, k_heads, v_heads], dtype=torch.int32, device=device)


def scan(query, key, value, g, beta, state, ib=4, tb=16):
    """query/key [B, T, Hk, 128] (normalized, query scaled), value [B, T, Hv, 128], g/beta [B, T, Hv]; float32.

    ``state`` [B, Hv, 128, 128] float32 is updated in place; returns the output [B, T, Hv, 128] float32."""
    batch, tokens, k_heads, _ = query.shape
    v_heads = value.shape[2]
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    gate_beta = torch.stack((g.exp(), beta), dim=-1).contiguous()
    kq = (query * key).sum(-1).contiguous()
    out = torch.empty_like(value)
    getattr(_library(), f"gdn_scan_{ib}_{tb}")(
        state,
        query,
        key,
        value,
        gate_beta,
        kq,
        out,
        _dims(tokens, k_heads, v_heads, query.device),
        threads=batch * v_heads * 4 * ib * 32,
        group_size=ib * 32,
    )
    return out
