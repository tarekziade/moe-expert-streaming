"""One decode step of QSA attention (Qwen3.8-Flash-Next) over the indexer's selected tokens only (MPS).

Past ``indexer_budget`` keys the model selects ``block_topk`` blocks of ``compress_ratio`` tokens plus the incomplete tail
block, builds a boolean mask over every key and runs SDPA over the whole cache (0.98 ms per layer at 6K keys on an M5,
plus the mask ops). Here the selected blocks are read directly: split-K flash decoding, a threadgroup per (query head,
slice of the selected tokens), a simdgroup per token (each lane holds ``head_dim / 32`` elements of q and of the output),
an online softmax in float32, then one merge kernel over the slices. Grouped heads: query head ``h`` reads key/value head
``h // (heads / kv_heads)``.
"""

import functools
import os

import torch


_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// token i of the selection: block blocks[i / R] (R tokens each), then the tail tokens tail_start.. (i >= nb * R)
// partial[(h * S + s)] = (max, sum, acc[D]) of slice s of query head h
template <typename T, int D>
kernel void qsa_decode_partial(
    device const T* q [[buffer(0)]],            // [H, D]
    device const T* k [[buffer(1)]],            // [KVH, L, D] (rows of D, head stride dims[5])
    device const T* v [[buffer(2)]],
    device const long* blocks [[buffer(3)]],    // [nb]
    device float* partial [[buffer(4)]],        // [H, S, 2 + D]
    device const uint* dims [[buffer(5)]],      // H, KVH, nb, R, tail_start, head stride, n_selected, S
    device const float* scale [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sgs [[simdgroups_per_threadgroup]]) {
  constexpr int E = D / 32;
  const uint H = dims[0], KVH = dims[1], nb = dims[2], R = dims[3], tail_start = dims[4], n = dims[6], S = dims[7];
  const ulong hs = dims[5];
  const uint h = tg / S, s = tg % S, kvh = h / (H / KVH);
  const uint per = (n + S - 1) / S, i0 = s * per, i1 = min(i0 + per, n);
  float qv[E];
  for (int e = 0; e < E; ++e) qv[e] = float(q[h * D + lane * E + e]) * scale[0];
  float m = -INFINITY, l = 0.0f, acc[E];
  for (int e = 0; e < E; ++e) acc[e] = 0.0f;
  device const T* kh = k + kvh * hs;
  device const T* vh = v + kvh * hs;
  for (uint i = i0 + sg; i < i1; i += sgs) {
    const uint b = i / R;
    const ulong t = b < nb ? ulong(blocks[b]) * R + (i % R) : ulong(tail_start + (i - nb * R));
    float d = 0.0f;
    for (int e = 0; e < E; ++e) d = fma(qv[e], float(kh[t * D + lane * E + e]), d);
    d = simd_sum(d);
    const float m_new = max(m, d);
    const float c = exp(m - m_new), p = exp(d - m_new);
    l = l * c + p;
    for (int e = 0; e < E; ++e) acc[e] = fma(acc[e], c, p * float(vh[t * D + lane * E + e]));
    m = m_new;
  }
  threadgroup float tm[8], tl[8];  // SIMDGROUPS (8) simdgroups
  threadgroup float tacc[8][D];
  if (lane == 0) { tm[sg] = m; tl[sg] = l; }
  for (int e = 0; e < E; ++e) tacc[sg][lane * E + e] = acc[e];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  float M = -INFINITY;
  for (ushort g = 0; g < sgs; ++g) M = max(M, tm[g]);
  float L = 0.0f, out[E];
  for (int e = 0; e < E; ++e) out[e] = 0.0f;
  for (ushort g = 0; g < sgs; ++g) {
    const float c = tm[g] == -INFINITY ? 0.0f : exp(tm[g] - M);
    L += tl[g] * c;
    for (int e = 0; e < E; ++e) out[e] = fma(tacc[g][lane * E + e], c, out[e]);
  }
  device float* p = partial + ulong(h * S + s) * (2 + D);
  if (lane == 0) { p[0] = M; p[1] = L; }
  for (int e = 0; e < E; ++e) p[2 + lane * E + e] = out[e];
}

template <typename T, int D>
kernel void qsa_decode_merge(
    device const float* partial [[buffer(0)]], device T* out [[buffer(1)]], device const uint* dims [[buffer(2)]],
    uint h [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int E = D / 32;
  const uint S = dims[7];
  device const float* p = partial + ulong(h) * S * (2 + D);
  float M = -INFINITY;
  for (uint s = 0; s < S; ++s) M = max(M, p[s * (2 + D)]);
  float L = 0.0f, acc[E];
  for (int e = 0; e < E; ++e) acc[e] = 0.0f;
  for (uint s = 0; s < S; ++s) {
    device const float* ps = p + s * (2 + D);
    const float c = ps[0] == -INFINITY ? 0.0f : exp(ps[0] - M);
    L += ps[1] * c;
    for (int e = 0; e < E; ++e) acc[e] = fma(ps[2 + lane * E + e], c, acc[e]);
  }
  for (int e = 0; e < E; ++e) out[h * D + lane * E + e] = T(acc[e] / L);
}

// indexer query of one token: per head (a simdgroup, 4 values a lane) RMSNorm in float32 times (1 + weight), rounded to
// T, then the rotary embedding on the first `rot` dims in T arithmetic (x*cos and rotate_half(x)*sin rounded, then
// their sum), as the model's Qwen4ExpTextRMSNorm + apply_rotary_pos_emb; out: float [heads, 128]
template <typename T>
kernel void qsa_index_q(
    device const T* qk [[buffer(0)]], device const T* weight [[buffer(1)]], device const T* cos [[buffer(2)]],
    device const T* sin [[buffer(3)]], device float* out [[buffer(4)]], device const float* params [[buffer(5)]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup T normed[8][128];
  const int rot = int(params[1]), half_rot = rot / 2;
  float x[4], ss = 0.0f;
  for (int e = 0; e < 4; ++e) { x[e] = float(qk[sg * 128 + lane * 4 + e]); ss += x[e] * x[e]; }
  const float r = rsqrt(simd_sum(ss) / 128.0f + params[0]);
  for (int e = 0; e < 4; ++e) {
    const int d = lane * 4 + e;
    normed[sg][d] = T(x[e] * r * (1.0f + float(weight[d])));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = 0; e < 4; ++e) {
    const int d = lane * 4 + e;
    T v = normed[sg][d];
    if (d < rot) {
      const T other = d < half_rot ? T(-float(normed[sg][d + half_rot])) : normed[sg][d - half_rot];
      v = T(float(T(float(v) * float(cos[d]))) + float(T(float(other) * float(sin[d]))));
    }
    out[sg * 128 + d] = float(v);
  }
}

// block scores: a simdgroup per block, score = sum over heads of relu(q_h . k) / sqrt(128)
template <typename T>
kernel void qsa_index_scores(
    device const float* q [[buffer(0)]], device const T* keys [[buffer(1)]], device float* scores [[buffer(2)]],
    device const uint* dims [[buffer(3)]],   // blocks, heads
    uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sgs [[simdgroups_per_threadgroup]]) {
  const uint n = tg * sgs + sg;
  if (n >= dims[0]) return;
  float k[4];
  for (int e = 0; e < 4; ++e) k[e] = float(keys[ulong(n) * 128 + lane * 4 + e]);
  float total = 0.0f;
  for (uint h = 0; h < dims[1]; ++h) {
    float d = 0.0f;
    for (int e = 0; e < 4; ++e) d = fma(q[h * 128 + lane * 4 + e], k[e], d);
    total += max(simd_sum(d), 0.0f);
  }
  if (lane == 0) scores[n] = total / sqrt(128.0f);
}
template [[host_name("qsa_index_q_bfloat")]] kernel decltype(qsa_index_q<bfloat>) qsa_index_q<bfloat>;
template [[host_name("qsa_index_q_half")]] kernel decltype(qsa_index_q<half>) qsa_index_q<half>;
template [[host_name("qsa_index_scores_bfloat")]] kernel decltype(qsa_index_scores<bfloat>) qsa_index_scores<bfloat>;
template [[host_name("qsa_index_scores_half")]] kernel decltype(qsa_index_scores<half>) qsa_index_scores<half>;

#define INST(T, NAME, D) \\
  template [[host_name("qsa_decode_partial_" NAME "_" #D)]] kernel decltype(qsa_decode_partial<T, D>) qsa_decode_partial<T, D>; \\
  template [[host_name("qsa_decode_merge_" NAME "_" #D)]] kernel decltype(qsa_decode_merge<T, D>) qsa_decode_merge<T, D>;
INST(bfloat, "bfloat", 256) INST(half, "half", 256) INST(float, "float", 256)
INST(bfloat, "bfloat", 128) INST(half, "half", 128) INST(float, "float", 128)
"""

_SUFFIX = {torch.bfloat16: "bfloat", torch.float16: "half", torch.float32: "float"}
SIMDGROUPS = 8
TOKENS_PER_GROUP = 256


@functools.cache
def _library():
    return torch.mps.compile_shader(_SOURCE)


@functools.lru_cache(maxsize=256)
def _dims(values, device):
    """Small int32 parameter tensors on the device, cached (decode steps repeat or slowly change them). Made with
    ``device=``: a ``non_blocking`` copy from pageable memory may read the host tensor after it is freed."""
    return torch.tensor(values, dtype=torch.int32, device=device)


@functools.cache
def _scale(value, device):
    return torch.tensor([value], dtype=torch.float32, device=device)


def supports(query, key):
    return (
        query.device.type == "mps"
        and query.dtype in _SUFFIX
        and key.dtype == query.dtype
        and query.shape[0] == 1
        and query.shape[2] == 1
        and query.shape[-1] in (128, 256)
        and query.shape[1] % key.shape[1] == 0
    )


def _rows(cache):
    """``cache`` [1, KVH, L, D] as a buffer the kernel can read (rows of D, a head stride): a view of the first L rows of
    a larger buffer (the grown cache) is passed as that buffer, without a copy."""
    if cache.is_contiguous():
        return cache, cache.shape[2] * cache.shape[3]
    base = cache._base
    if (
        base is not None and base.is_contiguous() and cache.storage_offset() == 0 and base.dim() == 4
        and cache.stride(3) == 1 and cache.stride(2) == cache.shape[3] and base.stride(1) == cache.stride(1)
    ):
        return base, cache.stride(1)
    return cache.contiguous(), cache.shape[2] * cache.shape[3]


def attention(query, key, value, blocks, ratio, tail_start, scaling):
    """``query`` [1, H, 1, D], ``key``/``value`` [1, KVH, L, D] (the cache), ``blocks`` [nb] int64 selected block indices
    (``ratio`` tokens each), plus every token from ``tail_start`` to ``L - 1``. Returns [1, H, 1, D] in the query dtype."""
    _, heads, _, dim = query.shape
    kv_heads, length = key.shape[1], key.shape[2]
    (key, head_stride), (value, value_stride) = _rows(key), _rows(value)
    if head_stride != value_stride:
        key, value, head_stride = key.contiguous(), value.contiguous(), length * dim
    blocks = blocks.contiguous()
    selected = blocks.numel() * ratio + (length - tail_start)
    splits = max(1, -(-selected // TOKENS_PER_GROUP))
    dims = _dims((heads, kv_heads, blocks.numel(), ratio, tail_start, head_stride, selected, splits), query.device)
    partial = torch.empty(heads * splits * (2 + dim), dtype=torch.float32, device=query.device)
    out = torch.empty(heads, dim, dtype=query.dtype, device=query.device)
    lib, name = _library(), f"{_SUFFIX[query.dtype]}_{dim}"
    getattr(lib, f"qsa_decode_partial_{name}")(
        query.reshape(heads, dim).contiguous(),
        key,
        value,
        blocks,
        partial,
        dims,
        _scale(float(scaling), query.device),
        threads=heads * splits * SIMDGROUPS * 32,
        group_size=SIMDGROUPS * 32,
    )
    getattr(lib, f"qsa_decode_merge_{name}")(partial, out, dims, threads=heads * 32, group_size=32)
    return out.view(1, heads, 1, dim)


def enabled():
    return os.environ.get("MOE_STREAMING_QSA_DECODE", "1") == "1"


def attention_forward(module, hidden_states, position_embeddings, past_key_values, blocks, tail_start):
    """``Qwen4ExpTextAttention.forward`` for one token, attending to ``blocks`` and the tail from ``tail_start``: the
    same projections, norms, rotary embedding, cache update, output gate and projection as the original."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    cos, sin = (x[:, -hidden_states.shape[1] :, :] for x in position_embeddings)
    query_states, gate = torch.chunk(
        module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx)
    attn_output = attention(
        query_states, key_states, value_states, blocks, module.indexer.compress_ratio, tail_start, module.scaling
    )
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
    attn_output = attn_output * torch.sigmoid(gate)
    return module.o_proj(attn_output), None


@functools.cache
def _index_params(eps, rot, device):
    return torch.tensor([eps, rot], dtype=torch.float32, device=device)


def index_query(qk, weight, cos, sin, heads, eps):
    """The indexer's query of one token: ``qk`` [heads * 128 (+ more)] projection output, ``cos``/``sin`` [rot] of its
    position; normalized and rotated as the model does, returned in float32 [heads, 128]."""
    out = torch.empty(heads, 128, dtype=torch.float32, device=qk.device)
    getattr(_library(), f"qsa_index_q_{_SUFFIX[qk.dtype]}")(
        qk, weight, cos.contiguous(), sin.contiguous(), out, _index_params(float(eps), cos.shape[-1], qk.device),
        threads=heads * 32, group_size=heads * 32,
    )
    return out


def index_scores(query, block_keys):
    """``query`` [heads, 128] float32, ``block_keys`` [blocks, 128]: sum over heads of relu(q . k) / sqrt(128)."""
    blocks, heads = block_keys.shape[0], query.shape[0]
    scores = torch.empty(blocks, dtype=torch.float32, device=query.device)
    dims = _dims((blocks, heads), query.device)
    getattr(_library(), f"qsa_index_scores_{_SUFFIX[block_keys.dtype]}")(
        query, block_keys.contiguous(), scores, dims,
        threads=-(-blocks // SIMDGROUPS) * SIMDGROUPS * 32, group_size=SIMDGROUPS * 32,
    )
    return scores
