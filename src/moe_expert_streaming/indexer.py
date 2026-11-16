"""Vectorized QSA indexer token selection (Qwen3.8-Flash-Next) for causal masks.

The model's ``Qwen4ExpTextQSAIndexer.forward`` selects tokens with a Python loop over queries, each with a
``torch.nonzero`` (a host synchronization) and ~15 small ops: ~10 s of a 4K-token prefill once there are more than
``token_budget`` keys. With a causal mask and no padding, query ``i`` sees tokens ``0..past + i`` and every complete
block of ``compress_ratio`` tokens pools the same keys for every query, so the selection is one scores matrix
(queries x blocks), one ``topk`` and two masks. :func:`select` returns ``None`` when the mask is not plain causal; the
caller then runs the original code.
"""

import math
import os

import torch


def enabled():
    return os.environ.get("MOE_STREAMING_FAST_INDEXER", "1") == "1"


def _causal(visible, past):
    rows, kv_length = visible.shape[-2:]
    positions = torch.arange(kv_length, device=visible.device)
    return positions[None, :] <= (past + torch.arange(rows, device=visible.device))[:, None]


def select(indexer, hidden_states, position_embeddings, attention_mask, past_key_values, apply_rotary_pos_emb):
    """``indexer.forward`` for a batch of one with a plain causal mask, else ``None`` (before any side effect)."""
    batch_size, seq_length, _ = hidden_states.shape
    if batch_size != 1:
        return None
    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    kv_length = visible.shape[-1]
    past = kv_length - seq_length
    expected = _causal(visible, past)
    if visible.shape[1] != 1 or not torch.equal(visible[0, 0], expected):
        return None

    # Same projections, norms, rotary embedding and cache update as the original.
    full_cos, full_sin = position_embeddings
    current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]
    head_dim = indexer.index_head_dim
    qk = indexer.index_qk_proj(hidden_states)
    q, token_k = torch.split(qk, [indexer.index_n_heads * head_dim, indexer.index_kv_heads * head_dim], dim=-1)
    hidden_shape = (batch_size, seq_length, -1, head_dim)
    q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q = indexer.q_layernorm(q)
    q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, indexer.layer_idx)

    ratio = indexer.compress_ratio
    num_blocks = kv_length // ratio
    device = hidden_states.device
    # Complete blocks visible to each query: (past + i + 1) // ratio.
    blocks_per_query = (past + torch.arange(seq_length, device=device) + 1) // ratio
    selected = torch.zeros(seq_length, kv_length + 1, dtype=torch.bool, device=device)
    if num_blocks > 0:
        key_groups = raw_keys[0, : num_blocks * ratio].view(num_blocks, ratio, head_dim)
        pooled = indexer.k_layernorm(key_groups.float().mean(dim=1).to(raw_keys.dtype))
        starts = torch.arange(num_blocks, device=device) * ratio
        block_keys = apply_rotary_pos_emb(
            pooled.unsqueeze(1), cos=full_cos[0].index_select(0, starts), sin=full_sin[0].index_select(0, starts)
        ).squeeze(1)
        # scores[i, j] = sum over heads of relu(q_i,h . k_j) / sqrt(head_dim), as the original per query.
        scores = torch.einsum("thd,nd->thn", q[0].float(), block_keys.float()).relu().sum(dim=1) / math.sqrt(head_dim)
        valid = torch.arange(num_blocks, device=device)[None, :] < blocks_per_query[:, None]
        scores = scores.masked_fill(~valid, float("-inf"))
        top = scores.topk(min(indexer.block_topk, num_blocks), dim=1).indices
        top_valid = valid.gather(1, top)
        tokens = top[..., None] * ratio + torch.arange(ratio, device=device)
        tokens = torch.where(top_valid[..., None], tokens, kv_length).reshape(seq_length, -1)
        selected.scatter_(1, tokens, True)
    tail = torch.arange(kv_length, device=device)[None, :] >= (blocks_per_query * ratio)[:, None]
    return (selected[:, :kv_length] | (tail & expected)).view(1, 1, seq_length, kv_length)
