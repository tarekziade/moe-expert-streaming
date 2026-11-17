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


def _block_keys(indexer, raw_keys, full_cos, full_sin, first, last, apply_rotary_pos_emb):
    """Pooled, normalized, rotated keys of complete blocks ``first..last-1`` (as the original computes them)."""
    ratio, head_dim = indexer.compress_ratio, indexer.index_head_dim
    key_groups = raw_keys[0, first * ratio : last * ratio].view(last - first, ratio, head_dim)
    pooled = indexer.k_layernorm(key_groups.float().mean(dim=1).to(raw_keys.dtype))
    starts = torch.arange(first, last, device=raw_keys.device) * ratio
    return apply_rotary_pos_emb(
        pooled.unsqueeze(1), cos=full_cos[0].index_select(0, starts), sin=full_sin[0].index_select(0, starts)
    ).squeeze(1)


def select(
    indexer, hidden_states, position_embeddings, attention_mask, past_key_values, apply_rotary_pos_emb, blocks_only=False
):
    """``indexer.forward`` for a batch of one with a plain causal mask, else ``None`` (before any side effect).

    ``blocks_only`` (one query token past the budget): return ``(selected block indices, tail start)`` instead of the
    mask: the query attends to the ``compress_ratio`` tokens of each selected block and to every token from the tail
    start on (the same tokens the mask selects)."""
    if blocks_only and (hidden_states.shape[1] != 1 or attention_mask.shape[-1] // indexer.compress_ratio == 0):
        return None
    batch_size, seq_length, _ = hidden_states.shape
    if batch_size != 1:
        return None
    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    kv_length = visible.shape[-1]
    past = kv_length - seq_length
    expected = _causal(visible, past)
    # Decode steps continuing a sequence whose previous mask was checked: one more key, no new padding, so still plain
    # causal; skip the check (a host synchronization) and reuse the pooled keys of the complete blocks.
    state = getattr(indexer, "_fast_state", None)
    continuing = (
        past_key_values is not None
        and seq_length == 1
        and state is not None
        and state[0] is past_key_values
        and state[1] == kv_length - 1
    )
    if visible.shape[1] != 1 or (not continuing and not torch.equal(visible[0, 0], expected)):
        indexer._fast_state = None
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
    cached = state[2] if continuing else None
    block_keys = None
    if num_blocks > 0:
        if cached is not None and cached.shape[0] == num_blocks:
            block_keys = cached
        elif cached is not None and cached.shape[0] == num_blocks - 1:
            new = _block_keys(indexer, raw_keys, full_cos, full_sin, num_blocks - 1, num_blocks, apply_rotary_pos_emb)
            block_keys = torch.cat([cached, new])
        else:
            block_keys = _block_keys(indexer, raw_keys, full_cos, full_sin, 0, num_blocks, apply_rotary_pos_emb)
        # scores[i, j] = sum over heads of relu(q_i,h . k_j) / sqrt(head_dim), as the original per query.
        scores = torch.einsum("thd,nd->thn", q[0].float(), block_keys.float()).relu().sum(dim=1) / math.sqrt(head_dim)
        valid = torch.arange(num_blocks, device=device)[None, :] < blocks_per_query[:, None]
        scores = scores.masked_fill(~valid, float("-inf"))
        top = scores.topk(min(indexer.block_topk, num_blocks), dim=1).indices
        if blocks_only:
            indexer._fast_state = (past_key_values, kv_length, block_keys) if past_key_values is not None else None
            return top[0], num_blocks * ratio
        top_valid = valid.gather(1, top)
        tokens = top[..., None] * ratio + torch.arange(ratio, device=device)
        tokens = torch.where(top_valid[..., None], tokens, kv_length).reshape(seq_length, -1)
        selected.scatter_(1, tokens, True)
    indexer._fast_state = (
        (past_key_values, kv_length, block_keys) if past_key_values is not None and block_keys is not None else None
    )
    tail = torch.arange(kv_length, device=device)[None, :] >= (blocks_per_query * ratio)[:, None]
    return (selected[:, :kv_length] | (tail & expected)).view(1, 1, seq_length, kv_length)
