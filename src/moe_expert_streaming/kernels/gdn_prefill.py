"""Gated DeltaNet prefill on our scan kernel (:mod:`.gdn_scan`, head dim 128) or ``ggml-org/ggml-gated-delta-net`` (MPS).

Without ``flash-linear-attention`` (Triton, not on Apple GPUs) Transformers runs the chunked gated delta rule as a PyTorch
reference: 6.7 s of a 4K-token prefill of Qwen3.8-Flash-Next. ggml's kernel runs the recurrence for a whole chunk of
tokens in one dispatch. :func:`wrap` returns a drop-in for ``torch_chunk_gated_delta_rule`` (same arguments and outputs):
float32 inside like the reference, q/k L2-normalized with the model's own ``l2norm``, q/k heads expanded to the value
heads in the model's (grouped) order, and the recurrent state transposed to and from ggml's ``[value][key]`` layout.
"""

import functools
import os

import torch

from . import gdn_scan


@functools.cache
def load_kernel():
    from kernels import get_kernel

    return get_kernel("ggml-org/ggml-gated-delta-net", version=1)


def enabled():
    return os.environ.get("MOE_STREAMING_GGML_GDN_PREFILL", "1") == "1"


def wrap(original, l2norm):
    def chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        head_dim = query.shape[-1]
        if (
            not enabled()
            or query.device.type != "mps"
            or kwargs.get("cu_seqlens") is not None
            or value.shape[-1] != head_dim
            or g.dim() != 3
            or not load_kernel().supports_gated_delta_net(head_dim)
        ):
            return original(
                query,
                key,
                value,
                g,
                beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                **kwargs,
            )
        dtype = query.dtype
        batch, _, k_heads, _ = query.shape
        v_heads = value.shape[2]
        query, key = query.float(), key.float()
        if use_qk_l2norm_in_kernel:
            query, key = l2norm(query, dim=-1, eps=1e-6), l2norm(key, dim=-1, eps=1e-6)
        if head_dim == gdn_scan.HEAD_DIM and v_heads % k_heads == 0:
            # Our scan kernel (3x ggml's at 2K tokens on an M5): same (key, value) state layout as the reference.
            if initial_state is None:
                state = torch.zeros(batch, v_heads, head_dim, head_dim, dtype=torch.float32, device=query.device)
            else:
                state = initial_state.float().contiguous()
                if state.data_ptr() == initial_state.data_ptr():
                    state = state.clone()
            out = gdn_scan.scan(query * head_dim**-0.5, key, value.float(), g.float(), beta.float(), state)
            return out.to(dtype), state if output_final_state else None
        if v_heads != k_heads:
            # Each query/key head serves v_heads // k_heads consecutive value heads, as in the reference.
            query = query.repeat_interleave(v_heads // k_heads, dim=2)
            key = key.repeat_interleave(v_heads // k_heads, dim=2)
        if initial_state is None:
            state = torch.zeros(batch, v_heads, head_dim, head_dim, dtype=torch.float32, device=query.device)
        else:
            state = initial_state.float().transpose(-1, -2).contiguous()
        out, final = load_kernel().gated_delta_net(
            query.contiguous(),
            key.contiguous(),
            value.float().contiguous(),
            g.float().contiguous(),
            beta.float().contiguous(),
            state,
        )
        final = final.transpose(-1, -2).contiguous() if output_final_state else None
        return out.to(dtype), final

    return chunk_gated_delta_rule
