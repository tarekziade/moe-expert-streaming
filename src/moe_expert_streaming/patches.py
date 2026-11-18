"""Patches shared by the Qwen3.8 (``qwen4_exp``) and Qwen3.6 (``qwen3_5_moe``) loaders: the Gated DeltaNet layers and
the norm after them on Metal kernels, quantized experts off MPS, and attention caches that grow in place."""

import torch

from . import mlx_affine
from .hub_kernels import gated_delta_net, gated_norm, gdn_scan
from .kernels import gdn_prefill


def _patch(cls, name, make):
    """Replace ``cls.name`` by ``make(original)`` once."""
    original = getattr(cls, name)
    if getattr(original, "_moe_expert_streaming", False):
        return
    patched = make(original)
    patched._moe_expert_streaming = True
    setattr(cls, name, patched)


def _append(layer, name, current, new, dim):
    """``torch.cat([current, new], dim)`` as a view of ``layer.<name>``, a buffer grown in steps (copied only then).
    Starts over from ``current`` when something else replaced it (crop, reorder, offload)."""
    buffer, view = getattr(layer, name, (None, None))
    length = current.shape[dim] if current.numel() else 0
    needed = length + new.shape[dim]
    if buffer is None or view is not current or buffer.shape[dim] < needed or buffer.device != new.device:
        capacity = max(needed + 1024, int(needed * 1.25)) // 256 * 256 + 256
        shape = list(new.shape)
        shape[dim] = capacity
        grown = torch.empty(shape, dtype=new.dtype, device=new.device)
        if length:
            grown.narrow(dim, 0, length).copy_(current)
        buffer = grown
    buffer.narrow(dim, length, new.shape[dim]).copy_(new)
    view = buffer.narrow(dim, 0, needed)
    setattr(layer, name, (buffer, view))
    return view


def patch_gather_qmm():
    """Off MPS, quantized experts run through :func:`mlx_affine.gather_qmm` (dequantize, then dense matmuls)."""
    from transformers.integrations import experts_offload

    def load_gather_qmm(original):
        # TODO(portability): replace the dequantize-then-matmul path with a fused CUDA (Triton) gather_qmm.
        def load(device):
            return original(device) if torch.device(device).type == "mps" else mlx_affine.gather_qmm

        return load

    if not getattr(experts_offload._load_gather_qmm, "_moe_expert_streaming", False):
        patched = load_gather_qmm(experts_offload._load_gather_qmm)
        patched._moe_expert_streaming = True
        experts_offload._load_gather_qmm = patched


def patch_gated_delta_net(m, gdn_cls, norm_cls):
    """Metal decode step, gated norm, chunked delta rule and causal conv for a model module ``m`` whose Gated DeltaNet
    class is ``gdn_cls`` and output norm class ``norm_cls``."""

    def gdn_forward(original):
        # TODO(portability): Metal-only decode step (MPS); other backends keep the original forward.
        def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
            if (
                hidden_states.device.type == "mps"
                and hidden_states.shape[1] == 1
                and gated_delta_net.supports(self, hidden_states, cache_params, attention_mask)
            ):
                return gated_delta_net.gated_delta_net_step(self, hidden_states, cache_params)
            return original(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)

        return forward

    _patch(gdn_cls, "forward", gdn_forward)

    def gated_norm_forward(original):
        # TODO(portability): Metal-only (MPS); other backends keep the original forward.
        def forward(self, hidden_states, gate):
            if gated_norm.supports(self, hidden_states, gate):
                return gated_norm.gated_rms_norm(self, hidden_states, gate)
            return original(self, hidden_states, gate)

        return forward

    _patch(norm_cls, "forward", gated_norm_forward)
    # Prefill (and chunked prompts): the chunked delta rule on ggml's kernel instead of the PyTorch reference.
    _patch(m, "torch_chunk_gated_delta_rule", lambda original: gdn_prefill.wrap(original, m.l2norm))

    def conv_forward(original):
        # TODO(portability): Metal-only (MPS); other backends keep the original function.
        # Token-major output: the q / k / v views the delta rule reads are then rows, not columns.
        def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
            if (
                hidden_states.device.type == "mps"
                and bias is None
                and activation == "silu"
                and weight.dim() == 2
                and weight.shape[-1] == 4
                and weight.dtype == hidden_states.dtype
                and hidden_states.dtype in (torch.bfloat16, torch.float16)
                and not any(v is not None for v in kwargs.values())
                and not torch.is_grad_enabled()
                and gdn_prefill.enabled()
            ):
                return gdn_scan.conv_silu(hidden_states, weight)
            return original(hidden_states, weight, bias, activation=activation, **kwargs)

        return causal_conv1d_fn

    _patch(m, "causal_conv1d_fn", conv_forward)


def patch_cache_append(*layer_classes):
    """Attention caches grow by ``torch.cat``, which copies the whole cache at every decode step. Keep them as views of
    buffers grown in steps; appends write in place."""

    def layer_update(original):
        def update(self, key_states, value_states, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            self.keys, self.values = (
                _append(self, "_key_buffer", self.keys, key_states, -2),
                _append(self, "_value_buffer", self.values, value_states, -2),
            )
            return self.keys, self.values

        return update

    for cls in layer_classes:
        _patch(cls, "update", layer_update)
