"""Qwen3.8-Flash-Next (``Qwen4ExpForCausalLM``) on top of transformers' ``experts_offload`` (huggingface/transformers#50717).

``experts_offload`` streams the routed experts. This module adds what this model needs beyond it, as class-level patches
applied by :func:`apply`:

- the 51 GB hashed n-gram table read from disk row by row (:class:`DiskEmbedding`), instead of materialized;
- on Apple GPUs: the Gated DeltaNet decode step, the hyper-connection step and the Gated DeltaNet output norm as Metal
  kernels;
- the QSA indexer skips token selection while every token fits its budget (2,048 keys), where it selects them all;
- off Apple GPUs (CUDA), quantized experts run through :func:`mlx_affine.gather_qmm` (dequantize, then dense matmuls)
  instead of the slow per-expert PyTorch reference.
"""

import contextlib
import json
from pathlib import Path

import torch

from . import indexer, metal_linear, mlx_affine
from .kernels import gated_delta_net, gated_norm, gdn_prefill, gdn_scan, hyper_connection, qsa_decode


class DiskEmbedding(torch.nn.Module):
    """An embedding table that stays on disk; each lookup reads only the rows it needs (through the OS page cache).

    ``directory`` holds ``ngram.json`` (``rows``, ``dim``, ``dtype``: ``float8_e4m3fn``, ``bfloat16`` or ``float32``,
    ``scale``, ``file``) and the raw row-major file, as ``scripts/convert_fp8.py`` writes them. Rows are dequantized as
    ``row * scale``.
    """

    _DTYPES = {"float8_e4m3fn": torch.float8_e4m3fn, "bfloat16": torch.bfloat16, "float32": torch.float32}

    def __init__(self, directory, dtype=torch.bfloat16, read_threads=16):
        super().__init__()
        import os
        from concurrent.futures import ThreadPoolExecutor

        info = json.loads((Path(directory) / "ngram.json").read_text(encoding="utf-8"))
        self.rows, self.dim = info["rows"], info["dim"]
        self.stored_dtype = self._DTYPES[info["dtype"]]
        self.scale = float(info.get("scale", 1.0))
        self.row_bytes = self.dim * self.stored_dtype.itemsize
        # One descriptor, rows read in parallel: a token's rows are random, and reading them one page fault at a time
        # costs one storage latency each (16 rows: 1.4 ms on a local SSD, 0.25 ms in parallel; far more on network
        # storage).
        self.fd = os.open(Path(directory) / info["file"], os.O_RDONLY)
        self.pool = ThreadPoolExecutor(read_threads, thread_name_prefix="disk-embedding") if read_threads else None
        self.dtype = dtype
        # Callers read `.weight.device` to place their indices; it follows the model to its device.
        self.register_buffer("weight", torch.empty(0, device="cpu", dtype=dtype), persistent=False)

    def _read_row(self, row):
        import os

        return os.pread(self.fd, self.row_bytes, row * self.row_bytes)

    def forward(self, ids):
        flat = ids.reshape(-1).tolist()
        reads = self.pool.map(self._read_row, flat) if self.pool is not None else map(self._read_row, flat)
        raw = bytearray(b"".join(reads))
        rows = torch.frombuffer(raw, dtype=torch.uint8).view(self.stored_dtype).view(len(flat), self.dim).float()
        if self.scale != 1.0:
            rows = rows * self.scale
        return rows.to(self.dtype).view(*ids.shape, self.dim).to(self.weight.device, non_blocking=True)


_DISK_NGRAM = {"directory": None}


@contextlib.contextmanager
def disk_ngram_table(directory):
    """While active, ``Qwen4ExpTextNGramEmbedding`` modules built (e.g. by ``from_pretrained``) read their table from
    ``directory`` instead of allocating it."""
    previous = _DISK_NGRAM["directory"]
    _DISK_NGRAM["directory"] = directory
    try:
        yield
    finally:
        _DISK_NGRAM["directory"] = previous


def _patch(cls, name, make):
    """Replace ``cls.name`` by ``make(original)`` once."""
    original = getattr(cls, name)
    if getattr(original, "_moe_expert_streaming", False):
        return
    patched = make(original)
    patched._moe_expert_streaming = True
    setattr(cls, name, patched)


def apply():
    """Install the Qwen3.8 patches and the Metal dense-layer patches (idempotent)."""
    from transformers.integrations import experts_offload
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    metal_linear.apply()

    def load_gather_qmm(original):
        # TODO(portability): replace the dequantize-then-matmul path with a fused CUDA (Triton) gather_qmm.
        def load(device):
            return original(device) if torch.device(device).type == "mps" else mlx_affine.gather_qmm

        return load

    if not getattr(experts_offload._load_gather_qmm, "_moe_expert_streaming", False):
        patched = load_gather_qmm(experts_offload._load_gather_qmm)
        patched._moe_expert_streaming = True
        experts_offload._load_gather_qmm = patched

    def ngram_init(original):
        def __init__(self, *args, **kwargs):
            original(self, *args, **kwargs)
            if _DISK_NGRAM["directory"] is not None:
                self.ngram_embedding = DiskEmbedding(_DISK_NGRAM["directory"])

        return __init__

    _patch(m.Qwen4ExpTextNGramEmbedding, "__init__", ngram_init)

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

    _patch(m.Qwen4ExpTextGatedDeltaNet, "forward", gdn_forward)

    def gated_norm_forward(original):
        # TODO(portability): Metal-only (MPS); other backends keep the original forward.
        def forward(self, hidden_states, gate):
            if gated_norm.supports(self, hidden_states, gate):
                return gated_norm.gated_rms_norm(self, hidden_states, gate)
            return original(self, hidden_states, gate)

        return forward

    _patch(m.Qwen4ExpTextRMSNormGated, "forward", gated_norm_forward)
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

    def residual_forward(original):
        # TODO(portability): Metal-only decode step (MPS); other backends keep the original forward.
        def forward(self, hyper_input):
            if hyper_input.device.type == "mps" and hyper_connection.supports(self, hyper_input):
                return hyper_connection.gated_residual_step(self, hyper_input)
            return original(self, hyper_input)

        return forward

    _patch(m.Qwen4ExpTextGatedResidual, "forward", residual_forward)

    def indexer_forward(original):
        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
            if attention_mask.shape[-1] > self.token_budget:
                if indexer.enabled():
                    mask = indexer.select(
                        self,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_values,
                        m.apply_rotary_pos_emb,
                    )
                    if mask is not None:
                        return mask
                return original(self, hidden_states, position_embeddings, attention_mask, past_key_values)
            # At most `token_budget` keys: every complete block fits in `block_topk` and the tail is always kept, so
            # every visible token is selected. Only the key cache update is needed; this skips the per-query selection
            # (and its host synchronizations).
            if past_key_values is not None:
                batch_size, seq_length, _ = hidden_states.shape
                keys = self.index_qk_proj(hidden_states)[..., self.index_n_heads * self.index_head_dim :]
                keys = keys.reshape(batch_size, seq_length, -1, self.index_head_dim).squeeze(2)
                past_key_values.update_indexer(keys, self.layer_idx)
            return attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0

        return forward

    _patch(m.Qwen4ExpTextQSAIndexer, "forward", indexer_forward)

    def attention_forward(original):
        # TODO(portability): Metal-only decode step (MPS); other backends keep the original forward.
        # One query token past the indexer's budget: attend to the selected blocks only (no mask over the whole cache).
        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
            if (
                hidden_states.device.type == "mps"
                and hidden_states.shape[:2] == (1, 1)
                and past_key_values is not None
                and attention_mask is not None
                and attention_mask.dim() == 4
                and attention_mask.shape[-1] > self.indexer.token_budget
                and hidden_states.dtype in (torch.bfloat16, torch.float16)
                and self.head_dim in (128, 256)
                and not torch.is_grad_enabled()
                and indexer.enabled()
                and qsa_decode.enabled()
            ):
                selection = indexer.select(
                    self.indexer,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_values,
                    m.apply_rotary_pos_emb,
                    blocks_only=True,
                )
                if selection is not None:
                    return qsa_decode.attention_forward(self, hidden_states, position_embeddings, past_key_values, *selection)
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

        return forward

    _patch(m.Qwen4ExpTextAttention, "forward", attention_forward)


def load_qwen38(
    converted,
    *,
    cache_gib=12.0,
    bits=3,
    dense_bits="auto",
    io_threads=8,
    device="mps",
    cache_policy="lru",
    bypass_page_cache=False,
    mps_commit=True,
    host_cache_gib=0.0,
    host_preload=False,
    gguf_dense=None,
    prefetch_depth=0,
    gguf_experts=False,
):
    """Load a checkpoint written by ``scripts/convert_fp8.py``: ``dense/``, ``experts-q{bits}/`` and ``ngram/``.

    ``host_cache_gib`` / ``host_preload``: a tier of experts in pinned host memory between storage and the device
    cache, as Strata keeps its experts in RAM (needs a transformers with the host tier; for discrete GPUs).
    ``prefetch_depth``: layers ahead to guess and read experts (``experts_offload`` option). 0 by default: on the M5,
    guessing costs more than it hides (decode 11.1 -> 11.7 tok/s without it).
    ``gguf_dense``: GGUF file(s) whose quantized dense tensors replace the model's (see :mod:`gguf_dense`; MPS).
    ``gguf_experts``: run the routed experts in the GGUF's quantization, from Strata's ``.experts`` file beside it (see
    :mod:`gguf_experts`; needs ``gguf_dense``). ``cache_gib`` then sizes their cache; ``experts-q{bits}/`` is only read
    for its layout.
    ``dense_bits``: ``"auto"`` = 8-bit ``MetalConfig`` on MPS, BF16 elsewhere; an int quantizes with ``MetalConfig``
    (MPS only); ``None`` keeps BF16.
    """
    from transformers import MetalConfig, Qwen4ExpForCausalLM

    apply()
    on_mps = torch.device(device).type == "mps"
    if dense_bits == "auto":
        dense_bits = 8 if on_mps else None
    if dense_bits and not on_mps:
        raise ValueError("dense_bits needs MPS (MetalConfig); use dense_bits=None (BF16) on other devices")
    converted = Path(converted)
    with disk_ngram_table(converted / "ngram"):
        model = Qwen4ExpForCausalLM.from_pretrained(
            converted / "dense",
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="sdpa",  # the indexer's sparse mask rules out the flash kernels
            quantization_config=(
                MetalConfig(bits=dense_bits, group_size=64, modules_to_not_convert=["shared_expert_gate"])
                if dense_bits
                else None
            ),
            experts_offload={
                "cache_bytes": 0 if gguf_experts else int(cache_gib * 2**30),
                "quantized_experts": converted / f"experts-q{bits}",
                "io_threads": io_threads,
                "cache_policy": cache_policy,
                "bypass_page_cache": bypass_page_cache,
                "mps_commit": mps_commit,
                "prefetch_depth": prefetch_depth,
                **(
                    {"host_cache_bytes": int(host_cache_gib * 2**30), "host_preload": host_preload}
                    if host_cache_gib
                    else {}
                ),
            },
        )
    if dense_bits and on_mps:
        metal_linear.kernelize_layers(model)
    if gguf_dense:
        from . import gguf_dense as gguf_dense_module

        paths = [gguf_dense] if isinstance(gguf_dense, (str, Path)) else list(gguf_dense)
        gguf_dense_module.replace_dense(model, paths)
    if gguf_experts:
        from . import gguf_experts as gguf_experts_module

        if not gguf_dense:
            raise ValueError("gguf_experts needs gguf_dense (the GGUF the .experts file was made from)")
        gguf_experts_module.attach(
            model, paths[0], int(cache_gib * 2**30), io_threads=io_threads, bypass_page_cache=bypass_page_cache
        )
    return model.eval()
