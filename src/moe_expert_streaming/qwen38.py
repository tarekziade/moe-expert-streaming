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

from . import indexer, metal_linear, patches
from .hub_kernels import hyper_connection, qsa_decode
from .patches import _append, _patch


class DiskEmbedding(torch.nn.Module):
    """An embedding table that stays on disk; each lookup reads only the rows it needs (through the OS page cache).

    ``directory`` holds ``ngram.json`` (``rows``, ``dim``, ``dtype``: ``float8_e4m3fn``, ``bfloat16`` or ``float32``,
    ``scale``, and either ``file`` or ``files`` + ``rows_per_file`` for a table split in row-aligned shards) and the raw
    row-major file(s), as ``scripts/convert_fp8.py`` writes them. Rows are dequantized as ``row * scale``.
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
        files = info["files"] if "files" in info else [info["file"]]
        self.rows_per_file = info.get("rows_per_file", self.rows)
        self.fds = [os.open(Path(directory) / f, os.O_RDONLY) for f in files]
        self.pool = ThreadPoolExecutor(read_threads, thread_name_prefix="disk-embedding") if read_threads else None
        self.dtype = dtype
        # Callers read `.weight.device` to place their indices; it follows the model to its device.
        self.register_buffer("weight", torch.empty(0, device="cpu", dtype=dtype), persistent=False)

    def _read_row(self, row):
        import os

        shard, row = divmod(row, self.rows_per_file)
        return os.pread(self.fds[shard], self.row_bytes, row * self.row_bytes)

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


def apply():
    """Install the Qwen3.8 patches and the Metal dense-layer patches (idempotent)."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    metal_linear.apply()

    patches.patch_gather_qmm()

    def ngram_init(original):
        def __init__(self, *args, **kwargs):
            original(self, *args, **kwargs)
            if _DISK_NGRAM["directory"] is not None:
                self.ngram_embedding = DiskEmbedding(_DISK_NGRAM["directory"])

        return __init__

    _patch(m.Qwen4ExpTextNGramEmbedding, "__init__", ngram_init)

    patches.patch_gated_delta_net(m, m.Qwen4ExpTextGatedDeltaNet, m.Qwen4ExpTextRMSNormGated)

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
                    return qsa_decode.attention_forward(
                        self, hidden_states, position_embeddings, past_key_values, *selection
                    )
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

        return forward

    _patch(m.Qwen4ExpTextAttention, "forward", attention_forward)

    # The full-attention layers' caches grow by torch.cat, which copies the whole cache at every decode step (1.5 ms
    # per token at 6K tokens over the 12 layers). Keep them as views of buffers grown in steps; appends write in place.
    from transformers.cache_utils import DynamicIndexedLayer

    def indexer_update(original):
        def update_indexer(self, indexer_key_states):
            if not self.is_indexer_initialized:
                self.lazy_initialization_indexer(indexer_key_states)
            self.indexer_keys = _append(self, "_indexer_buffer", self.indexer_keys, indexer_key_states, 1)
            return self.indexer_keys

        return update_indexer

    patches.patch_cache_append(DynamicIndexedLayer)
    _patch(DynamicIndexedLayer, "update_indexer", indexer_update)


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
    prefetch_depth=1,
    gguf_experts=False,
    preload_profile=None,
    guess_limit=1,
):
    """Load a checkpoint written by ``scripts/convert_fp8.py``: ``dense/``, ``experts-q{bits}/`` and ``ngram/``.

    ``host_cache_gib`` / ``host_preload``: a tier of experts in pinned host memory between storage and the device
    cache, as Strata keeps its experts in RAM (needs a transformers with the host tier; for discrete GPUs).
    ``prefetch_depth`` / ``guess_limit``: guess the next layer's experts with its router and read the best
    ``guess_limit`` non-resident ones once this layer's reads are done (``experts_offload`` options). 1 / 1 by default:
    +3% decode, +3% after a 6K document on the M5; reading every guess (``guess_limit=0``) costs more than it hides.
    ``gguf_dense``: GGUF file(s) whose quantized dense tensors replace the model's (see :mod:`gguf_dense`; MPS).
    ``gguf_experts``: run the routed experts in the GGUF's quantization, from Strata's ``.experts`` file beside it (see
    :mod:`gguf_experts`; needs ``gguf_dense``). ``cache_gib`` then sizes their cache (``experts-q{bits}/`` is not read).
    ``preload_profile``: a routing profile (JSON, see ``scripts/routing_profile.py``) whose most used experts fill the
    cache at the first forward.
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
    paths = [gguf_dense] if isinstance(gguf_dense, (str, Path)) else list(gguf_dense or [])
    if gguf_experts:
        from transformers import AutoConfig

        from . import gguf_experts as gguf_experts_module

        if not gguf_dense:
            raise ValueError("gguf_experts needs gguf_dense (the GGUF the .experts file was made from)")
        experts_options = gguf_experts_module.offload_options(
            paths[0], AutoConfig.from_pretrained(converted / "dense"), io_threads, bypass_page_cache
        )
    else:
        experts_options = {"quantized_experts": converted / f"experts-q{bits}"}
    with disk_ngram_table(converted / "ngram"):
        model = Qwen4ExpForCausalLM.from_pretrained(
            converted / "dense",
            dtype=torch.bfloat16,
            device_map={"": device, "experts.*": "disk"},  # experts stream through experts_offload
            attn_implementation="sdpa",  # the indexer's sparse mask rules out the flash kernels
            quantization_config=(
                MetalConfig(bits=dense_bits, group_size=64, modules_to_not_convert=["shared_expert_gate"])
                if dense_bits
                else None
            ),
            experts_offload={
                "cache_bytes": int(cache_gib * 2**30),
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
                **experts_options,
                **({"preload_profile": preload_profile} if preload_profile else {}),
                **({"guess_limit": guess_limit} if guess_limit else {}),
            },
        )
    if dense_bits and on_mps:
        metal_linear.kernelize_layers(model)
    if gguf_dense:
        from . import gguf_dense as gguf_dense_module

        gguf_dense_module.replace_dense(model, paths)
    if gguf_experts:
        store = model.experts_offload_store
        print(
            f"gguf_experts: {len(store.pools)} size classes, "
            f"{store.pools[0].num_slots * store.layouts[next(iter(store.layouts))][2] // store.pools[0].num_experts} "
            f"slots per layer, {sum(p.num_slots * p.slot_bytes for p in store.pools) / 2**30:.2f} GiB",
            flush=True,
        )
    return model.eval()
