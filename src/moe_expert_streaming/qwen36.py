"""Qwen3.6-35B-A3B (``Qwen3_5MoeForCausalLM``) on top of transformers' ``experts_offload``.

The same machinery as :mod:`qwen38`, without what only Qwen3.8 has (n-gram table, QSA indexer, hyper-connections): the
routed experts stream through ``experts_offload`` from ``experts-q8/`` (written by ``scripts/convert_qwen36.py``), the
dense layers run 8-bit ``MetalConfig`` on the v2 MLX kernels, and the Gated DeltaNet layers use the Metal kernels.
"""

from pathlib import Path

import torch

from . import metal_linear, patches


def apply():
    """Install the Qwen3.6 patches and the Metal dense-layer patches (idempotent)."""
    from transformers.cache_utils import DynamicLayer
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m

    metal_linear.apply()
    patches.patch_gather_qmm()
    patches.patch_gated_delta_net(m, m.Qwen3_5MoeGatedDeltaNet, m.Qwen3_5MoeRMSNormGated)
    patches.patch_cache_append(DynamicLayer)


def load_qwen36(
    converted,
    *,
    cache_gib=12.0,
    bits=8,
    dense_bits="auto",
    io_threads=8,
    device="mps",
    cache_policy="lru",
    bypass_page_cache=False,
    mps_commit=True,
    prefetch_depth=1,
    guess_limit=1,
    preload_profile=None,
):
    """Load a checkpoint written by ``scripts/convert_qwen36.py``: ``dense/`` and ``experts-q{bits}/``.

    Options as :func:`qwen38.load_qwen38`.
    """
    from transformers import MetalConfig, Qwen3_5MoeForCausalLM

    apply()
    on_mps = torch.device(device).type == "mps"
    if dense_bits == "auto":
        dense_bits = 8 if on_mps else None
    if dense_bits and not on_mps:
        raise ValueError("dense_bits needs MPS (MetalConfig); use dense_bits=None (BF16) on other devices")
    converted = Path(converted)
    model = Qwen3_5MoeForCausalLM.from_pretrained(
        converted / "dense",
        dtype=torch.bfloat16,
        device_map={"": device, "experts.*": "disk"},  # experts stream through experts_offload
        attn_implementation="sdpa",
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
            "quantized_experts": converted / f"experts-q{bits}",
            **({"preload_profile": preload_profile} if preload_profile else {}),
            **({"guess_limit": guess_limit} if guess_limit else {}),
        },
    )
    if dense_bits and on_mps:
        metal_linear.kernelize_layers(model)
    return model.eval()
