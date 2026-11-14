"""Run MoE models larger than memory with transformers by streaming experts from the SSD.

Builds on ``from_pretrained(..., experts_offload=...)`` (huggingface/transformers#50717) and adds what is not (yet)
upstream: Qwen3.8-Flash-Next support (disk-backed n-gram table, Metal decode kernels, indexer shortcut) and the v2
MLX quantization kernels for ``MetalConfig`` dense layers (from huggingface/transformers#49467).
"""

from .qwen38 import DiskEmbedding, apply, disk_ngram_table, load_qwen38


__all__ = ["DiskEmbedding", "apply", "disk_ngram_table", "load_qwen38"]
__version__ = "0.1.0"
