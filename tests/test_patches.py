"""CPU tests: the patches keep the original results (Metal kernels are only exercised on MPS)."""

import json

import numpy as np
import pytest
import torch

from moe_expert_streaming import DiskEmbedding, apply
from moe_expert_streaming.metal_linear import affine_quantize


def test_disk_embedding_reads_rows(tmp_path):
    table = (np.arange(6 * 4, dtype=np.float32).reshape(6, 4) / 7).astype(np.float32)
    table.tofile(tmp_path / "table.bin")
    (tmp_path / "ngram.json").write_text(
        json.dumps({"rows": 6, "dim": 4, "dtype": "float32", "scale": 2.0, "file": "table.bin"})
    )
    embedding = DiskEmbedding(tmp_path, dtype=torch.float32)
    ids = torch.tensor([[5, 0], [2, 2]])
    torch.testing.assert_close(embedding(ids), torch.from_numpy(table)[ids] * 2.0)


@pytest.mark.parametrize("bits", [4, 8])
def test_affine_quantize_round_trip(bits):
    from transformers.integrations.metal_quantization import _affine_dequantize_tensor

    torch.manual_seed(0)
    weight = torch.randn(16, 128)
    packed, scales, biases = affine_quantize(weight, 64, bits)
    restored = _affine_dequantize_tensor(packed, scales, biases, 64, bits).float()
    # MLX snaps each group's larger-magnitude edge onto the grid, so the other edge may be off by up to one step.
    step = scales.abs().repeat_interleave(64, dim=1)
    assert ((restored - weight).abs() <= step + 1e-6).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Metal kernel")
@pytest.mark.parametrize("bits", [4, 8])
def test_affine_quantize_reference_matches_mlx(bits):
    torch.manual_seed(0)
    weight = torch.randn(64, 256, dtype=torch.bfloat16)
    expected = affine_quantize(weight.to("mps"), 64, bits)
    for got, want in zip(affine_quantize(weight, 64, bits), expected):
        torch.testing.assert_close(got, want.cpu(), rtol=0, atol=0)


def tiny_qwen38():
    from transformers import Qwen4ExpForCausalLM
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    # As transformers' Qwen4ExpTextModelTester, with an indexer budget the prompt fits first and outgrows later.
    config = Qwen4ExpTextConfig(
        vocab_size=99,
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 1]},
        layer_types=["linear_attention", "indexed_attention"],
        linear_conv_kernel_dim=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        hc_count=2,
        hc_lowrank=8,
        ple_layer_ids=[1],
        ple_embed_dim=16,
        ple_conv_kernel_size=2,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=16,
        indexer_compress_ratio=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        eos_token_id=100099,
        pad_token_id=0,
    )
    return Qwen4ExpForCausalLM(config)


def test_patched_model_matches_original_on_cpu():
    torch.manual_seed(0)
    reference = tiny_qwen38().eval()
    ids = torch.randint(1, 99, (1, 12))
    with torch.no_grad():
        expected = reference.generate(ids, max_new_tokens=8, do_sample=False)
        apply()  # CPU: Metal steps fall through to the originals; the indexer shortcut (12 keys, budget 16) keeps the same tokens
        actual = reference.generate(ids, max_new_tokens=8, do_sample=False)
    torch.testing.assert_close(actual, expected)
