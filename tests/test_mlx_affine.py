"""mlx_affine on CPU: packing, dequantization and gather_qmm (bit-exactness with MLX is checked on MPS separately)."""

import pytest
import torch

from moe_expert_streaming import mlx_affine


@pytest.mark.parametrize("bits", mlx_affine.BITS)
def test_pack_unpack_round_trip(bits):
    torch.manual_seed(bits)
    codes = torch.randint(0, 1 << bits, (5, 3, 128))
    packed = mlx_affine.pack(codes, bits)
    assert packed.dtype == torch.uint32 and packed.shape == (5, 3, 128 * bits // 32)
    assert torch.equal(mlx_affine.unpack(packed, 128, bits), codes)


@pytest.mark.parametrize("bits", mlx_affine.BITS)
def test_quantize_dequantize(bits):
    torch.manual_seed(0)
    weight = torch.randn(32, 256)
    packed, scales, biases = mlx_affine.quantize(weight, 64, bits)
    restored = mlx_affine.dequantize(packed, scales, biases, 64, bits, dtype=torch.float32)
    # MLX snaps each group's larger-magnitude edge onto the grid: the other edge may be off by up to one step.
    assert ((restored - weight).abs() <= scales.abs().repeat_interleave(64, dim=1) + 1e-6).all()


@pytest.mark.parametrize("rows", [6, 200])  # the batched (decode) and per-expert (prefill) paths
def test_gather_qmm_matches_dense(rows):
    torch.manual_seed(rows)
    experts, n, k, bits = 5, 48, 128, 3
    quantized = [mlx_affine.quantize(torch.randn(n, k), 64, bits) for _ in range(experts)]
    w, scales, biases = (torch.stack(t) for t in zip(*quantized))
    dense = torch.stack([mlx_affine.dequantize(*q, 64, bits, dtype=torch.float32) for q in quantized])
    x = torch.randn(rows, 1, k)
    rhs = torch.randint(0, experts, (rows,))
    got = mlx_affine.gather_qmm(x, w, scales, biases, rhs_indices=rhs, group_size=64, bits=bits)
    expected = torch.einsum("sk,snk->sn", x[:, 0], dense[rhs]).unsqueeze(1)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_experts_offload_runs_through_gather_qmm(tmp_path):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from transformers.integrations.experts_offload import quantize_experts

    from moe_expert_streaming import apply

    apply()
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
    )
    reference = Qwen3MoeForCausalLM(config).eval()
    reference.save_pretrained(tmp_path / "model")
    quantize_experts(tmp_path / "model", tmp_path / "q4", bits=4, group_size=64)
    model = Qwen3MoeForCausalLM.from_pretrained(
        tmp_path / "model",
        experts_offload={"cache_bytes": 1 << 20, "quantized_experts": tmp_path / "q4", "io_threads": 2},
    ).eval()
    assert model.experts_offload_store.gather_qmm is mlx_affine.gather_qmm
    # Same model with its experts replaced by their dequantized 4-bit values: the offloaded run must match it.
    for layer in reference.model.layers:
        experts = layer.mlp.experts
        for name in ("gate_up_proj", "down_proj"):
            bank = getattr(experts, name).data
            for e in range(bank.shape[0]):
                bank[e] = mlx_affine.dequantize(*mlx_affine.quantize(bank[e], 64, 4), 64, 4, dtype=bank.dtype)
    ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, reference(ids).logits, rtol=2e-3, atol=2e-3)
