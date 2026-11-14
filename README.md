# moe-expert-streaming

Run Mixture-of-Experts models larger than memory with 🤗 Transformers by streaming the experts from the SSD through a
bounded device cache. The flagship target: **Qwen3.8-Flash-Next** (48 layers × 512 experts, 121 GB of FP8 experts)
on a **32 GB MacBook Pro M5**.

Results, architecture and analysis: <https://huggingface.co/spaces/tarekziade/moe-expert-streaming>

| Qwen3.8-Flash-Next, M5 32 GB | |
|---|---|
| Decode | ~10 tok/s (median of 4 different 200-token answers, cold cache first) |
| Prefill, 4K-token prompt | 67–70 tok/s (1K chunks) |
| Device memory after load | ~18 GiB with a 12 GiB expert cache |
| Experts | 3-bit (49 GB on disk); dense layers 8-bit (perplexity +0.2% vs BF16) |

## How it fits with Transformers

The generic part lives in Transformers itself; this project adds what is not (yet) upstream.

| Piece | Where |
|---|---|
| `from_pretrained(..., experts_offload={...})`: expert cache, SSD reads, prefetch, quantized experts, early MPS commits | [huggingface/transformers#50717](https://github.com/huggingface/transformers/pull/50717) |
| `MetalConfig` dense layers on the v2 MLX kernels (`quantized_matmul`), MLX-exact quantization, ggml layer kernels | [huggingface/transformers#49467](https://github.com/huggingface/transformers/pull/49467), carried in `metal_linear.py` until merged |
| Qwen3.8: the 51 GB n-gram table read from disk, Metal decode kernels (Gated DeltaNet, hyper-connections), QSA indexer shortcut | this package (`qwen38.py`, `kernels/`) |
| Converter, benchmark, profiling and analysis scripts | this package (`scripts/`, `repro/`) |

Patches are applied at the class level by `moe_expert_streaming.apply()`; nothing in Transformers is edited.

## Install

Apple silicon, Python ≥ 3.10:

```bash
pip install -e .   # pulls transformers from the experts-offload branch (PR #50717)
```

## Use

Convert once (downloads `Qwen/Qwen3.8-Flash-Next-FP8`, ~130 GB, then writes ~105 GB; resumable, ~5 h):

```bash
python scripts/convert_fp8.py /path/to/qwen38 --bits 3
```

Then:

```python
from transformers import AutoTokenizer
from moe_expert_streaming import load_qwen38

model = load_qwen38("/path/to/qwen38", cache_gib=12, dense_bits=8)
tok = AutoTokenizer.from_pretrained("/path/to/qwen38/dense")
ids = tok.apply_chat_template([{"role": "user", "content": "Hello!"}], add_generation_prompt=True,
                              return_tensors="pt").to("mps")
print(tok.decode(model.generate(ids, max_new_tokens=100)[0, ids.shape[1]:]))
```

Other MoEs (OLMoE, Qwen3-MoE, ...) need nothing from this package: use `experts_offload` from Transformers directly.

## Benchmark

Close other apps first (a browser costs ~0.7 tok/s and pushes macOS into swap). The guarded runner kills a run on
critical memory pressure instead of freezing the Mac:

```bash
./scripts/run_guarded.sh decode.log scripts/bench.py /path/to/qwen38 --cache-gib 12 --dense-bits 8 \
    --io-threads 8 --skip-prefill --output decode.json
./scripts/run_guarded.sh prefill.log scripts/bench.py /path/to/qwen38 --cache-gib 10 --dense-bits 8 \
    --io-threads 8 --skip-story --prefill-chunk-size 1024 --output prefill.json
```

| Script | What it measures |
|---|---|
| `scripts/bench.py` | decode on 4 different prompts (cold cache first), 4K-token prefill |
| `scripts/decode_ppl.py` | teacher-forced perplexity through the decode path (one token per step) |
| `scripts/dense_ablation.py` | dense-layer cost per module category (experts zeroed, replay ablation) |
| `scripts/decode_profile.py` | host-side time split of the expert path (sync, read waits, bookkeeping) |
| `scripts/cache_policy_trace.py`, `scripts/cache_policy_sim.py` | routing traces replayed through LRU / LFU / pinning / Belady |
| `repro/mps_commit.py` | PyTorch MPS keeps small kernels uncommitted until the next sync |
| `repro/quantized_matmul_bf16.py` | M5 bf16 bug in the v2 dense quantized matmul (kernels-community#1332) |

Some scripts still hold the paths of the original experiments; adjust them to your checkout.

## Portability

Everything Metal-specific is marked `TODO(portability)`: the decode kernels run only on MPS, other devices fall back to
the original Transformers code paths.

## License

Apache-2.0
