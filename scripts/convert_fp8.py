"""Convert Qwen/Qwen3.8-Flash-Next-FP8 for SSD streaming on a 32 GB Mac, one shard at a time (peak disk: one shard).

Output, under OUT_DIR:
- ``dense/``: the text model without experts and without the n-gram table, dequantized to BF16 (~9.9 GB), keys renamed
  from ``model.language_model.*`` to ``model.*``, a text-only config, and the tokenizer files. Vision and MTP are skipped.
- ``experts-q{bits}/``: every routed expert quantized with MLX affine quantization (``quantize`` from
  kernels-community/mlx-quantization-metal-kernels on MPS), one contiguous row per expert, in the layout
  ``transformers.integrations.ssd_moe.quantize_experts`` writes (``model.layers.N.mlp.experts.experts`` [512, bytes]).
- ``ngram/``: the 51.2 GB hashed n-gram table, kept in FP8 E4M3 exactly as published, as one raw [rows, 160] file
  plus its scalar scale, for a disk-backed embedding that reads only the rows each token needs.

FP8 weights carry 128x128 block scales (``weight_scale_inv``): weight = fp8 * scale, per block.
Usage: convert_fp8.py OUT_DIR [--bits 3] [--group-size 64] [--revision REV]
"""

import argparse
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

from moe_expert_streaming import mlx_affine


REPO = "Qwen/Qwen3.8-Flash-Next-FP8"
EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
NGRAM = re.compile(r"^model\.language_model\.layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: the first available)")
    args = parser.parse_args()
    default = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out, device = args.out, torch.device(args.device or default)
    staging, dense_dir = out / "staging", out / "dense"
    experts_dir, ngram_dir = out / f"experts-q{args.bits}", out / "ngram"
    for directory in (staging, dense_dir, experts_dir, ngram_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # FP8 E4M3 -> float32 for all 256 bit patterns, so dequantization is a gather on the GPU.
    fp8_values = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().to(device)

    def dequantize(raw, scale_inv):
        weight = fp8_values[raw.to(device).view(torch.uint8).int()]
        if scale_inv is not None:
            rows, columns = weight.shape
            scale = (
                scale_inv.to(device).float().repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :columns]
            )
            weight = weight * scale
        return weight

    index = json.loads(
        Path(hf_hub_download(REPO, "model.safetensors.index.json", revision=args.revision)).read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    config = json.loads(Path(hf_hub_download(REPO, "config.json", revision=args.revision)).read_text(encoding="utf-8"))
    text = dict(config["text_config"])
    text.pop("quantization_config", None)
    text["architectures"] = ["Qwen4ExpForCausalLM"]
    num_experts, num_layers = text["num_experts"], text["num_hidden_layers"]
    (dense_dir / "config.json").write_text(json.dumps(text, indent=2))
    for name in TOKENIZER_FILES:
        try:
            shutil.copy(hf_hub_download(REPO, name, revision=args.revision), dense_dir / name)
        except Exception as error:  # not every repo has every tokenizer file
            print(f"skipping {name}: {type(error).__name__}")

    # Resumable per file. Experts are buffered per layer until all of a layer's 512 are quantized (a layer spans two
    # files), so a restart reprocesses every file holding experts of a layer that is not written yet.
    progress_path = out / "progress.json"
    progress = {"done": [], "layers_done": [], "dense_map": {}, "ngram_parts": []}
    if progress_path.exists():
        progress.update(json.loads(progress_path.read_text(encoding="utf-8")))
    layers_done = set(progress["layers_done"])
    layer_files = defaultdict(set)
    for name, filename in weight_map.items():
        if match := EXPERT.match(name):
            layer_files[int(match[1])].add(filename)
    redo = {f for layer, names in layer_files.items() if layer not in layers_done for f in names}
    progress["done"] = [f for f in progress["done"] if f not in redo]
    pending_experts = defaultdict(dict)  # layer -> expert -> {projection: bf16 weight on device}
    expert_rows = defaultdict(dict)  # layer -> expert -> uint8 blob (CPU)
    dense, dense_bytes, dense_map = {}, 0, progress["dense_map"]
    dense_files = len(list(dense_dir.glob("dense-*.safetensors")))
    ngram_rows = text["split_ngram_parts"] * 2500012  # checked against the first part below
    ngram_shape, ngram_file = (ngram_rows, 160), ngram_dir / "ngram.fp8"

    def flush_dense(final=False):
        nonlocal dense, dense_bytes, dense_files
        if dense and (final or dense_bytes > 2e9):
            name = f"dense-{dense_files:05d}.safetensors"
            save_file(dense, dense_dir / name, {"format": "pt"})
            dense_map.update(dict.fromkeys(dense, name))
            dense_files += 1
            dense, dense_bytes = {}, 0

    def finish_expert(layer, expert):
        parts = pending_experts[layer].pop(expert)
        blob = []
        for weight in (torch.cat((parts["gate"], parts["up"])), parts["down"]):
            packed, scales, biases = mlx_affine.quantize(weight.to(torch.bfloat16), args.group_size, args.bits)
            blob += [packed, scales, biases]
        expert_rows[layer][expert] = torch.cat([t.cpu().contiguous().view(torch.uint8).reshape(-1) for t in blob])
        if len(expert_rows[layer]) == num_experts:
            rows = expert_rows.pop(layer)
            layers_done.add(layer)
            name = f"experts-{layer:05d}.safetensors"
            save_file(
                {f"model.layers.{layer}.mlp.experts.experts": torch.stack([rows[e] for e in range(num_experts)])},
                experts_dir / name,
                {"format": "pt"},
            )
            print(f"  experts of layer {layer} written ({rows[0].numel() / 1e6:.2f} MB per expert)", flush=True)

    files = sorted(set(weight_map.values()))
    started = time.perf_counter()
    for count, filename in enumerate(files, 1):
        if filename in progress["done"]:
            continue
        if shutil.disk_usage(out).free < 8e9:
            raise RuntimeError("Less than 8 GB free; stopping before the next shard")
        path = Path(hf_hub_download(REPO, filename, revision=args.revision, local_dir=staging))
        with safe_open(path, framework="pt") as handle:
            names = list(handle.keys())
            for name in names:
                if name.endswith("weight_scale_inv") or name.startswith(("mtp.", "model.visual.")):
                    continue
                expert = EXPERT.match(name)
                ngram = NGRAM.match(name)
                if expert:
                    layer, index_, projection = int(expert[1]), int(expert[2]), expert[3]
                    if layer in layers_done:
                        continue
                    scale = handle.get_tensor(name + "_scale_inv") if name + "_scale_inv" in names else None
                    weight = dequantize(handle.get_tensor(name), scale).to(torch.bfloat16)
                    pending_experts[layer].setdefault(index_, {})[projection] = weight
                    if len(pending_experts[layer][index_]) == 3:
                        finish_expert(layer, index_)
                elif ngram:
                    tensor = handle.get_tensor(name)
                    if tuple(tensor.shape) != (ngram_rows // text["split_ngram_parts"], ngram_shape[1]):
                        raise ValueError(f"Unexpected n-gram part shape {tuple(tensor.shape)}")
                    if not ngram_file.exists():
                        with ngram_file.open("wb") as raw:
                            raw.truncate(ngram_shape[0] * ngram_shape[1])
                    with ngram_file.open("r+b") as raw:
                        raw.seek(int(ngram[2]) * tensor.numel())
                        raw.write(tensor.view(torch.uint8).numpy().tobytes())
                    progress["ngram_parts"] = sorted(set(progress["ngram_parts"]) | {int(ngram[2])})
                elif name.endswith("ngram_embedding.weight_scale"):
                    (ngram_dir / "scale.json").write_text(
                        json.dumps({"scale": handle.get_tensor(name).float().item()})
                    )
                else:
                    tensor = handle.get_tensor(name)
                    if tensor.dtype == torch.float8_e4m3fn:
                        scale = handle.get_tensor(name + "_scale_inv") if name + "_scale_inv" in names else None
                        tensor = dequantize(tensor, scale).to(torch.bfloat16).cpu()
                    key = name.replace("model.language_model.", "model.", 1)
                    dense[key] = tensor.contiguous()
                    dense_bytes += tensor.numel() * tensor.element_size()
                    flush_dense()
        flush_dense(final=True)  # dense tensors of a done file are on disk before it is marked done
        path.unlink()
        progress["done"].append(filename)
        progress["layers_done"] = sorted(layers_done)
        progress_path.write_text(json.dumps(progress))
        elapsed = time.perf_counter() - started
        print(f"[{count}/{len(files)}] {filename} done, {elapsed / 60:.1f} min", flush=True)
    if any(pending_experts.values()) or expert_rows or len(layers_done) != num_layers:
        raise RuntimeError(f"Incomplete experts: {sorted(pending_experts)} {sorted(expert_rows)}")
    if len(progress["ngram_parts"]) != text["split_ngram_parts"]:
        raise RuntimeError(f"Missing n-gram parts: {len(progress['ngram_parts'])} of {text['split_ngram_parts']}")
    (dense_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": dense_map}))
    expert_map = {
        f"model.layers.{layer}.mlp.experts.experts": f"experts-{layer:05d}.safetensors" for layer in range(num_layers)
    }
    (experts_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": expert_map}))
    info = {
        "format": "mlx-affine-expert-blob",
        "bits": args.bits,
        "group_size": args.group_size,
        "quantizer": "mlx-affine (moe_expert_streaming.mlx_affine, bit-exact with mx.quantize)",
        "source": f"{REPO}@{args.revision}",
    }
    (experts_dir / "quantization.json").write_text(json.dumps(info, indent=2))
    scale = json.loads((ngram_dir / "scale.json").read_text(encoding="utf-8"))["scale"]
    (ngram_dir / "ngram.json").write_text(
        json.dumps(
            {
                "rows": ngram_shape[0],
                "dim": ngram_shape[1],
                "dtype": "float8_e4m3fn",
                "scale": scale,
                "file": "ngram.fp8",
            },
            indent=2,
        )
    )
    shutil.rmtree(staging, ignore_errors=True)
    print("done")


if __name__ == "__main__":
    main()
