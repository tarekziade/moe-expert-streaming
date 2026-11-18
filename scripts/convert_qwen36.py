"""Build a Qwen3.6-35B-A3B checkpoint for SSD streaming whose experts are byte-for-byte those of an MLX 8-bit conversion.

The point is a same-weights comparison with engines that serve the MLX checkpoint (vllm-metal): the routed experts are
repacked from ``mlx-community/Qwen3.6-35B-A3B-8bit`` (MLX affine, 8 bits, groups of 64) into the layout
``experts_offload`` reads, without requantizing. The dense layers are taken in BF16 from ``Qwen/Qwen3.6-35B-A3B``
(only their byte ranges are downloaded, ~4.9 GB) and quantized at load time by ``MetalConfig`` (MLX-exact 8-bit).

Output, under OUT_DIR:
- ``dense/``: the text model without experts, vision or MTP, in BF16, keys renamed ``model.language_model.*`` ->
  ``model.*``, a text-only config (``Qwen3_5MoeForCausalLM``) and the tokenizer files.
- ``experts-q8/``: ``model.layers.N.mlp.experts.experts`` [256, bytes] per layer, one contiguous row per expert:
  gate|up packed, scales, biases, then down packed, scales, biases (the ``quantize_experts`` layout).

``--check-experts N``: also quantize N experts of layer 0 from the BF16 source with :mod:`mlx_affine` and compare.

Usage: convert_qwen36.py OUT_DIR [--check-experts 4]
"""

import argparse
import json
import shutil
import struct
from pathlib import Path

import torch
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


BF16_REPO = "Qwen/Qwen3.6-35B-A3B"
MLX_REPO = "mlx-community/Qwen3.6-35B-A3B-8bit"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
)
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def remote_tensors(fs, repo, filename):
    """(name -> (dtype, shape, absolute byte range)) of a remote safetensors file, from its header only."""
    path = f"{repo}/{filename}"
    (length,) = struct.unpack("<Q", fs.cat_file(path, start=0, end=8))
    header = json.loads(fs.cat_file(path, start=8, end=8 + length))
    header.pop("__metadata__", None)
    return {
        name: (
            info["dtype"],
            info["shape"],
            (8 + length + info["data_offsets"][0], 8 + length + info["data_offsets"][1]),
        )
        for name, info in header.items()
    }


def read_remote(fs, repo, filename, dtype, shape, byte_range):
    raw = bytearray(fs.cat_file(f"{repo}/{filename}", start=byte_range[0], end=byte_range[1]))
    return torch.frombuffer(raw, dtype=DTYPES[dtype]).reshape(shape)


def is_dense(name):
    return name.startswith("model.language_model.") and ".mlp.experts." not in name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--check-experts", type=int, default=4)
    args = parser.parse_args()
    out = args.out
    dense_dir, experts_dir = out / "dense", out / "experts-q8"
    dense_dir.mkdir(parents=True, exist_ok=True)
    experts_dir.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()

    # Dense layers, BF16, by byte range.
    config = json.loads(Path(hf_hub_download(BF16_REPO, "config.json")).read_text(encoding="utf-8"))
    text = dict(config["text_config"])
    text["architectures"] = ["Qwen3_5MoeForCausalLM"]
    (dense_dir / "config.json").write_text(json.dumps(text, indent=2))
    for name in TOKENIZER_FILES:
        try:
            shutil.copy(hf_hub_download(BF16_REPO, name), dense_dir / name)
        except Exception as error:  # not every repo has every tokenizer file
            print(f"skipping {name}: {type(error).__name__}")
    index = json.loads(Path(hf_hub_download(BF16_REPO, "model.safetensors.index.json")).read_text(encoding="utf-8"))
    files = sorted({f for n, f in index["weight_map"].items() if is_dense(n) or n == "lm_head.weight"})
    weight_map, part = {}, 0
    for filename in files:
        tensors = {}
        for name, (dtype, shape, byte_range) in remote_tensors(fs, BF16_REPO, filename).items():
            if is_dense(name) or name == "lm_head.weight":
                key = name.replace("model.language_model.", "model.", 1)
                tensors[key] = read_remote(fs, BF16_REPO, filename, dtype, shape, byte_range).clone()
        if tensors:
            shard = f"dense-{part:05d}.safetensors"
            save_file(tensors, dense_dir / shard, {"format": "pt"})
            weight_map.update(dict.fromkeys(tensors, shard))
            part += 1
            print(f"dense {filename}: {len(tensors)} tensors", flush=True)
    (dense_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))

    # Experts, repacked from the MLX checkpoint.
    mlx = Path(snapshot_download(MLX_REPO, allow_patterns=["*.json", "*.safetensors"]))
    mlx_config = json.loads((mlx / "config.json").read_text(encoding="utf-8"))
    quantization = mlx_config["quantization"]
    bits, group_size = quantization["bits"], quantization["group_size"]
    mlx_map = json.loads((mlx / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    num_layers = text["num_hidden_layers"]
    expert_map = {}
    for layer in range(num_layers):
        prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp"
        parts = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for component in ("weight", "scales", "biases"):
                name = f"{prefix}.{projection}.{component}"
                with safe_open(mlx / mlx_map[name], framework="pt") as handle:
                    parts[projection, component] = handle.get_tensor(name)
        num_experts = parts["gate_proj", "weight"].shape[0]
        blob = []
        for component in ("weight", "scales", "biases"):
            blob.append(torch.cat((parts["gate_proj", component], parts["up_proj", component]), dim=1))
        for component in ("weight", "scales", "biases"):
            blob.append(parts["down_proj", component])
        rows = torch.cat([t.contiguous().view(torch.uint8).reshape(num_experts, -1) for t in blob], dim=1)
        name = f"experts-{layer:05d}.safetensors"
        save_file(
            {f"model.layers.{layer}.mlp.experts.experts": rows.contiguous()}, experts_dir / name, {"format": "pt"}
        )
        expert_map[f"model.layers.{layer}.mlp.experts.experts"] = name
        print(f"experts of layer {layer}: {rows.shape[1] / 1e6:.2f} MB per expert", flush=True)
    (experts_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": expert_map}))
    (experts_dir / "quantization.json").write_text(
        json.dumps(
            {
                "format": "mlx-affine-expert-blob",
                "bits": bits,
                "group_size": group_size,
                "quantizer": f"repacked from {MLX_REPO} (no requantization)",
                "source": MLX_REPO,
            },
            indent=2,
        )
    )

    if args.check_experts:
        from moe_expert_streaming import mlx_affine

        name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        filename = index["weight_map"][name]
        dtype, shape, (begin, _) = remote_tensors(fs, BF16_REPO, filename)[name]
        per_expert = shape[1] * shape[2] * 2
        down_name = name.replace("gate_up_proj", "down_proj")
        down_file = index["weight_map"][down_name]
        _, down_shape, (down_begin, _) = remote_tensors(fs, BF16_REPO, down_file)[down_name]
        down_per_expert = down_shape[1] * down_shape[2] * 2
        with safe_open(experts_dir / "experts-00000.safetensors", framework="pt") as handle:
            rows = handle.get_tensor("model.layers.0.mlp.experts.experts")
        exact = 0
        for expert in range(args.check_experts):
            gate_up = read_remote(
                fs,
                BF16_REPO,
                filename,
                dtype,
                shape[1:],
                (begin + expert * per_expert, begin + (expert + 1) * per_expert),
            )
            down = read_remote(
                fs,
                BF16_REPO,
                down_file,
                dtype,
                down_shape[1:],
                (down_begin + expert * down_per_expert, down_begin + (expert + 1) * down_per_expert),
            )
            mine = []
            for weight in (gate_up, down):
                mine += list(mlx_affine.quantize(weight, group_size, bits))
            mine = torch.cat([t.cpu().contiguous().view(torch.uint8).reshape(-1) for t in mine])
            same = torch.equal(mine, rows[expert])
            exact += same
            if not same:
                diff = (mine != rows[expert]).float().mean().item()
                print(f"expert {expert}: {diff:.4%} of bytes differ from mlx_affine.quantize of the BF16 source")
        print(f"check: {exact}/{args.check_experts} experts of layer 0 bit-exact with mlx_affine.quantize(BF16)")
    print("done")


if __name__ == "__main__":
    main()
