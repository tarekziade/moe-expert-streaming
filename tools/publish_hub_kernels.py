"""Publish this package's Metal kernels to the Kernel Hub, one repository each.

Every kernel here is Metal source compiled at first use with ``torch.mps.compile_shader``, so a Hub build is the
Python module as is: a noarch ``build/torch-metal`` variant (no kernel-builder, no binaries). Each repository gets
that variant on ``main`` and on the ``v1`` branch, so ``get_kernel(repo, version=1, trust_remote_code=True)`` loads it.

    python tools/publish_hub_kernels.py --stage-only        # build the trees under --out and stop
    python tools/publish_hub_kernels.py                     # build and upload
    python tools/publish_hub_kernels.py --verify            # load each from the Hub and compile it on MPS
"""

import argparse
import ast
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "moe_expert_streaming" / "kernels"
NAMESPACE = "tarekziade"
VERSION = 1

# module -> (repo name, license, extra files, one-line summary)
KERNELS = {
    "gguf_mm": (
        "gguf-mm-metal",
        "MIT",
        ["gguf_mm.metal"],
        "Tile GEMM over GGUF-quantized expert banks (Metal 4 `matmul2d`, blocks dequantized in the kernel, rows read "
        "through an index), plus the per-token route combine. Built for MoE expert prefill.",
    ),
    "q2_0": (
        "q2-0-metal",
        "Apache-2.0",
        [],
        "Q2_0 (ggml type 42) gather-gemv for routed experts (`mul_mat_id`) and a dequantize kernel.",
    ),
    "gated_delta_net": (
        "gated-delta-net-decode-metal",
        "Apache-2.0",
        [],
        "One Gated DeltaNet decode step in two kernels: conv state + conv + SiLU + gates, then delta rule + gated "
        "RMSNorm.",
    ),
    "gdn_scan": (
        "gdn-scan-metal",
        "MIT",
        [],
        "Gated DeltaNet prefill: causal conv + SiLU, q/k L2 norm, and the gated delta rule as one scan (a lane per "
        "state column).",
    ),
    "qsa_decode": (
        "qsa-decode-metal",
        "Apache-2.0",
        [],
        "Split-K flash decoding over a sparse set of selected key blocks plus the tail, and the QSA indexer's query "
        "norm/rotary and block scores.",
    ),
    "gated_norm": (
        "gated-rms-norm-metal",
        "Apache-2.0",
        [],
        "Gated RMSNorm (SiLU or sigmoid gate) in one kernel.",
    ),
    "hyper_connection": (
        "hyper-connection-metal",
        "Apache-2.0",
        [],
        "Gated hyper-connection (Qwen3.8 `GatedResidual`): norm, activation and mix kernels; one-token fused "
        "down+inject projection through the MLX quantized matmul.",
    ),
}

# The package-relative import two modules use for the MLX quantized matmul, replaced by a Hub load.
_RELATIVE = "from ..metal_linear import load_metal_kernel"
_HUB = (
    "from kernels import get_kernel\n\n"
    "        def load_metal_kernel():\n"
    '            return get_kernel("kernels-community/mlx-quantization-metal-kernels", version=2)'
)


def _git_commit():
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", str(SRC)], capture_output=True, text=True, check=True
    )
    return out.stdout.strip(), bool(dirty.stdout.strip())


def _module_source(module):
    text = (SRC / f"{module}.py").read_text()
    if _RELATIVE in text:
        # Keep the original indentation of each import site.
        lines = []
        for line in text.splitlines(keepends=True):
            if line.strip() == _RELATIVE:
                indent = line[: len(line) - len(line.lstrip())]
                hub = _HUB.replace("\n        ", "\n" + indent)
                lines.append(indent + hub + "\n")
            else:
                lines.append(line)
        text = "".join(lines)
    assert "from .." not in text and "from . " not in text, f"{module}: package-relative import left"
    return text


def _readme(module, repo, license, summary, public):
    funcs = "\n".join(f"- `{name}`" for name in public)
    return f"""---
license: {license.lower()}
tags:
- kernels
- metal
---

# {repo}

{summary}

Metal source compiled at first use with `torch.mps.compile_shader` (Apple silicon, PyTorch MPS). Published from
[moe-expert-streaming](https://huggingface.co/spaces/tarekziade/moe-expert-streaming), where it is
`moe_expert_streaming/kernels/{module}.py`.

```python
from kernels import get_kernel

{module} = get_kernel("{NAMESPACE}/{repo}", version={VERSION}, trust_remote_code=True)
```

Functions:

{funcs}

See the module docstrings for shapes and the conditions each kernel handles (`supports(...)` returns False when the
caller should keep its own code path).
"""


def _public(source):
    tree = ast.parse(source)
    return [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]


def stage(out):
    from kernels._rust import Digest, DigestAlgorithm

    commit, dirty = _git_commit()
    if out.exists():
        shutil.rmtree(out)
    for module, (repo, license, extra, summary) in KERNELS.items():
        variant = out / repo / "build" / "torch-metal"
        variant.mkdir(parents=True)
        source = _module_source(module)
        (variant / "__init__.py").write_text(source)
        for name in extra:
            shutil.copy(SRC / name, variant / name)
        digest = Digest.hash_variant(DigestAlgorithm.SHA256, variant)
        metadata = {
            "name": repo,
            "id": "_" + repo.replace("-", "_"),
            "version": VERSION,
            "kernels-minver": "0.14.0",
            "license": license,
            "python-depends": [],
            "kernel-depends": [],
            "backend": {"type": "metal"},
            "digest": {"algorithm": "sha256", "files": digest.files},
        }
        (variant / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (out / repo / "README.md").write_text(_readme(module, repo, license, summary, _public(source)))
        print(f"staged {NAMESPACE}/{repo} ({len(digest.files)} files)")
    if dirty:
        print(
            "warning: uncommitted changes under kernels/; the commit in the upload message does not identify the source"
        )


def upload(out):
    from huggingface_hub import HfApi

    api = HfApi()
    commit, _ = _git_commit()
    for repo in (KERNELS[m][0] for m in KERNELS):
        repo_id = f"{NAMESPACE}/{repo}"
        api.create_repo(repo_id, repo_type="kernel", exist_ok=True)
        info = api.upload_folder(
            repo_id=repo_id,
            repo_type="kernel",
            folder_path=out / repo,
            commit_message=f"Build from moe-expert-streaming {commit[:7]}",
            delete_patterns=["build/**"],
        )
        api.create_branch(repo_id, repo_type="kernel", branch=f"v{VERSION}", revision=info.oid, exist_ok=True)
        # An existing v1 branch: move it to the new commit.
        api.upload_folder(
            repo_id=repo_id,
            repo_type="kernel",
            folder_path=out / repo,
            revision=f"v{VERSION}",
            commit_message=f"Build from moe-expert-streaming {commit[:7]}",
            delete_patterns=["build/**"],
        )
        print(f"uploaded https://huggingface.co/kernels/{repo_id}")


def verify():
    import torch
    from kernels import get_kernel

    assert torch.backends.mps.is_available(), "verification compiles the Metal source: needs MPS"
    for module, (repo, *_rest) in KERNELS.items():
        kernel = get_kernel(f"{NAMESPACE}/{repo}", version=VERSION, trust_remote_code=True)
        local = _module_source(module)
        assert Path(kernel.__file__).read_text() == local, f"{repo}: Hub source differs from the local module"
        library = kernel._library()
        print(f"ok {NAMESPACE}/{repo}: {type(library).__name__} compiled, functions {_public(local)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "hub-kernels")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--verify", action="store_true", help="only load from the Hub and compile")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    stage(args.out)
    if not args.stage_only:
        upload(args.out)


if __name__ == "__main__":
    main()
