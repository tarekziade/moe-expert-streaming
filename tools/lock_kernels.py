"""Write kernels.lock: `kernels lock .`, with this package's own repositories trusted.

`kernels lock` (0.18) checks every repository's publisher with trust_remote_code=False and has no option to change
it; our kernels live under a user namespace, which that check cannot verify. This does the same lock (every Hub
kernel in `[tool.kernels.dependencies]` of pyproject.toml, at the commit its version branch points to now) with the
repositories of `hub_kernels.REPOS` allowlisted.

    python tools/lock_kernels.py        # then commit kernels.lock; `kernels download .` fetches the pinned builds
"""

import sys
from pathlib import Path

import kernels.locking as locking
from kernels._rust import KernelDependency, KernelVersion
from kernels.compat import tomllib
from kernels.hf_hub import _check_trust_remote_code, _get_hf_api


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moe_expert_streaming.hub_kernels import NAMESPACE, REPOS  # noqa: E402


def main():
    trusted = [f"{NAMESPACE}/{repo}" for repo in REPOS.values()]
    locking._check_trust_remote_code = lambda repo_id, local_files_only, trust_remote_code: _check_trust_remote_code(
        repo_id, local_files_only, trusted
    )
    with open(ROOT / "pyproject.toml", "rb") as f:
        versions = tomllib.load(f)["tool"]["kernels"]["dependencies"]
    depends = [KernelDependency(repo_id=repo, version=KernelVersion.Version(v)) for repo, v in versions.items()]
    locks = locking.extract_dependency_locks(depends, api=_get_hf_api())
    (ROOT / "kernels.lock").write_text(locks.to_json())
    print(f"locked {len(depends)} kernels in {ROOT / 'kernels.lock'}")


if __name__ == "__main__":
    main()
