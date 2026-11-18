"""Hub kernels, pinned by ``kernels.lock``, and this package's Metal kernels loaded from the Hub.

Every Hub kernel the package uses is declared in ``pyproject.toml`` (``[tool.kernels.dependencies]``) and pinned to a
commit in ``kernels.lock`` (``kernels lock .``); :func:`get_locked` loads that commit, from the cache when
``HF_HUB_OFFLINE=1`` (``kernels download .`` fetches them all beforehand).

Each module of :mod:`.kernels` that holds Metal source is published as its own Hub repository
(``tools/publish_hub_kernels.py``). The names at the bottom are lazy stand-ins for those modules: the first attribute
access loads the locked Hub build when MPS is available, and the local module otherwise (CPU/CUDA, where only their
pure-PyTorch parts and constants are used) or when ``MOE_STREAMING_LOCAL_KERNELS=1`` (to try a change to ``kernels/``
before publishing it).
"""

import functools
import importlib
import os
from pathlib import Path


NAMESPACE = "tarekziade"
REPOS = {
    "gguf_mm": "gguf-mm-metal",
    "q2_0": "q2-0-metal",
    "gated_delta_net": "gated-delta-net-decode-metal",
    "gdn_scan": "gdn-scan-metal",
    "qsa_decode": "qsa-decode-metal",
    "gated_norm": "gated-rms-norm-metal",
    "hyper_connection": "hyper-connection-metal",
}


def lockfile():
    """``kernels.lock``: inside the package for a wheel install, at the repository root for an editable one."""
    for path in (Path(__file__).parent / "kernels.lock", Path(__file__).parents[2] / "kernels.lock"):
        if path.is_file():
            return path
    raise FileNotFoundError("kernels.lock not found; run `kernels lock .` at the repository root")


@functools.cache
def get_locked(repo_id):
    """The Hub kernel ``repo_id`` at the commit ``kernels.lock`` pins."""
    from kernels import get_locked_kernel

    trusted = [f"{NAMESPACE}/{repo}" for repo in REPOS.values()]
    return get_locked_kernel(repo_id, lockfile=lockfile(), trust_remote_code=trusted)


def use_hub():
    import torch

    return os.environ.get("MOE_STREAMING_LOCAL_KERNELS", "0") != "1" and torch.backends.mps.is_available()


@functools.cache
def load(name):
    """The kernel module ``name`` (a key of :data:`REPOS`): the locked Hub build on MPS, else the local module."""
    if use_hub():
        return get_locked(f"{NAMESPACE}/{REPOS[name]}")
    return importlib.import_module(f"{__package__}.kernels.{name}")


class _Lazy:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        return getattr(load(self._name), attr)

    def __repr__(self):
        return f"<kernel {self._name} (lazy)>"


gguf_mm = _Lazy("gguf_mm")
q2_0 = _Lazy("q2_0")
gated_delta_net = _Lazy("gated_delta_net")
gdn_scan = _Lazy("gdn_scan")
qsa_decode = _Lazy("qsa_decode")
gated_norm = _Lazy("gated_norm")
hyper_connection = _Lazy("hyper_connection")
