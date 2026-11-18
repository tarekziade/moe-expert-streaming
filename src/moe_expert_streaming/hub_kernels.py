"""This package's Metal kernels, loaded from the Kernel Hub.

Each module of :mod:`.kernels` that holds Metal source is published as its own Hub repository
(``tools/publish_hub_kernels.py``). The names here are lazy stand-ins for those modules: the first attribute access
loads the Hub build (``get_kernel(..., version=1)``) when MPS is available, and the local module otherwise (CPU/CUDA,
where only their pure-PyTorch parts and constants are used) or when ``MOE_STREAMING_LOCAL_KERNELS=1`` (to try a change
to ``kernels/`` before publishing it).
"""

import functools
import importlib
import os


NAMESPACE, VERSION = "tarekziade", 1
REPOS = {
    "gguf_mm": "gguf-mm-metal",
    "q2_0": "q2-0-metal",
    "gated_delta_net": "gated-delta-net-decode-metal",
    "gdn_scan": "gdn-scan-metal",
    "qsa_decode": "qsa-decode-metal",
    "gated_norm": "gated-rms-norm-metal",
    "hyper_connection": "hyper-connection-metal",
}


def use_hub():
    import torch

    return os.environ.get("MOE_STREAMING_LOCAL_KERNELS", "0") != "1" and torch.backends.mps.is_available()


@functools.cache
def load(name):
    """The kernel module ``name`` (a key of :data:`REPOS`): the Hub build on MPS, else the local module."""
    if use_hub():
        from kernels import get_kernel

        repo_id = f"{NAMESPACE}/{REPOS[name]}"
        return get_kernel(repo_id, version=VERSION, trust_remote_code=[repo_id])
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
