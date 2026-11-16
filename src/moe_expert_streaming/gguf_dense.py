"""Dense layers from a GGUF file (e.g. Strata's Qwen3.8-Flash-Next quantization), run on ggml's kernels.

``replace_dense(model, gguf_paths)`` swaps the model's large dense projections (Gated DeltaNet and attention
projections, shared experts, ``lm_head``) for :class:`GgufLinear` modules that keep the GGUF's packed blocks as they are
(K-quants, IQ quants, ...) and multiply with ``ggml-org/ggml-quantization``: the fused dequantize-gemv for decode (up to
``MAX_GEMV_ROWS`` rows), dequantize + matmul above that (prefill). Tensors in a type the kernel's gemv does not cover on
this device, or stored unquantized (BF16), keep the model's own weights.

llama.cpp stores the Gated DeltaNet value heads in "tiled" order (value head ``r * num_k_heads + k`` is the model's head
``k * ratio + r``). Rows are put back in the model's order at load (free: packed blocks are row-major); the out
projection, permuted along its packed input dimension, gathers its input into tiled order instead.
"""

import functools
import os
import struct

import torch


# ggml type -> (values per block, bytes per block), for the types read here.
_BLOCKS = {
    8: (32, 34),  # Q8_0
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
}

# model module suffix -> GGUF tensor suffix
_LAYER_MAP = {
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.out_proj": "ssm_out",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.shared_expert.gate_proj": "ffn_gate_shexp",
    "mlp.shared_expert.up_proj": "ffn_up_shexp",
    "mlp.shared_expert.down_proj": "ffn_down_shexp",
}


@functools.cache
def load_kernel():
    from kernels import get_kernel

    return get_kernel("ggml-org/ggml-quantization", version=1)


class GgufLinear(torch.nn.Module):
    """``y = x @ W.T`` with ``W`` kept as GGUF blocks ``(out_features, bytes_per_row)`` uint8."""

    def __init__(self, blocks, ggml_type, in_features, out_features, input_perm=None):
        super().__init__()
        self.register_buffer("blocks", blocks, persistent=False)
        self.register_buffer("input_perm", input_perm, persistent=False)
        self.ggml_type, self.in_features, self.out_features = ggml_type, in_features, out_features
        self.bias = None

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, ggml_type={self.ggml_type}"

    def forward(self, x):
        kernel = load_kernel()
        flat = x.reshape(-1, self.in_features)
        if self.input_perm is not None:
            flat = flat.index_select(1, self.input_perm)
        if flat.shape[0] <= kernel.MAX_GEMV_ROWS:
            # f32 out; cast here
            y = kernel.mul_mat_vec(self.blocks, flat, self.ggml_type, self.out_features).to(x.dtype)
        else:
            weight = kernel.dequantize(self.blocks, self.ggml_type, self.out_features, self.in_features, x.dtype)
            y = flat @ weight.T
        return y.view(*x.shape[:-1], self.out_features)


class GgufFile:
    """Tensor directory of one or more GGUF shards: ``name -> (path, absolute offset, dims, ggml type)``."""

    def __init__(self, paths):
        self.tensors = {}
        for path in paths:
            self._read_header(str(path))

    def _read_header(self, path):
        sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        with open(path, "rb") as f:

            def rd(fmt):
                return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

            def rstr():
                return f.read(rd("Q")).decode("utf-8", "replace")

            def rval(kind):
                if kind == 8:
                    return rstr()
                if kind == 9:
                    item, count = rd("I"), rd("Q")
                    return [rval(item) for _ in range(count)]
                return rd(sizes[kind])

            if f.read(4) != b"GGUF":
                raise ValueError(f"{path} is not a GGUF file")
            rd("I")
            n_tensors, n_kv = rd("Q"), rd("Q")
            alignment = 32
            for _ in range(n_kv):
                key = rstr()
                value = rval(rd("I"))
                if key == "general.alignment":
                    alignment = value
            infos = []
            for _ in range(n_tensors):
                name = rstr()
                dims = [rd("Q") for _ in range(rd("I"))]
                infos.append((name, dims, rd("I"), rd("Q")))
            data_start = (f.tell() + alignment - 1) // alignment * alignment
        for name, dims, ggml_type, offset in infos:
            self.tensors[name] = (path, data_start + offset, dims, ggml_type)

    def read_blocks(self, name):
        """``(rows, bytes_per_row)`` uint8 for a 2D quantized tensor, with its type and column count."""
        path, offset, dims, ggml_type = self.tensors[name]
        cols, rows = dims
        per_block, block_bytes = _BLOCKS[ggml_type]
        row_bytes = cols // per_block * block_bytes
        fd = os.open(path, os.O_RDONLY)
        try:
            raw = bytearray(os.pread(fd, rows * row_bytes, offset))
        finally:
            os.close(fd)
        return torch.frombuffer(raw, dtype=torch.uint8).view(rows, row_bytes), ggml_type, cols


def _tiled_head_order(module):
    """``order[h]`` = tiled position of the model's value head ``h``."""
    kh, vh = module.num_k_heads, module.num_v_heads
    ratio = vh // kh
    return torch.tensor([(h % ratio) * kh + h // ratio for h in range(vh)])


def _rows_by_head(blocks, order, head_rows):
    """Model-order rows from tiled-order rows (blocks of ``head_rows`` per head)."""
    return blocks.view(len(order), head_rows, -1)[order].reshape(blocks.shape)


def _set_module(model, name, module):
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, module)


def replace_dense(model, gguf_paths, include_lm_head=True, verbose=True):
    """Replace the mapped dense projections of a ``Qwen4ExpForCausalLM`` by :class:`GgufLinear`; returns a summary."""
    kernel = load_kernel()
    gguf = GgufFile(gguf_paths)
    device = next(model.parameters()).device
    replaced, kept, saved = {}, {}, 0
    targets = []
    for layer_idx, layer in enumerate(model.model.layers):
        for suffix, gguf_suffix in _LAYER_MAP.items():
            try:
                layer.get_submodule(suffix)
            except AttributeError:
                continue
            targets.append((f"model.layers.{layer_idx}.{suffix}", f"blk.{layer_idx}.{gguf_suffix}.weight", layer))
    if include_lm_head:
        targets.append(("lm_head", "output.weight", None))
    for name, gguf_name, layer in targets:
        if gguf_name not in gguf.tensors:
            kept[name] = "missing"
            continue
        ggml_type = gguf.tensors[gguf_name][3]
        if ggml_type not in _BLOCKS or ggml_type not in kernel.GEMV_TYPES:
            kept[name] = f"type {ggml_type}"
            continue
        old = model.get_submodule(name)
        blocks, ggml_type, cols = gguf.read_blocks(gguf_name)
        input_perm = None
        if ".linear_attn." in name:
            gdn = layer.linear_attn
            order = _tiled_head_order(gdn)
            if name.endswith("in_proj_qkv"):
                v = blocks[2 * gdn.key_dim :]
                blocks = torch.cat([blocks[: 2 * gdn.key_dim], _rows_by_head(v, order, gdn.head_v_dim)])
            elif name.endswith("in_proj_z"):
                blocks = _rows_by_head(blocks, order, gdn.head_v_dim)
            elif name.endswith("out_proj"):
                # tiled position t holds model head perm[t]
                perm = torch.empty_like(order)
                perm[order] = torch.arange(len(order))
                input_perm = (perm[:, None] * gdn.head_v_dim + torch.arange(gdn.head_v_dim)).reshape(-1).to(device)
        new = GgufLinear(blocks.contiguous().to(device), ggml_type, cols, blocks.shape[0], input_perm)
        saved += sum(t.numel() * t.element_size() for t in old.state_dict().values()) - new.blocks.numel()
        _set_module(model, name, new)
        replaced[name] = ggml_type
    if device.type == "mps":
        torch.mps.empty_cache()
    summary = {"replaced": len(replaced), "kept": kept, "bytes_saved": saved}
    if verbose:
        print(
            f"gguf_dense: replaced {len(replaced)} projections, kept {len(kept)} ({sorted(set(kept.values()))}), "
            f"{saved / 1e9:.2f} GB less",
            flush=True,
        )
    return summary
