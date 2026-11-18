"""Routed experts in their GGUF quantization (Strata's IQ2/IQ3 gate/up, IQ4_NL or Q2_0 down), streamed by the
``experts_offload`` store from Strata's ``.experts`` file.

Strata's ``<gguf>.experts`` file holds every expert's gate, up and down GGUF blocks contiguous (one read per expert):
header ``"STRX"``, u32 version (1), u32 layers, u32 experts, u64 source GGUF size; at byte 64 one ``(offset, gate
bytes, up bytes, down bytes)`` u64 row per layer; expert ``e`` of layer ``l`` at ``offset + e * (gate + up + down)``.
Gate and up share a type in every layer, so ``gate | up`` is one ``(2 * intermediate, row_bytes)`` matrix.

:class:`GgufExpertSource` is the store's ``expert_source``: expert sizes differ by layer (1.36 to 2.33 MB for
Qwen3.8-Flash-Next), so the store keeps one pool per layout (as Strata's arena), with the same number of slots per
layer. :func:`compute` is its ``expert_compute``: decode runs ggml's ``mul_mat_id`` (``ggml-org/ggml-quantization``)
on the pools directly, and :mod:`.kernels.q2_0` for Q2_0 down projections; prefill runs the tile kernels of
:mod:`.kernels.gguf_mm` (or, off MPS, dequantizes a group of experts at a time). :func:`offload_options` gives the
``experts_offload`` options.
"""

import fcntl
import os
import re
import struct
import threading
from pathlib import Path

import torch

from .gguf_dense import GgufFile, load_kernel
from .hub_kernels import q2_0


# ggml type -> (values per block, bytes per block), expert types.
_BLOCKS = {16: (256, 66), 17: (256, 74), 18: (256, 98), 20: (32, 18), 21: (256, 110), 22: (256, 82), 42: (64, 18)}

# Routes up to which the experts run as gemvs on the pools (decode, short prompts); above, groups of experts are
# dequantized (each dequantized expert is 9.8 MB of bf16, so groups stay small).
GEMV_MAX_ROUTES = 512
# Calls with at least this many tokens (long prefill chunks) leave the experts they install at the cold end of the LRU
# and do not refresh the ones they hit: a chunk cycles through nearly every expert of a layer, more than its pool holds, so
# plain LRU evicts each expert just before the next chunk needs it, and leaves decode a cache of scan leftovers. Replay
# of a 6K-token document (2K chunks): prefill misses 50.2K -> 36.2K, misses per decode token after it 66.6 -> 61.5, the
# story prompts unchanged (their prompts are short). 0 turns it off.
COLD_PREFILL_TOKENS = int(os.environ.get("GGUF_EXPERTS_COLD_PREFILL", "512"))
PREFILL_EXPERT_GROUP = 32


class ExpertsFile:
    """Strata's ``.experts`` file: ``layers[l] = (offset, gate_bytes, up_bytes, down_bytes)``."""

    def __init__(self, path, bypass_page_cache=True):
        self.path = str(path)
        with open(self.path, "rb") as f:
            header = f.read(64)
            if header[:4] != b"STRX":
                raise ValueError(f"{path} is not a Strata .experts file")
            version, self.num_layers, self.num_experts = struct.unpack("<III", header[4:16])
            if version != 1:
                raise ValueError(f"Unsupported .experts version {version}")
            self.layers = [struct.unpack("<4Q", f.read(32)) for _ in range(self.num_layers)]
        self.fd = os.open(self.path, os.O_RDONLY)
        if bypass_page_cache and hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(self.fd, fcntl.F_NOCACHE, 1)
        self.read_bytes = self.read_calls = 0
        self.lock = threading.Lock()

    def expert_bytes(self, layer):
        return sum(self.layers[layer][1:])

    def read(self, layer, expert, out):
        """Read one expert's gate | up | down blocks into ``out`` (a CPU uint8 tensor of ``expert_bytes(layer)``)."""
        offset, *sizes = self.layers[layer]
        size = sum(sizes)
        view = memoryview(out.numpy()).cast("B")
        start, done = offset + expert * size, 0
        while done < size:
            count = os.preadv(self.fd, [view[done:size]], start + done)
            if not count:
                raise OSError(f"Short read: {self.path}, layer {layer} expert {expert}")
            done += count
        with self.lock:
            self.read_bytes += size
            self.read_calls += 1
        return out


class GgufExpertSource:
    """The ``experts_offload`` store's ``expert_source`` for Strata's ``.experts`` file: per layer, the experts' ggml
    types (from the GGUF) and their layout, ``[(2 * intermediate, gate row bytes), (hidden, down row bytes)]`` uint8."""

    def __init__(self, experts_file, gguf, hidden, intermediate):
        self.file = experts_file
        self.gguf = gguf
        self.hidden, self.intermediate = hidden, intermediate
        self.info = {}  # prefix -> (layer, gate type, gate row bytes, down type, down row bytes)

    @property
    def read_bytes(self):
        return self.file.read_bytes

    @property
    def read_calls(self):
        return self.file.read_calls

    def prefix(self, module_name):
        return module_name

    def _info(self, prefix):
        info = self.info.get(prefix)
        if info is None:
            layer = int(re.search(r"layers\.(\d+)\.", prefix).group(1))
            _, gate_bytes, up_bytes, down_bytes = self.file.layers[layer]
            types = [self.gguf.tensors[f"blk.{layer}.ffn_{p}_exps.weight"][3] for p in ("gate", "up", "down")]
            if types[0] != types[1] or gate_bytes != up_bytes:
                raise ValueError(f"layer {layer}: gate and up differ ({types[:2]})")
            rows = [_row_bytes(types[0], self.hidden), _row_bytes(types[2], self.intermediate)]
            if (gate_bytes, down_bytes) != (rows[0] * self.intermediate, rows[1] * self.hidden):
                raise ValueError(f"layer {layer}: .experts sizes do not match the GGUF types {types}")
            info = self.info[prefix] = (layer, types[0], rows[0], types[2], rows[1])
        return info

    def layout(self, prefix):
        _, _, gate_row, _, down_row = self._info(prefix)
        return [((2 * self.intermediate, gate_row), torch.uint8), ((self.hidden, down_row), torch.uint8)]

    def pool_key(self, prefix):
        return self._info(prefix)[1:]

    def read(self, prefix, expert, out):
        return self.file.read(self._info(prefix)[0], expert, out)


def _row_bytes(ggml_type, columns):
    values, size = _BLOCKS[ggml_type]
    return columns // values * size


_KERNEL = []


def _kernel():
    if not _KERNEL:
        _KERNEL.append(load_kernel())
    return _KERNEL[0]


def matmul_routes(blocks, ggml_type, x, ids, out_features):
    """``out[r, s] = x[r] @ blocks[ids[r, s]].T``: ``(routes, slots, out_features)`` float32."""
    if ggml_type == q2_0.GGML_TYPE:
        rows, used = ids.shape
        inputs = x if rows == ids.numel() else x.repeat_interleave(used, dim=0)
        return q2_0.mul_mat_id(blocks, inputs, ids.reshape(-1), out_features).view(rows, used, out_features)
    return _kernel().mul_mat_id(blocks, x, ids, ggml_type, out_features)


def dequantize(blocks, ggml_type, columns, dtype):
    """``(experts, rows, row_bytes)`` blocks -> ``(experts, rows, columns)``."""
    if ggml_type == q2_0.GGML_TYPE:
        return q2_0.dequantize(blocks, columns, dtype)
    experts, rows, _ = blocks.shape
    flat = _kernel().dequantize(blocks.reshape(experts * rows, -1), ggml_type, experts * rows, columns, dtype)
    return flat.view(experts, rows, columns)


class _Layer:
    """What the compute paths need of one layer: its pool's banks and slot count, the ggml types, the dimensions."""

    def __init__(self, module, store):
        prefix = module.offload_prefix
        pool = store.pool_of[prefix]
        _, self.gate_type, _, self.down_type, _ = store.source._info(prefix)
        self.gate_up, self.down = pool.banks
        self.num_slots = pool.num_slots
        self.table = store.host_slot_tables[prefix]
        self.hidden, self.intermediate, self.num_experts = (
            module.hidden_dim,
            module.intermediate_dim,
            module.num_experts,
        )

    def blocks(self, store, prefix, expert):
        """``(gate_up, down)`` blocks of one expert on the device: pool views when cached, else read for one use."""
        return store.expert_tensors(prefix, expert)


def compute(module, store, hidden_states, top_k_index, top_k_weights, routing, experts, pinned):
    """The store's ``expert_compute``: gemvs on the pools when every routed expert is resident (decode), else the
    prefill paths."""
    layer = module.__dict__.get("_gguf_layer")
    if layer is None:
        layer = module.__dict__["_gguf_layer"] = _Layer(
            module, store
        )  # pool banks exist once the store is materialized
    if top_k_index.numel() <= GEMV_MAX_ROUTES and len(pinned) == len(experts):
        return _gemv_forward(module, store, layer, hidden_states, routing, top_k_weights)
    if hidden_states.device.type == "mps":
        return _tiled_forward_v2(module, store, layer, hidden_states, top_k_index, top_k_weights, experts)
    return _grouped_forward(module, store, layer, hidden_states, top_k_index, top_k_weights, experts)


def _gemv_forward(module, store, layer, hidden_states, routing, top_k_weights):
    """Every routed expert resident: ``mul_mat_id`` on the class pools."""
    tokens, top_k = routing.shape
    slots = layer.table[routing].to(torch.int32)
    store.inflight.append((slots,))  # read by the non-blocking copy below until the next sync
    slots = slots.to(hidden_states.device, non_blocking=True)
    gate_up = matmul_routes(layer.gate_up, layer.gate_type, hidden_states, slots, 2 * layer.intermediate)
    activated = module._apply_gate(gate_up.view(tokens * top_k, -1))
    down = matmul_routes(layer.down, layer.down_type, activated, slots.view(-1, 1), layer.hidden).view(
        tokens, top_k, -1
    )
    return (down * top_k_weights[..., None].float()).sum(dim=1).to(hidden_states.dtype)


def _grouped_forward(module, store, layer, hidden_states, top_k_index, top_k_weights, experts):
    """Prefill: groups of experts dequantized, routes sorted by expert, padded batched matmuls."""
    prefix, top_k = module.offload_prefix, top_k_index.shape[1]
    device, dtype = hidden_states.device, hidden_states.dtype
    result = torch.zeros(hidden_states.shape, dtype=torch.float32, device=device)
    flat_index = top_k_index.reshape(-1)
    for start in range(0, len(experts), PREFILL_EXPERT_GROUP):
        group = experts[start : start + PREFILL_EXPERT_GROUP]
        blocks = [layer.blocks(store, prefix, expert) for expert in group]
        gate_up = dequantize(torch.stack([b[0] for b in blocks]), layer.gate_type, layer.hidden, dtype)
        down = dequantize(torch.stack([b[1] for b in blocks]), layer.down_type, layer.intermediate, dtype)
        del blocks
        lookup = torch.full((layer.num_experts,), -1, dtype=torch.long, device=device)
        lookup[torch.tensor(group, device=device)] = torch.arange(len(group), device=device)
        local = lookup[flat_index]
        selected = torch.nonzero(local >= 0).squeeze(1)
        rhs, permutation = torch.sort(local[selected], stable=True)
        order = selected[permutation]
        counts = torch.bincount(rhs, minlength=len(group))
        width = int(counts.max())
        position = torch.arange(rhs.numel(), device=device) - (torch.cumsum(counts, 0) - counts)[rhs]
        padded = hidden_states.new_zeros(len(group), width, layer.hidden)
        padded[rhs, position] = hidden_states[order // top_k]
        activated = module._apply_gate(torch.bmm(padded, gate_up.transpose(1, 2)))
        output = torch.bmm(activated, down.transpose(1, 2))[rhs, position].float()
        result.index_add_(0, order // top_k, output * top_k_weights.reshape(-1)[order, None].float())
        if store.inflight:
            store.synchronize()
    return result.to(dtype)


def _tiled_forward(module, store, layer, hidden_states, top_k_index, top_k_weights, experts):
    """Prefill on MPS: routes sorted by expert, one tile GEMM per projection over the class pool (dequantizing in the
    kernel, :mod:`.kernels.gguf_mm`); experts that did not fit in the pool go through a temporary bank."""
    from .hub_kernels import gguf_mm

    prefix, top_k = module.offload_prefix, top_k_index.shape[1]
    device = hidden_states.device
    flat = top_k_index.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tokens = order // top_k
    x = hidden_states[tokens].float()
    counts = torch.bincount(flat, minlength=layer.num_experts)
    starts = torch.cumsum(counts, 0) - counts
    table = layer.table
    resident = table < layer.num_slots
    snapshot = table.clone()  # the copy below runs later; install_ahead may change `table` before then
    store.inflight.append((snapshot,))
    slots = snapshot.to(device, non_blocking=True).to(torch.int64)
    calls = [(layer.gate_up, layer.down, gguf_mm.tiles_for(counts * resident.to(device), slots, starts))]
    extra = [expert for expert in experts if not resident[expert]]
    if extra:
        blocks = [layer.blocks(store, prefix, expert) for expert in extra]
        index = torch.zeros(layer.num_experts, dtype=torch.int64)
        index[extra] = torch.arange(len(extra))
        mask = torch.zeros(layer.num_experts, dtype=counts.dtype)
        mask[extra] = 1
        tiles = gguf_mm.tiles_for(counts * mask.to(device), index.to(device), starts)
        calls.append((torch.stack([b[0] for b in blocks]), torch.stack([b[1] for b in blocks]), tiles))
    gate_up = torch.empty(x.shape[0], 2 * layer.intermediate, dtype=torch.float32, device=device)
    for bank, _, tiles in calls:
        gguf_mm.mm_tiles(bank, layer.gate_type, x, tiles, out=gate_up)
    activated = module._apply_gate(gate_up)
    del gate_up, x
    down = torch.empty(activated.shape[0], layer.hidden, dtype=torch.float32, device=device)
    for _, bank, tiles in calls:
        gguf_mm.mm_tiles(bank, layer.down_type, activated, tiles, out=down)
    down *= top_k_weights.reshape(-1)[order, None].float()
    result = torch.zeros(hidden_states.shape, dtype=torch.float32, device=device)
    result.index_add_(0, tokens, down)
    return result.to(hidden_states.dtype)


PREFILL_TILE_ROWS = int(os.environ.get("GGUF_EXPERTS_TILE_ROWS", "32"))
# Prefill: queue the next layer's finished reads into the pool behind this layer's compute.
PREFILL_INSTALL_AHEAD = os.environ.get("GGUF_EXPERTS_INSTALL_AHEAD", "1") == "1"
# 16-row tiles for each expert's last <= 16 routes (0 disables).
PREFILL_TAIL_ROWS = int(os.environ.get("GGUF_EXPERTS_TAIL_ROWS", "16"))


def _tiled_forward_v2(module, store, layer, hidden_states, top_k_index, top_k_weights, experts):
    """Prefill on MPS: routes sorted by expert; the tile GEMMs read the hidden states in place through a row index, and
    a combine kernel sums each token's routes (no gathered copy, no scatter)."""
    from .hub_kernels import gguf_mm

    prefix, (tokens, top_k) = module.offload_prefix, top_k_index.shape
    device, tile_rows = hidden_states.device, PREFILL_TILE_ROWS
    flat = top_k_index.reshape(-1)
    order = torch.argsort(flat, stable=True)
    source = (order // top_k).to(torch.int32)
    position = torch.empty_like(order)
    position[order] = torch.arange(order.numel(), device=device)
    counts = torch.bincount(flat, minlength=layer.num_experts)
    starts = torch.cumsum(counts, 0) - counts
    table = layer.table
    resident = table < layer.num_slots
    snapshot = table.clone()  # the copy below runs later; install_ahead may change `table` before then
    store.inflight.append((snapshot,))
    slots = snapshot.to(device, non_blocking=True).to(torch.int64)

    def tile_lists(selected_counts, bank_slots):
        if PREFILL_TAIL_ROWS:
            tiles, tails = gguf_mm.split_tiles(selected_counts, bank_slots, starts, tile_rows, PREFILL_TAIL_ROWS)
            return [(tiles, tile_rows), (tails, PREFILL_TAIL_ROWS)]
        return [(gguf_mm.tiles_for(selected_counts, bank_slots, starts, tile_rows), tile_rows)]

    calls = [(layer.gate_up, layer.down, tile_lists(counts * resident.to(device), slots))]
    extra = [expert for expert in experts if not resident[expert]]
    if extra:
        blocks = [layer.blocks(store, prefix, expert) for expert in extra]
        index = torch.zeros(layer.num_experts, dtype=torch.int64)
        index[extra] = torch.arange(len(extra))
        mask = torch.zeros(layer.num_experts, dtype=counts.dtype)
        mask[extra] = 1
        lists = tile_lists(counts * mask.to(device), index.to(device))
        calls.append((torch.stack([b[0] for b in blocks]), torch.stack([b[1] for b in blocks]), lists))
    gate_up = torch.empty(order.numel(), 2 * layer.intermediate, dtype=torch.float32, device=device)
    for bank, _, lists in calls:
        for tiles, rows in lists:
            gguf_mm.mm_rows(bank, layer.gate_type, hidden_states, source, tiles, rows, out=gate_up)
    activated = module._apply_gate(gate_up)  # float32: +0.4 ms per 2K-token layer, rel err 1.7e-3 instead of 2.3e-3
    del gate_up
    identity = torch.arange(order.numel(), device=device, dtype=torch.int32)
    down = torch.empty(order.numel(), layer.hidden, dtype=torch.float32, device=device)
    for _, bank, lists in calls:
        for tiles, rows in lists:
            gguf_mm.mm_rows(bank, layer.down_type, activated, identity, tiles, rows, out=down)
    return gguf_mm.combine(down, position.view(tokens, top_k), top_k_weights, hidden_states.dtype)


def experts_path(gguf_path):
    """Strata's ``.experts`` file beside its first GGUF shard."""
    path = Path(gguf_path)
    first = Path(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-00001-of-\1.gguf", str(path)))
    return Path(f"{first}.experts")


def gguf_shards(gguf_path):
    """Every shard of a split GGUF (``-0000N-of-0000M.gguf``), or the file itself."""
    path = str(gguf_path)
    match = re.search(r"-\d{5}-of-(\d{5})\.gguf$", path)
    if not match:
        return [path]
    total = int(match.group(1))
    return [re.sub(r"-\d{5}-of-", f"-{i:05d}-of-", path) for i in range(1, total + 1)]


def offload_options(gguf_path, config, io_threads=8, bypass_page_cache=True, experts_file=None):
    """``experts_offload`` options (without ``cache_bytes``) that stream the GGUF's experts from its ``.experts`` file."""
    text = getattr(config, "text_config", config)
    source = GgufExpertSource(
        ExpertsFile(experts_file or experts_path(gguf_path), bypass_page_cache),
        GgufFile(gguf_shards(gguf_path)),
        text.hidden_size,
        text.moe_intermediate_size,
    )
    return {
        "expert_source": source,
        "expert_compute": compute,
        "io_threads": io_threads,
        "cold_prefill_routes": COLD_PREFILL_TOKENS * text.num_experts_per_tok,
        "prefill_install_ahead": PREFILL_INSTALL_AHEAD,
    }
