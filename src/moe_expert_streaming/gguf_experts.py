"""Routed experts in their GGUF quantization (Strata's IQ2/IQ3 gate/up, IQ4_NL or Q2_0 down), streamed from its
``.experts`` file into a device cache of per-size-class slot pools.

Strata's ``<gguf>.experts`` file holds every expert's gate, up and down GGUF blocks contiguous (one read per expert):
header ``"STRX"``, u32 version (1), u32 layers, u32 experts, u64 source GGUF size; at byte 64 one ``(offset, gate
bytes, up bytes, down bytes)`` u64 row per layer; expert ``e`` of layer ``l`` at ``offset + e * (gate + up + down)``.
Gate and up share a type in every layer, so ``gate | up`` is one ``(2 * intermediate, row_bytes)`` matrix.

Expert sizes differ by layer (1.36 to 2.33 MB for Qwen3.8-Flash-Next), so the cache is one pool per size class (as
Strata's arena), each with an LRU and as many slots per layer on average: ``cache_bytes // sum(expert bytes over
layers)`` slots per layer.

Decode runs ggml's ``mul_mat_id`` (``ggml-org/ggml-quantization``) on the pools directly, and :mod:`.kernels.q2_0` for
Q2_0 down projections. Prefill dequantizes a group of experts at a time and runs dense matmuls.

:func:`attach` replaces the ``experts_offload`` store of a loaded model by a :class:`GgufExpertStore`.
"""

import fcntl
import os
import re
import struct
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .gguf_dense import GgufFile, load_kernel
from .kernels import q2_0


# ggml type -> (values per block, bytes per block), expert types.
_BLOCKS = {16: (256, 66), 17: (256, 74), 18: (256, 98), 20: (32, 18), 21: (256, 110), 22: (256, 82), 42: (64, 18)}

# Routes up to which the experts run as gemvs on the pools (decode, short prompts); above, groups of experts are
# dequantized (each dequantized expert is 9.8 MB of bf16, so groups stay small).
GEMV_MAX_ROUTES = 512
PREFILL_EXPERT_GROUP = 32
IMPLEMENTATION = "experts_offload_gguf"


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


class _SizeClass:
    """The slot pools of the layers whose experts share a layout."""

    def __init__(self, layout, hidden, intermediate, num_slots, device):
        self.gate_type, self.gate_row_bytes, self.down_type, self.down_row_bytes = layout
        self.gate_up_bytes = 2 * intermediate * self.gate_row_bytes
        self.expert_bytes = self.gate_up_bytes + hidden * self.down_row_bytes
        self.num_slots = num_slots
        self.gate_up = torch.empty((num_slots, 2 * intermediate, self.gate_row_bytes), dtype=torch.uint8, device=device)
        self.down = torch.empty((num_slots, hidden, self.down_row_bytes), dtype=torch.uint8, device=device)
        self.cache = OrderedDict()  # (prefix, expert) -> slot, least recently used first
        self.free = list(range(num_slots - 1, -1, -1))


class GgufExpertStore:
    """The ``experts_offload`` store's role for GGUF experts: per-size-class LRU pools on the device."""

    def __init__(self, experts_file, gguf, modules, cache_bytes, device, io_threads=8):
        """``modules``: ``[(prefix, experts module)]``; ``gguf``: a :class:`GgufFile` with the expert tensor types."""
        self.file = experts_file
        self.device = torch.device(device)
        self.layer_of, self.class_of = {}, {}
        layouts = {}
        for prefix, module in modules:
            layer = int(re.search(r"layers\.(\d+)\.", prefix).group(1))
            hidden, intermediate = module.hidden_dim, module.intermediate_dim
            _, gate_bytes, up_bytes, down_bytes = experts_file.layers[layer]
            types = [gguf.tensors[f"blk.{layer}.ffn_{p}_exps.weight"][3] for p in ("gate", "up", "down")]
            if types[0] != types[1] or gate_bytes != up_bytes:
                raise ValueError(f"layer {layer}: gate and up differ ({types[:2]})")
            row_bytes = [self._row_bytes(t, c) for t, c in ((types[0], hidden), (types[2], intermediate))]
            if (gate_bytes, down_bytes) != (row_bytes[0] * intermediate, row_bytes[1] * hidden):
                raise ValueError(f"layer {layer}: .experts sizes do not match the GGUF types {types}")
            if module.num_experts != experts_file.num_experts:
                raise ValueError(f"layer {layer}: {module.num_experts} experts, the file has {experts_file.num_experts}")
            self.layer_of[prefix] = layer
            layouts.setdefault((types[0], row_bytes[0], types[2], row_bytes[1]), []).append(prefix)
        self.hidden, self.intermediate, self.num_experts = hidden, intermediate, module.num_experts
        per_layer = cache_bytes // sum(experts_file.expert_bytes(layer) for layer in self.layer_of.values())
        self.slots_per_layer = per_layer
        self.classes = []
        for layout, prefixes in layouts.items():
            size_class = _SizeClass(layout, hidden, intermediate, per_layer * len(prefixes), self.device)
            self.classes.append(size_class)
            for prefix in prefixes:
                self.class_of[prefix] = size_class
        # expert -> slot in its class (num_slots when absent), on the host
        # Host buffers all of the largest expert size: one size keeps the pinned-memory allocator's cache small.
        self.host_buffer_bytes = max(c.expert_bytes for c in self.classes)
        self.host_buffers = []
        self.slot_tables = {p: torch.full((self.num_experts,), c.num_slots, dtype=torch.int32) for p, c in self.class_of.items()}
        self.executor = ThreadPoolExecutor(io_threads, thread_name_prefix="gguf-experts-io") if io_threads else None
        self.window = 2 * io_threads
        self.pending = {}
        self.inflight = []
        self.commit = True
        self.hits = self.misses = self.evictions = self.not_admitted = 0
        self.prefetched = self.prefetch_used = 0
        self.kernel = load_kernel()

    @staticmethod
    def _row_bytes(ggml_type, columns):
        values, size = _BLOCKS[ggml_type]
        return columns // values * size

    def all_resident(self):
        return False

    # ---- reads
    def _read_host(self, prefix, expert):
        try:
            buffer = self.host_buffers.pop()
        except IndexError:
            buffer = torch.empty(self.host_buffer_bytes, dtype=torch.uint8, pin_memory=self.device.type != "cpu")
        return self.file.read(self.layer_of[prefix], expert, buffer[: self.class_of[prefix].expert_bytes])

    def prefetch(self, prefix, experts):
        if self.executor is None:
            return
        size_class = self.class_of[prefix]
        for expert in experts:
            key = (prefix, expert)
            if key not in size_class.cache and key not in self.pending:
                self.pending[key] = self.executor.submit(self._read_host, prefix, expert)
                self.prefetched += 1

    def discard(self, prefix):
        for key in [key for key in self.pending if key[0] == prefix]:
            self.pending.pop(key).cancel()  # a read already done just drops its buffer

    def _take_host(self, prefix, expert):
        future = self.pending.pop((prefix, expert), None)
        if future is not None and not future.cancelled():
            self.prefetch_used += 1
            buffer = future.result()
        else:
            buffer = self._read_host(prefix, expert)
        self.inflight.append(buffer)  # read by queued copies until the next sync
        return buffer

    def release_host_buffers(self):
        """Called right after a host synchronization: queued copies have read their host buffers."""
        for buffer in self.inflight:
            if buffer.dtype == torch.uint8:
                self.host_buffers.append(buffer._base if buffer._base is not None else buffer)
        self.inflight.clear()

    def synchronize(self):
        torch.accelerator.synchronize()
        self.release_host_buffers()

    # ---- cache
    def acquire(self, prefix, expert, pinned=()):
        """The expert's slot in its class pool (installed on a miss), or None when every slot is pinned."""
        size_class, key = self.class_of[prefix], (prefix, expert)
        slot = size_class.cache.get(key)
        if slot is not None:
            self.hits += 1
            size_class.cache.move_to_end(key)
            return slot
        self.misses += 1
        if size_class.free:
            slot = size_class.free.pop()
        else:
            victim = next((k for k in size_class.cache if k not in pinned), None)
            if victim is None:
                self.not_admitted += 1
                return None
            slot = size_class.cache.pop(victim)
            self.slot_tables[victim[0]][victim[1]] = size_class.num_slots
            self.evictions += 1
        buffer = self._take_host(prefix, expert)
        split = size_class.gate_up_bytes
        size_class.gate_up[slot].view(-1).copy_(buffer[:split], non_blocking=True)
        size_class.down[slot].view(-1).copy_(buffer[split:], non_blocking=True)
        size_class.cache[key] = slot
        self.slot_tables[prefix][expert] = slot
        return slot

    def expert_blocks(self, prefix, expert):
        """``(gate_up, down)`` blocks of one expert on the device: pool views when cached, else read for one use."""
        size_class = self.class_of[prefix]
        slot = size_class.cache.get((prefix, expert))
        if slot is not None:
            return size_class.gate_up[slot], size_class.down[slot]
        buffer = self._take_host(prefix, expert).to(self.device, non_blocking=True)
        split = size_class.gate_up_bytes
        return (
            buffer[:split].view(2 * self.intermediate, size_class.gate_row_bytes),
            buffer[split:].view(self.hidden, size_class.down_row_bytes),
        )

    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "not_admitted": self.not_admitted,
            "policy": "lru",
            "slots": sum(c.num_slots for c in self.classes),
            "slots_per_layer": self.slots_per_layer,
            "size_classes": len(self.classes),
            "prefetched": self.prefetched,
            "prefetch_used": self.prefetch_used,
            "resident_bytes": sum(len(c.cache) * c.expert_bytes for c in self.classes),
            "read_bytes": self.file.read_bytes,
            "read_calls": self.file.read_calls,
        }

    # ---- compute
    def matmul_routes(self, blocks, ggml_type, x, ids, out_features):
        """``out[r, s] = x[r] @ blocks[ids[r, s]].T``: ``(routes, slots, out_features)`` float32."""
        if ggml_type == q2_0.GGML_TYPE:
            rows, used = ids.shape
            inputs = x if rows == ids.numel() else x.repeat_interleave(used, dim=0)
            return q2_0.mul_mat_id(blocks, inputs, ids.reshape(-1), out_features).view(rows, used, out_features)
        return self.kernel.mul_mat_id(blocks, x, ids, ggml_type, out_features)

    def dequantize(self, blocks, ggml_type, columns, dtype):
        """``(experts, rows, row_bytes)`` blocks -> ``(experts, rows, columns)``."""
        if ggml_type == q2_0.GGML_TYPE:
            return q2_0.dequantize(blocks, columns, dtype)
        experts, rows, _ = blocks.shape
        flat = self.kernel.dequantize(blocks.reshape(experts * rows, -1), ggml_type, experts * rows, columns, dtype)
        return flat.view(experts, rows, columns)


def _gemv_forward(module, store, size_class, hidden_states, routing, top_k_weights):
    """Every routed expert resident: ``mul_mat_id`` on the class pools."""
    tokens, top_k = routing.shape
    slots = store.slot_tables[module.offload_prefix][routing]
    store.inflight.append(slots)  # read by the non-blocking copy below until the next sync
    slots = slots.to(hidden_states.device, non_blocking=True)
    gate_up = store.matmul_routes(
        size_class.gate_up, size_class.gate_type, hidden_states, slots, 2 * store.intermediate
    )
    activated = module._apply_gate(gate_up.view(tokens * top_k, -1))
    down = store.matmul_routes(
        size_class.down, size_class.down_type, activated, slots.view(-1, 1), store.hidden
    ).view(tokens, top_k, -1)
    return (down * top_k_weights[..., None].float()).sum(dim=1).to(hidden_states.dtype)


def _grouped_forward(module, store, size_class, hidden_states, top_k_index, top_k_weights, experts):
    """Prefill: groups of experts dequantized, routes sorted by expert, padded batched matmuls."""
    prefix, top_k = module.offload_prefix, top_k_index.shape[1]
    device, dtype = hidden_states.device, hidden_states.dtype
    result = torch.zeros(hidden_states.shape, dtype=torch.float32, device=device)
    flat_index = top_k_index.reshape(-1)
    for start in range(0, len(experts), PREFILL_EXPERT_GROUP):
        group = experts[start : start + PREFILL_EXPERT_GROUP]
        blocks = [store.expert_blocks(prefix, expert) for expert in group]
        gate_up = store.dequantize(
            torch.stack([b[0] for b in blocks]), size_class.gate_type, store.hidden, dtype
        )
        down = store.dequantize(torch.stack([b[1] for b in blocks]), size_class.down_type, store.intermediate, dtype)
        del blocks
        lookup = torch.full((store.num_experts,), -1, dtype=torch.long, device=device)
        lookup[torch.tensor(group, device=device)] = torch.arange(len(group), device=device)
        local = lookup[flat_index]
        selected = torch.nonzero(local >= 0).squeeze(1)
        rhs, permutation = torch.sort(local[selected], stable=True)
        order = selected[permutation]
        counts = torch.bincount(rhs, minlength=len(group))
        width = int(counts.max())
        position = torch.arange(rhs.numel(), device=device) - (torch.cumsum(counts, 0) - counts)[rhs]
        padded = hidden_states.new_zeros(len(group), width, store.hidden)
        padded[rhs, position] = hidden_states[order // top_k]
        activated = module._apply_gate(torch.bmm(padded, gate_up.transpose(1, 2)))
        output = torch.bmm(activated, down.transpose(1, 2))[rhs, position].float()
        result.index_add_(0, order // top_k, output * top_k_weights.reshape(-1)[order, None].float())
        if store.inflight:
            store.synchronize()
    return result.to(dtype)


def gguf_experts_forward(module, hidden_states, top_k_index, top_k_weights):
    """The experts forward for a model whose store is a :class:`GgufExpertStore`."""
    from transformers.integrations.experts_offload import _commit_mps

    store, prefix = module.offload_store, module.offload_prefix
    routing = top_k_index.cpu()  # also waits for earlier host-to-device copies
    store.release_host_buffers()
    experts = torch.unique(routing).tolist()
    store.prefetch(prefix, experts[: store.window])
    readahead = module.offload_readahead_routes
    if readahead and module.offload_next_prefix is not None and top_k_index.numel() >= readahead:
        store.prefetch(module.offload_next_prefix, range(store.num_experts))
    size_class = store.class_of[prefix]
    try:
        pinned = set()
        for position, expert in enumerate(experts):
            store.prefetch(prefix, experts[position + 1 : position + 1 + store.window])
            if store.acquire(prefix, expert, pinned) is not None:
                pinned.add((prefix, expert))
        if top_k_index.numel() <= GEMV_MAX_ROUTES and len(pinned) == len(experts):
            result = _gemv_forward(module, store, size_class, hidden_states, routing, top_k_weights)
        else:
            result = _grouped_forward(module, store, size_class, hidden_states, top_k_index, top_k_weights, experts)
    finally:
        store.discard(prefix)
    if store.commit:
        _commit_mps(store)
    return result


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


def attach(model, gguf_path, cache_bytes, io_threads=8, bypass_page_cache=True, experts_file=None):
    """Replace the model's ``experts_offload`` store by a :class:`GgufExpertStore` on the GGUF's experts."""
    from transformers.integrations.moe import ExpertsInterface

    ExpertsInterface.register(IMPLEMENTATION, gguf_experts_forward)
    old = model.experts_offload_store
    modules = [(m.offload_prefix, m) for _, m in model.named_modules() if getattr(m, "offload_store", None) is old]
    store = GgufExpertStore(
        ExpertsFile(experts_file or experts_path(gguf_path), bypass_page_cache),
        GgufFile(gguf_shards(gguf_path)),
        modules,
        cache_bytes,
        old.device,
        io_threads,
    )
    store.commit = old.commit
    if old.executor is not None:
        old.executor.shutdown()
    for _, module in modules:
        module.offload_store = store
        module.config._experts_implementation = IMPLEMENTATION
    model.experts_offload_store = store
    print(
        f"gguf_experts: {len(store.classes)} size classes, {store.slots_per_layer} slots per layer, "
        f"{sum(c.num_slots * c.expert_bytes for c in store.classes) / 2**30:.2f} GiB",
        flush=True,
    )
    return store
