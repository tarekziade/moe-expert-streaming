"""Evaluate next-layer expert predictors offline on traces from predict_trace.py, through an LRU expert cache.

Replays the decode trace (all MoE calls in order, so prompts' prefill calls also fill the cache) with the real
prefetch scheme: guesses for layer L+1 (made at layer L) are read and installed before L+1; guesses for L+2 are only
read (installed if used). Per decode token:
  critical  routed experts neither cached nor read ahead: the layer waits for them
  covered   routed experts not cached but read ahead
  wasted    reads of guesses that were not used
Predictors: `router` (next layer's router on this layer's experts input: what the code does), `ridge` (least squares
map from this layer's input to the target layer's router logits; trained on the prefill trace + the other decode
prompts, evaluated on the held-out prompt).

Usage: predict_sim.py DECODE_TRACE PREFILL_TRACE DENSE_DIR [CACHE_GIB=12]
"""

import glob
import sys
from collections import OrderedDict, defaultdict

import torch
from safetensors import safe_open


decode = torch.load(sys.argv[1])
prefill = torch.load(sys.argv[2])
DENSE = sys.argv[3]
CACHE_GIB = float(sys.argv[4]) if len(sys.argv) > 4 else 12
SLOTS = int(CACHE_GIB * 2**30) // decode["slot_bytes"]
LAYERS, TOPK = 48, 10
torch.set_num_threads(8)

routers = {}
for f in sorted(glob.glob(f"{DENSE}/*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if k.endswith("mlp.gate.weight"):
                routers[int(k.split("layers.")[1].split(".")[0])] = s.get_tensor(k).float()


def steps(calls):
    """Group calls into forward steps (layer 0..47); yields (gen, [calls by layer])."""
    group = []
    for call in calls:
        if call[1] == 0 and group:
            yield group
            group = []
        group.append(call)
    if group:
        yield group


# sanity: the router on the recorded input reproduces the recorded routing
g, layer, h, idx, _ = next(c for c in decode["calls"] if c[2].shape[0] == 1 and c[1] == 5)
check = (h.float() @ routers[5].T).topk(TOPK).indices.sort().values
print("router sanity:", torch.equal(check, idx.long().sort().values), flush=True)


# ---------------------------------------------------------------- training pairs (src layer, dist) -> (X, Y logits)
def pairs(calls, gens=None):
    out = defaultdict(lambda: [[], []])
    for group in steps(calls):
        if gens is not None and group[0][0] not in gens:
            continue
        if len(group) != LAYERS:
            continue
        for src in range(LAYERS):
            for dist in (1, 2):
                dst = src + dist
                if dst >= LAYERS:
                    continue
                x, y = group[src][2].float(), group[dst][2].float()
                if x.shape[0] != y.shape[0]:
                    continue
                out[(src, dist)][0].append(x)
                out[(src, dist)][1].append(y @ routers[dst].T)
    return {k: (torch.cat(v[0]), torch.cat(v[1])) for k, v in out.items()}


decode_gens = sorted({c[0] for c in decode["calls"]})
prefill_pairs = pairs(prefill["calls"])
print("prefill rows per pair:", next(iter(prefill_pairs.values()))[0].shape[0], flush=True)


def fit(train, lam):
    maps = {}
    for key, (x, y) in train.items():
        mu = x.mean(0)
        xc = x - mu
        gram = xc.T @ xc
        gram += lam * gram.diagonal().mean() * torch.eye(gram.shape[0])
        w = torch.linalg.solve(gram, xc.T @ (y - y.mean(0)))
        maps[key] = (mu, w, y.mean(0))
    return maps


# ---------------------------------------------------------------- simulation
def simulate(gen, predict, widths):
    """Replay all calls of generation `gen`, warm cache from earlier generations (as in the bench process)."""
    cache = SIM_CACHE  # shared across generations, in order
    stats = defaultdict(float)
    tokens = 0
    pending = set()  # read ahead, not installed (dist 2)
    for group in steps(c for c in decode["calls"] if c[0] == gen):
        decode_step = group[0][2].shape[0] == 1
        tokens += decode_step
        installed_guess = set()
        for call in group:
            _, layer, h, idx, experts = call
            key_set = {(layer, e) for e in experts}
            for key in key_set:
                if key in cache:
                    cache.move_to_end(key)
                    if decode_step:
                        stats["hits"] += 1
                    continue
                if decode_step:
                    stats["covered" if key in pending else "critical"] += 1
                pending.discard(key)
                cache[key] = None
                if len(cache) > SLOTS:
                    cache.popitem(last=False)
            if decode_step:
                # unused read-ahead for this layer
                stats["wasted"] += sum(1 for k in pending if k[0] == layer) + len(
                    [k for k in installed_guess if k[0] == layer and k not in key_set]
                )
            pending = {k for k in pending if k[0] > layer}
            if not decode_step or predict is None:
                continue
            for dist in (1, 2):
                dst = layer + dist
                if dst >= LAYERS:
                    continue
                for e in predict(layer, dist, h.float()[0]).tolist():
                    k = (dst, e)
                    if k in cache or k in pending or k in installed_guess:
                        continue
                    stats["reads_ahead"] += 1
                    if dist == 1:  # installed before the next layer runs
                        installed_guess.add(k)
                        cache[k] = None
                        if len(cache) > SLOTS:
                            cache.popitem(last=False)
                    else:
                        pending.add(k)
    return {k: v / max(tokens, 1) for k, v in stats.items()}, tokens


def router_predictor(width):
    def predict(src, dist, h):
        return (routers[src + dist] @ h).topk(width).indices

    return predict


def ridge_predictor(maps, width):
    def predict(src, dist, h):
        mu, w, ymu = maps[(src, dist)]
        return ((h - mu) @ w + ymu).topk(width).indices

    return predict


def run(name, make):
    global SIM_CACHE
    SIM_CACHE = OrderedDict()
    total = defaultdict(float)
    n = 0
    for gen in decode_gens:
        predict = make(gen)
        stats, tokens = simulate(gen, predict, None)
        for k, v in stats.items():
            total[k] += v * tokens
        n += tokens
    t = {k: v / n for k, v in total.items()}
    mb = decode["slot_bytes"] / 1e6
    reads = t.get("critical", 0) + t.get("reads_ahead", 0)
    print(
        f"{name:28s} critical {t.get('critical', 0):5.1f}  covered {t.get('covered', 0):5.1f}  "
        f"wasted {t.get('wasted', 0):5.1f}  hit {t.get('hits', 0) / 480 * 100:5.1f}%  "
        f"reads {reads:5.1f}/token = {reads * mb:5.0f} MB",
        flush=True,
    )


print(f"cache {CACHE_GIB} GiB = {SLOTS} slots; per decode token (48 layers x {TOPK} routes = 480 lookups)")
run("no prediction", lambda gen: None)
for width in (10, 12, 16):
    run(f"router k={width}", lambda gen, w=width: router_predictor(w))
decode_pairs_by_gen = {g: pairs(decode["calls"], {g}) for g in decode_gens}


def residual(train):
    """Targets minus the plain router guess (router of the target layer on the source input)."""
    return {(src, d): (x, y - x @ routers[src + d].T) for (src, d), (x, y) in train.items()}


def ridge_res_predictor(maps, width):
    def predict(src, dist, h):
        mu, w, ymu = maps[(src, dist)]
        return (routers[src + dist] @ h + (h - mu) @ w + ymu).topk(width).indices

    return predict


for lam in (1.0, 10.0, 100.0):
    fitted = {}
    for gen in decode_gens:
        train = {}
        for key, (x, y) in prefill_pairs.items():
            xs, ys = [x], [y]
            for other in decode_gens:
                if other != gen and key in decode_pairs_by_gen[other]:
                    xs.append(decode_pairs_by_gen[other][key][0])
                    ys.append(decode_pairs_by_gen[other][key][1])
            train[key] = (torch.cat(xs), torch.cat(ys))
        fitted[gen] = fit(residual(train), lam)
    for width in (10, 12):
        run(f"ridge-res lam={lam} k={width}", lambda gen, w=width, f=fitted: ridge_res_predictor(f[gen], w))
