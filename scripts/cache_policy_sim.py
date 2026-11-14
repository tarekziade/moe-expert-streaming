"""Replay the expert demand trace through cache policies. Hit rate = hits / (unique experts per MoE call)."""

import heapq
import json
import sys
from collections import Counter, OrderedDict


t = json.load(open(sys.argv[1]))
SLOTS = t["slots"]
gens = sorted({g for g, _, _ in t["calls"]})


def accesses(gs):
    return [(l, e) for g, l, es in t["calls"] if g in gs for e in es]


def lru(acc, slots, pinned=frozenset(), warm=None):
    cache = OrderedDict((k, None) for k in (warm or []))
    cap = slots - len(pinned)
    hits = 0
    for k in acc:
        if k in pinned:
            hits += 1
            continue
        if k in cache:
            hits += 1
            cache.move_to_end(k)
            continue
        if cap <= 0:
            continue
        if len(cache) >= cap:
            cache.popitem(last=False)
        cache[k] = None
    return hits / len(acc), list(cache)


def lfu(acc, slots):  # perfect counts, no decay, always admit
    counts, cache, heap, hits = Counter(), set(), [], 0
    for i, k in enumerate(acc):
        counts[k] += 1
        if k in cache:
            hits += 1
            heapq.heappush(heap, (counts[k], i, k))
            continue
        if len(cache) >= slots:
            while True:
                c, _, v = heapq.heappop(heap)
                if v in cache and counts[v] == c:
                    cache.discard(v)
                    break
        cache.add(k)
        heapq.heappush(heap, (counts[k], i, k))
    return hits / len(acc)


def belady(acc, slots, bypass=True):
    INF = float("inf")
    nxt = [INF] * len(acc)
    last = {}
    for i in range(len(acc) - 1, -1, -1):
        nxt[i] = last.get(acc[i], INF)
        last[acc[i]] = i
    res, heap, hits = {}, [], 0
    for i, k in enumerate(acc):
        if k in res:
            hits += 1
        elif len(res) >= slots:
            while True:
                n, v = heapq.heappop(heap)
                n = -n
                if res.get(v) == n:
                    break
            if bypass and nxt[i] > n:  # the newcomer is needed later than everything resident: don't cache it
                heapq.heappush(heap, (-n, v))
                continue
            del res[v]
        res[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return hits / len(acc)


print(f"slots {SLOTS}, experts {len(t['layers'])}x512 = {len(t['layers']) * 512}")
for g in gens:
    a = accesses({g})
    u = len(set(a))
    print(f"\ngen {g}: {t['prompts'][g][:60]!r}\n  lookups {len(a)}, distinct experts {u} ({u / SLOTS:.2f}x cache)")
    others = Counter(accesses(set(gens) - {g} - ({0, 4} if g in (0, 4) else set())))
    prev, _ = None, None
    r = {"LRU cold": lru(a, SLOTS)[0], "LFU cold": lfu(a, SLOTS), "Belady (OPT)": belady(a, SLOTS)}
    if g > 0:
        _, warm = lru(accesses({g - 1}), SLOTS)
        r["LRU warm (prev prompt)"] = lru(a, SLOTS, warm=warm)[0]
    for frac in (0.25, 0.5, 0.75):
        hot = frozenset(k for k, _ in others.most_common(int(frac * SLOTS)))
        r[f"pin {int(frac * 100)}% hot (other prompts) + LRU"] = lru(a, SLOTS, pinned=hot)[0]
    _, warm_self = lru(a, SLOTS)
    r["LRU warm (same prompt, = bench run 2)"] = lru(a, SLOTS, warm=warm_self)[0]
    for k, v in r.items():
        print(f"  {k:42s} {v:6.2%}")
# Cache-size sweep on the story for LRU vs OPT
a = accesses({0})
print("\nstory, cache-size sweep (slots: LRU / OPT)")
for s in (SLOTS // 2, SLOTS, int(SLOTS * 1.5), SLOTS * 2):
    print(f"  {s:6d} ({s * SLOTS and s / SLOTS:.1f}x): {lru(a, s)[0]:6.2%} / {belady(a, s):6.2%}")
