import time

import torch


dev = "mps"
W = torch.randn(2560, 2560, device=dev, dtype=torch.bfloat16) * 0.01
x0 = torch.randn(1, 2560, device=dev, dtype=torch.bfloat16)


def busy(ms):
    t = time.perf_counter()
    while time.perf_counter() - t < ms / 1e3:
        pass


def work(n):  # decode-like: small matvecs + elementwise
    x = x0
    for _ in range(n):
        x = torch.nn.functional.silu(x @ W) + x0
    return x


def t(fn, N=20):
    fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
        torch.mps.synchronize()
    return round(1e3 * (time.perf_counter() - t0) / N, 2)


for n in (200, 1000):
    t0 = time.perf_counter()
    work(n)
    launch = 1e3 * (time.perf_counter() - t0)
    torch.mps.synchronize()
    gpu = t(lambda: work(n))
    print(
        f"{n} steps: launch {launch:.1f} ms; launch+run+sync {gpu} ms; "
        f"+host busy {gpu} ms after launch: {t(lambda: (work(n), busy(gpu)))} ms (overlap ~{gpu}, none ~{2 * gpu:.1f})"
    )

n = 1000
gpu = t(lambda: work(n))


def ev():
    e = torch.mps.Event()
    e.record()


def commit_variants():
    for name, after in [
        ("nothing", lambda: None),
        ("Event().record()", ev),
        ("non_blocking 1-elt copy to pinned", lambda: PIN.copy_(x0[0, :1].float(), non_blocking=True)),
    ]:
        print(
            f"  commit via {name:36s}: {t(lambda: (work(n), after(), busy(gpu)))} ms (overlap ~{gpu}, none ~{2 * gpu:.0f})"
        )


PIN = torch.empty(1, pin_memory=True)
commit_variants()
