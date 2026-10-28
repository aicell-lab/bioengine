"""Single-flight gate for the soup merge — re-runnable evidence for #0083.

``aggregate()``'s admission check models ONE resident merge (``soup_admission``
predicts ``base + 3x`` the largest member), but ``CellposeRuntime`` sets
``max_ongoing_requests=10`` and ``materialize_soup`` held no lock, so N merges
could allocate concurrently while each passed a check sized for one. This
measures that they no longer can.

Drives N concurrent ``materialize_soup`` calls at one real ``CellposeRuntime``
and reports the maximum number that were ever inside the allocating region at
once, plus peak RSS. Run::

    python3.9 bench_soup_concurrency.py          # needs 3.9+ for asyncio.to_thread

Exits non-zero if more than one merge gets in, so it is a regression gate and not
just a report. ``--mode before`` replaces the semaphore with a no-op to reproduce
the pre-fix behaviour; it is the discrimination control for the single-flight row
and is expected to FAIL the gate. If it does not, the harness cannot tell the two
apart and its pass means nothing.

Members are byte-accurate stand-ins rather than real cpsam tensors, and torch,
numpy and the PUT are stubbed — what is under test is the semaphore around the
allocating region, so the harness runs without the runtime env.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Dict

MEMBER_BYTES = 1024 ** 3
PAGE = os.sysconf("SC_PAGE_SIZE")


def _rss_bytes() -> int:
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * PAGE


def _load_runtime():
    """Import runtime_cellpose.py with its heavy deps stubbed out."""
    fake_be = types.ModuleType("bioengine")
    fake_be.app = lambda **kw: (lambda cls: cls)
    fake_be.async_init = lambda f: f
    fake_be.logger = logging.getLogger("bench")
    sys.modules["bioengine"] = fake_be

    fake_np = types.ModuleType("numpy")
    for attr in ("ndarray", "floating", "integer"):
        setattr(fake_np, attr, type(attr, (), {}))
    sys.modules["numpy"] = fake_np

    fake_pool = types.ModuleType("pool")
    fake_pool.sha256_file = lambda p: "0" * 64
    sys.modules["pool"] = fake_pool

    fake_torch = types.ModuleType("torch")
    fake_torch.save = lambda obj, path: Path(path).write_bytes(obj)
    sys.modules["torch"] = fake_torch

    import urllib.request

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""

    urllib.request.urlopen = lambda *a, **kw: _Resp()

    path = Path(__file__).with_name("runtime_cellpose.py")
    spec = importlib.util.spec_from_file_location("_rtc_bench", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rtc_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


class _NoopSem:
    """Stands in for ``_soup_sem`` under ``--mode before``."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def locked(self) -> bool:
        return False


async def run(rtc, n_merges: int, scratch: Path, bypass: bool) -> Dict[str, Any]:
    scratch.mkdir(parents=True, exist_ok=True)
    for stale in scratch.iterdir():
        stale.unlink()

    runtime = rtc.CellposeRuntime()
    runtime._soup_dir = lambda: scratch
    runtime._replica_id = lambda: "bench"
    if bypass:
        runtime._soup_sem = _NoopSem()

    live = 0
    max_live = 0
    counter_lock = threading.Lock()

    def build(members):
        nonlocal live, max_live
        with counter_lock:
            live += 1
            max_live = max(max_live, live)
        blob = bytearray(MEMBER_BYTES)
        blob[0] = blob[-1] = 1
        time.sleep(0.6)
        try:
            return blob
        finally:
            with counter_lock:
                live -= 1

    runtime._build_soup = build

    peak = _rss_bytes()
    stop = False

    async def sample():
        nonlocal peak
        while not stop:
            peak = max(peak, _rss_bytes())
            await asyncio.sleep(0.01)

    sampler = asyncio.ensure_future(sample())
    started = time.time()
    results = await asyncio.gather(*[
        runtime.materialize_soup([{"id": f"m{i}"}], "http://bench.invalid/put")
        for i in range(n_merges)
    ])
    wall = time.time() - started
    stop = True
    await asyncio.gather(sampler, return_exceptions=True)

    return {
        "max_concurrent": max_live,
        "peak_bytes": peak,
        "peak_members": peak / MEMBER_BYTES,
        "wall_s": round(wall, 2),
        "completed": len(results),
        "leaked_files": [p.name for p in scratch.iterdir()],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("current", "before"), default="current",
                    help="'before' bypasses the semaphore — the discrimination control")
    ap.add_argument("--merges", type=int, default=4)
    ap.add_argument("--scratch", default="/tmp/bench_soup_concurrency")
    args = ap.parse_args()

    rtc = _load_runtime()
    res = asyncio.get_event_loop().run_until_complete(
        run(rtc, args.merges, Path(args.scratch), bypass=args.mode == "before")
    )

    print(f"mode                : {args.mode}")
    print(f"merges driven       : {args.merges}")
    print(f"merges completed    : {res['completed']}")
    print(f"MAX CONCURRENT      : {res['max_concurrent']}")
    print(f"peak RSS            : {res['peak_bytes'] / 1024 ** 3:.3f} GiB "
          f"({res['peak_members']:.2f} members)")
    print(f"wall                : {res['wall_s']}s")
    print(f"leaked soup files   : {res['leaked_files'] or 'none'}")

    failures = []
    if res["completed"] != args.merges:
        failures.append(f"only {res['completed']}/{args.merges} merges completed")
    if res["max_concurrent"] > 1:
        failures.append(
            f"{res['max_concurrent']} merges were inside the allocating region at once; "
            "aggregate()'s admission check sizes ONE, so the replica can exceed its "
            "memory_mb reservation and be OOM-killed mid-merge"
        )
    if res["leaked_files"]:
        failures.append(f"soup files left behind: {res['leaked_files']}")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        if args.mode == "before":
            print("  (expected for --mode before: this is the discrimination control)")
        return 1
    print("\nPASS — the merge is single-flight per replica")
    return 0


if __name__ == "__main__":
    sys.exit(main())
