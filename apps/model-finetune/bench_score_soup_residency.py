"""Resident-window gate for ``score_soup`` — re-runnable evidence for #0084.

#0083 made the *merge* single-flight, but ``_build_soup`` has two callers and
``score_soup`` was the other one: it built a soup BEFORE taking ``_gpu_lock`` and
kept the local alive through the whole segmentation loop, with no bound on how
many did so at once. ``max_ongoing_requests=10`` admits ten, ``_gpu_lock``
serialises one, so nine could sit in the lock queue each holding a full averaged
state_dict. In the field the two calls implicated in an OOM kill were a 430 s and
a 440 s ``score_soup``, not merges.

This measures peak RSS against the number of **concurrent scoring calls**, one
fresh subprocess per data point, reporting ``ru_maxrss`` as a multiple of one
member. Run::

    python3.9 bench_score_soup_residency.py     # needs 3.9+ for asyncio.to_thread

Gated on the slope, not an absolute ceiling, for the reason spelled out in
``bench_soup_memory.py``: an absolute ratio also measures fixed interpreter
overhead and does not transfer across hosts or member sizes. Growth does.

``--mode before`` swaps in ``_old_score_soup`` — the pre-fix body copied verbatim
from ``c08aebf`` — and is the discrimination control. It is expected to FAIL both
checks. If it passes, the harness cannot tell the arms apart and the ``after``
pass means nothing. Note the control is a reordering, not a missing semaphore, so
a no-op semaphore alone would not reproduce it.

Members are byte-accurate stand-ins rather than real cpsam tensors, and torch,
numpy and the segmenter are stubbed — what is under test is where the soup is
built, saved and released relative to the two locks, so the harness runs without
the runtime env.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import resource
import subprocess
import sys
import threading
import time
import types
import uuid
from pathlib import Path
from typing import Any, Dict, List

# One member's worth of headroom across the whole sweep. The pre-fix body spends
# one per concurrent scorer admitted.
MAX_GROWTH = 1.0

# Long enough that every scorer is queued on _gpu_lock while one segments, which
# is the window the defect holds a soup through.
SEGMENT_SECONDS = 0.4
# Without a build window the unguarded arm can serialise by thread-pool luck and
# report one concurrent build, so the single-flight check would pass vacuously.
# Settable because the build:segment ratio is what decides the starvation cost of
# the semaphore: a real build is download-bound and those downloads used to
# overlap across callers.
BUILD_SECONDS = 0.3


def _load_runtime():
    """Import runtime_cellpose.py with its heavy deps stubbed out."""
    fake_be = types.ModuleType("bioengine")
    fake_be.app = lambda **kw: (lambda cls: cls)
    fake_be.logger = logging.getLogger("bench")
    sys.modules["bioengine"] = fake_be

    fake_np = types.ModuleType("numpy")
    for attr in ("ndarray", "floating", "integer"):
        setattr(fake_np, attr, type(attr, (), {}))
    fake_np.asarray = lambda x: x
    sys.modules["numpy"] = fake_np

    fake_pool = types.ModuleType("pool")
    fake_pool.sha256_file = lambda p: "0" * 64
    fake_pool.mean_instance_f1 = lambda masks, labels: 0.0
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


async def _old_score_soup(
    self, members: List[Dict[str, str]], images: List[Any],
    labels: List[Any], diam_mean: float = 30.0,
) -> Dict[str, Any]:
    """The pre-fix ``score_soup`` body, kept only as this benchmark's control.

    Copied from ``c08aebf``. Two properties make it the defect: ``soup`` is bound
    before ``_gpu_lock`` is acquired, so the lock WAIT happens with a full soup
    resident; and the local is never dropped, so it also survives the segmentation
    loop even though ``_segment`` reads the checkpoint from disk.
    """
    import pool

    import numpy as np

    dl_before = self._soup_dl_bytes
    soup = await asyncio.to_thread(self._build_soup, members)
    ckpt = None
    async with self._gpu_lock:
        if soup is not None:
            import torch

            ckpt = self._soup_dir() / f"soup-{uuid.uuid4().hex}.pt"
            await asyncio.to_thread(lambda: torch.save(soup, str(ckpt)))
        try:
            gk = {"diameter": diam_mean}
            masks = []
            for image in images:
                masks.append(await asyncio.to_thread(
                    self._segment, image, gk, str(ckpt) if ckpt else None, "cpsam"))
        finally:
            if ckpt and ckpt.exists():
                ckpt.unlink()
    return {
        "f1": pool.mean_instance_f1(masks, [np.asarray(lbl) for lbl in labels]),
        "replica": self._replica_id(),
        "soup_dl_bytes": self._soup_dl_bytes - dl_before,
    }


async def _drive(rtc, n_calls: int, member_bytes: int, scratch: Path,
                 mode: str, build_s: float, segment_s: float) -> Dict[str, Any]:
    scratch.mkdir(parents=True, exist_ok=True)
    for stale in scratch.iterdir():
        stale.unlink()

    runtime = rtc.CellposeRuntime()
    runtime._soup_dir = lambda: scratch
    runtime._replica_id = lambda: "bench"

    live = 0
    max_live = 0
    counter_lock = threading.Lock()

    def build(members):
        """Allocate one member-sized blob and keep it reachable by its caller."""
        nonlocal live, max_live
        with counter_lock:
            live += 1
            max_live = max(max_live, live)
        try:
            blob = bytearray(member_bytes)
            blob[0] = blob[-1] = 1
            time.sleep(build_s)
            return blob
        finally:
            with counter_lock:
                live -= 1

    def segment(image, gk, ckpt_path, model):
        """Stand-in for the GPU pass: reads the checkpoint from DISK, as the real
        one does, which is why the resident copy is dead weight by this point."""
        if ckpt_path:
            with open(ckpt_path, "rb") as fh:
                fh.read(1)
        time.sleep(segment_s)
        return image

    runtime._build_soup = build
    runtime._segment = segment

    if mode == "before":
        scorer = lambda: _old_score_soup(runtime, [{"key": "k", "url": "u"}], [0], [0])
    else:
        scorer = lambda: runtime.score_soup([{"key": "k", "url": "u"}], [0], [0])

    started = time.time()
    results = await asyncio.gather(*[scorer() for _ in range(n_calls)])
    wall = time.time() - started

    return {
        "peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "max_concurrent_builds": max_live,
        "wall_s": round(wall, 2),
        "completed": len(results),
        "leaked_files": [p.name for p in scratch.iterdir()],
    }


def _run_point(n_calls: int, member_bytes: int, scratch: str, mode: str,
               build_s: float, segment_s: float) -> None:
    rtc = _load_runtime()
    res = asyncio.get_event_loop().run_until_complete(
        _drive(rtc, n_calls, member_bytes, Path(scratch), mode, build_s, segment_s)
    )
    print(json.dumps(res))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("after", "before"), default="after",
                    help="'before' runs the pre-fix body — the discrimination control")
    ap.add_argument("--member-mb", type=int, default=256)
    ap.add_argument("--calls", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--scratch", default="/tmp/bench_score_soup_residency")
    ap.add_argument("--build-seconds", type=float, default=BUILD_SECONDS,
                    help="stand-in for member download + average; raise it to size "
                         "the semaphore's starvation cost against segmentation")
    ap.add_argument("--segment-seconds", type=float, default=SEGMENT_SECONDS)
    ap.add_argument("--_point", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    member_bytes = args.member_mb * 1024 * 1024
    if args._point:
        _run_point(args._point, member_bytes, args.scratch, args.mode,
                   args.build_seconds, args.segment_seconds)
        return 0

    print(f"mode={args.mode}  member={args.member_mb} MiB  "
          f"build={args.build_seconds}s segment={args.segment_seconds}s  "
          f"peak RSS as a multiple of one member")
    rows, failures = [], []
    for n in args.calls:
        out = subprocess.run(
            [sys.executable, __file__, "--mode", args.mode, "--_point", str(n),
             "--member-mb", str(args.member_mb), "--scratch", args.scratch,
             "--build-seconds", str(args.build_seconds),
             "--segment-seconds", str(args.segment_seconds)],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            print(f"  calls={n:<3} CHILD FAILED rc={out.returncode}\n{out.stderr}")
            return 1
        point = json.loads(out.stdout.strip().splitlines()[-1])
        ratio = point["peak_bytes"] / member_bytes
        rows.append((n, ratio, point))
        print(f"  calls={n:<3} peak={point['peak_bytes'] / 1024 ** 3:6.2f} GiB  "
              f"{ratio:5.2f}x   max concurrent builds={point['max_concurrent_builds']}  "
              f"wall={point['wall_s']}s")
        if point["completed"] != n:
            failures.append(f"calls={n}: only {point['completed']}/{n} completed")
        if point["max_concurrent_builds"] > 1:
            failures.append(
                f"calls={n}: {point['max_concurrent_builds']} soups were being built at "
                "once; aggregate()'s admission check sizes ONE resident soup"
            )
        if point["leaked_files"]:
            failures.append(f"calls={n}: soup files left behind: {point['leaked_files']}")

    growth = rows[-1][1] - rows[0][1]
    print(f"  growth {rows[0][0]}->{rows[-1][0]} concurrent calls = {growth:+.2f}x "
          f"(budget {MAX_GROWTH}x)")
    if growth > MAX_GROWTH:
        failures.append(
            f"peak grew {growth:+.2f}x from {rows[0][0]} to {rows[-1][0]} concurrent "
            f"calls, budget {MAX_GROWTH}x — a queued scorer still holds a soup"
        )

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        if args.mode == "before":
            print("  (expected for --mode before: this is the discrimination control)")
        return 1
    print("\nPASS — scoring peak RSS is flat in concurrent callers, "
          "and soup builds are single-flight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
