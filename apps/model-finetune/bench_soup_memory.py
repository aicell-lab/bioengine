"""Peak-RSS-vs-member-count benchmark for the soup combine — re-runnable evidence.

``aggregate()`` OOM-killed its own CellposeRuntime replica on a 16-member merge:
27.16 GB resident against the ``memory_mb=12 * 1024`` reservation Ray was given.
The cause was that the combine needed every member resident at once, so peak RSS
was O(members) against a constant reservation. This measures that it no longer is.

Each data point runs in a fresh subprocess and reports ``ru_maxrss``, expressed as
a multiple of one member's size so the number is independent of member size. Run::

    python bench_soup_memory.py

Exits non-zero if the current combine's peak grows with member count, so it is a
regression gate and not just a report. ``--mode before`` reproduces the pre-fix
algorithm for contrast; it is the discrimination control for the flat row, and it
is expected to FAIL the flatness check.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# The defect is that peak RSS GREW with member count, so the gate is on the slope,
# not on the absolute ratio. An absolute ceiling measures the fixed interpreter +
# torch-import overhead as much as the merge: the same flat implementation reads
# 2.87x with 256 MiB members and 5.49x with 64 MiB ones, because that fixed cost is
# divided by a smaller member. Growth is the property and is size-robust.
# Budgeted as one member's worth across the whole sweep; the pre-fix combine spends
# one per member added (+14.00x from 2 to 16).
MAX_GROWTH = 1.0


def _old_uniform_soup(state_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The pre-fix combine, kept only as this benchmark's control.

    Loops keys-outer / members-inner over a materialised list, so it indexes every
    member on every key and all N must be resident for the whole call. This is the
    shape that measured 27.16 GB at 16 members.
    """
    import torch

    keys = list(state_dicts[0].keys())
    out: Dict[str, Any] = {}
    n = len(state_dicts)
    for key in keys:
        ref = state_dicts[0][key]
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for sd in state_dicts:
            acc += sd[key].detach().float()
        out[key] = (acc / n).to(ref.dtype)
    return out


def _write_member(path: Path, member_mb: int, value: float) -> None:
    import torch

    n_elems = (member_mb * 1024 * 1024) // 4
    torch.save({"w": torch.full((n_elems,), value, dtype=torch.float32)}, str(path))


def _checksum(sd: Dict[str, Any]) -> str:
    import torch

    return f"{float(torch.stack([v.double().sum() for v in sd.values()]).sum()):.6f}"


def _run_point(member_file: str, n_members: int, mode: str) -> None:
    """Child process: merge ``n_members`` copies and print its own peak RSS."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pool

    def load() -> Dict[str, Any]:
        return torch.load(member_file, map_location="cpu", weights_only=False)

    if mode == "before":
        merged = _old_uniform_soup([load() for _ in range(n_members)])
    else:
        merged = pool.uniform_soup([load] * n_members)

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"peak_bytes": peak_kb * 1024, "checksum": _checksum(merged)}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["after", "before"], default="after")
    ap.add_argument("--member-mb", type=int, default=256)
    ap.add_argument("--members", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--_point", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--_file", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._point:
        _run_point(args._file, args._point, args.mode)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        member = Path(tmp) / "member.pt"
        _write_member(member, args.member_mb, 2.0)
        member_bytes = member.stat().st_size

        print(f"mode={args.mode}  member={member_bytes / 1024 ** 2:.0f} MiB  "
              f"peak RSS as a multiple of one member")
        rows, checksums = [], set()
        for n in args.members:
            out = subprocess.run(
                [sys.executable, __file__, "--mode", args.mode,
                 "--_point", str(n), "--_file", str(member)],
                capture_output=True, text=True, check=True,
            )
            point = json.loads(out.stdout.strip().splitlines()[-1])
            ratio = point["peak_bytes"] / member_bytes
            rows.append((n, ratio))
            checksums.add(point["checksum"])
            print(f"  members={n:<3} peak={point['peak_bytes'] / 1024 ** 3:6.2f} GiB  {ratio:5.2f}x")

        # A merge is a mean, so the result must not depend on member count. If this
        # ever splits, the memory numbers are measuring two different computations.
        if len(checksums) != 1:
            print(f"FAIL: merged checksum differs across member counts: {checksums}")
            return 1
        print(f"  merged checksum identical at every member count: {checksums.pop()}")

        worst = max(ratio for _, ratio in rows)
        growth = rows[-1][1] - rows[0][1]
        print(f"  worst={worst:.2f}x  growth {rows[0][0]}->{rows[-1][0]} members = {growth:+.2f}x")
        if growth > MAX_GROWTH:
            print(f"FAIL: peak grew {growth:+.2f}x from {rows[0][0]} to {rows[-1][0]} members, "
                  f"budget {MAX_GROWTH}x — peak memory still scales with member count")
            return 1
        print(f"OK: peak grew {growth:+.2f}x from {rows[0][0]} to {rows[-1][0]} members "
              f"(budget {MAX_GROWTH}x) — flat in member count")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
