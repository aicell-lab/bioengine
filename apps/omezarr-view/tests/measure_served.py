"""Measure the served OME-Zarr endpoint: build time, cold and warm chunk latency.

Everything here is measured through the public HTTP endpoint, i.e. what a
viewer or a Python client actually experiences — not an in-process shortcut.
Cold and warm are distinguished by the server's own ``X-View-Cache`` header
rather than assumed from ordering.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List

import httpx


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    v = sorted(values)
    return {
        "n": len(v),
        "median_ms": round(statistics.median(v) * 1000, 1),
        "min_ms": round(v[0] * 1000, 1),
        "max_ms": round(v[-1] * 1000, 1),
        "p25_ms": round(v[len(v) // 4] * 1000, 1),
        "p75_ms": round(v[(3 * len(v)) // 4] * 1000, 1),
    }


def _provenance(base: str) -> Dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", "/data/nmechtel/bioengine", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = "unknown"
    return {
        "endpoint": base,
        "client_host": platform.node(),
        "bioengine_commit": commit,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "client and server are the same host (europa); the endpoint is "
                "reached through its public tunnel, so each request pays a "
                "tunnel round trip that a co-located client would not.",
    }


def chunk_keys(meta: Dict[str, Any], level: int, limit: int) -> List[str]:
    """Enumerate chunk keys for a level, spread across the array."""
    shape = meta["shape"]
    chunks = meta["chunks"]
    grid = [max(1, -(-s // c)) for s, c in zip(shape, chunks)]
    keys = []
    # Walk a diagonal so the sample is not all from one corner or one plane.
    for i in range(limit * 4):
        idx = [(i * (d + 1) // 3) % g for d, g in enumerate(grid)]
        key = f"{level}/" + "/".join(str(n) for n in idx)
        if key not in keys:
            keys.append(key)
        if len(keys) >= limit:
            break
    return keys


def measure(base: str, dataset: str, n: int, timeout: float) -> Dict[str, Any]:
    client = httpx.Client(timeout=timeout, follow_redirects=True)
    out: Dict[str, Any] = {"dataset": dataset}

    t0 = time.perf_counter()
    r = client.get(f"{base}/api/datasets/{dataset}", timeout=timeout)
    r.raise_for_status()
    info = r.json()
    out["build_wall_s"] = round(time.perf_counter() - t0, 3)
    out["recipe"] = info.get("recipe")
    out["location"] = info.get("location")
    out["shape"] = info.get("shape")
    out["levels"] = info.get("levels")
    out["server_build_timings_s"] = info.get("build_timings_s")
    out["source_bytes"] = info.get("size_bytes")
    out["pixel_bytes_copied"] = info.get("pixel_bytes_copied")

    # first tile a viewer would actually draw: the coarsest level
    coarsest = (info.get("levels") or 1) - 1
    zmeta = client.get(f"{base}/zarr/{dataset}/{coarsest}/.zarray").json()
    first_key = chunk_keys(zmeta, coarsest, 1)[0]
    t0 = time.perf_counter()
    rr = client.get(f"{base}/zarr/{dataset}/{first_key}")
    out["first_viewable_tile"] = {
        "key": first_key,
        "status": rr.status_code,
        "seconds": round(time.perf_counter() - t0, 3),
        "bytes": len(rr.content),
        "cache": rr.headers.get("X-View-Cache"),
    }

    per_level: Dict[str, Any] = {}
    for level in {0, coarsest}:
        meta = client.get(f"{base}/zarr/{dataset}/{level}/.zarray").json()
        keys = chunk_keys(meta, level, n)
        cold, warm, sizes = [], [], []
        for key in keys:
            t0 = time.perf_counter()
            resp = client.get(f"{base}/zarr/{dataset}/{key}")
            dt = time.perf_counter() - t0
            if resp.status_code != 200:
                continue
            sizes.append(len(resp.content))
            (cold if resp.headers.get("X-View-Cache") == "miss" else warm).append(dt)
        for key in keys:
            t0 = time.perf_counter()
            resp = client.get(f"{base}/zarr/{dataset}/{key}")
            dt = time.perf_counter() - t0
            if resp.status_code == 200 and resp.headers.get("X-View-Cache") == "hit":
                warm.append(dt)
        per_level[f"L{level}"] = {
            "chunk_shape": meta["chunks"],
            "cold": _stats(cold),
            "warm_cache_hit": _stats(warm),
            "mean_served_chunk_bytes": int(sum(sizes) / len(sizes)) if sizes else None,
        }
    out["per_chunk"] = per_level
    client.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default="served_measurements.json")
    ap.add_argument("datasets", nargs="+")
    args = ap.parse_args()

    results = {"provenance": _provenance(args.base), "datasets": []}
    for ds in args.datasets:
        print(f"measuring {ds} ...", flush=True)
        try:
            results["datasets"].append(measure(args.base, ds, args.n, args.timeout))
        except Exception as e:
            results["datasets"].append({"dataset": ds, "error": f"{type(e).__name__}: {e}"})
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
