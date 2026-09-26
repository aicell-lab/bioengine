"""Benchmark a lazy view against an actually-converted OME-Zarr copy.

The question the figure needs answered is not "is Zarr fast" — the NGFF paper
settled that — but "is a view over an UNCONVERTED file fast enough to be
practical". The only comparison that tests it is the converted copy, which is
what this measures.

A naive two-arm design (view-over-remote vs converted-copy-on-local-disk)
confounds two different effects, and the local copy would win for the boring
reason. So the arms form a 2x2 over {route} x {storage}:

    A  view, Tier B (served, transcoded)          remote (GCS)
    B  view, Tier A (client range-reads original) remote (GCS)
    C  view, Tier A                               local copy of the ORIGINAL
    D  converted OME-Zarr, read directly          local

    B vs C isolates STORAGE LOCALITY with the route held constant.
    C vs D isolates ROUTE/FORMAT with locality held constant.

The missing cell — a converted copy on the same remote bucket — is not
measured, because we cannot write to that bucket. That is stated, not hidden.

Identical chunk sample across all arms, cold and warm reported separately.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def provenance(extra: Dict[str, Any]) -> Dict[str, Any]:
    def _git(*a: str) -> str:
        try:
            return subprocess.run(["git", "-C", "/data/nmechtel/bioengine", *a],
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return "unknown"
    out = {
        "host": platform.node(),
        "bioengine_commit": _git("rev-parse", "--short", "HEAD"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "apps/omezarr-view/tests/convert_and_benchmark.py",
    }
    out.update(extra)
    return out


def stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    v = sorted(values)
    q = lambda f: round(v[min(len(v) - 1, int(len(v) * f))] * 1000, 2)  # noqa: E731
    return {
        "n": len(v),
        "median_ms": round(statistics.median(v) * 1000, 2),
        "mean_ms": round(statistics.fmean(v) * 1000, 2),
        "p10_ms": q(0.10), "p25_ms": q(0.25), "p75_ms": q(0.75), "p90_ms": q(0.90),
        "min_ms": round(v[0] * 1000, 2), "max_ms": round(v[-1] * 1000, 2),
        # The distribution is the point; a mean alone hides the tail.
        "all_ms": [round(x * 1000, 2) for x in v],
    }


# ---------------------------------------------------------------------------


def download(url: str, dest: Path) -> Dict[str, Any]:
    import httpx

    if dest.exists() and dest.stat().st_size > 0:
        return {"seconds": None, "bytes": dest.stat().st_size, "note": "already present"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_bytes(1 << 22):
                fh.write(chunk)
    return {"seconds": round(time.perf_counter() - t0, 1), "bytes": dest.stat().st_size}


def convert(src: Path, dest: Path) -> Dict[str, Any]:
    """Convert the local original to a real OME-Zarr pyramid on disk."""
    import numcodecs
    import tifffile
    import zarr

    if dest.exists():
        shutil.rmtree(dest)
    t0 = time.perf_counter()
    compressor = numcodecs.Blosc(cname="zstd", clevel=5,
                                 shuffle=numcodecs.Blosc.SHUFFLE)
    tf = tifffile.TiffFile(str(src))
    series = tf.series[0]
    root = zarr.open_group(str(dest), mode="w")
    axes = series.axes.lower()
    yi, xi = len(axes) - 2, len(axes) - 1

    datasets = []
    for lvl, level in enumerate(series.levels):
        shape = tuple(int(n) for n in level.shape)
        chunks = tuple(1 if i not in (yi, xi) else min(512, shape[i])
                       for i in range(len(shape)))
        arr = root.create_dataset(str(lvl), shape=shape, chunks=chunks,
                                  dtype=level.dtype, compressor=compressor,
                                  dimension_separator="/")
        store = level.aszarr()
        source = zarr.open(store, mode="r")
        if isinstance(source, zarr.hierarchy.Group):
            # A level store can present as a group with the array under a key.
            source = source[sorted(source.array_keys())[0]]
        # Copy in row BANDS, not whole planes: one plane of this file is
        # 36040 x 52660 uint16 = 3.8 GB, which would defeat the point of
        # streaming the conversion at all.
        lead = shape[:-2]
        band = max(chunks[yi], 2048)
        for index in (np.ndindex(*lead) if lead else [()]):
            for y0 in range(0, shape[yi], band):
                y1 = min(y0 + band, shape[yi])
                sel = tuple(index) + (slice(y0, y1), slice(None))
                arr[sel] = np.asarray(source[sel])
        store.close()
        datasets.append({"path": str(lvl), "coordinateTransformations": [
            {"type": "scale", "scale": [
                (series.levels[0].shape[i] / shape[i]) if i in (yi, xi) else 1.0
                for i in range(len(shape))]}]})
    root.attrs["multiscales"] = [{
        "version": "0.4", "name": src.stem,
        "axes": [{"name": a, "type": {"c": "channel", "t": "time"}.get(a, "space")}
                 for a in axes],
        "datasets": datasets}]
    seconds = time.perf_counter() - t0

    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    return {
        "seconds": round(seconds, 1),
        "on_disk_bytes": total,
        "codec": "blosc/zstd level 5 with shuffle",
        "levels": len(series.levels),
        "path": str(dest),
    }


# ---------------------------------------------------------------------------


def sample_chunks(shape: Tuple[int, ...], chunks: Tuple[int, ...], axes: str,
                  n: int, seed: int) -> List[Tuple[int, ...]]:
    """N distinct random chunk indices at full resolution."""
    rng = np.random.default_rng(seed)
    grid = [max(1, -(-s // c)) for s, c in zip(shape, chunks)]
    seen, out = set(), []
    while len(out) < n and len(seen) < np.prod(grid):
        idx = tuple(int(rng.integers(0, g)) for g in grid)
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def time_reads(getter, indices, repeat_warm: bool) -> Tuple[List[float], List[float]]:
    cold, warm = [], []
    for idx in indices:
        t0 = time.perf_counter()
        getter(idx)
        cold.append(time.perf_counter() - t0)
    if repeat_warm:
        for idx in indices:
            t0 = time.perf_counter()
            getter(idx)
            warm.append(time.perf_counter() - t0)
    return cold, warm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://viv-demo.storage.googleapis.com/"
                                     "Vanderbilt-Spraggins-Kidney-MxIF.ome.tif")
    ap.add_argument("--dataset", default="spraggins-kidney")
    ap.add_argument("--base", required=True, help="served view base url (Tier B)")
    ap.add_argument("--work", default="/data2/nmechtel/omezarr-bench")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="/data/nmechtel/bioengine-paper/analysis/"
                                     "results/omezarr_view/figure_assets/"
                                     "converted_copy_benchmark.json")
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    local_tif = work / "source.ome.tif"
    zarr_copy = work / "converted.ome.zarr"

    result: Dict[str, Any] = {"onboarding": {}, "arms": {}}

    print("[1/5] download original ...", flush=True)
    result["onboarding"]["download_original"] = download(args.url, local_tif)
    print("   ", result["onboarding"]["download_original"], flush=True)

    if not args.skip_convert:
        print("[2/5] convert to OME-Zarr ...", flush=True)
        result["onboarding"]["conversion"] = convert(local_tif, zarr_copy)
        print("   ", {k: v for k, v in result["onboarding"]["conversion"].items()
                      if k != "path"}, flush=True)

    import httpx
    import imagecodecs.numcodecs
    import zarr
    imagecodecs.numcodecs.register_codecs(verbose=False)

    # --- geometry from the served view, so all arms sample the same grid ---
    meta = httpx.get(f"{args.base}/zarr/{args.dataset}/0/.zarray", timeout=900).json()
    shape, chunks = tuple(meta["shape"]), tuple(meta["chunks"])
    axes = httpx.get(f"{args.base}/api/datasets/{args.dataset}",
                     timeout=900).json()["axes"].lower()
    indices = sample_chunks(shape, chunks, axes, args.n, args.seed)
    result["sample"] = {"n": len(indices), "seed": args.seed, "level": 0,
                        "shape": list(shape), "chunks": list(chunks),
                        "note": "identical chunk indices used by every arm"}

    # --- A: Tier B, served view over HTTP, remote source ---
    print("[3/5] arm A: Tier B served view (remote) ...", flush=True)
    client = httpx.Client(timeout=900)
    def get_a(idx):
        key = "/".join(str(i) for i in idx)
        r = client.get(f"{args.base}/zarr/{args.dataset}/0/{key}")
        return r.content if r.status_code == 200 else b""
    cold, warm = time_reads(get_a, indices, repeat_warm=True)
    result["arms"]["A_view_tierB_remote"] = {
        "description": "served view; platform range-reads GCS, decodes, re-encodes",
        "storage": "remote (GCS)", "route": "view Tier B",
        "cold": stats(cold), "warm_server_cache": stats(warm)}

    # --- B and C: Tier A reference index, remote then local ---
    from tiff_arms import tier_a_arms  # noqa: E402
    print("[4/5] arms B and C: Tier A index, remote then local ...", flush=True)
    result["arms"].update(tier_a_arms(args.url, local_tif, indices, stats, time_reads))

    # --- D: converted copy on local disk ---
    print("[5/5] arm D: converted OME-Zarr, local ...", flush=True)
    g = zarr.open_group(str(zarr_copy), mode="r")["0"]
    def get_d(idx):
        sel = tuple(slice(i * c, (i + 1) * c) for i, c in zip(idx, chunks))
        return np.asarray(g[sel])
    cold, warm = time_reads(get_d, indices, repeat_warm=True)
    result["arms"]["D_converted_local"] = {
        "description": "a real converted OME-Zarr copy, read directly from disk",
        "storage": "local disk", "route": "native zarr",
        "cold": stats(cold), "warm_os_page_cache": stats(warm)}

    result["provenance"] = provenance({
        "source_url": args.url, "dataset": args.dataset, "served_base": args.base})
    result["binding_rules"] = [
        "Every arm reads the SAME 100 chunk indices at full resolution.",
        "B vs C isolates storage locality with the route held constant; "
        "C vs D isolates route/format with locality held constant. Quoting "
        "A or B against D alone conflates the two.",
        "The missing cell — a converted copy on the same remote bucket — was "
        "NOT measured, because we cannot write to that bucket. Any statement "
        "about converted-copy performance on remote storage is unsupported.",
        "'Warm' differs per arm and is labelled per arm: server chunk cache for "
        "A, client-side for B and C, OS page cache for D. They are not the "
        "same cache and should not be pooled.",
        "Conversion wall-clock and on-disk size are measured here, not estimated.",
    ]
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
