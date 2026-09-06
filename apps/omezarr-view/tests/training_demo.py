"""Train a small model whose data loader reads through the served OME-Zarr view.

The claim this produces evidence for is narrow and deliberate: **training
consumes the view**, and here is what the loader costs. No claim is made about
what the model learned — the objective is self-supervised denoising precisely
so that no label, and no quality claim, is involved.

Throughput is measured three ways on DISJOINT crop sets of equal size, so each
condition is equally cold. Measuring them on the same crops would let the
server's chunk cache make whichever ran second look fast.

  lazy        one crop at a time, straight through the served view
  prefetch    the same reads, overlapped by a small thread pool (the async
              pipeline class: computation running ahead of consumption)
  local       the same crops after materialising them to a local file, with
              the materialisation cost reported rather than hidden

CPU only by design.
"""
from __future__ import annotations

import argparse
import json
import platform
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def provenance(base: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    def _git(*a: str) -> str:
        try:
            return subprocess.run(["git", "-C", "/data/nmechtel/bioengine", *a],
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return "unknown"

    out = {
        "endpoint": base,
        "host": platform.node(),
        "device": "cpu",
        "cpu_count": __import__("os").cpu_count(),
        "bioengine_commit": _git("rev-parse", "--short", "HEAD"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "apps/omezarr-view/tests/training_demo.py",
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# crops


def plan_crops(shape: List[int], axes: str, n: int, size: int,
               seed: int, zs: List[Any]) -> List[Tuple[int, ...]]:
    """Pick crop origins over a GIVEN set of z planes.

    The z planes are passed in rather than drawn here so that each measured
    condition can be given its own, disjoint, set. Sharing z planes across
    conditions would let the first condition warm the source chunks the next
    one needs, and the second would then be measuring a cache.
    """
    rng = np.random.default_rng(seed)
    yi, xi = len(axes) - 2, len(axes) - 1
    crops = []
    for i in range(n):
        origin: List[Any] = []
        for ax_i, a in enumerate(axes):
            if ax_i == yi:
                origin.append(int(rng.integers(0, max(1, shape[ax_i] - size))))
            elif ax_i == xi:
                origin.append(int(rng.integers(0, max(1, shape[ax_i] - size))))
            elif a == "z":
                origin.append(int(zs[i % len(zs)]))  # noqa: E501
            elif a == "c":
                origin.append(-1)  # all channels
            else:
                origin.append(0)
        crops.append(tuple(origin))
    return crops


def read_crop(arr, origin: Tuple[int, ...], axes: str, size: int) -> np.ndarray:
    yi, xi = len(axes) - 2, len(axes) - 1
    sel: List[Any] = []
    for i, a in enumerate(axes):
        if i == yi or i == xi:
            sel.append(slice(origin[i], origin[i] + size))
        elif a == "c":
            sel.append(slice(None))
        else:
            sel.append(origin[i])
    out = np.asarray(arr[tuple(sel)], dtype=np.float32)
    if out.ndim == 2:
        out = out[None]
    return out


# ---------------------------------------------------------------------------
# loaders


def run_lazy(arr, crops, axes, size) -> Tuple[List[np.ndarray], float]:
    t0 = time.perf_counter()
    got = [read_crop(arr, c, axes, size) for c in crops]
    return got, time.perf_counter() - t0


def run_prefetch(arr, crops, axes, size, workers: int,
                 depth: int) -> Tuple[List[np.ndarray], float]:
    """Fetch ahead into a bounded queue while the consumer works.

    The queue is bounded so this stays a prefetcher and does not quietly become
    'download everything first', which would measure something else.
    """
    q: "queue.Queue" = queue.Queue(maxsize=depth)
    done = object()

    def produce():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fut in [pool.submit(read_crop, arr, c, axes, size) for c in crops]:
                q.put(fut.result())
        q.put(done)

    t0 = time.perf_counter()
    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    got = []
    while True:
        item = q.get()
        if item is done:
            break
        got.append(item)
    thread.join()
    return got, time.perf_counter() - t0


def materialise(arr, crops, axes, size, path: Path) -> Dict[str, Any]:
    """Write the crops to a local file — the cost a 'just copy it' answer pays."""
    t0 = time.perf_counter()
    stack = np.stack([read_crop(arr, c, axes, size) for c in crops])
    np.save(path, stack)
    return {
        "seconds": time.perf_counter() - t0,
        "bytes": path.stat().st_size,
        "path": str(path),
    }


def run_local(path: Path) -> Tuple[List[np.ndarray], float]:
    t0 = time.perf_counter()
    stack = np.load(path, mmap_mode=None)
    got = [np.asarray(stack[i]) for i in range(stack.shape[0])]
    return got, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# model


def build_unet(in_ch: int):
    import torch.nn as nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(inplace=True),
                             nn.Conv2d(b, b, 3, padding=1), nn.ReLU(inplace=True))

    class TinyUNet(nn.Module):
        def __init__(self, ch=16):
            super().__init__()
            self.d1, self.d2 = block(in_ch, ch), block(ch, ch * 2)
            self.bott = block(ch * 2, ch * 4)
            self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
            self.u2 = block(ch * 4, ch * 2)
            self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
            self.u1 = block(ch * 2, ch)
            self.out = nn.Conv2d(ch, in_ch, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            import torch
            c1 = self.d1(x)
            c2 = self.d2(self.pool(c1))
            b = self.bott(self.pool(c2))
            u2 = self.u2(torch.cat([self.up2(b), c2], 1))
            u1 = self.u1(torch.cat([self.up1(u2), c1], 1))
            return self.out(u1)

    return TinyUNet()


def train(crops_data: List[np.ndarray], steps: int, batch: int,
          seed: int) -> Dict[str, Any]:
    """Self-supervised denoising. No label, and therefore no quality claim."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4) // 2))
    stack = torch.from_numpy(np.stack(crops_data).astype(np.float32))
    stack = stack / max(1.0, float(stack.max()))
    model = build_unet(stack.shape[1])
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)

    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        idx = torch.randint(0, stack.shape[0], (batch,), generator=gen)
        clean = stack[idx]
        noisy = clean + 0.1 * torch.randn(clean.shape, generator=gen)
        loss = torch.nn.functional.mse_loss(model(noisy), clean)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(round(float(loss.item()), 6))
    seconds = time.perf_counter() - t0

    return {
        "objective": "self-supervised denoising (MSE of reconstruction from a "
                     "noised copy); no labels, and no claim is made about what "
                     "the model learned",
        "model": "tiny U-Net, 2 down / 2 up, base 16 channels",
        "parameters": params,
        "steps": steps,
        "batch_size": batch,
        "samples_seen": steps * batch,
        "seconds": round(seconds, 2),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_trace": losses,
    }


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", default="idr0106-training-10um")
    ap.add_argument("--crops", type=int, default=24)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--z-planes", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--queue-depth", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--scratch", default=".training_demo")
    ap.add_argument("--out", default="/data/nmechtel/bioengine-paper/analysis/"
                                     "results/omezarr_view/figure_assets/"
                                     "training_demo.json")
    args = ap.parse_args()

    import httpx
    import zarr

    info = httpx.get(f"{args.base}/api/datasets/{args.dataset}", timeout=900).json()
    axes = info["axes"].lower()
    shape = info["level_shapes"][0]
    group = zarr.open_group(zarr.storage.FSStore(f"{args.base}/zarr/{args.dataset}"),
                            mode="r")
    arr = group["0"]

    # Three disjoint sets over DISJOINT Z PLANES, same size, same sampling.
    # Disjoint (y, x) alone is not enough: crops on a shared z plane hit the
    # same source chunks, so whichever condition ran second would be measuring
    # the first one's cache.
    n = args.crops
    rng = np.random.default_rng(args.seed)
    zi = axes.index("z") if "z" in axes else None
    if zi is not None:
        pool = rng.choice(shape[zi], size=min(args.z_planes * 3, shape[zi]),
                          replace=False)
        groups = np.array_split(pool, 3)
    else:
        groups = [[None]] * 3
    sets = {
        name: plan_crops(shape, axes, n, args.size, args.seed + k, list(groups[k]))
        for k, name in enumerate(("lazy", "prefetch", "local"))
    }
    z_used = {name: sorted(int(z) for z in groups[k])
              for k, name in enumerate(("lazy", "prefetch", "local"))}

    scratch = Path(args.scratch)
    scratch.mkdir(exist_ok=True)
    results: Dict[str, Any] = {}

    print("lazy ...", flush=True)
    lazy_data, lazy_s = run_lazy(arr, sets["lazy"], axes, args.size)
    results["lazy"] = {"crops": n, "seconds": round(lazy_s, 2),
                       "samples_per_s": round(n / lazy_s, 3)}

    print("prefetch ...", flush=True)
    pf_data, pf_s = run_prefetch(arr, sets["prefetch"], axes, args.size,
                                 args.workers, args.queue_depth)
    results["prefetch"] = {"crops": n, "seconds": round(pf_s, 2),
                           "samples_per_s": round(n / pf_s, 3),
                           "workers": args.workers, "queue_depth": args.queue_depth}

    print("materialise + local ...", flush=True)
    mat = materialise(arr, sets["local"], axes, args.size, scratch / "crops.npy")
    local_data, local_s = run_local(scratch / "crops.npy")
    results["local_copy"] = {
        "crops": n, "seconds": round(local_s, 3),
        "samples_per_s": round(n / local_s, 1),
        "materialisation": {
            "seconds": round(mat["seconds"], 2),
            "bytes": mat["bytes"],
            "note": "the one-off cost of making the local copy, paid before any "
                    "of its throughput can be enjoyed; the lazy view pays none "
                    "of it",
        },
    }

    print("training ...", flush=True)
    results["training"] = train(lazy_data + pf_data, args.steps, args.batch,
                                args.seed)

    cache = httpx.get(f"{args.base}/api/datasets/{args.dataset}",
                      timeout=900).json().get("source_chunk_cache")

    payload = {
        "provenance": provenance(args.base, {
            "dataset": args.dataset,
            "dataset_recipe": info.get("recipe"),
            "dataset_shape": shape,
            "crop_size": args.size,
            "crops_per_condition": n,
            "z_planes_per_condition": args.z_planes,
            "z_planes_used": z_used,
        }),
        "what_this_shows": (
            "A model trained on batches read through the served OME-Zarr view of "
            "a remote file that was never converted. The training is real and the "
            "loss trace is recorded; no claim is made about model quality, and "
            "the objective is label-free for that reason."),
        "throughput": results,
        "source_chunk_cache_after": cache,
        "binding_rules": [
            "The three conditions use disjoint crop sets over DISJOINT Z PLANES, "
            "so none inherits another's warm source chunks. Disjoint (y, x) "
            "alone is not sufficient: crops sharing a z plane share source "
            "chunks, and an earlier condition would warm them for a later one.",
            "The local-copy row is not free: its materialisation time and bytes "
            "are reported beside it and must be quoted with it.",
            "CPU only. No GPU was used.",
            "No training-quality claim follows from this run.",
            "Throughput is site-specific: europa to Google Cloud Storage.",
        ],
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "training"}, indent=1))
    print("training:", {k: v for k, v in results["training"].items()
                        if k != "loss_trace"})
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
