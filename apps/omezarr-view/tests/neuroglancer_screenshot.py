"""Drive Neuroglancer against a served view, to show the view is not Vizarr-specific.

Neuroglancer reads NGFF 0.4 multiscale zarr v2 with a zlib compressor, which is
what these views serve, so if the view is genuinely standard this needs no
special-casing on either side. Same evidence rule as the Vizarr tool: record
every request the viewer made to us, and count lit pixels, because a black
canvas is not distinguishable from a working render in a screenshot alone.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.parse
from pathlib import Path

from playwright.async_api import async_playwright

NEUROGLANCER = "https://neuroglancer-demo.appspot.com/"


def _ink(path: Path) -> dict:
    from PIL import Image

    img = Image.open(path).convert("L")
    w, h = img.size
    img = img.crop((int(w * 0.02), int(h * 0.10), int(w * 0.98), int(h * 0.98)))
    import numpy as np

    arr = np.asarray(img, dtype=float)
    hist = img.histogram()
    total = sum(hist)
    # Lit fraction alone cannot tell an image from a flat background: a canvas
    # of uniform grey scores 1.0. Structure (pixel spread) is the half that
    # distinguishes them.
    return {
        "lit_fraction": round(sum(hist[12:]) / total, 4),
        "stdev": round(float(arr.std()), 2),
        "pixels": total,
    }


def state_for(source: str, info: dict, window: dict | None) -> dict:
    """A viewer state: one image layer over the served view.

    Neuroglancer reads the NGFF axes and scales, but it does not read the OME
    `omero` block, so its default contrast is the full dtype range and a 16-bit
    image opens near black. The same measured percentiles the view already
    publishes are passed in as shader controls — the display setting is stated
    here rather than left to a default that misrepresents the data.
    """
    layer = {
        "type": "image",
        "name": "view",
        "source": f"zarr://{source}",
        "opacity": 1.0,
    }
    if window:
        layer["shaderControls"] = {
            "normalized": {"range": [float(window["start"]), float(window["end"])]}
        }
    # No crossSectionScale override: Neuroglancer's own default framing fits
    # the volume, and forcing a scale zoomed so far out that the canvas was
    # uniform background grey — which the lit-pixel check happily called a
    # successful render.
    return {"layers": [layer], "layout": "xy", "showDefaultAnnotations": False}


async def shoot(base: str, dataset: str, out_dir: Path, scale: int,
                settle_s: int, timeout_s: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = f"{base}/zarr/{dataset}"

    import httpx

    info = httpx.get(f"{base}/api/datasets/{dataset}", timeout=900).json()
    attrs = httpx.get(f"{base}/zarr/{dataset}/.zattrs", timeout=900).json()
    channels = (attrs.get("omero") or {}).get("channels") or []
    window = channels[0]["window"] if channels else None
    state = state_for(source, info, window)
    url = NEUROGLANCER + "#!" + urllib.parse.quote(json.dumps(state))

    calls: list[dict] = []
    errors: list[str] = []

    async with async_playwright() as pw:
        # Neuroglancer is WebGL2; headless Chromium has no WebGL at all, so this
        # must run under xvfb-run with a headful browser.
        browser = await pw.chromium.launch(headless=False, args=[
            "--use-gl=angle", "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        page = await browser.new_page(viewport={"width": 1280, "height": 860},
                                      device_scale_factor=scale)
        page.on("response", lambda r: calls.append(
            {"key": r.url.split(f"/zarr/{dataset}/")[-1], "status": r.status})
            if f"/zarr/{dataset}" in r.url else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        await page.goto(url, wait_until="load", timeout=timeout_s * 1000)
        t0 = time.perf_counter()
        last, stable = 0, time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            await asyncio.sleep(0.5)
            if len(calls) != last:
                last, stable = len(calls), time.perf_counter()
            elif calls and time.perf_counter() - stable > settle_s:
                break

        shot = out_dir / f"neuroglancer-{dataset}.png"
        await page.screenshot(path=str(shot))
        await browser.close()

    chunks = [c for c in calls
              if not c["key"].endswith((".zarray", ".zattrs", ".zgroup"))]
    ok = [c for c in chunks if c["status"] == 200]
    ink = _ink(shot)
    return {
        "viewer": "neuroglancer",
        "dataset": dataset,
        "url": url,
        "screenshot": str(shot),
        "metadata_requests": [c for c in calls if c not in chunks],
        "chunk_requests": len(chunks),
        "chunk_requests_ok": len(ok),
        "canvas": ink,
        "page_errors": errors[:8],
        "rendered": bool(ok) and ink["lit_fraction"] > 0.005 and ink["stdev"] > 3.0,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="../../.dev/omezarr-probe/shots")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("datasets", nargs="+")
    args = ap.parse_args()

    results = []
    for ds in args.datasets:
        r = await shoot(args.base, ds, Path(args.out), args.scale, args.settle,
                        args.timeout)
        results.append(r)
        print(json.dumps(r, indent=1))
    Path(args.out, "neuroglancer_results.json").write_text(
        json.dumps(results, indent=1))
    return 0 if all(r["rendered"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
