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
    """Is anything actually drawn?

    Lit-fraction alone passes a black canvas. Whole-frame spread alone passes a
    flat grey one, because the viewer's panel dividers carry the variance — a
    broken Neuroglancer render scored stdev 24.45 that way. Spread over the
    MIDDLE HALF is the metric that separates them: 8 distinct levels on the
    broken frame, 256 on a working one.
    """
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("L")
    w, h = img.size
    img = img.crop((int(w * 0.02), int(h * 0.10), int(w * 0.98), int(h * 0.98)))
    arr = np.asarray(img, dtype=float)
    h, w = arr.shape
    central = arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    hist = img.histogram()
    total = sum(hist)
    return {
        "lit_fraction": round(sum(hist[12:]) / total, 4),
        "stdev": round(float(arr.std()), 2),
        "central_stdev": round(float(central.std()), 2),
        "central_distinct_levels": int(np.unique(central.astype(np.uint8)).size),
        "pixels": total,
    }


def state_for(source: str, info: dict, window: dict | None,
              force_dimensions: bool = False) -> dict:
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
    state = {"layers": [layer], "layout": "xy", "showDefaultAnnotations": False}
    # An override is OFF by default. What made Neuroglancer pick t as a display
    # axis was a unitless t of scale 1.0; once the view maps the real time
    # increment, the override is unnecessary — and leaving it on would hide
    # whether that mapping actually works. Opt in with --force-dimensions only
    # to measure the delta against a view that still has the bug.
    axes = (info.get("axes") or "").lower()
    if force_dimensions and "t" in axes:
        state["dimensions"] = {a: [1, ""] for a in axes if a in "xyz"}
        state["displayDimensions"] = ["x", "y"]
    return state


async def shoot(base: str, dataset: str, out_dir: Path, scale: int,
                settle_s: int, timeout_s: int,
                force_dimensions: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = f"{base}/zarr/{dataset}"

    import httpx

    info = httpx.get(f"{base}/api/datasets/{dataset}", timeout=900).json()
    attrs = httpx.get(f"{base}/zarr/{dataset}/.zattrs", timeout=900).json()
    channels = (attrs.get("omero") or {}).get("channels") or []
    window = channels[0]["window"] if channels else None
    state = state_for(source, info, window, force_dimensions)
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
        # 128, not 32. Against nine labelled frames the broken set scored
        # [8, 8, 8, 37] and the working set [216, 227, 231, 256, 256], so the
        # real gap is [37, 216]: a 4-panel Neuroglancer layout of nothing but
        # grey panels and section rules scores 37 and passed the old threshold.
        "rendered": (bool(ok) and ink["lit_fraction"] > 0.005
                     and ink["central_distinct_levels"] > 128),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="../../.dev/omezarr-probe/shots")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--force-dimensions", action="store_true",
                    help="emit an x,y,z dimensions override; only useful to "
                         "measure the delta against a view with an unmapped t axis")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("datasets", nargs="+")
    args = ap.parse_args()

    results = []
    for ds in args.datasets:
        r = await shoot(args.base, ds, Path(args.out), args.scale, args.settle,
                        args.timeout, args.force_dimensions)
        results.append(r)
        print(json.dumps(r, indent=1))
    Path(args.out, "neuroglancer_results.json").write_text(
        json.dumps(results, indent=1))
    return 0 if all(r["rendered"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
