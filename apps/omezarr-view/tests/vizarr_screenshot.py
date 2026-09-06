"""Drive Vizarr against a served view and record what it actually rendered.

A screenshot alone does not prove pixels arrived — Vizarr shows a black canvas
just as happily for a failed fetch. So this also records every network request
Vizarr made back to the view server, with status codes, and fails if no chunk
came back 200.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

VIZARR = "https://hms-dbmi.github.io/vizarr/"


def _ink(path: Path) -> dict:
    """Fraction of the canvas that is not black.

    A screenshot cannot distinguish 'rendered' from 'failed' on its own —
    Vizarr paints a black canvas just as happily for a view that never
    delivered a tile, and during development it did exactly that three times
    for three different reasons. Counting lit pixels is the actual evidence.
    """
    from PIL import Image

    img = Image.open(path).convert("L")
    # Ignore the control panel in the top-left corner.
    w, h = img.size
    img = img.crop((int(w * 0.25), 0, w, h))
    import numpy as np

    # Whole-frame spread is not enough: a flat grey canvas with two panel
    # dividers scores well because the DIVIDERS carry all the variance. Measure
    # the middle half, where only pixels can be.
    arr = np.asarray(img, dtype=float)
    h, w = arr.shape
    central = arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    hist = img.histogram()
    total = sum(hist)
    lit = sum(hist[12:])
    return {
        "lit_fraction": round(lit / total, 4),
        "central_stdev": round(float(central.std()), 2),
        "central_distinct_levels": int(np.unique(central.astype(np.uint8)).size),
        "pixels": total,
    }


async def shoot(base: str, dataset: str, out_dir: Path, timeout_s: int = 120,
                warm: bool = True, scale: int = 1, zoom: int = 0,
                suffix: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = f"{base}/zarr/{dataset}"
    url = f"{VIZARR}?source={source}"

    warm_seconds = None
    if warm:
        # Build the view first. A cold deep pyramid takes tens of seconds to
        # index, and screenshotting during that measures the build, not the
        # render.
        import httpx

        t0 = time.perf_counter()
        httpx.get(f"{base}/api/datasets/{dataset}", timeout=timeout_s * 2)
        warm_seconds = round(time.perf_counter() - t0, 2)
    calls: list[dict] = []
    errors: list[str] = []

    async with async_playwright() as pw:
        # Vizarr renders through deck.gl/WebGL, and Playwright's headless
        # Chromium has no WebGL at all — it fails at luma init and paints
        # nothing whether or not the data arrived. Only a headful browser under
        # Xvfb gets a working context here, so run this via `xvfb-run`.
        browser = await pw.chromium.launch(headless=False, args=[
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--ignore-gpu-blocklist",
        ])
        page = await browser.new_page(viewport={"width": 1280, "height": 860},
                                      device_scale_factor=scale)

        def on_response(r):
            if source in r.url:
                calls.append({
                    "key": r.url.split(f"/zarr/{dataset}/")[-1],
                    "status": r.status,
                })

        page.on("response", on_response)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                if m.type == "error" else None)

        t0 = time.perf_counter()
        await page.goto(url, wait_until="load", timeout=timeout_s * 1000)

        # Wait until chunk traffic stops arriving rather than a fixed sleep —
        # first paint is the measurement, so it must not be guessed.
        first_chunk_at = None
        stable_since = time.perf_counter()
        last_count = 0
        while time.perf_counter() - t0 < timeout_s:
            await asyncio.sleep(0.5)
            chunks = [c for c in calls if not c["key"].endswith((".zarray", ".zattrs",
                                                                ".zgroup"))]
            if chunks and first_chunk_at is None:
                first_chunk_at = time.perf_counter() - t0
            if len(calls) != last_count:
                last_count = len(calls)
                stable_since = time.perf_counter()
            elif chunks and time.perf_counter() - stable_since > 8:
                # Vizarr fetches metadata, pauses while it resolves the
                # pyramid, then bursts the tiles — a short settle window
                # screenshots the gap and reports a black canvas as a failure.
                break

        if zoom:
            # Zooming in forces the finest levels to load, which is what proves
            # the full-resolution tiles are really being served rather than a
            # thumbnail standing in for them.
            await page.mouse.move(820, 430)
            for _ in range(zoom):
                await page.mouse.wheel(0, -240)
                await asyncio.sleep(0.7)
            settle = time.perf_counter()
            last = len(calls)
            while time.perf_counter() - settle < 25:
                await asyncio.sleep(0.5)
                if len(calls) != last:
                    last = len(calls)
                    settle = time.perf_counter()
                elif time.perf_counter() - settle > 8:
                    break

        shot = out_dir / f"vizarr-{dataset}{suffix}.png"
        await page.screenshot(path=str(shot))
        await browser.close()

    chunks = [c for c in calls if not c["key"].endswith((".zarray", ".zattrs", ".zgroup"))]
    ok = [c for c in chunks if c["status"] == 200]
    ink = _ink(shot)
    return {
        "dataset": dataset,
        "vizarr_url": url,
        "screenshot": str(shot),
        "warm_build_seconds": warm_seconds,
        "seconds_to_first_chunk_request": round(first_chunk_at, 2) if first_chunk_at else None,
        "metadata_requests": [c for c in calls if c not in chunks],
        "chunk_requests": len(chunks),
        "chunk_requests_ok": len(ok),
        "chunk_statuses": sorted({c["status"] for c in chunks}),
        "canvas": ink,
        "page_errors": errors[:10],
        # Both halves are required: tiles arrived AND the canvas is lit.
        "rendered": (bool(ok) and ink["lit_fraction"] > 0.005
                     and ink["central_distinct_levels"] > 32),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="../../.dev/omezarr-probe/shots")
    ap.add_argument("--scale", type=int, default=1,
                    help="device pixel ratio; use 2 for figure-grade assets")
    ap.add_argument("--zoom", type=int, default=0,
                    help="wheel ticks to zoom in before the shot, forcing the "
                         "finest pyramid levels to load")
    ap.add_argument("--suffix", default="")
    ap.add_argument("datasets", nargs="+")
    args = ap.parse_args()

    results = []
    for ds in args.datasets:
        r = await shoot(args.base, ds, Path(args.out), scale=args.scale,
                        zoom=args.zoom, suffix=args.suffix)
        results.append(r)
        print(json.dumps(r, indent=1))
    Path(args.out, f"vizarr_results{args.suffix or ''}.json").write_text(
        json.dumps(results, indent=1))
    return 0 if all(r["rendered"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
