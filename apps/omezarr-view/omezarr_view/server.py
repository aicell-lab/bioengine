"""HTTP server presenting each catalog entry as an OME-Zarr endpoint.

Routes:
  ``GET /``                          the catalog page
  ``GET /api/datasets``              catalog as JSON
  ``GET /api/datasets/{id}``         one dataset, indexing it if needed
  ``GET /api/datasets/{id}/reference.json``  the byte-offset index, where the
                                     recipe has one — the client can read the
                                     original file with it and never call us again
  ``GET /api/datasets/{id}/thumbnail.png``
  ``GET /zarr/{id}/{key}``           the OME-Zarr store itself

CORS is open because a viewer runs on someone else's origin. Protected
datasets take a bearer token, or ``?token=`` for clients that cannot set a
header — the same two-door scheme ``bioengine/datasets/proxy_server.py`` uses.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .catalog import Catalog, DatasetEntry
from .recipes import ViewError
from .tiles import TileRenderer

logger = logging.getLogger("omezarr-view")

def _find_frontend() -> Optional[Path]:
    """Locate the static pages.

    A BioEngine replica receives the app source through Hypha's per-file
    transport, and a __file__-relative sibling directory is not guaranteed to
    land where it sits in the repo. Check the plausible roots rather than
    assuming one, and let /health report the answer instead of failing with a
    500 that says nothing.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent / "frontend",
                      here.parent / "frontend",
                      Path.cwd() / "frontend",
                      Path.cwd() / "omezarr-view" / "frontend"):
        if (candidate / "index.html").is_file():
            return candidate
    return None


FRONTEND = _find_frontend()
VIZARR = "https://hms-dbmi.github.io/vizarr/"


class ChunkCache:
    """Bounded LRU over served chunk bytes, shared across datasets.

    Without it every viewer pan re-reads the source, and 'warm' latency has no
    meaning to measure. Hits and misses are counted so the served numbers can
    say which they were.
    """

    def __init__(self, max_bytes: int = 512 << 20):
        self.max_bytes = max_bytes
        self._items: "OrderedDict[str, bytes]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            value = self._items.get(key)
            if value is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            if key in self._items:
                self._bytes -= len(self._items.pop(key))
            self._items[key] = value
            self._bytes += len(value)
            while self._bytes > self.max_bytes and self._items:
                _, dropped = self._items.popitem(last=False)
                self._bytes -= len(dropped)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"entries": len(self._items), "bytes": self._bytes,
                    "hits": self.hits, "misses": self.misses}


def _is_authorised(entry: DatasetEntry, authorization: Optional[str],
                   token: Optional[str]) -> bool:
    if not entry.is_protected:
        return True
    supplied = token
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    return supplied == entry.token


def _authorise(entry: DatasetEntry, authorization: Optional[str],
               token: Optional[str]) -> None:
    if not _is_authorised(entry, authorization, token):
        raise HTTPException(401, f"dataset '{entry.id}' requires a token")


def _redacted(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Catalog stub for a protected dataset the caller cannot open.

    Gating the pixels is not enough — dimensions, pixel sizes and channel
    names are themselves the sensitive part of an image, so an unauthorised
    listing must not carry the view's report at all.
    """
    # The title is author-written free text and routinely contains exactly what
    # redaction is for — "3-channel pyramid (42 GB)", "T=3_Z=5_CH=2". Keeping it
    # while stripping the shape fields is redaction in name only.
    return {
        "id": summary["id"],
        "location": summary["location"],
        "protected": True,
        "indexed": False,
        "redacted": True,
        "note": "title and structure withheld; supply a token to see them. "
                "NOTE the id itself is public — every route needs it — and ids "
                "derived from filenames leak what the filename said, so set an "
                "explicit id for anything sensitive.",
    }


def _thumbnail_png(array: np.ndarray, max_side: int = 320) -> bytes:
    """Render (C, Y, X) as an 8-bit RGB composite, contrast-stretched per channel."""
    from PIL import Image

    channels = array[:3] if array.shape[0] >= 3 else array[:1]
    rgb = np.zeros(channels.shape[1:] + (3,), dtype=np.float32)
    colors = [(0, 1, 0), (1, 0, 0), (0, 0, 1)]
    for i, plane in enumerate(channels):
        plane = plane.astype(np.float32)
        lo, hi = np.percentile(plane, (1, 99.5))
        if hi <= lo:
            lo, hi = float(plane.min()), float(plane.max() or 1)
        norm = np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1)
        if channels.shape[0] == 1:
            rgb += norm[..., None]
        else:
            for c in range(3):
                rgb[..., c] += norm * colors[i % 3][c]
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_app(config_path: str | Path, public_url: Optional[str] = None) -> FastAPI:
    catalog = Catalog.from_config(config_path)
    public_url = (public_url or os.getenv("OMEZARR_VIEW_PUBLIC_URL", "")).rstrip("/")
    thumbnails: Dict[str, bytes] = {}
    cache = ChunkCache(int(os.getenv("OMEZARR_VIEW_CACHE_BYTES", 512 << 20)))
    renderers: Dict[str, TileRenderer] = {}
    annotations_dir = Path(os.getenv(
        "OMEZARR_VIEW_ANNOTATIONS",
        Path(config_path).resolve().parent / "annotations"))

    app = FastAPI(title="BioEngine OME-Zarr view", docs_url="/api/docs")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Length", "Content-Range"],
    )

    def base_url(request: Request) -> str:
        """Public root to advertise in zarr and Vizarr URLs.

        Set OMEZARR_VIEW_PUBLIC_URL to pin it. Otherwise derive it from the
        request, forcing https for any non-local host: the tunnel terminates
        TLS and then forwards plain http, so ``x-forwarded-proto`` says
        ``http`` even for an https client. Trusting it hands a browser an
        http:// zarr URL, which it refuses as mixed content.
        """
        if public_url:
            return public_url
        root = str(request.base_url).rstrip("/")
        host = request.headers.get("x-forwarded-host") or request.url.hostname or ""
        if root.startswith("http://") and not host.startswith(("localhost", "127.")):
            root = "https://" + root[len("http://"):]
        return root

    def entry_or_404(dataset_id: str) -> DatasetEntry:
        entry = catalog.get(dataset_id)
        if entry is None:
            raise HTTPException(404, f"no dataset '{dataset_id}'")
        return entry

    async def build_view(entry: DatasetEntry):
        """Index off the event loop — a deep pyramid takes tens of seconds."""
        try:
            return await asyncio.to_thread(entry.ensure_view)
        except ViewError as e:
            raise HTTPException(422, str(e)) from e

    async def renderer_for(entry: DatasetEntry) -> TileRenderer:
        view = await build_view(entry)
        if entry.id not in renderers:
            renderers[entry.id] = TileRenderer(view)
        return renderers[entry.id]

    def _indices(request: Request) -> Dict[str, int]:
        """Non-spatial axis positions (z, t, ...) from the query string."""
        out: Dict[str, int] = {}
        for axis in ("t", "z"):
            raw = request.query_params.get(axis)
            if raw is not None:
                try:
                    out[axis] = int(raw)
                except ValueError:
                    raise HTTPException(400, f"{axis} must be an integer")
        return out

    @app.get("/api/datasets/{dataset_id}/tilegrid.json")
    async def tilegrid(dataset_id: str,
                       authorization: Optional[str] = Header(None),
                       token: Optional[str] = Query(None)):
        entry = entry_or_404(dataset_id)
        _authorise(entry, authorization, token)
        renderer = await renderer_for(entry)
        return await asyncio.to_thread(renderer.grid)

    @app.get("/tiles/{dataset_id}/{zoom}/{tx}/{ty}.png")
    async def tile(request: Request, dataset_id: str, zoom: int, tx: int, ty: int,
                   channels: Optional[str] = Query(None),
                   authorization: Optional[str] = Header(None),
                   token: Optional[str] = Query(None)):
        """RGB tiles, so tile-based clients (OpenLayers, Leaflet) can consume
        the same lazy view that Zarr clients read directly."""
        entry = entry_or_404(dataset_id)
        _authorise(entry, authorization, token)
        renderer = await renderer_for(entry)
        picked = ([int(c) for c in channels.split(",") if c.strip().isdigit()]
                  if channels else None)
        png = await asyncio.to_thread(renderer.render, zoom, tx, ty,
                                      _indices(request), picked)
        if png is None:
            raise HTTPException(404, "no such tile")
        return Response(png, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/api/datasets/{dataset_id}/annotations")
    async def get_annotations(dataset_id: str,
                              authorization: Optional[str] = Header(None),
                              token: Optional[str] = Query(None)):
        entry = entry_or_404(dataset_id)
        _authorise(entry, authorization, token)
        path = annotations_dir / f"{entry.id}.geojson"
        if not path.exists():
            return {"type": "FeatureCollection", "features": []}
        return json.loads(path.read_text())

    @app.put("/api/datasets/{dataset_id}/annotations")
    async def put_annotations(dataset_id: str, payload: Dict[str, Any],
                              authorization: Optional[str] = Header(None),
                              token: Optional[str] = Query(None)):
        """Annotations are stored beside the catalog, never in the image.

        The source file stays the untouched record — that is the whole claim —
        so anything a user draws lands in its own GeoJSON file.
        """
        entry = entry_or_404(dataset_id)
        _authorise(entry, authorization, token)
        if payload.get("type") != "FeatureCollection":
            raise HTTPException(400, "expected a GeoJSON FeatureCollection")
        annotations_dir.mkdir(parents=True, exist_ok=True)
        path = annotations_dir / f"{entry.id}.geojson"
        path.write_text(json.dumps(payload, indent=1))
        return {"saved": len(payload.get("features", [])), "path": str(path)}

    @app.get("/annotate", response_class=HTMLResponse)
    async def annotate():
        return _page("annotate.html")

    @app.get("/health")
    async def health():
        return {"status": "ok", "datasets": len(catalog.entries),
                "chunk_cache": cache.stats(),
                "pages": "on-disk" if FRONTEND else "bundled"}

    def _page(name: str) -> HTMLResponse:
        # On disk when running from a checkout, so editing the HTML is live.
        # Bundled when deployed, because a replica only receives Python modules.
        if FRONTEND is not None:
            return HTMLResponse((FRONTEND / name).read_text())
        try:
            from ._pages import PAGES
        except ImportError:
            raise HTTPException(
                503,
                "static pages are neither on disk nor bundled; run "
                "`python -m omezarr_view.bundle`. The JSON API at /api/datasets "
                "and the zarr endpoints at /zarr/{id} work regardless")
        if name not in PAGES:
            raise HTTPException(404, f"no page {name!r}")
        return HTMLResponse(PAGES[name])

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _page("index.html")

    @app.get("/api/datasets")
    async def list_datasets(request: Request,
                            authorization: Optional[str] = Header(None),
                            token: Optional[str] = Query(None)):
        root = base_url(request)
        out = []
        for entry in catalog.entries.values():
            summary = entry.summary()
            if not _is_authorised(entry, authorization, token):
                summary = _redacted(summary)
            summary["zarr_url"] = f"{root}/zarr/{summary['id']}"
            summary["vizarr_url"] = f"{VIZARR}?source={summary['zarr_url']}"
            out.append(summary)
        return {"title": catalog.title, "datasets": out, "vizarr": VIZARR}

    @app.get("/api/datasets/{dataset_id}")
    async def get_dataset(request: Request, dataset_id: str,
                          authorization: Optional[str] = Header(None),
                          token: Optional[str] = Query(None)):
        entry = entry_or_404(dataset_id)
        _authorise(entry, authorization, token)
        await build_view(entry)
        summary = entry.summary()
        root = base_url(request)
        summary["zarr_url"] = f"{root}/zarr/{entry.id}"
        summary["vizarr_url"] = f"{VIZARR}?source={summary['zarr_url']}"
        view = entry.view
        channels = ((view.attrs.get("omero") or {}).get("channels") or []) if view else []
        if channels:
            # Neuroglancer ignores the omero block, so a client building a
            # Neuroglancer link needs the measured window handed to it.
            summary["omero_window"] = channels[0]["window"]
        return summary

    @app.get("/api/datasets/{dataset_id}/reference.json")
    async def reference(dataset_id: str,
                        authorization: Optional[str] = Header(None),
                        token: Optional[str] = Query(None)):
        entry = entry_or_404(dataset_id)
        _authorise(entry, authorization, token)
        view = await build_view(entry)
        if not hasattr(view, "raw_reference"):
            raise HTTPException(
                404,
                f"the '{view.recipe}' recipe has no byte-offset index; this "
                "dataset can only be read through the server",
            )
        return JSONResponse(view.raw_reference())

    @app.get("/api/datasets/{dataset_id}/thumbnail.png")
    async def thumbnail(dataset_id: str,
                        authorization: Optional[str] = Header(None),
                        token: Optional[str] = Query(None)):
        entry = entry_or_404(dataset_id)
        _authorise(entry, authorization, token)
        if dataset_id not in thumbnails:
            view = await build_view(entry)
            array = await asyncio.to_thread(view.thumbnail_array)
            thumbnails[dataset_id] = await asyncio.to_thread(_thumbnail_png, array)
        return Response(thumbnails[dataset_id], media_type="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/t/{path_token}/zarr/{dataset_id}/{key:path}")
    async def zarr_key_path_token(request: Request, path_token: str,
                                  dataset_id: str, key: str):
        """Token in the path, not the query.

        A viewer builds chunk URLs by appending the key to the store root, so
        ``?token=`` lands before the key and every chunk 404s. Putting the
        token in a path prefix is the only spelling that survives that, which
        is what makes access-controlled data openable in a third-party viewer.
        """
        return await zarr_key(request, dataset_id, key, None, path_token)

    @app.get("/zarr/{dataset_id}")
    @app.get("/zarr/{dataset_id}/")
    async def zarr_root(request: Request, dataset_id: str,
                        authorization: Optional[str] = Header(None),
                        token: Optional[str] = Query(None)):
        # Must declare its own credential params and pass real values through:
        # calling zarr_key as a plain function handed FastAPI's Header/Query
        # sentinel objects to the authoriser, which then raised instead of
        # answering 401 for a missing credential.
        return await zarr_key(request, dataset_id, ".zgroup", authorization, token)

    @app.get("/zarr/{dataset_id}/{key:path}")
    async def zarr_key(request: Request, dataset_id: str, key: str,
                       authorization: Optional[str] = Header(None),
                       token: Optional[str] = Query(None)):
        entry = entry_or_404(dataset_id)
        _authorise(entry, authorization, token)
        view = await build_view(entry)
        key = key.strip("/")
        cache_key = f"{dataset_id}/{key}"
        t0 = time.perf_counter()
        value = cache.get(cache_key)
        warm = value is not None
        if not warm:
            try:
                value = await asyncio.to_thread(view.zarr_key, key)
            except Exception as e:
                logger.exception("chunk %s/%s failed", dataset_id, key)
                raise HTTPException(500, str(e)) from e
            if value is not None:
                cache.put(cache_key, value)
        if value is None:
            # Zarr treats 404 as "absent, use fill_value", which is correct for
            # a chunk the source never wrote.
            raise HTTPException(404, f"no key '{key}'")
        return Response(
            value,
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "public, max-age=3600",
                "X-View-Cache": "hit" if warm else "miss",
                "Server-Timing": f"view;dur={(time.perf_counter() - t0) * 1000:.1f}",
            },
        )

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = os.getenv("OMEZARR_VIEW_CONFIG",
                       str(Path(__file__).resolve().parent.parent / "datasets.yaml"))
    # Behind a tunnel/proxy the app must build its own public URLs from the
    # forwarded headers, otherwise every zarr_url it hands out points at
    # localhost and nothing outside the host can open the view.
    uvicorn.run(create_app(config), host="0.0.0.0",
                port=int(os.getenv("PORT", "8842")), log_level="info",
                proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
