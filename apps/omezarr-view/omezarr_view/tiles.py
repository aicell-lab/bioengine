"""Render RGB map tiles from a view, so tile-based clients can consume it too.

Vizarr and Neuroglancer speak Zarr and read the view directly. Map libraries —
OpenLayers, Leaflet — speak image tiles, which is also what an annotation UI
usually wants: it needs a picture to draw on, not an array.

So this renders PNG tiles on demand from the same lazy view. It is a second
consumer of the same bytes, not a second copy of them: a tile request pulls
exactly the source chunks it overlaps and nothing is stored.
"""
from __future__ import annotations

import io
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

TILE = 512

# omero channel colours are hex without the leading '#'.
_FALLBACK = ["FFFFFF"]


def _hex_rgb(value: str) -> Tuple[float, float, float]:
    value = (value or "FFFFFF").lstrip("#")
    if len(value) != 6:
        value = "FFFFFF"
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore


class TileRenderer:
    """Renders tiles for one view, caching decoded planes it has to re-read."""

    def __init__(self, view, cache_tiles: int = 128):
        import zarr

        from .recipes import _ViewStore

        self.view = view
        self._group = zarr.open_group(
            zarr.storage.KVStore(_ViewStore(view)), mode="r")
        self._arrays: Dict[int, Any] = {}
        self._png: "OrderedDict[str, bytes]" = OrderedDict()
        self._max = cache_tiles
        self._lock = threading.Lock()
        self.axes = view.axes
        self.level_shapes = [list(s) for s in view.level_shapes]
        attrs = view.attrs
        self.channels = (attrs.get("omero") or {}).get("channels") or []
        self.rdefs = (attrs.get("omero") or {}).get("rdefs") or {}

    # -- geometry ---------------------------------------------------------

    def grid(self) -> Dict[str, Any]:
        """What a tile client needs to build its own tile grid.

        Levels are listed coarsest-first because that is the order a map
        library expects its zoom levels in, which is the reverse of the way a
        pyramid is stored.
        """
        yi, xi = len(self.axes) - 2, len(self.axes) - 1
        base_y, base_x = self.level_shapes[0][yi], self.level_shapes[0][xi]
        levels = []
        for lvl in range(len(self.level_shapes) - 1, -1, -1):
            shape = self.level_shapes[lvl]
            levels.append({
                "zoom": len(levels),
                "view_level": lvl,
                "width": shape[xi],
                "height": shape[yi],
                "resolution": base_x / shape[xi],
                "tiles_x": -(-shape[xi] // TILE),
                "tiles_y": -(-shape[yi] // TILE),
            })
        sizes = self.view.mapping.mapped.get("physical_pixel_sizes") or {}
        return {
            "width": base_x,
            "height": base_y,
            "tile_size": TILE,
            "levels": levels,
            "resolutions": [lv["resolution"] for lv in levels],
            "axes": self.axes.upper(),
            "index_axes": [a for a in self.axes if a not in "yxc"],
            "index_sizes": {a: self.level_shapes[0][i]
                            for i, a in enumerate(self.axes) if a not in "yxc"},
            "channels": [
                {"label": c.get("label"), "color": c.get("color"),
                 "window": c.get("window"), "active": c.get("active")}
                for c in self.channels],
            "physical_pixel_sizes": sizes,
            "note": "tiles are rendered on demand from the lazy view; nothing "
                    "is pre-rendered or stored",
        }

    # -- rendering --------------------------------------------------------

    def _array(self, level: int):
        if level not in self._arrays:
            self._arrays[level] = self._group[str(level)]
        return self._arrays[level]

    def _plane_region(self, level: int, channel: int, indices: Dict[str, int],
                      y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        sel: List[Any] = []
        for i, a in enumerate(self.axes):
            if i == len(self.axes) - 2:
                sel.append(slice(y0, y1))
            elif i == len(self.axes) - 1:
                sel.append(slice(x0, x1))
            elif a == "c":
                sel.append(channel)
            else:
                sel.append(int(indices.get(a, 0)))
        return np.asarray(self._array(level)[tuple(sel)], dtype=np.float32)

    def render(self, zoom: int, tx: int, ty: int, indices: Dict[str, int],
               channels: Optional[Sequence[int]] = None) -> Optional[bytes]:
        grid = self.grid()
        if not 0 <= zoom < len(grid["levels"]):
            return None
        info = grid["levels"][zoom]
        level = info["view_level"]
        if not (0 <= tx < info["tiles_x"] and 0 <= ty < info["tiles_y"]):
            return None

        key = f"{zoom}/{tx}/{ty}/{sorted(indices.items())}/{channels}"
        with self._lock:
            hit = self._png.get(key)
            if hit is not None:
                self._png.move_to_end(key)
                return hit

        yi, xi = len(self.axes) - 2, len(self.axes) - 1
        shape = self.level_shapes[level]
        y0, x0 = ty * TILE, tx * TILE
        y1, x1 = min(y0 + TILE, shape[yi]), min(x0 + TILE, shape[xi])

        n_c = shape[self.axes.index("c")] if "c" in self.axes else 1
        wanted = list(channels) if channels else [
            i for i, c in enumerate(self.channels) if c.get("active")] or [0]
        wanted = [c for c in wanted if 0 <= c < n_c] or [0]

        rgb = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
        for ci in wanted:
            plane = self._plane_region(level, ci, indices, y0, y1, x0, x1)
            spec = self.channels[ci] if ci < len(self.channels) else {}
            window = spec.get("window") or {}
            lo = float(window.get("start", 0.0))
            hi = float(window.get("end", 1.0))
            if hi <= lo:
                lo, hi = float(plane.min()), float(plane.max() or 1.0)
            norm = np.clip((plane - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            colour = _hex_rgb(spec.get("color") or
                              _FALLBACK[ci % len(_FALLBACK)])
            for k in range(3):
                rgb[..., k] += norm * colour[k]

        out = np.zeros((TILE, TILE, 3), dtype=np.uint8)
        out[: y1 - y0, : x1 - x0] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(out, "RGB").save(buf, format="PNG", optimize=False)
        png = buf.getvalue()
        with self._lock:
            self._png[key] = png
            while len(self._png) > self._max:
                self._png.popitem(last=False)
        return png
