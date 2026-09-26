"""Recipe class 2: a DERIVED view — a computed transformation, served lazily.

A raw view re-addresses the original's own pixels. A derived view computes new
ones: rescaled to a target physical spacing, channel-selected, dtype-normalised
— a "training view" shaped to what a model wants, produced per request from the
raw view and never written back.

The honesty line this file has to hold: **the original file is still never
migrated or rewritten, but what comes out of here is a computed product, not
the source's pixels.** Every derived view says so in its own report and caption,
names its transform chain, and reports its cache statistics as measured facts.
Nothing here may be described as zero-copy.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ngff import MetadataMapping
from .recipes import OUT_COMPRESSOR, BaseView, ViewError, normalise_chunk_key

TILE = 512

_DTYPES = {"uint8": np.uint8, "uint16": np.uint16, "float32": np.float32}


def _resize_plane(plane: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a 2D plane. Float throughout; the caller casts."""
    from PIL import Image

    if plane.shape == out_hw:
        return plane.astype(np.float32, copy=False)
    img = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32), mode="F")
    return np.asarray(img.resize((out_hw[1], out_hw[0]), Image.BILINEAR))


class _CachingSourceStore(dict):
    """Zarr store over a base view, with an LRU on SOURCE CHUNK keys.

    Caching whole requested regions is useless here: every output tile asks
    for a different rectangle, so a region-keyed cache never hits. The unit
    that actually repeats is the source chunk — one 512x512 source tile is
    touched by every output tile that overlaps it — so the LRU has to sit at
    the chunk key, which is exactly what zarr asks this store for.
    """

    def __init__(self, view: BaseView, max_chunks: int = 256):
        super().__init__()
        self._view = view
        self._cache: "OrderedDict[str, bytes]" = OrderedDict()
        self._max = max_chunks
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __getitem__(self, key: str) -> bytes:
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                self.hits += 1
                return hit
        value = self._view.zarr_key(key)
        if value is None:
            raise KeyError(key)
        with self._lock:
            self.misses += 1
            self._cache[key] = value
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
        return value

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


class _SourceReader:
    """Reads source regions through the base view, over a chunk-caching store."""

    def __init__(self, base: BaseView, max_chunks: int = 256):
        import zarr

        self._store = _CachingSourceStore(base, max_chunks)
        self._group = zarr.open_group(zarr.storage.KVStore(self._store), mode="r")
        self._arrays: Dict[int, Any] = {}

    def array(self, level: int):
        if level not in self._arrays:
            self._arrays[level] = self._group[str(level)]
        return self._arrays[level]

    def region(self, level: int, index: Tuple[Any, ...]) -> np.ndarray:
        return np.asarray(self.array(level)[index])

    def stats(self) -> Dict[str, int]:
        total = self._store.hits + self._store.misses
        return {
            "source_chunk_hits": self._store.hits,
            "source_chunk_misses": self._store.misses,
            "hit_rate": round(self._store.hits / total, 3) if total else None,
        }


class DerivedView(BaseView):
    """A computed view over another view.

    Level 0 is exactly the requested target — the resolution a training loader
    consumes. Coarser levels are added only when the target is large enough
    that a viewer could not otherwise draw it, and they are navigation aids,
    not part of what was asked for.
    """

    recipe = "derived"

    def __init__(
        self,
        dataset_id: str,
        base: BaseView,
        title: Optional[str] = None,
        target_spacing_um: Optional[float] = None,
        target_scale: Optional[float] = None,
        channels: Optional[Sequence[Any]] = None,
        dtype: str = "uint8",
        normalise: str = "percentile",
    ):
        super().__init__(dataset_id, base.source, title or f"{base.title} (derived)")
        self.base = base
        self.axes = base.axes
        if dtype not in _DTYPES:
            raise ViewError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
        if normalise not in ("percentile", "none"):
            raise ViewError("normalise must be 'percentile' or 'none'")
        self.dtype = np.dtype(_DTYPES[dtype])
        self.normalise = normalise

        t0 = time.perf_counter()
        self._reader = _SourceReader(base)
        base_shape = list(base.level_shapes[0])
        self._yi, self._xi = len(self.axes) - 2, len(self.axes) - 1

        # --- channel selection ---
        self._channel_index: Optional[List[int]] = None
        base_names = list(
            base.mapping.mapped.get("channel_names")
            or ([None] * base_shape[self.axes.index("c")] if "c" in self.axes else [])
        )
        if channels is not None:
            if "c" not in self.axes:
                raise ViewError("channel selection requested but the source has no channel axis")
            picked = []
            for want in channels:
                if isinstance(want, int):
                    picked.append(want)
                else:
                    if want not in base_names:
                        raise ViewError(
                            f"no channel named {want!r}; source has {base_names}")
                    picked.append(base_names.index(want))
            n_c = base_shape[self.axes.index("c")]
            bad = [i for i in picked if not 0 <= i < n_c]
            if bad:
                raise ViewError(f"channel index out of range: {bad} (source has {n_c})")
            self._channel_index = picked

        # --- spatial rescale ---
        sizes = base.mapping.mapped.get("physical_pixel_sizes") or {}
        self.source_spacing = {a: sizes[a]["size"] for a in ("x", "y") if a in sizes}
        if target_spacing_um is not None:
            if len(self.source_spacing) < 2:
                raise ViewError(
                    "target_spacing_um needs the source's physical pixel size, which "
                    "this file does not declare; use target_scale instead")
            fy = target_spacing_um / self.source_spacing["y"]
            fx = target_spacing_um / self.source_spacing["x"]
        elif target_scale is not None:
            if not 0 < target_scale <= 1:
                raise ViewError("target_scale must be in (0, 1]")
            fy = fx = 1.0 / target_scale
        else:
            fy = fx = 1.0
        if fy < 1 or fx < 1:
            raise ViewError(
                "this view only downsamples; the requested target is finer than the "
                "source, which would invent detail")
        self._factor = (fy, fx)
        self.target_spacing_um = target_spacing_um

        out_shape = list(base_shape)
        if self._channel_index is not None:
            out_shape[self.axes.index("c")] = len(self._channel_index)
        out_shape[self._yi] = max(1, int(round(base_shape[self._yi] / fy)))
        out_shape[self._xi] = max(1, int(round(base_shape[self._xi] / fx)))
        self.shape = out_shape

        # A derived view carries its own pyramid. A single level is what a
        # training loader wants, but it makes a large view unusable in a
        # viewer: with nothing coarser to draw, the viewer must fetch every
        # tile of the full grid before it can show anything. Level 0 is still
        # exactly the requested target; the extra levels are only for
        # navigation.
        self.level_shapes = [list(out_shape)]
        while (max(self.level_shapes[-1][self._yi], self.level_shapes[-1][self._xi])
               > 1024 and len(self.level_shapes) < 6):
            prev = self.level_shapes[-1]
            nxt = list(prev)
            nxt[self._yi] = max(1, prev[self._yi] // 2)
            nxt[self._xi] = max(1, prev[self._xi] // 2)
            self.level_shapes.append(nxt)

        self.chunks = [1] * (len(out_shape) - 2) + [
            min(TILE, out_shape[self._yi]), min(TILE, out_shape[self._xi])]

        # Per output level, read from the coarsest source level that still
        # holds the detail it needs; resizing from level 0 every time would
        # read hundreds of times more bytes than the result contains.
        self._source_level: List[int] = []
        for shape in self.level_shapes:
            chosen = 0
            for i, src in enumerate(base.level_shapes):
                if src[self._yi] >= shape[self._yi] and src[self._xi] >= shape[self._xi]:
                    chosen = i
            self._source_level.append(chosen)
        self._level = self._source_level[0]
        self._level_shape = list(base.level_shapes[self._level])

        # --- normalisation bounds ---
        # Reuse the percentiles the base view already measured for display; they
        # are data-derived, and the record says so rather than implying the file
        # declared them.
        self._bounds: List[Tuple[float, float]] = []
        base_channels = (base.attrs.get("omero") or {}).get("channels") or []
        selected = self._channel_index if self._channel_index is not None \
            else list(range(max(1, len(base_channels))))
        for ci in selected:
            if normalise == "percentile" and ci < len(base_channels):
                w = base_channels[ci]["window"]
                self._bounds.append((float(w["start"]), float(w["end"])))
            else:
                self._bounds.append((0.0, 0.0))  # sentinel: pass through

        self.attrs, self.mapping = self._build_attrs(base_names, selected)
        self.build_timings["configure_s"] = time.perf_counter() - t0

        self._meta = {
            ".zgroup": json.dumps({"zarr_format": 2}).encode(),
            ".zattrs": json.dumps(self.attrs).encode(),
        }
        for lvl, shape in enumerate(self.level_shapes):
            self._meta[f"{lvl}/.zarray"] = json.dumps({
                "zarr_format": 2,
                "shape": shape,
                "chunks": self.chunks,
                "dtype": self.dtype.str,
                "compressor": OUT_COMPRESSOR.get_config(),
                "dimension_separator": "/",
                "fill_value": 0,
                "filters": None,
                "order": "C",
            }).encode()
            self._meta[f"{lvl}/.zattrs"] = json.dumps(
                {"_ARRAY_DIMENSIONS": list(self.axes.upper())}).encode()

    # ------------------------------------------------------------------

    def _build_attrs(self, base_names: List[Optional[str]],
                     selected: List[int]) -> Tuple[Dict[str, Any], MetadataMapping]:
        mapping = MetadataMapping()
        sizes = self.base.mapping.mapped.get("physical_pixel_sizes") or {}
        fy, fx = self._factor
        out_sizes = {}
        for axis, factor in (("y", fy), ("x", fx)):
            if axis in sizes:
                out_sizes[axis] = {"size": sizes[axis]["size"] * factor,
                                   "unit": sizes[axis]["unit"]}
        if "z" in sizes:
            # z is not resampled by this view; the source spacing carries over.
            out_sizes["z"] = dict(sizes["z"])

        ngff_axes = []
        for a in self.axes:
            entry: Dict[str, Any] = {
                "name": a,
                "type": {"t": "time", "c": "channel"}.get(a, "space")}
            if entry["type"] == "space" and a in out_sizes and out_sizes[a]["unit"]:
                entry["unit"] = out_sizes[a]["unit"]
            ngff_axes.append(entry)
        base_scale = [out_sizes.get(a, {}).get("size", 1.0) for a in self.axes]
        datasets = []
        for lvl, shape in enumerate(self.level_shapes):
            ratio_y = self.level_shapes[0][self._yi] / shape[self._yi]
            ratio_x = self.level_shapes[0][self._xi] / shape[self._xi]
            scale = list(base_scale)
            scale[self._yi] *= ratio_y
            scale[self._xi] *= ratio_x
            datasets.append({"path": str(lvl), "coordinateTransformations": [
                {"type": "scale", "scale": scale}]})

        names = [base_names[i] if i < len(base_names) else None for i in selected]
        attrs: Dict[str, Any] = {
            "multiscales": [{
                "version": "0.4",
                "name": self.title,
                "axes": ngff_axes,
                "datasets": datasets,
            }]
        }
        if "c" in self.axes:
            hi = 1.0 if self.dtype == np.float32 else float(np.iinfo(self.dtype).max)
            attrs["omero"] = {
                "version": "0.4",
                "name": self.title,
                "channels": [{
                    "label": names[i] or f"Channel {i}",
                    "color": ["00FF00", "FF0000", "0000FF", "FFFF00"][i % 4],
                    "window": {"start": 0, "end": hi, "min": 0, "max": hi},
                    "active": i < 3,
                } for i in range(len(selected))],
                "rdefs": {"model": "color"},
            }
            if "z" in self.axes:
                attrs["omero"]["rdefs"]["defaultZ"] = \
                    self.shape[self.axes.index("z")] // 2

        mapping.add("derived_from", self.base.id)
        mapping.add("transform_chain", self.transform_chain())
        mapping.add("dimensions", {a: self.shape[i] for i, a in enumerate(self.axes)})
        mapping.add("physical_pixel_sizes", out_sizes)
        if any(names):
            mapping.add("channel_names", names)
        mapping.add("dtype", self.dtype.str)
        mapping.add("output_levels", len(self.level_shapes))
        mapping.add("source_level_read", self._source_level)
        mapping.miss("pixel values",
                     "COMPUTED, not the source's own pixels — resampled and "
                     "rescaled per request; the original file is untouched")
        if self.normalise == "percentile":
            mapping.miss("intensity calibration",
                         "normalisation bounds are 1-99.8 percentiles measured "
                         "from a coarse level of the source, not declared by it; "
                         "absolute intensities are not preserved")
        mapping.miss("z spacing", "not resampled by this view; source z spacing carried over")
        for absent in ("channel colours", "objective / instrument metadata",
                       "stage position", "plate / well context",
                       "ROIs and annotations"):
            mapping.miss(absent, "not mapped by this recipe")
        return attrs, mapping

    def transform_chain(self) -> List[str]:
        chain = []
        if self._channel_index is not None:
            chain.append(f"select channels {self._channel_index}")
        fy, fx = self._factor
        if fy != 1.0 or fx != 1.0:
            if self.target_spacing_um:
                chain.append(
                    f"resample to {self.target_spacing_um:g} µm/px "
                    f"(x{1/fx:.3g} in x, x{1/fy:.3g} in y, bilinear)")
            else:
                chain.append(f"rescale x{1/fx:.3g} (bilinear)")
        if self.normalise == "percentile":
            chain.append("normalise per channel to 1-99.8 percentiles of the source")
        chain.append(f"cast to {self.dtype.name}")
        return chain

    # ------------------------------------------------------------------

    def _compute_tile(self, level: int, index: Tuple[int, ...]) -> np.ndarray:
        """Compute one output chunk of ``level``, padded to the full chunk shape."""
        out_shape = self.level_shapes[level]
        src_level = self._source_level[level]
        src_shape = self.base.level_shapes[src_level]

        cy, cx = self.chunks[-2], self.chunks[-1]
        y0, x0 = index[self._yi] * cy, index[self._xi] * cx
        y1 = min(y0 + cy, out_shape[self._yi])
        x1 = min(x0 + cx, out_shape[self._xi])

        # Output pixel [y] samples source rows [y * sy, (y+1) * sy).
        sy = src_shape[self._yi] / out_shape[self._yi]
        sx = src_shape[self._xi] / out_shape[self._xi]
        syi0 = int(np.floor(y0 * sy))
        syi1 = min(int(np.ceil(y1 * sy)), src_shape[self._yi])
        sxi0 = int(np.floor(x0 * sx))
        sxi1 = min(int(np.ceil(x1 * sx)), src_shape[self._xi])
        syi1 = max(syi1, syi0 + 1)
        sxi1 = max(sxi1, sxi0 + 1)

        sel: List[Any] = []
        out_c = 0
        for i, a in enumerate(self.axes):
            if i == self._yi:
                sel.append(slice(syi0, syi1))
            elif i == self._xi:
                sel.append(slice(sxi0, sxi1))
            elif a == "c":
                # The output channel index is the request's own index whether
                # or not a selection is in play; only the SOURCE index differs.
                out_c = index[i]
                sel.append(self._channel_index[index[i]]
                           if self._channel_index is not None else index[i])
            else:
                sel.append(index[i])
        region = self._reader.region(src_level, tuple(sel))
        resized = _resize_plane(np.asarray(region, dtype=np.float32), (y1 - y0, x1 - x0))

        if self.normalise == "percentile" and out_c < len(self._bounds):
            lo, hi = self._bounds[out_c]
            if hi > lo:
                resized = np.clip((resized - lo) / (hi - lo), 0.0, 1.0)
                if self.dtype != np.float32:
                    resized = resized * float(np.iinfo(self.dtype).max)

        tile = np.zeros(self.chunks, dtype=self.dtype)
        tile[..., : y1 - y0, : x1 - x0] = resized.astype(self.dtype, copy=False)
        return tile

    def contains(self, key: str) -> bool:
        if key in self._meta:
            return True
        head, _, rest = normalise_chunk_key(key).partition("/")
        try:
            level, index = int(head), tuple(int(p) for p in rest.split("."))
        except ValueError:
            return False
        if not 0 <= level < len(self.level_shapes):
            return False
        shape = self.level_shapes[level]
        return (len(index) == len(shape)
                and not any(i < 0 or i * c >= s
                            for i, c, s in zip(index, self.chunks, shape)))

    def zarr_key(self, key: str) -> Optional[bytes]:
        if key in self._meta:
            return self._meta[key]
        key = normalise_chunk_key(key)
        head, _, rest = key.partition("/")
        try:
            level = int(head)
            index = tuple(int(p) for p in rest.split("."))
        except ValueError:
            return None
        if not 0 <= level < len(self.level_shapes):
            return None
        shape = self.level_shapes[level]
        if len(index) != len(shape):
            return None
        if any(i < 0 or i * c >= s for i, c, s in zip(index, self.chunks, shape)):
            return None
        return OUT_COMPRESSOR.encode(
            np.ascontiguousarray(self._compute_tile(level, index)))

    def thumbnail_array(self) -> np.ndarray:
        sample = self.base.thumbnail_array()
        if self._channel_index is not None:
            keep = [i for i in self._channel_index if i < len(sample)]
            sample = sample[keep] if keep else sample[:1]
        return sample

    def report(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.base.source,
            "location": "computed",
            "recipe": self.recipe,
            "recipe_label": "Derived view (computed per request from the raw view)",
            "derived_from": self.base.id,
            "transform_chain": self.transform_chain(),
            "axes": self.axes.upper(),
            "shape": self.shape,
            "levels": len(self.level_shapes),
            "level_shapes": self.level_shapes,
            "dtype": str(self.dtype),
            "chunks": self.chunks,
            "source_level_read": self._source_level,
            "source_level_shape": self._level_shape,
            "served_compression": "zlib",
            "build_reads": (
                "no file headers of its own — configured from the raw view; each "
                "chunk is computed on request from source pixels"),
            "index_copies_pixel_data": False,
            "direct_client_access": False,
            "computed_product": True,
            "source_chunk_cache": self._reader.stats(),
            "build_timings_s": {k: round(v, 3) for k, v in self.build_timings.items()},
            "metadata_mapping": self.mapping.as_dict(),
            "caption": {
                "access": (
                    "A DERIVED view: served lazily like the raw view, but its "
                    "pixels are COMPUTED per request rather than re-addressed. "
                    f"Transform chain: {'; '.join(self.transform_chain())}. The "
                    "original file is neither migrated nor rewritten; this "
                    "output is a computed product and is not zero-copy."),
                **self.mapping.as_sentences(),
            },
        }
