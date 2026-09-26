"""The two recipe families for serving an existing image file as OME-Zarr.

Both present the same interface — answer a Zarr key, describe yourself — so the
server does not care which one backs a dataset. They differ in one way that
matters and is not tunable:

  ``ReferenceView``  maps each Zarr chunk to a byte range in the original file.
                     Nothing is decoded to build it, and a client that can read
                     the source codec can bypass the server entirely.

  ``BioIOView``      has no byte-offset mapping; every read goes through the
                     reader, so the server is always in the data path and the
                     chunk size is whatever the reader reads at once.

Chunks are re-encoded to zlib on the way out because that is the one codec
every Zarr client has, including numcodecs.js in the browser. The source
codecs (LZW here) have no browser implementation.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numcodecs
import numpy as np

from .czi_meta import read_czi_extras
from .ngff import SourceMetadata, build_ngff_attrs, from_ome_xml

# Every Zarr client has zlib; blosc/zstd coverage in the browser is patchier.
OUT_COMPRESSOR = numcodecs.Zlib(level=1)


class ViewError(Exception):
    """The source could not be opened as a view by this recipe."""


def normalise_chunk_key(key: str) -> str:
    """Accept both chunk key spellings and return the dot-separated one.

    NGFF 0.4 asks for ``/`` between chunk indices (``0/1/2/0/0``) and
    ome-zarr-py requests exactly that, but tifffile's reference index and plain
    zarr default to ``.`` (``0/1.2.0.0``). Serving only one spelling makes the
    other client silently read fill_value instead of pixels, so accept both.
    """
    head, _, rest = key.partition("/")
    if not rest or rest.startswith("."):
        return key
    return f"{head}/{rest.replace('/', '.')}"


class BaseView:
    """Common surface: a read-only Zarr v2 group over a non-Zarr source."""

    recipe = "base"

    def __init__(self, dataset_id: str, source: str, title: Optional[str] = None):
        self.id = dataset_id
        self.source = source
        self.title = title or dataset_id
        self.build_timings: Dict[str, float] = {}

    def zarr_key(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    def contains(self, key: str) -> bool:
        """Does this key exist, WITHOUT fetching its bytes?

        zarr asks `key in store` before reading it. Answering that by doing the
        read and throwing it away doubles every chunk fetch on the build path —
        measured at exactly 2.0x. Recipes that can answer from their index must.
        """
        return self.zarr_key(key) is not None

    def report(self) -> Dict[str, Any]:
        raise NotImplementedError

    def thumbnail_array(self) -> np.ndarray:
        raise NotImplementedError

    def _apply_display_windows(self) -> None:
        """Set omero contrast limits from a sample of the coarsest level.

        The full dtype range is a useless default — 16-bit microscopy rarely
        uses more than a few percent of it, so a viewer opens on black. The
        source files carry no display settings, so this is measured from the
        pixels, and the mapping report says so rather than implying the file
        declared it.
        """
        channels = self.attrs.get("omero", {}).get("channels")
        if not channels:
            return
        t0 = time.perf_counter()
        try:
            # Bounded sample, NOT the thumbnail: on a single-scale source the
            # thumbnail path reads a whole plane per channel, which is the
            # entire file.
            level, n_chunks, affordable = _display_sample_plan(self)
            if not affordable:
                self.display_sample_note = (
                    f"NOT measured: full coverage of level {level} would need "
                    f"{n_chunks} chunk reads, over the budget of "
                    f"{FULL_COVERAGE_CHUNK_BUDGET}. A partial sample of a tissue "
                    "section is spatially biased at any pixel count, so no "
                    "window is published and the range shown is the declared "
                    "bit depth. Raise OMEZARR_VIEW_DISPLAY_SCAN_CHUNKS to measure it.")
                self.mapping.unmapped = [m for m in self.mapping.unmapped
                                         if not m.startswith("display windows")]
                self.mapping.miss("display windows (contrast limits)",
                                  self.display_sample_note)
                self.build_timings["sample_display_range_s"] = time.perf_counter() - t0
                return
            sample = _sample_for_display(self, level)
        except Exception:
            return
        for i, channel in enumerate(channels):
            if i >= len(sample):
                break
            plane = sample[i].astype(np.float64)
            lo, hi = (float(v) for v in np.percentile(plane, (1.0, 99.8)))
            if hi <= lo:
                lo, hi = float(plane.min()), float(plane.max())
            if hi > lo:
                channel["window"]["start"] = lo
                channel["window"]["end"] = hi
        # Open mid-stack: the first plane of a z-stack is usually empty.
        axes = [a["name"] for a in self.attrs["multiscales"][0]["axes"]]
        for axis, key in (("z", "defaultZ"), ("t", "defaultT")):
            if axis in axes:
                size = self.level_shapes[0][axes.index(axis)]
                self.attrs["omero"].setdefault("rdefs", {})[key] = size // 2
        # Say what was DONE without asserting anything about the source: the
        # builder already recorded whether the file declared windows, and
        # overwriting that line with a hardcoded "not declared" is how a false
        # claim survived even after the extractor learned to read them.
        declared = bool(self.mapping.mapped.get("declared_display_windows"))
        self.mapping.unmapped = [
            m for m in self.mapping.unmapped if not m.startswith("display windows")
        ]
        self.mapping.miss(
            "display windows (contrast limits)",
            ("the source declares them; this view publishes windows measured "
             "from the pixels instead (1-99.8 percentiles of a coarse level), "
             "and the declared values are recorded but not applied")
            if declared else
            ("not declared in the source; view computes 1-99.8 percentiles "
             "from a coarse level"))
        self.build_timings["sample_display_range_s"] = time.perf_counter() - t0
        self.display_sample_note = (
            f"every chunk of level {level} ({n_chunks} chunk reads, in "
            "parallel) — full spatial coverage, because a partial sample of a "
            "tissue section is biased at any pixel count")


# ---------------------------------------------------------------------------
# Recipe (b): reference index
# ---------------------------------------------------------------------------


class ReferenceView(BaseView):
    """Zero-copy view over a tiled TIFF, local or remote, via a byte-offset index."""

    recipe = "reference-index"

    def __init__(
        self,
        dataset_id: str,
        source: str,
        title: Optional[str] = None,
        series: int = 0,
        fetcher=None,
    ):
        super().__init__(dataset_id, source, title)
        import fsspec
        import tifffile

        self.series_index = series
        self._fetch = fetcher or _default_fetcher()
        self.remote = "://" in source

        t0 = time.perf_counter()
        if self.remote:
            proto = source.split("://")[0]
            handle = fsspec.filesystem(proto, block_size=2 << 20).open(source, "rb")
        else:
            handle = source
        try:
            tf = tifffile.TiffFile(handle)
        except Exception as e:
            raise ViewError(f"not readable as TIFF: {e}") from e
        self.build_timings["open_source_s"] = time.perf_counter() - t0

        # Series access walks the whole IFD chain — one IFD per page per level.
        # On a deep pyramid this dominates the build, not the index write.
        t0 = time.perf_counter()
        try:
            s = tf.series[series]
        except IndexError as e:
            raise ViewError(f"no series {series} in {source}") from e
        self.n_pages = sum(len(lv.pages) for lv in s.levels)
        self.build_timings["walk_ifd_chain_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        buf = io.StringIO()
        s.aszarr().write_fsspec(buf, url=source.rsplit("/", 1)[0])
        refs: Dict[str, Any] = json.loads(buf.getvalue())
        self.build_timings["build_index_s"] = time.perf_counter() - t0

        self.level_shapes = [list(lv.shape) for lv in s.levels]
        self.axes = s.axes.lower()
        self.dtype = np.dtype(s.dtype)
        self.n_series = len(tf.series)

        t0 = time.perf_counter()
        source_md = from_ome_xml(
            ome_xml=tf.ome_metadata,
            axes=s.axes,
            level_shapes=self.level_shapes,
            dtype=str(self.dtype.str),
            name=self.title,
        )
        # A multi-series file serves ONE series; silently picking the first and
        # saying nothing would hide the rest from the reader entirely.
        source_md.mark("scenes")
        source_md.scenes = [getattr(x, "name", None) or f"series {i}"
                            for i, x in enumerate(tf.series)]
        source_md.served_scene = source_md.scenes[series]
        attrs, self.mapping = build_ngff_attrs(source_md)
        self.build_timings["synthesise_metadata_s"] = time.perf_counter() - t0

        if ".zgroup" not in refs:
            # A single-level series is written as a bare array; wrap it so the
            # view always presents the same multiscale shape.
            refs = {(k if k.startswith(".") else f"0/{k}"): v for k, v in refs.items()}
            refs["0/.zarray"] = refs.pop(".zarray", refs.get("0/.zarray"))
            refs.pop(".zattrs", None)
            refs["0/.zattrs"] = json.dumps({"_ARRAY_DIMENSIONS": list(self.axes.upper())})
        refs[".zgroup"] = json.dumps({"zarr_format": 2})
        refs[".zattrs"] = json.dumps(attrs)

        # The served .zarray must advertise the codec we hand out, not the
        # source's — chunks are transcoded on the way through.
        self._source_codec: Dict[str, Optional[Dict[str, Any]]] = {}
        for lvl in range(len(self.level_shapes)):
            zk = f"{lvl}/.zarray"
            if zk not in refs:
                continue
            meta = json.loads(refs[zk])
            self._source_codec[str(lvl)] = meta.get("compressor")
            meta["compressor"] = OUT_COMPRESSOR.get_config()
            meta["dimension_separator"] = "/"
            refs[zk] = json.dumps(meta)
            if lvl == 0:
                self.chunks = meta["chunks"]

        self.refs = refs
        self.attrs = attrs
        self.compression = str(s.levels[0].keyframe.compression)
        tf.close()
        self._apply_display_windows()
        self.refs[".zattrs"] = json.dumps(self.attrs)

    def _decode_source_chunk(self, level: str, raw: bytes) -> bytes:
        config = self._source_codec.get(level)
        if not config:
            return raw
        import imagecodecs.numcodecs

        imagecodecs.numcodecs.register_codecs(verbose=False)
        decoded = numcodecs.get_codec(config).decode(raw)
        return decoded.tobytes() if hasattr(decoded, "tobytes") else bytes(decoded)

    def zarr_key(self, key: str) -> Optional[bytes]:
        entry = self.refs.get(normalise_chunk_key(key))
        if entry is None:
            return None
        if isinstance(entry, str):
            return entry.encode()
        url, offset, length = entry
        raw = self._fetch(url, offset, length)
        level = key.split("/", 1)[0]
        return OUT_COMPRESSOR.encode(self._decode_source_chunk(level, raw))

    def contains(self, key: str) -> bool:
        return normalise_chunk_key(key) in self.refs

    def raw_reference(self) -> Dict[str, Any]:
        """The index as the client would use it to bypass the server entirely.

        This restores the SOURCE codec in ``.zarray``, because a client reading
        the original file directly gets the original bytes, not our transcode.
        """
        refs = dict(self.refs)
        for lvl, config in self._source_codec.items():
            zk = f"{lvl}/.zarray"
            if zk in refs:
                meta = json.loads(refs[zk])
                meta["compressor"] = config
                meta.pop("dimension_separator", None)
                refs[zk] = json.dumps(meta)
        return refs

    def thumbnail_array(self) -> np.ndarray:
        # Coarsest level that is still big enough to look like something; the
        # very last level of a deep pyramid is often only ~150 px across.
        level = len(self.level_shapes) - 1
        while level > 0 and max(self.level_shapes[level][-2:]) < 256:
            level -= 1
        return _read_level(self, level)

    def report(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "location": "remote" if self.remote else "local",
            "recipe": self.recipe,
            "recipe_label": "Reference index (zero-copy byte ranges)",
            "axes": self.axes.upper(),
            "shape": self.level_shapes[0],
            "levels": len(self.level_shapes),
            "level_shapes": self.level_shapes,
            "dtype": str(self.dtype),
            "chunks": self.chunks,
            "source_compression": self.compression,
            "served_compression": "zlib (transcoded)",
            "source_pages_indexed": self.n_pages,
            "index_keys": len(self.refs),
            "build_reads": (
                "file headers only to build the index; then a bounded pixel "
                "sample to set display range — " + getattr(
                    self, "display_sample_note",
                    "a few small windows per channel, not a whole level")),
            "index_copies_pixel_data": False,
            "build_timings_s": {k: round(v, 3) for k, v in self.build_timings.items()},
            "direct_client_access": True,
            "metadata_mapping": self.mapping.as_dict(),
            "caption": {
                "access": (
                    "Served as a lazy OME-Zarr view over the original "
                    f"{self.compression.split('.')[-1].lower()}-compressed TIFF; "
                    "the file was not converted or copied. The view is a "
                    "byte-offset index, so a client holding it reads the "
                    "original file directly and the server is not in the data "
                    "path (zero-copy). Browsers cannot decode the source codec, "
                    "so the HTTP endpoint they use transcodes each chunk to "
                    "zlib and is not zero-copy."),
                **self.mapping.as_sentences(),
            },
        }


# ---------------------------------------------------------------------------
# Recipe (a): BioIO reader
# ---------------------------------------------------------------------------


class BioIOView(BaseView):
    """View over any BioIO-readable file. Local files only — see README."""

    recipe = "bioio"

    def __init__(self, dataset_id: str, source: str, title: Optional[str] = None):
        super().__init__(dataset_id, source, title)
        from bioio import BioImage

        if "://" in source:
            # bioio-czi rejects any non-local filesystem outright, and the other
            # plugins vary; fail here rather than deep inside a reader.
            raise ViewError(
                "the BioIO recipe reads local files only; fetch the file first"
            )
        t0 = time.perf_counter()
        try:
            self.img = BioImage(source)
        except Exception as e:
            raise ViewError(f"BioIO cannot read it: {e}") from e
        self.build_timings["open_source_s"] = time.perf_counter() - t0

        self.axes = "".join(self.img.dims.order).lower()
        self.shape = [int(n) for n in self.img.shape]
        self.dtype = np.dtype(self.img.dtype)
        # One YX plane per chunk: it is what the reader reads, so a smaller
        # tile would decode the whole plane and throw most of it away.
        self.chunks = [1] * (len(self.shape) - 2) + self.shape[-2:]
        self.level_shapes = [self.shape]

        t0 = time.perf_counter()
        self.attrs, self.mapping = build_ngff_attrs(self._source_metadata())
        self.build_timings["synthesise_metadata_s"] = time.perf_counter() - t0

        self._meta = {
            ".zgroup": json.dumps({"zarr_format": 2}).encode(),
            ".zattrs": json.dumps(self.attrs).encode(),
            "0/.zarray": json.dumps({
                "zarr_format": 2,
                "shape": self.shape,
                "chunks": self.chunks,
                "dtype": self.dtype.str,
                "compressor": OUT_COMPRESSOR.get_config(),
                "dimension_separator": "/",
                "fill_value": 0,
                "filters": None,
                "order": "C",
            }).encode(),
            "0/.zattrs": json.dumps(
                {"_ARRAY_DIMENSIONS": list(self.axes.upper())}
            ).encode(),
        }
        self._apply_display_windows()
        self._meta[".zattrs"] = json.dumps(self.attrs).encode()

    def _source_metadata(self) -> SourceMetadata:
        pps = self.img.physical_pixel_sizes
        sizes: Dict[str, Tuple[float, Optional[str]]] = {}
        for axis, value in (("x", pps.X), ("y", pps.Y), ("z", pps.Z)):
            if value is not None:
                # BioIO normalises physical sizes to micrometres across reader
                # plugins; the source unit string is not preserved.
                sizes[axis] = (float(value), "µm")
        names: Optional[List[Optional[str]]] = None
        if self.img.channel_names:
            names = [str(n) for n in self.img.channel_names]

        md = SourceMetadata(
            axes=self.axes,
            level_shapes=[list(self.shape)],
            dtype=self.dtype.str,
            name=self.title,
            physical_sizes=sizes,
            channel_names=names,
        )
        md.mark("physical pixel sizes", "channel names")

        # BioIO's accessors stop at names and pixel sizes. Anything beyond them
        # has to be read from the raw metadata, or the view must say it did not
        # look — claiming "not declared" for a field nobody inspected is how
        # three false statements got into published captions.
        try:
            extras, inspected = read_czi_extras(self.img.metadata)
        except Exception:
            extras, inspected = {}, []
        md.mark(*inspected)
        for attr, value in extras.items():
            setattr(md, attr, value)

        # BioIO's scene list is what is actually REACHABLE through this view, so
        # it wins over the names in the raw metadata, which may be spelled
        # differently and would otherwise disagree with served_scene.
        scenes = [str(x) for x in (self.img.scenes or [])]
        if scenes:
            md.mark("scenes")
            md.scenes = scenes
            md.served_scene = str(self.img.current_scene)
        return md

    def _plane(self, index: Tuple[int, ...]) -> np.ndarray:
        sel = tuple(slice(i, i + 1) for i in index[:-2])
        return np.ascontiguousarray(self.img.dask_data[sel].compute())

    def _chunk_index(self, key: str) -> Optional[Tuple[int, ...]]:
        key = normalise_chunk_key(key)
        if not key.startswith("0/"):
            return None
        try:
            index = tuple(int(p) for p in key[2:].split("."))
        except ValueError:
            return None
        if len(index) != len(self.shape):
            return None
        if any(i < 0 or i * c >= s for i, c, s in zip(index, self.chunks, self.shape)):
            return None
        return index

    def contains(self, key: str) -> bool:
        return key in self._meta or self._chunk_index(key) is not None

    def zarr_key(self, key: str) -> Optional[bytes]:
        if key in self._meta:
            return self._meta[key]
        index = self._chunk_index(key)
        if index is None:
            return None
        return OUT_COMPRESSOR.encode(self._plane(index))

    def thumbnail_array(self) -> np.ndarray:
        return _read_level(self, 0)

    def report(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "location": "local",
            "recipe": self.recipe,
            "recipe_label": "BioIO reader (server in the data path)",
            "axes": self.axes.upper(),
            "shape": self.shape,
            "levels": 1,
            "level_shapes": [self.shape],
            "dtype": str(self.dtype),
            "chunks": self.chunks,
            "source_compression": "handled by the reader",
            "served_compression": "zlib",
            "pixel_bytes_copied": 0,
            "build_timings_s": {k: round(v, 3) for k, v in self.build_timings.items()},
            "direct_client_access": False,
            "chunk_semantics": "one YX plane per chunk (the reader's own granularity)",
            "metadata_mapping": self.mapping.as_dict(),
            "caption": {
                "access": (
                    "Served as a lazy OME-Zarr view read through BioIO; the "
                    "file was not converted or copied. This format has no "
                    "chunk layout that can be addressed as byte ranges, so "
                    "there is no zero-copy tier: every read goes through the "
                    "server, one image plane at a time."),
                **self.mapping.as_sentences(),
            },
        }


# ---------------------------------------------------------------------------


def _default_fetcher(attempts: int = 5, max_inflight: int = 8):
    """Range-read bytes from a URL or a local path.

    Two things this has to survive, both learned the hard way against Google
    Cloud Storage. Object stores drop pooled connections freely, so a dropped
    connection must be retried rather than surfaced as a 500 that renders a
    black tile. And HTTP/2 multiplexes every range read onto ONE connection,
    so a single reset takes out all of them at once; under a viewer's or a
    training loader's parallel reads that happened often enough to fail whole
    requests. Pooled HTTP/1.1 with bounded concurrency isolates the failures
    to one read, which the retry then absorbs.
    """
    import httpx

    def _new() -> "httpx.Client":
        return httpx.Client(
            http2=False,
            timeout=120,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=max_inflight * 2,
                                max_keepalive_connections=max_inflight),
        )

    state = {"client": _new()}
    lock = threading.Lock()
    inflight = threading.Semaphore(max_inflight)

    def _reset(dead) -> None:
        """Swap in a fresh client. Deliberately does NOT close the old one.

        httpx.Client is safe to share across threads, but closing one is not:
        a sibling thread mid-request on the same client gets "Cannot send a
        request, as the client has been closed". Dropping the reference lets
        it be collected once its in-flight requests finish.
        """
        with lock:
            if state["client"] is dead:
                state["client"] = _new()

    def fetch(url: str, offset: int, length: int) -> bytes:
        if "://" not in url or url.startswith("file://"):
            path = url[7:] if url.startswith("file://") else url
            with open(path, "rb") as fh:
                fh.seek(offset)
                return fh.read(length)
        headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
        last: Optional[Exception] = None
        for attempt in range(attempts):
            client = state["client"]
            try:
                with inflight:
                    r = client.get(url, headers=headers)
                r.raise_for_status()
                return r.content
            except (httpx.TransportError, httpx.HTTPStatusError, RuntimeError) as e:
                last = e
                if isinstance(e, httpx.HTTPStatusError) and \
                        e.response.status_code not in (429, 500, 502, 503, 504):
                    raise
                if isinstance(e, RuntimeError) and "closed" not in str(e):
                    raise
                _reset(client)
                time.sleep(0.25 * (2 ** attempt))
        raise RuntimeError(f"range read failed after {attempts} attempts: {last}")

    return fetch


# A display-range sample must never scale with the file. On a SINGLE-SCALE
# source the "coarsest level" IS level 0, so reading a whole plane per channel
# read the entire image: 529 s of a 634 s first request on a 116 MB file, and
# an eleven-minute first click for whoever opens a cold view. The sample only
# needs enough pixels for a percentile, so it is bounded by construction.
# An UNBIASED percentile needs spatial COVERAGE, not pixel count. An
# independent sweep found a 2-chunk diagonal sample (131,072 px) was 33x less
# accurate than striding across every chunk (82,628 px) — fewer pixels, more
# coverage, better answer — because intensity in a tissue section is structured
# in space, so any partial sample stays biased however many pixels it draws.
# So: read a level in FULL when that is affordable, and when it is not, say the
# window is unmeasured rather than publish one that is wrong on most channels.
FULL_COVERAGE_CHUNK_BUDGET = int(os.environ.get(
    "OMEZARR_VIEW_DISPLAY_SCAN_CHUNKS", "512"))
SAMPLE_FETCH_WORKERS = 12


def _level_array(view: BaseView, level: int):
    import zarr

    store = _ViewStore(view)
    return zarr.open_group(zarr.storage.KVStore(store), mode="r")[str(level)]


def _read_level(view: BaseView, level: int, max_side: int = 4096) -> np.ndarray:
    """Read the middle plane of each channel at ``level``, as (C, Y, X).

    Bounded: past ``max_side`` this returns a centred crop rather than the whole
    plane, because the caller wants something to look at, not the whole file.
    """
    arr = _level_array(view, level)
    axes = view.axes
    shape = arr.shape
    yi, xi = len(axes) - 2, len(axes) - 1
    sel: List[Any] = []
    for i, a in enumerate(axes):
        if i in (yi, xi):
            extent = shape[i]
            if extent > max_side:
                start = (extent - max_side) // 2
                sel.append(slice(start, start + max_side))
            else:
                sel.append(slice(None))
        elif a == "c":
            sel.append(slice(None))
        else:
            sel.append(shape[i] // 2)
    out = np.asarray(arr[tuple(sel)])
    if "c" not in axes:
        out = out[None]
    return out


def _display_sample_plan(view: "BaseView"):
    """Pick a level to sample and say whether full coverage is affordable."""
    level = len(view.level_shapes) - 1
    while level > 0 and max(view.level_shapes[level][-2:]) < 256:
        level -= 1
    arr = _level_array(view, level)
    axes = view.axes
    yi, xi = len(axes) - 2, len(axes) - 1
    grid_y = max(1, -(-arr.shape[yi] // arr.chunks[yi]))
    grid_x = max(1, -(-arr.shape[xi] // arr.chunks[xi]))
    n_c = arr.shape[axes.index("c")] if "c" in axes else 1
    total = grid_y * grid_x * n_c
    return level, total, total <= FULL_COVERAGE_CHUNK_BUDGET


def _sample_for_display(view: "BaseView", level: int) -> np.ndarray:
    """Read EVERY chunk of one plane per channel at ``level``, in parallel.

    Full spatial coverage is what makes the percentile unbiased. Cost is bounded
    by CHOOSING the level, never by skipping parts of it.
    """
    from concurrent.futures import ThreadPoolExecutor

    arr = _level_array(view, level)
    axes = view.axes
    shape, chunks = arr.shape, arr.chunks
    yi, xi = len(axes) - 2, len(axes) - 1
    n_c = shape[axes.index("c")] if "c" in axes else 1
    grid_y = max(1, -(-shape[yi] // chunks[yi]))
    grid_x = max(1, -(-shape[xi] // chunks[xi]))

    jobs = []
    for c in range(n_c):
        for gy in range(grid_y):
            for gx in range(grid_x):
                sel = []
                for i, a in enumerate(axes):
                    if i == yi:
                        sel.append(slice(gy * chunks[yi],
                                         min((gy + 1) * chunks[yi], shape[yi])))
                    elif i == xi:
                        sel.append(slice(gx * chunks[xi],
                                         min((gx + 1) * chunks[xi], shape[xi])))
                    elif a == "c":
                        sel.append(c)
                    else:
                        sel.append(shape[i] // 2)
                jobs.append((c, tuple(sel)))

    def fetch(job):
        c, sel = job
        return c, np.asarray(arr[sel]).reshape(-1)

    per_channel = {c: [] for c in range(n_c)}
    with ThreadPoolExecutor(max_workers=SAMPLE_FETCH_WORKERS) as pool:
        for c, values in pool.map(fetch, jobs):
            per_channel[c].append(values)
    return np.stack([np.concatenate(per_channel[c]) for c in range(n_c)])


class _ViewStore(dict):
    """Minimal Mapping adaptor so zarr can read a view in-process."""

    def __init__(self, view: BaseView):
        super().__init__()
        self._view = view

    def __getitem__(self, key: str) -> bytes:
        value = self._view.zarr_key(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: object) -> bool:
        # Ask, do not fetch: reading the chunk to answer an existence question
        # made every build-path chunk arrive twice.
        return isinstance(key, str) and self._view.contains(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


def open_view(dataset_id: str, source: str, title: Optional[str] = None,
              recipe: str = "auto") -> BaseView:
    """Open ``source`` with the best available recipe.

    ``auto`` prefers the reference index (zero-copy, and it works remotely),
    and falls back to BioIO for formats or layouts it cannot index.
    """
    errors = []
    order = {
        "auto": [ReferenceView, BioIOView],
        "reference": [ReferenceView],
        "bioio": [BioIOView],
    }[recipe]
    for cls in order:
        try:
            return cls(dataset_id, source, title)
        except ViewError as e:
            errors.append(f"{cls.recipe}: {e}")
        except Exception as e:  # a reader raising something of its own
            errors.append(f"{cls.recipe}: {type(e).__name__}: {e}")
    raise ViewError(f"no recipe could open {source} ({'; '.join(errors)})")
