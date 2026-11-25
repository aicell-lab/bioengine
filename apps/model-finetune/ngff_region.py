"""Read one region of a remote OME-Zarr image, without fetching the rest of it.

Training used to receive whole images: ``_load_image_from_source`` GETs a URL and
hands the entire body to ``imread``. For a whole-slide source that is the only
thing standing between a 1 GB read and a 4096x4096 crop the annotator actually
drew on, so this module exists to read the crop and nothing else.

WHAT IT DOES NOT DO: parse OME-NGFF. The dataset paths and the axis roles come
from the broker's ``probe_ngff_source``, which already parses ``multiscales`` for
v0.4, v0.5 and the pre-v0.4 implied order. Two implementations of that parsing
already have to agree -- ``broker_core.parse_ngff_multiscales`` and
``ngffLayout.ts``, which it says it mirrors -- and a third here is how three
copies drift. This module consumes the interpretation; it never forms one.

THE SPLIT IN HERE IS DELIBERATE. Everything above ``read_region`` is pure: given
the probe's numbers it computes an index expression, and it is tested directly.
``read_region`` is the only part that touches a store, and it takes its opener as
an argument so the arithmetic can be tested without zarr installed. That matters
concretely: CI installs ``requirements-test.txt`` and the datasets requirements,
NOT ``requirements-entry.txt``, so a module-level ``import zarr`` here would turn
this file's tests into collection ERRORS in CI rather than failures -- the shape
that once let three skipped cases pass for a guard.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

NGFF_SOURCE_KIND = "ngff"


class RegionSourceError(ValueError):
    """A region reference that cannot be read as written."""


def is_region_source(source: Any) -> bool:
    """True for a reference this module can read.

    Checked by SHAPE, not by trying to read it: a caller passing a plain URL or
    an array must fall through to the existing file path untouched.
    """
    return (
        isinstance(source, dict)
        and str(source.get("kind") or "") == NGFF_SOURCE_KIND
        and bool(str(source.get("store_root") or "").strip())
    )


def parse_region_source(source: Any) -> Dict[str, Any]:
    """Normalise a region reference, or raise with the reason.

    Accepts the shape the broker already persists for a remote image -- the
    ``source`` sub-document of ``remote_image_entry``: ``kind``, ``store_root``,
    ``multiscale_path`` and ``region``. Reading the same record the viewer reads
    is the point; a second description of a crop is a second thing to keep true.
    """
    if not is_region_source(source):
        raise RegionSourceError(
            "Not an OME-Zarr region reference: expected a mapping with "
            f"kind={NGFF_SOURCE_KIND!r} and a non-empty store_root."
        )
    region = source.get("region")
    if region is not None:
        if not isinstance(region, dict):
            raise RegionSourceError("region must be a mapping or omitted.")
        missing = [k for k in ("x", "y", "width", "height") if k not in region]
        if missing:
            raise RegionSourceError(
                f"region is missing {', '.join(missing)}; it needs x, y, width "
                f"and height in full-resolution pixels."
            )
        region = {k: int(region[k]) for k in ("x", "y", "width", "height")}
        if region["width"] <= 0 or region["height"] <= 0:
            raise RegionSourceError(
                f"region has non-positive extent "
                f"({region['width']}x{region['height']})."
            )
        if region["x"] < 0 or region["y"] < 0:
            raise RegionSourceError("region origin is negative.")
    return {
        "kind": NGFF_SOURCE_KIND,
        "store_root": str(source["store_root"]).strip(),
        "multiscale_path": source.get("multiscale_path") or None,
        # None means the whole image, which is how the broker records it: its
        # normalise_region returns None when the box covers the store.
        "region": region,
    }


def region_index(
    axis_roles: Dict[str, Any],
    rank: int,
    shape: Sequence[int],
    region: Optional[Dict[str, int]],
) -> Tuple[Any, ...]:
    """The index expression that selects *region* from a level-0 array.

    ``axis_roles`` is the broker's ``{"y", "x", "c"}`` axis indices. Every axis
    that is not y, x or c is reduced to its FIRST element rather than kept: a
    5D tczyx store would otherwise come back with singleton t and z dimensions
    that neither trainer accepts. The channel axis is kept whole, because
    dropping it would silently turn an RGB crop into one band.
    """
    y, x = axis_roles.get("y"), axis_roles.get("x")
    c = axis_roles.get("c")
    if not isinstance(y, int) or not isinstance(x, int):
        raise RegionSourceError(f"axis_roles has no usable y/x: {axis_roles!r}")
    if rank != len(shape):
        raise RegionSourceError(
            f"rank {rank} does not match shape {list(shape)!r}; the probe and "
            f"the array disagree, so neither is trusted."
        )
    for name, idx in (("y", y), ("x", x)):
        if not (0 <= idx < rank):
            raise RegionSourceError(
                f"axis_roles[{name!r}]={idx} is outside an array of rank {rank}."
            )

    if region is None:
        y_sel: Any = slice(None)
        x_sel: Any = slice(None)
    else:
        h, w = int(shape[y]), int(shape[x])
        y0, x0 = region["y"], region["x"]
        y1, x1 = y0 + region["height"], x0 + region["width"]
        # REFUSED rather than clamped. A crop silently shrunk at the edge
        # trains on a different area than the annotator drew on, and the label
        # mask is rasterised to the region's declared width/height, so a
        # clamped image would no longer be the same size as its labels.
        if y1 > h or x1 > w:
            raise RegionSourceError(
                f"region x={x0},y={y0},{region['width']}x{region['height']} "
                f"runs past the level-0 array ({w}x{h}). It is refused rather "
                f"than clamped: a shrunk crop would not match its rasterised "
                f"labels."
            )
        y_sel, x_sel = slice(y0, y1), slice(x0, x1)

    out: List[Any] = []
    for axis in range(rank):
        if axis == y:
            out.append(y_sel)
        elif axis == x:
            out.append(x_sel)
        elif isinstance(c, int) and axis == c:
            out.append(slice(None))
        else:
            out.append(0)
    return tuple(out)


def to_hw_or_hwc(array: Any, axis_roles: Dict[str, Any], rank: int) -> Any:
    """Move a kept channel axis last, so the result is HxW or HxWxC.

    Both trainers read HxW or HxWxC. ``region_index`` drops every axis except
    y, x and c, so what comes back is already 2D or 3D -- but in the store's
    axis order, which for a tczyx store puts channels FIRST. Transposing here
    rather than in the caller keeps the store's layout from leaking outward.
    """
    import numpy as np

    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr
    c = axis_roles.get("c")
    if arr.ndim != 3 or not isinstance(c, int):
        return arr
    y, x = int(axis_roles["y"]), int(axis_roles["x"])
    # Among the three surviving axes, their relative order is preserved.
    order = sorted([("c", c), ("y", y), ("x", x)], key=lambda t: t[1])
    names = [n for n, _ in order]
    want = ["y", "x", "c"]
    if names == want:
        return arr
    return np.transpose(arr, [names.index(n) for n in want])


def level0_path(dataset_paths: Sequence[str]) -> str:
    """The full-resolution level.

    OME-NGFF orders ``datasets`` coarsest-last, so level 0 is the first entry.
    Taken from the probe rather than guessed as ``"0"``: a store is free to name
    its levels anything, and several in the wild use ``"s0"``.
    """
    if not dataset_paths:
        raise RegionSourceError("the probe returned no dataset paths.")
    return str(dataset_paths[0]).strip("/")


def read_region(
    source: Any,
    probe: Dict[str, Any],
    opener: Any = None,
) -> Any:
    """Read *source*'s region out of its store and return HxW or HxWxC.

    *probe* is the broker's ``probe_ngff_source`` result; this reads
    ``dataset_paths``, ``axis_roles`` and ``level0_shape`` from it and parses
    nothing itself.

    *opener* takes ``(store_root, array_path)`` and returns something
    numpy-sliceable. It defaults to a zarr-backed opener, imported HERE rather
    than at module scope so that the pure functions above can be tested in an
    environment without zarr.
    """
    ref = parse_region_source(source)
    for key in ("dataset_paths", "axis_roles", "level0_shape"):
        if key not in probe:
            raise RegionSourceError(
                f"the probe is missing {key!r}; this needs a probe_ngff_source "
                f"result from a broker that returns the store layout."
            )
    shape = list(probe["level0_shape"])
    index = region_index(probe["axis_roles"], len(shape), shape, ref["region"])

    if opener is None:
        opener = _zarr_opener
    array = opener(ref["store_root"], level0_path(probe["dataset_paths"]))
    return to_hw_or_hwc(array[index], probe["axis_roles"], len(shape))


def _zarr_opener(store_root: str, array_path: str) -> Any:
    """Open one array inside a remote Zarr group, read-only.

    Imported lazily: see this module's docstring on why zarr must not be a
    module-level import here.
    """
    import zarr

    root = store_root if store_root.endswith("/") else store_root + "/"
    return zarr.open_array(root + array_path, mode="r")
