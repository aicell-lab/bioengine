"""Read one region of a remote OME-Zarr source, without fetching the rest of it.

Why training needs this. A whole-slide source is added to a dataset as a PATCH:
the annotation broker records ``{store_root, array_path, axes, region}`` and
copies no pixels, because a 100000x80000 slide can never be materialised into the
artifact — the browser reads level 0 whole to encode an OME-TIFF and would
exhaust the tab long before it produced a file. Such a stem was annotatable and
not trainable: training resolved every image by fetching a URL whole, and the
broker refused to mint one for an image that had no file behind it.

So the crop is read here instead, out of the source, over HTTP range requests:
only the chunks the region overlaps are fetched, which is what makes the cost a
function of the REGION rather than of the slide. ``HttpZarrStore``
(``bioengine.datasets.http_zarr_store``) does the transport and shares one LRU
chunk cache across every open store in the process.

**The read plan comes from the broker, deliberately.** Selecting the multiscale,
resolving the pyramid level and normalising the axes already exist, once, in
``annotation-broker``'s ``broker_core`` (and once more in the frontend's
``ngffLayout.ts``). A third copy here would be a third thing to keep in step, so
this module takes ``array_path`` and ``axes`` as given and refuses a source that
does not carry them, naming what is missing.

The axis-role fallback below is the one piece that is repeated, because it
depends on the array's rank, which is only known once the array is open: it
mirrors ``broker_core.ngff_axis_roles``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Decoded bytes one region read may cost. The entry actor is declared with 4 GiB
# and pinned to one replica, and a training set resolves several images before
# any of them is written out, so an unbounded read would take the whole app down
# rather than fail its own call.
#
# The floor on this number is the broker's MAX_MATERIALISE_BYTES (64 MiB): a
# region small enough for the annotator to materialise must always be trainable,
# or the two limits would disagree about the same patch. 128 MiB clears that with
# room to spare and is still 1/32 of the actor.
#
# This bounds the REGION, never the source. A whole-slide store stays readable —
# taking a patch out of one is the entire point.
MAX_REGION_BYTES = 128 * 1024 * 1024

_ITEMSIZE = {
    "|b1": 1,
    "|i1": 1,
    "|u1": 1,
    "<i2": 2,
    "<u2": 2,
    "<f2": 2,
    "<i4": 4,
    "<u4": 4,
    "<f4": 4,
    "<i8": 8,
    "<u8": 8,
    "<f8": 8,
}


class NgffRegionError(ValueError):
    """A region read that cannot be performed, with the reason a caller can act on."""


def axis_roles(axes: Sequence[str], rank: int) -> Dict[str, Optional[int]]:
    """``{"y", "x", "c"}`` axis indices for an array of *rank* dimensions.

    Mirrors ``broker_core.ngff_axis_roles``, including its fallback: when the
    declared axes cannot be trusted (wrong length, or no y/x) Y and X are taken
    to be the two fastest-varying dimensions, which holds for every image layout
    in practice. Repeated here rather than imported because the rank is only
    known once the array is open, and because the two apps ship separately.
    """
    names = [str(a).lower() for a in axes]
    y = names.index("y") if "y" in names else -1
    x = names.index("x") if "x" in names else -1
    c = names.index("c") if "c" in names else -1
    if not (len(names) == rank and y >= 0 and x >= 0):
        if rank < 2:
            raise NgffRegionError(
                f"OME-Zarr array has rank {rank}; a region needs at least 2 "
                f"dimensions."
            )
        y, x, c = rank - 2, rank - 1, -1
    return {"y": y, "x": x, "c": c if c >= 0 else None}


def region_selection(
    axes: Sequence[str], shape: Sequence[int], region: Optional[Dict[str, int]]
) -> Tuple[Any, ...]:
    """The index tuple that reads *region* out of an array of *shape*.

    Y and X become slices, a channel axis is kept whole, and every other axis
    (t, z, or anything unrecognised) is taken at index 0 — one plane, which is
    what a 2D annotation was drawn on. ``None`` for *region* reads the whole of
    the Y/X plane, which is how a whole-image import is recorded (the broker
    normalises a region covering the store entirely to ``None``, so the two
    cases produce one manifest entry rather than two).

    The far edge is clamped to the array, and an origin outside it is refused —
    the same asymmetry ``broker_core.normalise_region`` applies, for the same
    reason: a picker on a scaled preview rounds the far edge past the end
    routinely, while an origin outside the store means the caller has the wrong
    coordinate system and must not be repaired into a valid-looking patch
    somewhere else.
    """
    roles = axis_roles(axes, len(shape))
    height, width = int(shape[roles["y"]]), int(shape[roles["x"]])

    if region is None:
        y0, x0, y1, x1 = 0, 0, height, width
    else:
        try:
            x0, y0 = int(region["x"]), int(region["y"])
            rw, rh = int(region["width"]), int(region["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise NgffRegionError(
                "A region needs integer x, y, width and height, in pixels of "
                "the level being read."
            ) from exc
        if rw <= 0 or rh <= 0:
            raise NgffRegionError(f"Region is empty: {rw}x{rh} pixels.")
        if x0 < 0 or y0 < 0:
            raise NgffRegionError(f"Region starts outside the image at ({x0}, {y0}).")
        if x0 >= width or y0 >= height:
            raise NgffRegionError(
                f"Region starts at ({x0}, {y0}), outside the {width}x{height} array."
            )
        x1, y1 = min(x0 + rw, width), min(y0 + rh, height)

    selection: List[Any] = [0] * len(shape)
    selection[roles["y"]] = slice(y0, y1)
    selection[roles["x"]] = slice(x0, x1)
    if roles["c"] is not None:
        selection[roles["c"]] = slice(None)
    return tuple(selection)


def selection_bytes(selection: Sequence[Any], shape: Sequence[int], dtype: Any) -> int:
    """Decoded size of what *selection* reads, in bytes.

    Computed from the selection rather than from the region the caller asked
    for, so the clamping above and the channel count are both already in it.
    An unknown dtype counts as 8 bytes per sample — the largest this reader
    supports — because a guess that under-counts turns the cap into no cap.
    """
    itemsize = _ITEMSIZE.get(_normalise_dtype(dtype), 8)
    samples = 1
    for sel, size in zip(selection, shape):
        if isinstance(sel, slice):
            start, stop, _ = sel.indices(int(size))
            samples *= max(0, stop - start)
    return samples * itemsize


def _normalise_dtype(dtype: Any) -> str:
    """A zarr dtype in either spelling as its v2 string, e.g. ``"<u2"``."""
    text = getattr(dtype, "str", None) or str(dtype)
    text = str(text)
    if text in _ITEMSIZE:
        return text
    # numpy's repr form ("uint16"), and the v3 names zarr 3 reports.
    named = {
        "bool": "|b1",
        "int8": "|i1",
        "uint8": "|u1",
        "int16": "<i2",
        "uint16": "<u2",
        "float16": "<f2",
        "int32": "<i4",
        "uint32": "<u4",
        "float32": "<f4",
        "int64": "<i8",
        "uint64": "<u8",
        "float64": "<f8",
    }
    return named.get(text, text)


def read_plan(source: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a training source and return what the reader needs from it.

    Refuses rather than guesses: a source with no ``array_path`` or no ``axes``
    came from something other than the broker's training payload, and inferring
    them here would mean a fourth copy of the multiscale parsing.
    """
    if not isinstance(source, dict):
        raise NgffRegionError("An NGFF training source must be an object.")
    inner = source.get("ngff") if isinstance(source.get("ngff"), dict) else source
    store_root = inner.get("store_root")
    if not isinstance(store_root, str) or not store_root:
        raise NgffRegionError(
            "An NGFF training source needs 'store_root', the URL of the "
            "OME-Zarr group holding the pyramid."
        )
    array_path = inner.get("array_path")
    if not isinstance(array_path, str) or not array_path:
        raise NgffRegionError(
            "An NGFF training source needs 'array_path', the pyramid level to "
            "read (the broker's get_training_urls returns it). Selecting the "
            "level from the store's own metadata is the broker's job, not this "
            "app's."
        )
    axes = inner.get("axes")
    if not isinstance(axes, (list, tuple)) or not axes:
        raise NgffRegionError(
            "An NGFF training source needs 'axes', the axis names of that "
            "level (the broker's get_training_urls returns them)."
        )
    region = inner.get("region")
    if region is not None and not isinstance(region, dict):
        raise NgffRegionError("'region' must be an object with x, y, width, height.")
    return {
        "store_root": store_root.rstrip("/"),
        "array_path": array_path,
        "axes": [str(a) for a in axes],
        "region": region,
        "token": inner.get("token") or None,
    }


def read_region(source: Dict[str, Any], max_bytes: int = MAX_REGION_BYTES):
    """The region as an array, read out of the remote store.

    Blocking: run it in a thread. Returns whatever the array's own dtype is,
    with the axes still in the store's order — the caller reshapes, because
    ``training._to_hwc_or_hw`` already owns that decision for every other
    source.
    """
    import zarr

    from bioengine.datasets.http_zarr_store import HttpZarrStore

    plan = read_plan(source)
    store = HttpZarrStore(base_url=plan["store_root"], token=plan["token"])
    try:
        group = zarr.open_group(store=store, mode="r")
        try:
            array = group[plan["array_path"]]
        except KeyError as exc:
            raise NgffRegionError(
                f"The OME-Zarr group at {plan['store_root']} has no array "
                f"{plan['array_path']!r}, which is the level the source names."
            ) from exc

        selection = region_selection(plan["axes"], array.shape, plan["region"])
        size = selection_bytes(selection, array.shape, array.dtype)
        if size > max_bytes:
            raise NgffRegionError(
                f"This region decodes to {size / (1 << 20):.0f} MiB, over the "
                f"{max_bytes / (1 << 20):.0f} MiB a single read may cost. Import "
                f"a smaller patch of the source and annotate that."
            )
        return array[selection]
    finally:
        # The store opens its own HTTP client, and one is created per read.
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - a leaked socket must not fail a read
                pass
