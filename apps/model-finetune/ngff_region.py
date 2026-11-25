"""Which chunks of an OME-Zarr array a region touches, and how to assemble them.

Training on a region annotation must read the region and nothing else. Today the
entry fetches each training image whole through a presigned URL and hands the buffer
to ``imageio.imread``, which caps what can be trained on at whatever fits in memory
and in a materialisation step — a whole-slide image fits in neither. It is also the
app's second most common production failure: a materialised temp object expires
between annotation and training.

A chunked store does not need any of that. A region of a 100000x100000 slide covers a
handful of chunks, and a chunked store addresses them individually, so the read is
proportional to the region rather than to the image.

**This module is the arithmetic only, and that is deliberate.** It computes which
chunks a region intersects and how they reassemble, taking the fetch as a callable.
So the property that matters for a whole-slide image — that the read touches a small
bounded number of chunks and never the whole array — is testable here with no store,
no network, no Zarr library and no GPU. ``read_region`` is a thin assembler over that
plan; whoever supplies ``fetch`` owns decompression.

Axes follow OME-Zarr (NGFF v0.4/v0.5): the last two are ``y`` then ``x``, and any
leading axes (``t``, ``c``, ``z``) are pinned to one index each, because a training
pair is a single 2D plane.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class RegionOutOfBounds(ValueError):
    """The requested region is not inside the array.

    Its own class because the caller's response differs from a malformed request: a
    region outside the image means the annotation and the store disagree, which is
    worth reporting to a user rather than clamping silently. Clamping would train on
    a different area than was annotated.
    """


def _spatial_axes(axes: Sequence[str]) -> Tuple[int, int]:
    """Indices of the y and x axes. NGFF puts them last, in that order."""
    if len(axes) < 2:
        raise ValueError(f"an image needs at least a y and an x axis, got {list(axes)}")
    names = [str(a).lower() for a in axes]
    if names[-2:] != ["y", "x"]:
        raise ValueError(
            f"expected the last two axes to be y then x, got {names[-2:]}; "
            "a transposed store would silently read the wrong area"
        )
    return len(axes) - 2, len(axes) - 1


def chunk_span(start: int, length: int, chunk: int) -> range:
    """The chunk indices covering ``[start, start+length)`` along one axis."""
    if chunk <= 0:
        raise ValueError(f"chunk size must be positive, got {chunk}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    first = start // chunk
    last = (start + length - 1) // chunk
    return range(first, last + 1)


def plan_region_read(
    shape: Sequence[int],
    chunks: Sequence[int],
    axes: Sequence[str],
    region: Dict[str, int],
    index: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Plan a region read without performing it.

    Returns ``{"height", "width", "chunks": [...], "chunks_touched", "axes"}`` where
    each entry of ``chunks`` is::

        {"key": (i0, i1, ...),          # chunk index per axis, NGFF key order
         "src": ((y0, y1), (x0, x1)),   # the part of that chunk the region needs
         "dst": ((y0, y1), (x0, x1))}   # where it lands in the output

    The plan is the unit of verification: a caller can assert on
    ``chunks_touched`` before fetching anything, which is how "this read is
    proportional to the region" becomes a check rather than a hope.
    """
    if len(shape) != len(chunks) or len(shape) != len(axes):
        raise ValueError(
            f"shape {list(shape)}, chunks {list(chunks)} and axes {list(axes)} "
            "must describe the same number of dimensions"
        )
    ay, ax = _spatial_axes(axes)
    for key in ("x", "y", "width", "height"):
        if key not in region:
            raise ValueError(f"a region needs x, y, width and height; missing {key!r}")
    x0, y0 = int(region["x"]), int(region["y"])
    w, h = int(region["width"]), int(region["height"])
    if w <= 0 or h <= 0:
        raise ValueError(f"a region needs a positive size, got {w}x{h}")
    if x0 < 0 or y0 < 0 or x0 + w > shape[ax] or y0 + h > shape[ay]:
        raise RegionOutOfBounds(
            f"region x={x0} y={y0} {w}x{h} is not inside an image of "
            f"{shape[ax]}x{shape[ay]} (width x height)"
        )

    index = dict(index or {})
    # Every non-spatial axis must be pinned: a training pair is one 2D plane, and
    # defaulting a missing one to 0 would quietly train on the first channel of a
    # multi-channel store without saying so.
    pinned: List[int] = []
    for i, name in enumerate(axes):
        if i in (ay, ax):
            pinned.append(-1)
            continue
        n = str(name).lower()
        if n not in index:
            raise ValueError(
                f"axis {n!r} has extent {shape[i]} and must be pinned to an index; "
                f"pass index={{{n!r}: 0}} rather than letting it default"
            )
        v = int(index[n])
        if not 0 <= v < shape[i]:
            raise RegionOutOfBounds(
                f"index {n}={v} is outside its extent {shape[i]}"
            )
        pinned.append(v)

    out: List[Dict[str, Any]] = []
    for cy in chunk_span(y0, h, chunks[ay]):
        cy_lo = cy * chunks[ay]
        sy0 = max(y0, cy_lo) - cy_lo
        sy1 = min(y0 + h, cy_lo + chunks[ay]) - cy_lo
        for cx in chunk_span(x0, w, chunks[ax]):
            cx_lo = cx * chunks[ax]
            sx0 = max(x0, cx_lo) - cx_lo
            sx1 = min(x0 + w, cx_lo + chunks[ax]) - cx_lo
            key: List[int] = []
            for i in range(len(shape)):
                if i == ay:
                    key.append(cy)
                elif i == ax:
                    key.append(cx)
                else:
                    key.append(pinned[i] // chunks[i])
            out.append(
                {
                    "key": tuple(key),
                    "src": ((sy0, sy1), (sx0, sx1)),
                    "dst": (
                        (cy_lo + sy0 - y0, cy_lo + sy1 - y0),
                        (cx_lo + sx0 - x0, cx_lo + sx1 - x0),
                    ),
                    # Where the pinned axes sit INSIDE the chunk, so a caller can
                    # slice a chunk that still carries them.
                    "pinned_in_chunk": {
                        str(axes[i]).lower(): pinned[i] % chunks[i]
                        for i in range(len(shape))
                        if i not in (ay, ax)
                    },
                }
            )
    return {
        "height": h,
        "width": w,
        "axes": [str(a).lower() for a in axes],
        "chunks": out,
        "chunks_touched": len(out),
    }


def chunk_key(key: Sequence[int], dimension_separator: str = ".") -> str:
    """The store path for a chunk, e.g. ``0.0.0.3.4``.

    Zarr v2 writes ``.`` by default and ``/`` when the store says so; the value is
    in ``.zarray`` as ``dimension_separator``, so it is passed in rather than
    assumed — guessing it yields a 404 on every chunk, which is indistinguishable
    from an empty region.
    """
    if dimension_separator not in (".", "/"):
        raise ValueError(
            f"dimension_separator must be '.' or '/', got {dimension_separator!r}"
        )
    return dimension_separator.join(str(int(i)) for i in key)


def read_region(
    plan: Dict[str, Any],
    fetch: Callable[[Tuple[int, ...]], Any],
    out: Any,
) -> Any:
    """Assemble a planned region into ``out``.

    ``fetch`` is given a chunk key and returns that chunk as something sliceable in
    (…, y, x); a missing chunk may come back as ``None``, which Zarr semantics treat
    as the fill value, so it is left as whatever ``out`` was initialised to rather
    than raising. ``out`` is supplied by the caller so this module needs no array
    library of its own.
    """
    for part in plan["chunks"]:
        block = fetch(part["key"])
        if block is None:
            continue
        (sy0, sy1), (sx0, sx1) = part["src"]
        (dy0, dy1), (dx0, dx1) = part["dst"]
        out[dy0:dy1, dx0:dx1] = block[..., sy0:sy1, sx0:sx1]
    return out
