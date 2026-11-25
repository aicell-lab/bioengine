"""Turn one stored OME-Zarr chunk into an array.

:mod:`ngff_region` decides WHICH chunks a region needs; this decides what a chunk's
bytes mean. Together they read a crop without fetching the image, which is what lets
a region annotation on a whole-slide image train at all.

**Why ``numcodecs`` and not ``zarr``.** The indexing — which chunks a region covers,
how they reassemble — is already written and tested in :mod:`ngff_region`, so the
only thing missing is the decompressor. ``zarr`` would bring its own indexing plus
``fsspec`` to duplicate work this app already owns; ``numcodecs`` is the part that is
actually absent. It is imported lazily and only for a chunk that is compressed, so an
uncompressed store needs no new dependency at all and the parsing here is testable
without one.

**The codec question is per reader, not per store.** The annotation broker has an
``undecodable_ngff_codecs`` check, but it answers "can the BROWSER decode this",
because the browser is what reads pixels on that path. A Python reader with
``numcodecs`` decodes strictly more. Reusing that list here would refuse stores that
train perfectly well, so this module states its own support and the two are kept
apart deliberately despite the shared vocabulary.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

# Codec ids this reader handles, as they appear in a .zarray "compressor"/"filters"
# id field. Everything here is pure numcodecs with no system library beyond what the
# wheels carry. An id outside this set is refused by name rather than attempted,
# because numcodecs raises a ValueError that does not say which store or which codec.
SUPPORTED_CODEC_IDS = frozenset({"blosc", "zstd", "lz4", "zlib", "gzip", "bz2"})


class UnsupportedChunkEncoding(ValueError):
    """The chunk cannot be decoded by this reader.

    Separate from a malformed-metadata error because the operator's response differs:
    an unsupported codec means "this store needs a different reader or a conversion",
    which is worth saying to a user, while malformed metadata means the store is
    broken.
    """


def parse_zarray(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a ``.zarray`` document and return what a read needs from it.

    Returns ``{"shape", "chunks", "dtype", "order", "fill_value", "compressor",
    "filters", "dimension_separator"}``. Raises on anything this reader would
    otherwise handle by guessing.
    """
    if not isinstance(meta, dict):
        raise ValueError(f".zarray must be an object, got {type(meta).__name__}")
    fmt = meta.get("zarr_format")
    # v3 moves the codec pipeline and the key syntax; silently treating a v3 array as
    # v2 would compute chunk paths that 404, which looks like an empty region.
    if fmt not in (2, None):
        raise UnsupportedChunkEncoding(
            f"this reader handles Zarr v2 arrays; .zarray declares zarr_format={fmt!r}"
        )
    shape = meta.get("shape")
    chunks = meta.get("chunks")
    for name, value in (("shape", shape), ("chunks", chunks)):
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f".zarray {name} must be a non-empty list, got {value!r}")
        if any(int(v) <= 0 for v in value):
            raise ValueError(f".zarray {name} must be positive, got {list(value)}")
    if len(shape) != len(chunks):
        raise ValueError(
            f".zarray shape {list(shape)} and chunks {list(chunks)} differ in rank"
        )
    dtype = meta.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(f".zarray dtype must be a string, got {dtype!r}")
    order = meta.get("order", "C")
    if order not in ("C", "F"):
        raise ValueError(f".zarray order must be 'C' or 'F', got {order!r}")
    sep = meta.get("dimension_separator", ".")
    if sep not in (".", "/"):
        raise ValueError(
            f".zarray dimension_separator must be '.' or '/', got {sep!r}"
        )

    compressor = meta.get("compressor")
    filters = meta.get("filters") or []
    for spec in ([compressor] if compressor else []) + list(filters):
        if not isinstance(spec, dict) or "id" not in spec:
            raise ValueError(f"a codec spec needs an 'id', got {spec!r}")
        if spec["id"] not in SUPPORTED_CODEC_IDS:
            raise UnsupportedChunkEncoding(
                f"chunk codec {spec['id']!r} is not supported by this reader; "
                f"supported: {sorted(SUPPORTED_CODEC_IDS)}"
            )
    return {
        "shape": [int(v) for v in shape],
        "chunks": [int(v) for v in chunks],
        "dtype": dtype,
        "order": order,
        "fill_value": meta.get("fill_value", 0),
        "compressor": compressor,
        "filters": list(filters),
        "dimension_separator": sep,
    }


def _default_codec(spec: Dict[str, Any]):
    """Resolve a codec spec through numcodecs, imported only when needed."""
    import numcodecs  # noqa: PLC0415 -- lazy on purpose; see the module docstring

    return numcodecs.get_codec(spec)


def decode_chunk(
    raw: bytes,
    spec: Dict[str, Any],
    get_codec: Optional[Callable[[Dict[str, Any]], Any]] = None,
):
    """Decode one chunk's stored bytes into an array of the full chunk shape.

    Zarr v2 stores every chunk at its full shape, edge chunks included — the
    out-of-bounds part is padding — so the result is always ``spec["chunks"]``
    shaped and :func:`ngff_region.read_region` slices the useful part out.

    ``get_codec`` exists so the decode path can be exercised without numcodecs
    installed; by default it resolves through numcodecs.
    """
    import numpy as np

    parsed = spec if "order" in spec and "chunks" in spec else parse_zarray(spec)
    get_codec = get_codec or _default_codec

    buf: Any = raw
    # Filters are applied on write in order, so they are undone in reverse; the
    # compressor is the outermost layer and comes off first.
    if parsed["compressor"]:
        buf = get_codec(parsed["compressor"]).decode(buf)
    for f in reversed(parsed["filters"]):
        buf = get_codec(f).decode(buf)

    dt = np.dtype(parsed["dtype"])
    expected = int(np.prod(parsed["chunks"])) * dt.itemsize
    got = len(bytes(buf)) if not hasattr(buf, "nbytes") else int(buf.nbytes)
    if got != expected:
        # Reshaping a short buffer either raises deep in numpy or, worse, succeeds
        # against a different chunk shape. Say which chunk and by how much.
        raise ValueError(
            f"decoded chunk is {got} bytes but {parsed['chunks']} of {dt.str} "
            f"needs {expected}; the store or the codec chain disagrees"
        )
    arr = np.frombuffer(bytes(buf), dtype=dt)
    return arr.reshape(parsed["chunks"], order=parsed["order"])


def fill_chunk(spec: Dict[str, Any]):
    """The array an ABSENT chunk stands for.

    Zarr omits a chunk that is entirely the fill value, so a 404 is data, not an
    error — and a sparse store is normal for an annotated region, where most of a
    slide was never written.
    """
    import numpy as np

    parsed = spec if "order" in spec and "chunks" in spec else parse_zarray(spec)
    fill = parsed["fill_value"]
    dt = np.dtype(parsed["dtype"])
    if fill is None:
        fill = 0
    return np.full(parsed["chunks"], fill, dtype=dt)


def chunk_url(base_url: str, key: Sequence[int], dimension_separator: str = ".") -> str:
    """Where a chunk lives, given the array's base URL.

    The separator comes from the array's own metadata; guessing it 404s every chunk,
    which is indistinguishable from a region that happens to be empty.
    """
    from ngff_region import chunk_key

    return f"{base_url.rstrip('/')}/{chunk_key(key, dimension_separator)}"
