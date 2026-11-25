"""Decoding one stored chunk, and the two modules together reading a real crop.

The end-to-end case at the bottom is the important one: it builds a ground-truth
array, splits it into chunks exactly as a Zarr v2 store would, serves them from a
dict, and reads a region back through the planner and the decoder. If the chunk
arithmetic and the byte layout disagree anywhere, the comparison against a direct
slice of the same array fails. Nothing here needs numcodecs, a network or a GPU: an
uncompressed chunk is raw bytes, so the full path is exercisable with numpy alone,
and the compressed path is covered by injecting a codec.
"""

import json

import numpy as np
import pytest

import ngff_chunk as nc
import ngff_region as nr

BASE = {"zarr_format": 2, "shape": [8, 8], "chunks": [4, 4], "dtype": "|u1",
        "order": "C", "fill_value": 0, "compressor": None, "filters": None}


def test_a_plain_zarray_parses_to_what_a_read_needs():
    spec = nc.parse_zarray(dict(BASE))
    assert spec["shape"] == [8, 8]
    assert spec["chunks"] == [4, 4]
    assert spec["dtype"] == "|u1"
    assert spec["order"] == "C"
    assert spec["dimension_separator"] == "."
    assert spec["compressor"] is None
    assert spec["filters"] == []


def test_a_v3_array_is_refused_by_name_rather_than_read_as_v2():
    """v3 moves the codec pipeline and the key syntax, so treating it as v2 computes
    chunk paths that 404 — which looks exactly like an empty region."""
    with pytest.raises(nc.UnsupportedChunkEncoding) as e:
        nc.parse_zarray(dict(BASE, zarr_format=3))
    assert "v2" in str(e.value)


def test_an_unsupported_codec_is_named_rather_than_attempted():
    with pytest.raises(nc.UnsupportedChunkEncoding) as e:
        nc.parse_zarray(dict(BASE, compressor={"id": "imagecodecs_jpeg2k"}))
    assert "imagecodecs_jpeg2k" in str(e.value)
    # A filter is part of the same pipeline and must be checked too.
    with pytest.raises(nc.UnsupportedChunkEncoding):
        nc.parse_zarray(dict(BASE, filters=[{"id": "some_future_filter"}]))


def test_malformed_metadata_is_refused_rather_than_defaulted():
    for bad in (
        dict(BASE, shape=[]),
        dict(BASE, chunks=[0, 4]),
        dict(BASE, shape=[8, 8], chunks=[4]),
        dict(BASE, dtype=None),
        dict(BASE, order="Z"),
        dict(BASE, dimension_separator="-"),
        dict(BASE, compressor={"no_id": True}),
    ):
        with pytest.raises(ValueError):
            nc.parse_zarray(bad)
    with pytest.raises(ValueError):
        nc.parse_zarray("not an object")


def test_an_uncompressed_chunk_round_trips_in_both_orders():
    for order in ("C", "F"):
        block = np.arange(16, dtype=np.uint8).reshape(4, 4)
        raw = block.tobytes(order=order)
        out = nc.decode_chunk(raw, nc.parse_zarray(dict(BASE, order=order)))
        assert np.array_equal(out, block), order


def test_a_big_endian_dtype_is_honoured():
    """Byte order lives in the dtype string; ignoring it silently halves or doubles
    every pixel value."""
    spec = nc.parse_zarray(dict(BASE, dtype=">u2"))
    block = np.array([[1, 256], [4096, 65535]], dtype=">u2")
    block = np.pad(block, ((0, 2), (0, 2)))
    out = nc.decode_chunk(block.tobytes(), spec)
    assert np.array_equal(out, block)
    assert out.dtype.byteorder in (">", "=")


def test_the_declared_codec_is_the_one_requested_and_filters_undo_in_reverse():
    """Filters are applied in order on write, so they must come off in reverse. A
    recording fake proves both the ids and the order without numcodecs."""
    calls = []

    class Fake:
        def __init__(self, spec):
            self.spec = spec

        def decode(self, buf):
            calls.append(self.spec["id"])
            return buf

    spec = nc.parse_zarray(
        dict(BASE, compressor={"id": "blosc"}, filters=[{"id": "zlib"}, {"id": "bz2"}])
    )
    block = np.zeros((4, 4), dtype=np.uint8)
    nc.decode_chunk(block.tobytes(), spec, get_codec=Fake)
    assert calls == ["blosc", "bz2", "zlib"], calls


def test_a_truncated_chunk_is_refused_with_both_byte_counts():
    """Reshaping a short buffer either raises deep inside numpy or, worse, succeeds
    against a different chunk shape."""
    spec = nc.parse_zarray(dict(BASE))
    with pytest.raises(ValueError) as e:
        nc.decode_chunk(b"\x00" * 15, spec)
    msg = str(e.value)
    assert "15" in msg and "16" in msg


def test_an_absent_chunk_stands_for_its_fill_value():
    """Zarr omits a chunk that is entirely the fill value, so a 404 is data. Most of
    an annotated slide was never written."""
    spec = nc.parse_zarray(dict(BASE, fill_value=7))
    f = nc.fill_chunk(spec)
    assert f.shape == (4, 4)
    assert (f == 7).all()
    # A null fill_value is spelled as absent in some writers and means zero.
    assert (nc.fill_chunk(nc.parse_zarray(dict(BASE, fill_value=None))) == 0).all()


def test_the_chunk_url_uses_the_declared_separator():
    assert nc.chunk_url("https://x/y/0", (1, 2)) == "https://x/y/0/1.2"
    assert nc.chunk_url("https://x/y/0/", (1, 2), "/") == "https://x/y/0/1/2"


def test_end_to_end_a_region_read_from_a_chunked_store_matches_a_direct_slice():
    """THE INTEGRATION CASE. A store, a plan, a decode, an assembly — compared with
    slicing the same array directly. Also exercises a sparse store: one chunk is
    deliberately absent and must come back as the fill value."""
    rng = np.random.default_rng(7)
    shape, chunks = (240, 320), (64, 64)
    full = rng.integers(1, 255, size=shape, dtype=np.uint8)  # 1.. so 0 marks the hole
    zarray = {
        "zarr_format": 2, "shape": list(shape), "chunks": list(chunks),
        "dtype": "|u1", "order": "C", "fill_value": 0,
        "compressor": None, "filters": None, "dimension_separator": "/",
    }
    spec = nc.parse_zarray(json.loads(json.dumps(zarray)))

    # Build the store exactly as Zarr v2 would: every chunk at FULL chunk shape,
    # edge chunks padded with the fill value.
    store = {}
    ny = -(-shape[0] // chunks[0])
    nx = -(-shape[1] // chunks[1])
    for cy in range(ny):
        for cx in range(nx):
            block = np.zeros(chunks, dtype=np.uint8)
            src = full[cy * chunks[0]:(cy + 1) * chunks[0],
                       cx * chunks[1]:(cx + 1) * chunks[1]]
            block[: src.shape[0], : src.shape[1]] = src
            store[nr.chunk_key((cy, cx), spec["dimension_separator"])] = block.tobytes()
    hole = nr.chunk_key((1, 1), spec["dimension_separator"])
    del store[hole]  # a sparse store: this chunk is all fill value
    full_with_hole = full.copy()
    full_with_hole[64:128, 64:128] = 0

    region = {"x": 40, "y": 30, "width": 200, "height": 150}
    plan = nr.plan_region_read(spec["shape"], spec["chunks"], ["y", "x"], region)
    fetched = []

    def fetch(key):
        name = nr.chunk_key(key, spec["dimension_separator"])
        fetched.append(name)
        raw = store.get(name)
        if raw is None:
            return nc.fill_chunk(spec)
        return nc.decode_chunk(raw, spec)

    out = np.zeros((region["height"], region["width"]), dtype=np.uint8)
    nr.read_region(plan, fetch, out)
    expect = full_with_hole[30:180, 40:240]
    assert np.array_equal(out, expect)

    # And the read was proportional to the region, not the store.
    assert len(fetched) == plan["chunks_touched"]
    assert len(fetched) < ny * nx, "the region read every chunk in the store"
    assert hole in fetched, "the sparse chunk was not even attempted"
