"""A sharded v3 store must not be downloaded whole to deliver one crop.

Sharding is the normal layout for OME-NGFF v0.5 whole-slide data: many small
inner chunks are packed into one large shard object, with an index at its end.
zarr reads such an array by asking for a BYTE RANGE of the shard — first a suffix
range for the index, then a range per inner chunk it needs.

A store that ignores those ranges and fetches the whole object each time is
correct and ruinous: the crop comes back right, so nothing fails, while a
multi-gigabyte shard is pulled down once per inner chunk. Measured on the first
version of the v3 support: a 2048x2048 store with one 2048x2048 shard served
8,391,579 bytes — twice the whole image — to deliver a 256x256 crop, in 3 GETs
none of which carried a Range header.

So these tests assert on BYTES SERVED and on the Range headers the store sent,
not on the pixels. The pixels were already right. That is the whole point: this
is a transfer-economy guarantee, and the only way to test one is to count.

The server here honours Range, because a server that ignores it cannot
distinguish a store that asks correctly from one that does not — the round-1
attempt at this measurement used Python's SimpleHTTPRequestHandler, which drops
Range silently, and the resulting control was worthless.
"""

from __future__ import annotations

import asyncio
import http.server
import re
import threading

import numpy as np
import pytest

zarr = pytest.importorskip("zarr", reason="entry-env dependency")
if int(zarr.__version__.split(".")[0]) < 3:
    pytest.skip(
        f"sharding is a zarr-3 feature; found {zarr.__version__}",
        allow_module_level=True,
    )

import ngff_region  # noqa: E402

SHAPE = (1024, 1024)
INNER = (128, 128)
SHARD = (1024, 1024)  # ONE shard covering the whole array -- the hostile case


def _write_sharded(tmp_path):
    root = tmp_path / "sharded.zarr"
    rng = np.random.default_rng(11)
    full = rng.integers(1, 255, size=SHAPE, dtype=np.uint8)
    z = zarr.create_array(
        store=str(root), shape=SHAPE, chunks=INNER, shards=SHARD,
        dtype="u1", zarr_format=3, compressors=None,
    )
    z[:] = full
    files = [p for p in root.rglob("*") if p.is_file()]
    on_disk = sum(p.stat().st_size for p in files)
    return root, full, on_disk


class _RangeServer:
    """Serves a directory, HONOURING Range, and counting what it sends.

    ``ignore_range`` makes it answer 200 with the whole body instead, which is
    what a store backed by a server without range support looks like.
    """

    def __init__(self, root):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                rel = self.path.split("?")[0].lstrip("/")
                p = outer.root / rel
                rng = self.headers.get("Range")
                outer.requests.append((rel, rng))
                if not p.is_file():
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = p.read_bytes()
                total = len(body)
                start = end = None
                if rng and not outer.ignore_range:
                    m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
                    assert m, f"unparsable Range: {rng!r}"
                    a, b = m.group(1), m.group(2)
                    if a == "":                      # suffix: bytes=-N
                        n = min(int(b), total)
                        start, end = total - n, total - 1
                    elif b == "":                    # open-ended: bytes=N-
                        start, end = int(a), total - 1
                    else:
                        start, end = int(a), min(int(b), total - 1)
                    body = body[start : end + 1]
                self.send_response(206 if start is not None else 200)
                if start is not None:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                outer.bytes_served += len(body)
                try:
                    self.wfile.write(body)
                except Exception:
                    pass

        self.root = root
        self.ignore_range = False
        self.bytes_served = 0
        self.requests = []
        self._srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self._srv.server_address[1]}"
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def reset(self):
        self.bytes_served = 0
        self.requests = []

    def close(self):
        self._srv.shutdown()


@pytest.fixture()
def sharded(tmp_path):
    root, full, on_disk = _write_sharded(tmp_path)
    srv = _RangeServer(root)
    try:
        yield srv, full, on_disk
    finally:
        srv.close()


def test_the_fixture_really_is_one_big_shard(sharded):
    """THE CONTROL on the fixture itself. If zarr ignored `shards=` and wrote
    plain chunks, every assertion below would pass for the wrong reason -- small
    objects fetched whole are small."""
    srv, full, on_disk = sharded
    chunk_objects = [
        p for p in srv.root.rglob("*") if p.is_file() and p.name != "zarr.json"
    ]
    assert len(chunk_objects) == 1, (
        f"expected ONE shard object, found {len(chunk_objects)}: "
        f"{[p.name for p in chunk_objects]} -- the fixture is not sharded"
    )
    biggest = max(p.stat().st_size for p in chunk_objects)
    assert biggest > 500_000, f"the shard is only {biggest} B; fixture too small to be a test"


def test_a_crop_from_a_sharded_store_does_not_download_the_shard(sharded):
    """THE FINDING. Asserted on BYTES SERVED, because the pixels were already
    correct while the whole shard came down twice."""
    srv, full, on_disk = sharded
    srv.reset()
    a = ngff_region._zarr_opener(srv.base, "")
    crop = a[0:128, 0:128]
    assert np.array_equal(crop, full[0:128, 0:128]), "the crop itself is wrong"

    inner_bytes = INNER[0] * INNER[1]  # 16384 B of uint8 pixels
    ranged = [r for r in srv.requests if r[1]]
    assert ranged, (
        f"the store sent NO Range header. Requests: {srv.requests}. Every one of "
        f"those is a whole-object GET, so one inner chunk costs a whole shard."
    )
    # Generous ceiling: the index plus a few inner chunks, nowhere near the shard.
    assert srv.bytes_served < 20 * inner_bytes, (
        f"served {srv.bytes_served} B to deliver {inner_bytes} B of pixels "
        f"(shard on disk {on_disk} B). Requests: {srv.requests}"
    )


def test_a_server_that_IGNORES_range_is_refused_by_name(sharded):
    """If the server answers 200 to a ranged request it has sent the whole
    object. Slicing that locally would be correct and would quietly cost a full
    download per inner chunk -- the exact thing this file exists to prevent -- so
    it is refused instead, naming what happened."""
    srv, full, on_disk = sharded
    srv.ignore_range = True
    srv.reset()
    a = ngff_region._zarr_opener(srv.base, "")
    with pytest.raises(ngff_region.RegionSourceError) as e:
        a[0:128, 0:128]
    msg = str(e.value)
    assert "range" in msg.lower(), f"the refusal does not say what went wrong: {msg}"


def test_an_unsharded_v3_store_is_unaffected(tmp_path):
    """Controls the change: a plain v3 array is read with no Range at all, so the
    unsharded path is not quietly carrying the ranged one's behaviour."""
    root = tmp_path / "plain.zarr"
    rng = np.random.default_rng(12)
    full = rng.integers(1, 255, size=(256, 256), dtype=np.uint8)
    z = zarr.create_array(
        store=str(root), shape=(256, 256), chunks=(128, 128),
        dtype="u1", zarr_format=3, compressors=None,
    )
    z[:] = full
    srv = _RangeServer(root)
    try:
        a = ngff_region._zarr_opener(srv.base, "")
        assert np.array_equal(a[0:128, 0:128], full[0:128, 0:128])
    finally:
        srv.close()


# ── the Range header construction, which can ask for the WRONG bytes ─────────


@pytest.mark.parametrize(
    "make, expected",
    [
        (lambda m: None, None),
        (lambda m: m.RangeByteRequest(2, 5), "bytes=2-4"),
        (lambda m: m.OffsetByteRequest(7), "bytes=7-"),
        (lambda m: m.SuffixByteRequest(3), "bytes=-3"),
    ],
)
def test_each_byte_request_becomes_the_right_range_header(make, expected):
    """``RangeByteRequest(start, end)`` is end-EXCLUSIVE and HTTP ranges are
    end-INCLUSIVE. An off-by-one here returns plausible bytes from the wrong
    offset, which decodes without complaint."""
    import zarr.abc.store as zs

    assert ngff_region._range_header(make(zs)) == expected


def test_an_unknown_byte_request_is_REFUSED_not_sent_unranged():
    """Falling back to no Range would silently fetch the whole object, which is
    this file's subject; falling back to a guessed range would read the wrong
    bytes. Neither is acceptable, so it refuses."""

    class _Future:
        pass

    with pytest.raises(ngff_region.RegionSourceError) as e:
        ngff_region._range_header(_Future())
    assert "_Future" in str(e.value)
