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
                    if outer.shift_range and rel == outer.victim and a != "":
                        # The hostile case: serve the NEIGHBOURING bytes and
                        # report them honestly. Content-Length and Content-Range
                        # agree with each other and with the body; only the
                        # REQUEST disagrees, which is why nothing but comparing
                        # the two can catch it.
                        #
                        # Explicit start-end ranges ONLY (`a != ""`), i.e. the
                        # inner-chunk reads. Shifting the shard INDEX instead is
                        # a different and already-covered story: a suffix range
                        # may legitimately return fewer bytes than asked, so the
                        # wrapper cannot call it wrong -- and it does not need
                        # to, because zarr puts a crc32c on the index and raises
                        # "Stored and computed checksum do not match". Measured
                        # here. The inner-chunk read is the one with no such
                        # backstop, which is why it is the case under test.
                        start, end = start + 1, min(end + 1, total - 1)
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
        self.shift_range = False
        self.victim = None
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


# ── the bytes actually served must be the bytes asked for ────────────────────


def test_a_206_carrying_the_WRONG_bytes_is_refused(sharded):
    """RED before Content-Range was checked. A server that answers a range with
    the NEIGHBOURING bytes, and reports those bytes honestly in Content-Length
    and Content-Range, passes every other check in this module: the body is
    complete, its length matches what was declared, and nothing is truncated.
    Only comparing the request against Content-Range catches it.

    Measured consequence without that comparison, on an uncompressed shard: the
    crop came back with NO error and the wrong pixels. Compression and the shard
    index's CRC catch most variants; neither is guaranteed to be there, and this
    store is uncompressed on purpose so the check is tested rather than a codec.
    """
    srv, full, on_disk = sharded
    srv.victim = "c/0/0"
    srv.shift_range = True
    srv.reset()
    a = ngff_region._zarr_opener(srv.base, "")
    with pytest.raises(ngff_region.RegionSourceError) as e:
        a[0:128, 0:128]
    msg = str(e.value)
    assert "Content-Range" in msg, f"the refusal does not name the header: {msg}"
    assert "not the bytes that were requested" in msg, msg


def test_an_honest_206_is_NOT_refused(sharded):
    """THE CONTROL for the test above. Without it, the refusal could be the
    Content-Range check rejecting every ranged read rather than the bad one."""
    srv, full, on_disk = sharded
    srv.shift_range = False
    srv.reset()
    a = ngff_region._zarr_opener(srv.base, "")
    assert np.array_equal(a[0:128, 0:128], full[0:128, 0:128])
    assert any(r[1] for r in srv.requests), "no ranged request was made at all"


@pytest.mark.parametrize(
    "requested, served, why",
    [
        ("bytes=10-19", "bytes 11-20/100", "shifted start"),
        ("bytes=10-19", "bytes 10-18/100", "short end, self-consistent"),
        ("bytes=-8", "bytes 80-87/100", "suffix not at the end of the object"),
        ("bytes=10-19", None, "206 with no Content-Range at all"),
        ("bytes=10-19", "garbage", "unparsable Content-Range"),
        # A SHORT suffix that ends at the last byte satisfies every other rule.
        # Legitimate only if the object is smaller than the suffix, i.e. the
        # whole object came back, i.e. it starts at 0. This one does not.
        ("bytes=-10", "bytes 4107-4111/4112", "short suffix from the middle"),
    ],
)
def test_each_wrong_content_range_is_refused(requested, served, why):
    """The checker directly, so each disagreement is pinned rather than only the
    one shape a server happened to produce."""
    n = {"bytes=10-19": 10, "bytes=-8": 8, "bytes=-10": 5}[requested]
    with pytest.raises(ngff_region.WrongRangeRead):
        ngff_region._check_content_range("k", requested, served, n)


@pytest.mark.parametrize(
    "requested, served, body_len",
    [
        ("bytes=10-19", "bytes 10-19/100", 10),
        ("bytes=10-", "bytes 10-99/100", 90),
        ("bytes=-8", "bytes 92-99/100", 8),
        ("bytes=-8", "bytes 0-4/5", 5),  # object SMALLER than the suffix asked for
    ],
)
def test_a_matching_content_range_passes(requested, served, body_len):
    """The accept side, including the legitimate case of an object shorter than
    the suffix requested -- refusing that would break small shard indexes."""
    ngff_region._check_content_range("k", requested, served, body_len)
