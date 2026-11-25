"""A chunk whose download is cut short is refused, not trained on.

Why this needs its own file, and why it asserts on an EXCEPTION rather than on
pixels: a truncated chunk decodes to the full chunk shape with the wrong content,
and the first pixels still match the source. Both of those were measured. So a test
that compared a corner of the crop, or even its shape, would pass on corrupt data.
The only reliable signal is the refusal itself.

Both layers underneath were measured blind — the decompressor does not notice a
truncated stream, and the HTTP layer under zarr does not notice either, even when
the response declares the full length. ``Content-Length`` is the last point the
truncation is visible, so :func:`ngff_region.LengthCheckedStore` checks it there
and these tests pin that it does.

The stores here are uncompressed on purpose: the length check runs BEFORE any
decode, so compression is irrelevant to what is being tested and leaving it out
keeps the test free of a codec dependency.

**Two of these tests exist because of specific review findings, and both would
have passed before the finding was understood:**

- the gzip case. ``Content-Length`` counts the ENCODED body; httpx hands back the
  DECODED one. Comparing those two called an intact object "cut short", with the
  body arriving LONGER than declared. Objects stored with
  ``Content-Encoding: gzip`` are a normal way OME-Zarr metadata is served, so
  every such source was unreadable with a misleading integrity error.
- the zarr_format=3 case. The store is addressed through zarr's v3 ``Store``
  interface, and no test opened a v3 store through the shipped opener, so losing
  OME-NGFF v0.5 would not have failed anything.
"""

from __future__ import annotations

import asyncio
import gzip
import http.server
import threading

import numpy as np
import pytest

zarr = pytest.importorskip("zarr", reason="entry-env dependency")
if int(zarr.__version__.split(".")[0]) < 3:
    pytest.skip(
        f"this module targets the zarr-3 Store interface; found {zarr.__version__}. "
        "requirements-entry.txt pins zarr>=3.1,<3.1.6 and test_entry_requirements_pins "
        "asserts that bound, so a zarr-2 environment is a misconfigured one.",
        allow_module_level=True,
    )

import ngff_region  # noqa: E402  (after the version guard, on purpose)

SHAPE = (128, 128)
CHUNKS = (64, 64)


def _write_store(tmp_path, zarr_format):
    """Write a real store with zarr itself, rather than hand-rolling metadata.

    Hand-written ``zarr.json`` was the alternative and it is the kind of fixture
    that passes by agreeing with my own misreading of the spec. Letting zarr
    write it means the bytes served are bytes zarr produced.
    """
    root = tmp_path / f"v{zarr_format}.zarr"
    rng = np.random.default_rng(5)
    full = rng.integers(1, 255, size=SHAPE, dtype=np.uint8)
    kwargs = dict(
        store=str(root),
        shape=SHAPE,
        chunks=CHUNKS,
        dtype="u1",
        zarr_format=zarr_format,
    )
    try:
        z = zarr.create_array(compressors=None, **kwargs)
    except TypeError:
        # older zarr 3.x spelling
        z = zarr.create_array(**kwargs)
    z[:] = full
    return root, full


def _victim_key(root, zarr_format):
    """The relative key of the (1, 1) chunk, discovered rather than assumed."""
    want = "1.1" if zarr_format == 2 else "c/1/1"
    hits = [
        p
        for p in root.rglob("*")
        if p.is_file() and str(p.relative_to(root)).replace("\\", "/") == want
    ]
    assert len(hits) == 1, (
        f"expected exactly one chunk file at {want!r} under {root}, found "
        f"{[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()]}"
    )
    return want


class _Server:
    """Serves a store directory, with two deliberate misbehaviours available.

    ``truncate`` cuts one key's body in half while still declaring its FULL
    length — what a transfer cut short looks like on the wire, and the case
    neither layer below catches. ``gzip_keys`` serves a key gzip-encoded with a
    CORRECT Content-Length, which is an intact object that must read fine.
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
                if not p.is_file():
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = p.read_bytes()
                encoding = None
                if rel in outer.gzip_keys:
                    body = gzip.compress(body)
                    encoding = "gzip"
                declared = len(body)
                cut = rel == outer.victim and outer.truncate
                if cut:
                    body = body[: declared // 2]
                self.send_response(200)
                if encoding:
                    self.send_header("Content-Encoding", encoding)
                self.send_header("Content-Length", str(declared))
                if cut:
                    # A transfer that is genuinely CUT SHORT ends with the
                    # connection going away -- that is what makes it short rather
                    # than slow. Simulating it with keep-alive left open instead
                    # produces a 60-SECOND READ TIMEOUT per chunk, not a
                    # truncation: the client is still correctly waiting for bytes
                    # the server promised. That was measured here, and it cost
                    # three red tests whose failure looked like the length check
                    # not firing. The distinction is real and worth keeping
                    # straight: a held-open short body is a timeout, and the
                    # module reports it as one.
                    self.send_header("Connection", "close")
                    self.close_connection = True
                self.end_headers()
                try:
                    self.wfile.write(body)
                except Exception:
                    pass

        self.root = root
        self.victim = None
        self.truncate = False
        self.gzip_keys = set()
        self._srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self._srv.server_address[1]}"
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self):
        self._srv.shutdown()


@pytest.fixture()
def served_v2(tmp_path):
    root, full = _write_store(tmp_path, 2)
    srv = _Server(root)
    srv.victim = _victim_key(root, 2)
    srv.truncate = True
    try:
        yield srv, full
    finally:
        srv.close()


@pytest.fixture()
def served_v3(tmp_path):
    root, full = _write_store(tmp_path, 3)
    srv = _Server(root)
    srv.victim = _victim_key(root, 3)
    try:
        yield srv, full
    finally:
        srv.close()


def _open(base):
    """Open through the module's own opener, so the wrapper is in the path."""
    return ngff_region._zarr_opener(base, "")


# ── the control, first ────────────────────────────────────────────────────────


def test_an_intact_chunk_still_reads_correctly(served_v2):
    """THE CONTROL. Without it, a refusal below could be the wrapper breaking
    every read rather than catching the bad one."""
    srv, full = served_v2
    a = _open(srv.base)
    assert np.array_equal(a[0:64, 0:64], full[0:64, 0:64])


# ── the truncation this class exists for ─────────────────────────────────────


def test_a_short_body_is_REFUSED_rather_than_decoded(served_v2):
    """The chunk served short must raise. Asserted on the exception, because the
    decoded array would have the right shape and its first pixels would match."""
    srv, full = served_v2
    a = _open(srv.base)
    with pytest.raises(ngff_region.ShortStoreRead) as e:
        a[64:128, 64:128]
    msg = str(e.value)
    assert "1.1" in msg, f"the failing key is not named: {msg}"
    assert "2048" in msg and "4096" in msg, f"both lengths are not reported: {msg}"
    assert "cut short" in msg, f"the direction is not stated: {msg}"


def test_with_truncation_off_the_same_read_succeeds(served_v2):
    """Pins that the refusal is caused by the truncation and not by the chunk's
    position: the same key, same store, same code path, served whole."""
    srv, full = served_v2
    srv.truncate = False
    a = _open(srv.base)
    assert np.array_equal(a[64:128, 64:128], full[64:128, 64:128])


def test_the_refusal_names_no_credential(served_v2):
    """The store root carries the credential as a path segment, and this message
    travels into the session status, which is served over HTTP."""
    srv, full = served_v2
    with pytest.raises(Exception) as e:
        a = ngff_region._zarr_opener(f"{srv.base}/t/SECRET-TOKEN-VALUE/", "")
        a[0:64, 0:64]
    assert "SECRET-TOKEN-VALUE" not in str(e.value), str(e.value)[:200]


# ── F2: an intact object served gzip-encoded must READ, not be refused ───────


def test_a_gzip_encoded_object_is_read_not_called_short(served_v2):
    """RED before the fix. ``Content-Length`` counts the ENCODED bytes and httpx
    returns the DECODED body, so comparing those two refused an intact object —
    and reported it as "cut short" while the body was in fact LONGER than
    declared. Served here with a correct Content-Length for the gzip body, which
    is what a stored-gzip object looks like.
    """
    srv, full = served_v2
    srv.truncate = False
    srv.gzip_keys = {".zarray", "0.0"}
    a = _open(srv.base)
    assert np.array_equal(a[0:64, 0:64], full[0:64, 0:64])


def test_a_short_body_is_still_caught_when_the_object_is_gzip_encoded(served_v2):
    """The fix must not be "stop checking". Same gzip encoding, but the victim is
    also truncated, so the wire count falls below the declared one and the
    refusal has to fire exactly as it does without encoding."""
    srv, full = served_v2
    srv.gzip_keys = {srv.victim}
    srv.truncate = True
    a = _open(srv.base)
    with pytest.raises(ngff_region.ShortStoreRead) as e:
        a[64:128, 64:128]
    assert "cut short" in str(e.value)


# ── F3: a zarr_format=3 store reads through the shipped opener ───────────────


def test_a_zarr_format_3_store_reads_through_the_default_opener(served_v3):
    """OME-NGFF v0.5 is a v3 store. zarr 2 cannot open one at all, so this is the
    test that fails if the entry is ever pinned back below zarr 3 — and it fails
    on a read, which is how that regression would reach a user."""
    srv, full = served_v3
    srv.truncate = False
    a = _open(srv.base)
    assert a.shape == SHAPE
    assert np.array_equal(a[0:64, 0:64], full[0:64, 0:64])


def test_a_short_body_is_refused_on_a_v3_store_too(served_v3):
    """The length check lives under both formats, not only the one it was written
    against."""
    srv, full = served_v3
    srv.truncate = True
    a = _open(srv.base)
    with pytest.raises(ngff_region.ShortStoreRead):
        a[64:128, 64:128]


# ── the store's own contract, at the zarr-3 interface ────────────────────────


def test_an_absent_key_is_None_so_zarr_can_use_its_fill_value(served_v2):
    """Zarr omits a chunk that is entirely the fill value, so absence is data, and
    the v3 Store interface signals it with None. A store that raised instead would
    turn every sparse source into a failed read."""
    srv, _ = served_v2
    from zarr.core.buffer import default_buffer_prototype

    store = ngff_region.LengthCheckedStore(f"{srv.base}/")
    got = asyncio.run(store.get("9.9", default_buffer_prototype()))
    assert got is None
    assert asyncio.run(store.exists("9.9")) is False
    assert asyncio.run(store.exists(".zarray")) is True


def test_listing_says_so_rather_than_answering_empty():
    """An empty listing would turn "I cannot list" into "there is nothing here" —
    the silent-wrong-answer shape this class exists to prevent."""
    store = ngff_region.LengthCheckedStore("http://127.0.0.1:1/")
    assert store.supports_listing is False
    for call in (
        lambda: store.list(),
        lambda: store.list_prefix(""),
        lambda: store.list_dir(""),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_the_store_is_read_only():
    store = ngff_region.LengthCheckedStore("http://127.0.0.1:1/")
    assert store.read_only is True
    assert store.supports_writes is False
    assert store.supports_deletes is False
    with pytest.raises(ngff_region.RegionSourceError):
        asyncio.run(store.set("x", b"y"))
    with pytest.raises(ngff_region.RegionSourceError):
        asyncio.run(store.delete("x"))
