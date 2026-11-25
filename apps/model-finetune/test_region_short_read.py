"""A chunk whose download is cut short is refused, not trained on.

Why this needs its own file, and why it asserts on an EXCEPTION rather than on
pixels: a truncated chunk decodes to the full chunk shape with the wrong content,
and the first pixels still match the source. Both of those were measured. So a test
that compared a corner of the crop, or even its shape, would pass on corrupt data.
The only reliable signal is the refusal itself.

Both layers underneath were measured blind — the decompressor does not notice a
truncated stream, and the HTTP layer under zarr does not notice either, even when
the response declares the full length. ``Content-Length`` is the last point the
truncation is visible, so :class:`ngff_region.LengthCheckedStore` checks it there
and these tests pin that it does.

The store here is uncompressed on purpose: the length check runs BEFORE any decode,
so compression is irrelevant to what is being tested and leaving it out keeps the
test free of a codec dependency.
"""

import http.server
import json
import pathlib
import threading

import numpy as np
import pytest

import ngff_region

SHAPE = (128, 128)
CHUNKS = (64, 64)
VICTIM = "/1.1"  # the chunk served short


def _build_store(tmp_path):
    """A minimal Zarr v2 array directory, uncompressed."""
    root = tmp_path / "array"
    root.mkdir()
    rng = np.random.default_rng(5)
    full = rng.integers(1, 255, size=SHAPE, dtype=np.uint8)
    (root / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": list(SHAPE),
                "chunks": list(CHUNKS),
                "dtype": "|u1",
                "order": "C",
                "fill_value": 0,
                "compressor": None,
                "filters": None,
                "dimension_separator": ".",
            }
        )
    )
    for cy in range(SHAPE[0] // CHUNKS[0]):
        for cx in range(SHAPE[1] // CHUNKS[1]):
            block = full[
                cy * CHUNKS[0] : (cy + 1) * CHUNKS[0],
                cx * CHUNKS[1] : (cx + 1) * CHUNKS[1],
            ]
            (root / f"{cy}.{cx}").write_bytes(block.tobytes())
    return root, full


class _Server:
    """Serves the store, truncating one chunk while declaring its FULL length.

    That combination is what a transfer cut short looks like on the wire, and it is
    the case neither layer below catches.
    """

    def __init__(self, root):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                rel = self.path.split("?")[0]
                p = outer.root / rel.lstrip("/")
                if not p.is_file():
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = p.read_bytes()
                declared = len(body)
                if rel == VICTIM and outer.truncate:
                    body = body[: declared // 2]
                self.send_response(200)
                self.send_header("Content-Length", str(declared))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except Exception:
                    pass

        self.root = root
        self.truncate = True
        self._srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self._srv.server_address[1]}"
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self):
        self._srv.shutdown()


@pytest.fixture()
def served(tmp_path):
    root, full = _build_store(tmp_path)
    srv = _Server(root)
    try:
        yield srv, full
    finally:
        srv.close()


def _open(base):
    """Open through the module's own opener, so the wrapper is in the path."""
    return ngff_region._zarr_opener(base, "")


def test_an_intact_chunk_still_reads_correctly(served):
    """THE CONTROL. Without it, a refusal below could be the wrapper breaking every
    read rather than catching the bad one."""
    srv, full = served
    a = _open(srv.base)
    block = a[0:64, 0:64]
    assert np.array_equal(block, full[0:64, 0:64])


def test_a_short_body_is_REFUSED_rather_than_decoded(served):
    """The chunk served short must raise. Asserted on the exception, because the
    decoded array would have the right shape and its first pixels would match."""
    srv, full = served
    a = _open(srv.base)
    with pytest.raises(ngff_region.ShortStoreRead) as e:
        a[64:128, 64:128]
    msg = str(e.value)
    assert "1.1" in msg, f"the failing key is not named: {msg}"
    # Both lengths, so an operator can see how short it was.
    assert "2048" in msg and "4096" in msg, f"both lengths are not reported: {msg}"


def test_the_refusal_names_no_credential(served):
    """The store root carries the credential as a path segment, and this message
    travels into the session status, which is served over HTTP."""
    srv, full = served
    # Opening may itself fail with a credential segment in the root, so the whole
    # call is inside the assertion — the leak would be just as bad either way.
    with pytest.raises(Exception) as e:
        a = ngff_region._zarr_opener(f"{srv.base}/t/SECRET-TOKEN-VALUE/", "")
        a[0:64, 0:64]
    assert "SECRET-TOKEN-VALUE" not in str(e.value), str(e.value)[:200]


def test_with_truncation_off_the_same_read_succeeds(served):
    """Pins that the refusal is caused by the truncation and not by the chunk's
    position: the same key, same store, same code path, served whole."""
    srv, full = served
    srv.truncate = False
    a = _open(srv.base)
    block = a[64:128, 64:128]
    assert np.array_equal(block, full[64:128, 64:128])


def test_an_absent_chunk_is_a_KeyError_so_zarr_can_use_its_fill_value(served):
    """Zarr omits a chunk that is entirely the fill value, so absence is data. If
    the wrapper raised instead, every sparse store would become a failed read."""
    srv, full = served
    store = ngff_region.LengthCheckedStore(f"{srv.base}/")
    with pytest.raises(KeyError):
        store["9.9"]
    assert "9.9" not in store


def test_listing_says_so_rather_than_answering_empty():
    """An empty listing would turn "I cannot list" into "there is nothing here" —
    the silent-wrong-answer shape this class exists to prevent."""
    store = ngff_region.LengthCheckedStore("http://127.0.0.1:1/")
    with pytest.raises(NotImplementedError):
        iter(store)
    with pytest.raises(NotImplementedError):
        len(store)


def test_the_store_is_read_only():
    store = ngff_region.LengthCheckedStore("http://127.0.0.1:1/")
    with pytest.raises(ngff_region.RegionSourceError):
        store["x"] = b"y"
    with pytest.raises(ngff_region.RegionSourceError):
        del store["x"]
