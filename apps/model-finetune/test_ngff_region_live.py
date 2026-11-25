"""The region read against a REAL store, over HTTP, through the default opener.

This file exists because of two findings against the first version of this work,
and it is written to fail if either comes back.

1. THE PRODUCTION PATH COULD NOT RUN. ``requirements-entry.txt`` listed ``zarr``
   alone, measured with ``pip install --dry-run``. A bare ``zarr`` resolves to
   3.1.6, the one release declaring ``numpy>=2.0``, which replaced the worker
   image's numpy 1.26.4 in the entry overlay and broke every numpy-1.x-compiled
   module in that image -- including the pandas ``ray.serve`` imports, so the
   replica died in its constructor and the deployment reported DEPLOY_FAILED with
   no logs at all. A dry run answers what RESOLVES, never what WORKS; only
   deploying it does. The pin is now ``zarr>=3.1,<3.1.6``.

2. THE TESTS COULD NOT TELL A CROP FROM A FULL DOWNLOAD. Every test passed
   against a mutant that read all of level 0 and then sliced it, because the one
   read test handed the reader a whole numpy array. "Never downloaded in full"
   is the whole claim of this work and it had no test at all.

So the test below uses ``opener=None`` -- the default, the one that ships -- and
COUNTS the chunks the server is asked for. A reader that fetches the level and
slices locally fails it on the count, with the values still correct.

WHY ``importorskip`` HERE, having argued against it elsewhere -- and why the
reason it used to give was wrong. It said zarr is absent in CI. zarr is NOT
absent in CI: ``requirements-datasets.txt`` pins ``zarr==3.1.3`` and
``test-suite.yml`` installs it, so these tests RUN there. They were failing
rather than skipping, which is how the store being written against the wrong zarr
major went unnoticed. The guard stays only so the file does not ERROR at
collection in an environment with no zarr at all; it is not expected to fire in
CI, and if it ever does, that is a finding about the environment and not a pass.
The pure arithmetic stays in ``test_ngff_region.py``, which needs nothing and
always runs; this file covers what only a served store can show.
"""

from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr", reason="entry-env dependency")
# NO fsspec guard. There used to be one, and by the time it was removed it was
# doing harm: the opener no longer reaches fsspec at all, so the guard could only
# SKIP these tests for a reason that had stopped being true — and a skipped guard
# and a passing one look the same in a CI summary.

import ngff_region as nr  # noqa: E402  (after the skip guard, on purpose)

SHAPE = (2, 512, 768)  # (c, y, x)
CHUNKS = (1, 128, 128)  # 1 x 4 x 6 = 24 chunks per channel, 48 total
ROLES = {"y": 1, "x": 2, "c": 0}


class _CountingHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the store and records every path asked for."""

    def log_message(self, *a):  # keep the test output readable
        pass

    def do_GET(self):  # noqa: N802
        type(self).seen.append(self.path)
        return super().do_GET()


@pytest.fixture
def served_store(tmp_path):
    """A zarr v2 array on disk, served over HTTP, with request accounting."""
    store_dir = tmp_path / "store"
    arr = zarr.open_array(
        store=str(store_dir / "0"),
        mode="w",
        shape=SHAPE,
        chunks=CHUNKS,
        dtype="uint16",
        zarr_format=2,
    )
    ground = np.arange(int(np.prod(SHAPE)), dtype=np.uint16).reshape(SHAPE) % 65535
    arr[:] = ground

    _CountingHandler.seen = []
    handler = functools.partial(_CountingHandler, directory=str(store_dir))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        yield {
            "root": f"http://127.0.0.1:{port}/",
            "ground": ground,
            "seen": _CountingHandler.seen,
        }
    finally:
        httpd.shutdown()
        httpd.server_close()


def _chunk_requests(seen):
    """Only the chunk objects, not .zarray / .zattrs / zarr.json metadata.

    A v2 chunk key is the chunk's grid coordinates joined by dots, "0.1.2", so a
    name made only of digits and dots is a chunk and anything else is metadata.
    """
    out = []
    for path in seen:
        name = path.rsplit("/", 1)[-1].split("?")[0]
        if not name or name.startswith(".") or name == "zarr.json":
            continue
        parts = name.split(".")
        if parts and all(part.isdigit() for part in parts):
            out.append(name)
    return out


def _probe():
    return {
        "dataset_paths": ["0"],
        "axis_roles": ROLES,
        "level0_shape": list(SHAPE),
    }


def test_the_default_opener_reads_a_real_store_over_http(served_store):
    """F1's red-maker: the path that ships, exercised.

    ``opener=None`` means the module's own ``_zarr_opener``. With ``zarr`` alone
    installed this raised ModuleNotFoundError: fsspec, so the entry could not
    read a single region in production while all 25 unit tests passed.
    """
    out = nr.read_region(
        {
            "kind": "ngff",
            "store_root": served_store["root"],
            "region": {"x": 300, "y": 100, "width": 64, "height": 32},
        },
        probe=_probe(),
        opener=None,
    )
    expect = served_store["ground"][:, 100:132, 300:364]
    # ground is (c, y, x); the reader returns HxWxC.
    assert out.shape == (32, 64, 2)
    assert np.array_equal(out, np.transpose(expect, (1, 2, 0)))


def test_only_the_chunks_the_region_touches_are_fetched(served_store):
    """F2's red-maker, and the claim of this whole change.

    The region 300..364 x 100..132 spans x-chunks 2 (256-383) and y-chunk 0
    (0-127) — one chunk column, one chunk row, times 2 channels = 2 chunks, out
    of 48 in the array. A reader that fetched the level and sliced locally would
    return the SAME VALUES and fail here, which is exactly the mutant the first
    version of this work could not detect.
    """
    nr.read_region(
        {
            "kind": "ngff",
            "store_root": served_store["root"],
            "region": {"x": 300, "y": 100, "width": 64, "height": 32},
        },
        probe=_probe(),
        opener=None,
    )
    chunks = _chunk_requests(served_store["seen"])
    total_chunks = 1
    for dim, chunk in zip(SHAPE, CHUNKS):
        total_chunks *= -(-dim // chunk)
    assert chunks, "no chunk was fetched at all; the read did not happen"
    assert len(set(chunks)) <= 4, (
        f"fetched {len(set(chunks))} distinct chunks {sorted(set(chunks))}; the "
        f"region intersects at most 2 (one per channel). A reader that pulls the "
        f"level and slices locally lands here."
    )
    assert len(set(chunks)) < total_chunks, (
        f"fetched every chunk ({total_chunks}); that is the full download this "
        f"module exists to avoid."
    )


def test_a_whole_level_read_of_a_big_level_is_refused_before_any_fetch(served_store):
    """F5's red-maker: omitting ``region`` must not become a full download.

    The cap is checked before the store is opened, so a refusal costs no request
    — asserted here, not assumed, by requiring the server saw nothing.
    """
    before = len(served_store["seen"])
    big = {
        "dataset_paths": ["0"],
        "axis_roles": ROLES,
        "level0_shape": [2, 20000, 20000],
    }
    with pytest.raises(nr.RegionSourceError, match="whole-level read"):
        nr.read_region(
            {"kind": "ngff", "store_root": served_store["root"]},
            probe=big,
            opener=None,
        )
    assert (
        len(served_store["seen"]) == before
    ), "the refusal fetched something; it must be decided from the probe alone"


def test_a_small_level_may_still_be_read_whole(served_store):
    """The cap must not forbid the ordinary case: this array is 512x768, far
    under it, so omitting the region still works."""
    out = nr.read_region(
        {"kind": "ngff", "store_root": served_store["root"]},
        probe=_probe(),
        opener=None,
    )
    assert out.shape == (512, 768, 2)
