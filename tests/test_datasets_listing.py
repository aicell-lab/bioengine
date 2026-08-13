"""Unit tests for Zarr store listing against the data server's catalog route.

Listing is what makes a plain (non-OME) Zarr hierarchy discoverable: its child
names are declared nowhere, so a reader can only find them by enumerating. The
data server answers that with ``recursive=false``; these tests stub the
transport and check the key mapping, which is the part that is easy to get
subtly wrong.
"""

import asyncio

import pytest

from bioengine.datasets.http_zarr_store import HttpZarrStore


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def stub_transport(monkeypatch, payload, status_code=200):
    """Replace the HTTP call and record the params the store sent."""
    seen = {}

    async def fake_get(url, params=None, **kwargs):
        seen["url"] = url
        seen["params"] = params
        return FakeResponse(payload, status_code)

    monkeypatch.setattr(
        "bioengine.datasets.http_zarr_store.get_url_with_retry", fake_get
    )
    return seen


def bioengine_store():
    return HttpZarrStore(
        base_url="http://server/data/ds1/plain.zarr",
        service_url="http://server",
        dataset_id="ds1",
        zarr_path="plain.zarr",
    )


def drain(aiterator):
    async def _collect():
        return [item async for item in aiterator]

    return asyncio.run(_collect())


# ---------------------------------------------------------------------------
# supports_listing is per-instance
# ---------------------------------------------------------------------------


def test_bioengine_store_supports_listing():
    assert bioengine_store().supports_listing is True


def test_external_root_does_not_support_listing():
    """Plain HTTP has no way to enumerate keys, so don't claim it does."""
    store = HttpZarrStore(base_url="https://example.org/public.zarr")
    assert store.supports_listing is False


def test_external_root_listing_raises():
    store = HttpZarrStore(base_url="https://example.org/public.zarr")
    with pytest.raises(NotImplementedError):
        drain(store.list_dir(""))


# ---------------------------------------------------------------------------
# Key mapping: server answers relative to the dataset root, zarr wants
# keys relative to the store root
# ---------------------------------------------------------------------------


def test_list_dir_yields_bare_child_names(monkeypatch):
    stub_transport(
        monkeypatch,
        ["plain.zarr/alpha/", "plain.zarr/beta/", "plain.zarr/zarr.json"],
    )
    assert drain(bioengine_store().list_dir("")) == ["alpha", "beta", "zarr.json"]


def test_list_dir_under_a_prefix_is_relative_to_it(monkeypatch):
    stub_transport(monkeypatch, ["plain.zarr/nested/gamma/", "plain.zarr/nested/zarr.json"])
    assert drain(bioengine_store().list_dir("nested")) == ["gamma", "zarr.json"]


def test_list_dir_requests_one_level_of_the_right_directory(monkeypatch):
    seen = stub_transport(monkeypatch, [])
    drain(bioengine_store().list_dir("nested"))
    assert seen["url"] == "http://server/datasets/ds1/files"
    assert seen["params"] == {"dir_path": "plain.zarr/nested", "recursive": "false"}


def test_list_prefix_keeps_the_prefix_on_each_key(monkeypatch):
    stub_transport(monkeypatch, ["plain.zarr/nested/gamma/zarr.json"])
    assert drain(bioengine_store().list_prefix("nested")) == ["nested/gamma/zarr.json"]


def test_list_walks_from_the_store_root(monkeypatch):
    seen = stub_transport(monkeypatch, ["plain.zarr/zarr.json", "plain.zarr/alpha/zarr.json"])
    assert drain(bioengine_store().list()) == ["zarr.json", "alpha/zarr.json"]
    assert seen["params"] == {"dir_path": "plain.zarr", "recursive": "true"}


def test_missing_prefix_lists_empty_rather_than_raising(monkeypatch):
    """Zarr probes for optional members; a 400 means absent, not broken."""
    stub_transport(monkeypatch, {"detail": "does not exist"}, status_code=400)
    assert drain(bioengine_store().list_dir("nope")) == []
