"""Unit tests for data-server authentication and path handling.

Covers the move from the ``?token=`` query parameter to an
``Authorization: Bearer`` header, and the traversal guard on the routes that
join a client-supplied path onto a dataset directory.

No running data server is needed — the server-side helpers are pure functions
and the client-side assertions only inspect the request that would be sent.
"""

from pathlib import Path

import pytest
from fastapi import HTTPException

from bioengine.datasets import BioEngineDatasets
from bioengine.datasets.http_zarr_store import HttpZarrStore
from bioengine.datasets.proxy_server import resolve_token, safe_join


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


def test_bearer_header_is_used():
    assert resolve_token("Bearer abc123", None) == "abc123"


def test_bearer_header_wins_over_query_param():
    assert resolve_token("Bearer abc123", "legacy") == "abc123"


def test_query_param_still_accepted():
    """Deprecated, but kept so an older client can talk to a newer server."""
    assert resolve_token(None, "legacy") == "legacy"


@pytest.mark.parametrize("header", ["", "Basic abc123", "Bearer", "Bearer   "])
def test_non_bearer_headers_fall_through(header):
    assert resolve_token(header, "legacy") == "legacy"
    assert resolve_token(header, None) is None


# ---------------------------------------------------------------------------
# Traversal guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["../secret", "a/../../secret", "/etc/passwd", "a/b/../../../etc/passwd", ""],
)
def test_safe_join_rejects_traversal(path, tmp_path):
    with pytest.raises(HTTPException) as excinfo:
        safe_join(path, tmp_path)
    assert excinfo.value.status_code == 400


@pytest.mark.parametrize("path", ["a.zarr/zarr.json", "nested/dir/c/0/0", "file.txt"])
def test_safe_join_allows_normal_paths(path, tmp_path):
    assert safe_join(path, tmp_path) == tmp_path / path


def test_safe_join_permits_symlinked_data(tmp_path):
    """Dataset dirs legitimately symlink out to other mounts — don't resolve()."""
    external = tmp_path / "external"
    external.mkdir()
    (external / "img.bin").write_bytes(b"x")
    base = tmp_path / "dataset"
    base.mkdir()
    (base / "link").symlink_to(external)

    joined = safe_join("link/img.bin", base)
    assert joined.read_bytes() == b"x"


# ---------------------------------------------------------------------------
# Client sends the header, not a query param
# ---------------------------------------------------------------------------


def test_zarr_store_keeps_token_out_of_the_url():
    """A query string would break key joining: '...zarr?token=x' + '/zarr.json'."""
    store = HttpZarrStore(base_url="http://server/data/ds/a.zarr", token="secret")
    url = store._build_url("zarr.json")
    assert url == "http://server/data/ds/a.zarr/zarr.json"
    assert "secret" not in url


def test_zarr_store_builds_bearer_header():
    store = HttpZarrStore(base_url="http://server/data/ds/a.zarr", token="secret")
    assert store._auth_headers == {"Authorization": "Bearer secret"}


def test_zarr_store_without_token_sends_no_auth_header():
    store = HttpZarrStore(base_url="http://server/data/ds/a.zarr")
    assert store._auth_headers == {}


def test_client_auth_headers():
    client = BioEngineDatasets(data_server_url=None, hypha_token="tok")
    assert client._auth_headers("tok") == {"Authorization": "Bearer tok"}
    assert client._auth_headers(None) == {}
