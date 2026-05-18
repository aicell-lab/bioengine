"""Unit tests for BioEngineDatasets lazy data-server discovery.

Regression test for a bug where a deployment that received a BioEngineDatasets
instance before the data server was running stayed permanently disconnected
even after the data server wrote its discovery file. See the changelog for
0.9.3 for the full incident.

The tests do not require a running data server — they exercise only the
client-side resolution logic by pointing `_CURRENT_SERVER_FILE` at a tmp
path and writing / deleting it directly.
"""

from pathlib import Path

import pytest

from bioengine.datasets import BioEngineDatasets


@pytest.fixture
def discovery_file(tmp_path, monkeypatch):
    """Redirect the class-level discovery file path to a tmp location."""
    path = tmp_path / "bioengine_current_server"
    monkeypatch.setattr(BioEngineDatasets, "_CURRENT_SERVER_FILE", path)
    return path


def test_auto_no_file_yet_leaves_service_url_none(discovery_file):
    """If the discovery file does not exist at construction, service_url is None."""
    client = BioEngineDatasets(data_server_url="auto")
    assert client.service_url is None
    assert client.http_client is None


@pytest.mark.asyncio
async def test_list_datasets_returns_empty_when_no_server_yet(discovery_file):
    """Without a data server, list_datasets returns {} (current behaviour preserved)."""
    client = BioEngineDatasets(data_server_url="auto")
    assert await client.list_datasets() == {}


@pytest.mark.asyncio
async def test_list_datasets_resolves_after_file_appears(discovery_file):
    """The fix: list_datasets re-reads the discovery file when service_url is None."""
    client = BioEngineDatasets(data_server_url="auto")
    assert client.service_url is None

    # Data server "starts" after the client was constructed and writes its URL.
    discovery_file.parent.mkdir(parents=True, exist_ok=True)
    discovery_file.write_text("http://example.invalid:9000\n")

    # Force resolution without making a real HTTP request by triggering the
    # lazy path directly; list_datasets() would also do this but then it
    # would hit the network. We assert the resolution side-effect.
    client._resolve_service_url()
    assert client.service_url == "http://example.invalid:9000"
    assert client.http_client is not None


@pytest.mark.asyncio
async def test_refresh_picks_up_changed_url(discovery_file):
    """refresh() drops the old client and re-resolves from the discovery file."""
    discovery_file.parent.mkdir(parents=True, exist_ok=True)
    discovery_file.write_text("http://first.invalid:9000")

    client = BioEngineDatasets(data_server_url="auto")
    assert client.service_url == "http://first.invalid:9000"
    first_client = client.http_client

    discovery_file.write_text("http://second.invalid:9001")
    new_url = await client.refresh()
    assert new_url == "http://second.invalid:9001"
    assert client.service_url == "http://second.invalid:9001"
    # New httpx.AsyncClient created; old one is closed and replaced.
    assert client.http_client is not first_client


@pytest.mark.asyncio
async def test_explicit_none_stays_disabled_after_resolve(discovery_file):
    """data_server_url=None means "disable remote" — never auto-discover."""
    discovery_file.parent.mkdir(parents=True, exist_ok=True)
    discovery_file.write_text("http://example.invalid:9000")

    client = BioEngineDatasets(data_server_url=None)
    assert client.service_url is None

    # Even with the discovery file present, calling list_datasets does not
    # opt the client into remote access; the user said None and meant None.
    assert await client.list_datasets() == {}
    assert client.service_url is None


@pytest.mark.asyncio
async def test_explicit_url_does_not_re_resolve(discovery_file):
    """A literal URL passed at construction is honored; subsequent calls don't re-read the file."""
    discovery_file.parent.mkdir(parents=True, exist_ok=True)
    discovery_file.write_text("http://discovered.invalid:9000")

    client = BioEngineDatasets(data_server_url="http://explicit.invalid:9000")
    assert client.service_url == "http://explicit.invalid:9000"

    # Even after the discovery file changes, the client keeps the explicit URL.
    discovery_file.write_text("http://changed.invalid:9001")
    client._resolve_service_url()
    assert client.service_url == "http://explicit.invalid:9000"
