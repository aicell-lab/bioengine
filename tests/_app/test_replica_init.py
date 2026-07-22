"""Unit tests for :mod:`bioengine._app.replica_init` — the replica-side
artifact materialisation (per-file incremental sync from Hypha).

These tests exercise the pure-Python branches (exclude filtering, snapshot
diffing, deletion of removed files, HOME/TMPDIR/sys.path setup) without a live
Hypha server. The network-facing helpers are either monkey-patched at the
``_list_source_files`` / ``_download_file`` seam (sync-logic tests) or driven
through a fake ``urlopen`` (listing/download tests).
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import pytest

from bioengine._app import replica_init

#: Canned artifact: ``{relpath: (content, last_modified)}``.
_ARTIFACT = {
    "entry.py": (b"def main(): return 42\n", "2026-01-01T00:00:00"),
    "subpkg/__init__.py": (b"", "2026-01-01T00:00:00"),
    "subpkg/util.py": (b"X = 1\n", "2026-01-01T00:00:00"),
    "config/settings.yaml": (b"key: value\n", "2026-01-01T00:00:00"),
}


@pytest.fixture
def fake_hypha(monkeypatch):
    """Fake ``_list_source_files`` + ``_download_file``.

    Returns a controller with a mutable ``remote`` map (``{rel: (content, lm)}``)
    and a ``downloads`` list recording every relpath fetched, so tests can
    mutate the remote between syncs and assert exactly what got re-downloaded.
    """
    remote = dict(_ARTIFACT)
    downloads: list = []

    def fake_list(files_url, version, token, logger):
        return {rel: lm for rel, (_content, lm) in remote.items()}

    def fake_download(files_url, relpath, version, token, dest, logger):
        downloads.append(relpath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(remote[relpath][0])

    monkeypatch.setattr(replica_init, "_list_source_files", fake_list)
    monkeypatch.setattr(replica_init, "_download_file", fake_download)
    return {"remote": remote, "downloads": downloads}


@pytest.fixture
def replica_env(monkeypatch, tmp_path) -> Tuple[Path, dict]:
    app_dir = tmp_path / "bioimage-io-model-runner"
    env = {
        "BIOENGINE_APP_DIR": str(app_dir),
        "BIOENGINE_ARTIFACT_ID": "bioimage-io/model-runner",
        "BIOENGINE_ARTIFACT_VERSION": "1.2.0",
        "BIOENGINE_ARTIFACT_FILES_URL": (
            "https://hypha.aicell.io/bioimage-io/artifacts/model-runner/files"
        ),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN", raising=False)
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)
    return app_dir, env


# ───────────────────────────── exclude filter ─────────────────────────────


def test_is_excluded_drops_docs_frontend_and_dotfiles() -> None:
    assert replica_init._is_excluded("manifest.yaml")
    assert replica_init._is_excluded("README.md")
    assert replica_init._is_excluded("tutorial.ipynb")
    assert replica_init._is_excluded("frontend/index.html")
    assert replica_init._is_excluded("__pycache__/x.pyc")
    assert replica_init._is_excluded(".env")
    assert replica_init._is_excluded(".git/config")


def test_is_excluded_keeps_python_and_config() -> None:
    assert not replica_init._is_excluded("entry.py")
    assert not replica_init._is_excluded("subpkg/util.py")
    assert not replica_init._is_excluded("config/settings.yaml")
    assert not replica_init._is_excluded("model.json")


# ───────────────────────────── setup + sync ───────────────────────────────


def test_setup_creates_dirs_and_extends_sys_path(replica_env, fake_hypha) -> None:
    app_dir, _env = replica_env

    replica_init.setup_replica_environment()

    source = app_dir / "source"
    assert (source / "entry.py").is_file()
    assert (source / "subpkg" / "util.py").is_file()
    assert (source / "config" / "settings.yaml").is_file()

    assert (app_dir / "home").is_dir()
    assert (app_dir / "tmp").is_dir()
    assert os.environ["HOME"] == str(app_dir / "home")
    assert os.environ["TMPDIR"] == str(app_dir / "tmp")
    assert str(source) in sys.path

    snapshot = json.loads((app_dir / ".source_snapshot.json").read_text())
    assert set(snapshot) == set(_ARTIFACT)
    assert (
        app_dir / ".version"
    ).read_text().strip() == "bioimage-io/model-runner@1.2.0"


def test_initial_sync_downloads_all(replica_env, fake_hypha) -> None:
    replica_init.setup_replica_environment()
    assert sorted(fake_hypha["downloads"]) == sorted(_ARTIFACT)


def test_unchanged_timestamps_skip_download(replica_env, fake_hypha) -> None:
    replica_init.setup_replica_environment()
    fake_hypha["downloads"].clear()

    replica_init.setup_replica_environment()  # nothing changed remotely
    assert fake_hypha["downloads"] == []


def test_changed_timestamp_redownloads_only_that_file(
    replica_env, fake_hypha
) -> None:
    app_dir, _env = replica_env
    replica_init.setup_replica_environment()
    fake_hypha["downloads"].clear()

    fake_hypha["remote"]["entry.py"] = (
        b"def main(): return 99\n",
        "2026-02-02T00:00:00",
    )
    replica_init.setup_replica_environment()

    assert fake_hypha["downloads"] == ["entry.py"]
    assert (app_dir / "source" / "entry.py").read_bytes() == b"def main(): return 99\n"


def test_removed_file_deleted_locally(replica_env, fake_hypha) -> None:
    app_dir, _env = replica_env
    replica_init.setup_replica_environment()
    assert (app_dir / "source" / "subpkg" / "util.py").is_file()

    del fake_hypha["remote"]["subpkg/util.py"]
    replica_init.setup_replica_environment()

    assert not (app_dir / "source" / "subpkg" / "util.py").exists()
    assert (app_dir / "source" / "entry.py").is_file()


def test_missing_local_file_redownloaded(replica_env, fake_hypha) -> None:
    """A file the snapshot claims is current is re-fetched if gone from disk."""
    app_dir, _env = replica_env
    replica_init.setup_replica_environment()
    fake_hypha["downloads"].clear()

    (app_dir / "source" / "entry.py").unlink()
    replica_init.setup_replica_environment()

    assert "entry.py" in fake_hypha["downloads"]
    assert (app_dir / "source" / "entry.py").is_file()


def test_setup_is_noop_without_app_dir(monkeypatch) -> None:
    monkeypatch.delenv("BIOENGINE_APP_DIR", raising=False)
    replica_init.setup_replica_environment()  # must not raise


def test_setup_raises_when_version_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(tmp_path / "app"))
    monkeypatch.delenv("BIOENGINE_ARTIFACT_VERSION", raising=False)
    with pytest.raises(RuntimeError, match="BIOENGINE_ARTIFACT_VERSION"):
        replica_init.setup_replica_environment()


def test_missing_files_url_raises(monkeypatch, tmp_path) -> None:
    app_dir = tmp_path / "app"
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(app_dir))
    monkeypatch.setenv("BIOENGINE_ARTIFACT_VERSION", "1.2.0")
    monkeypatch.delenv("BIOENGINE_ARTIFACT_FILES_URL", raising=False)
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)
    monkeypatch.delenv("BIOENGINE_ARTIFACT_ID", raising=False)
    with pytest.raises(RuntimeError, match="BIOENGINE_ARTIFACT_FILES_URL"):
        replica_init.setup_replica_environment()


def test_local_artifact_path_shortcircuits_sync(
    replica_env, monkeypatch, tmp_path
) -> None:
    app_dir, _env = replica_env
    local_root = tmp_path / "local-dev"
    artifact_subdir = local_root / "model-runner"
    artifact_subdir.mkdir(parents=True)
    for rel, (content, _lm) in _ARTIFACT.items():
        out = artifact_subdir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)

    monkeypatch.setenv("BIOENGINE_LOCAL_ARTIFACT_PATH", str(local_root))

    def boom(*a, **kw):
        raise AssertionError("Hypha sync must not run when local path exists")

    monkeypatch.setattr(replica_init, "_sync_source_from_hypha", boom)

    replica_init.setup_replica_environment()
    assert (app_dir / "source" / "entry.py").is_file()
    assert (
        app_dir / ".version"
    ).read_text().strip() == "bioimage-io/model-runner@1.2.0"


# ───────────────────────── network-facing helpers ─────────────────────────


class _FakeResp:
    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *a) -> bool:
        return False


def test_list_source_files_walks_and_excludes(monkeypatch) -> None:
    tree = {
        "": [
            {"name": "entry.py", "type": "file", "size": 10, "last_modified": "t1"},
            {"name": "manifest.yaml", "type": "file", "size": 5, "last_modified": "t1"},
            {"name": "subpkg", "type": "directory"},
            {"name": "frontend", "type": "directory"},
        ],
        "subpkg/": [
            {"name": "util.py", "type": "file", "size": 3, "last_modified": "t2"},
        ],
    }

    def fake_urlopen(url, timeout=None):
        after = url.split("?")[0].split("/files", 1)[1].lstrip("/")
        return _FakeResp(json.dumps(tree.get(after, [])).encode())

    monkeypatch.setattr(replica_init.urllib.request, "urlopen", fake_urlopen)
    files = replica_init._list_source_files(
        "https://h/ws/artifacts/x/files", "1.0", None, logging.getLogger("t")
    )
    # manifest.yaml + frontend/ excluded; subpkg walked recursively.
    assert files == {"entry.py": "t1", "subpkg/util.py": "t2"}


def test_download_file_writes_content_and_builds_url(monkeypatch, tmp_path) -> None:
    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"] = url
        return _FakeResp(b"hello world")

    monkeypatch.setattr(replica_init.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "a" / "b" / "f.py"
    replica_init._download_file(
        "https://h/ws/artifacts/x/files",
        "a/b/f.py",
        "1.0",
        "tok",
        dest,
        logging.getLogger("t"),
    )
    assert dest.read_bytes() == b"hello world"
    assert seen["url"].startswith("https://h/ws/artifacts/x/files/a/b/f.py?")
    assert "version=1.0" in seen["url"] and "token=tok" in seen["url"]


def test_artifact_url_appends_version_and_token() -> None:
    files = "https://h/ws/artifacts/x/files"
    url = replica_init._artifact_url(files, "entry.py", "1.2.0", "abc 123")
    assert url.startswith("https://h/ws/artifacts/x/files/entry.py?")
    assert "version=1.2.0" in url
    assert "token=abc%20123" in url


def test_artifact_url_public_no_token() -> None:
    files = "https://h/ws/artifacts/x/files"
    url = replica_init._artifact_url(files, "entry.py", "1.2.0", None)
    assert url == "https://h/ws/artifacts/x/files/entry.py?version=1.2.0"
    assert "token" not in url
