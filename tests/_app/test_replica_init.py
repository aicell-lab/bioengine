"""Unit tests for :mod:`bioengine._app.replica_init` — the replica-side
artifact materialisation that v0.11.4 substitutes for the worker's old
zip-and-ship-via-file:// path.

These tests exercise the pure-Python branches (exclude filtering, version
marker reuse, atomic source/ rename, HOME/TMPDIR/sys.path setup) without
needing a live Hypha server. The HTTP download path is monkey-patched so
the tests don't reach the network.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Tuple

import pytest

from bioengine._app import replica_init


def _make_artifact_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


@pytest.fixture
def fake_artifact_files() -> dict[str, bytes]:
    return {
        "entry.py": b"def main(): return 42\n",
        "subpkg/__init__.py": b"",
        "subpkg/util.py": b"X = 1\n",
        "config/settings.yaml": b"key: value\n",
        "manifest.yaml": b"name: My App\n",  # excluded
        "README.md": b"# excluded\n",  # excluded
        "tutorial.ipynb": b"{}",  # excluded
        "frontend/index.html": b"<html/>",  # excluded
        ".env": b"SECRET=1",  # excluded (hidden)
        "__pycache__/x.pyc": b"\x00",  # excluded
    }


@pytest.fixture
def patched_download(monkeypatch, fake_artifact_files):
    """Replace ``_download_and_extract`` so the test never touches the network.

    The fake writes a zip with the canned artifact bytes into the dest dir
    using the real extract+filter logic, so exclude semantics still get
    exercised.
    """
    captured: dict = {}

    def fake(url, token, dest, logger):
        captured["url"] = url
        captured["token"] = token
        zip_bytes = _make_artifact_zip(fake_artifact_files)
        tmp = dest.parent / f".source.new.{os.getpid()}"
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if replica_init._is_excluded(info.filename):
                    continue
                zf.extract(info, tmp)
        if dest.exists():
            import shutil
            shutil.rmtree(dest)
        os.replace(tmp, dest)

    monkeypatch.setattr(replica_init, "_download_and_extract", fake)
    return captured


@pytest.fixture
def replica_env(monkeypatch, tmp_path) -> Tuple[Path, dict]:
    app_dir = tmp_path / "bioimage-io-model-runner"
    env = {
        "BIOENGINE_APP_DIR": str(app_dir),
        "BIOENGINE_ARTIFACT_ID": "bioimage-io/model-runner",
        "BIOENGINE_ARTIFACT_VERSION": "1.2.0",
        "BIOENGINE_ARTIFACT_DOWNLOAD_URL": (
            "https://hypha.aicell.io/bioimage-io/artifacts/model-runner"
            "/create-zip-file?version=1.2.0"
        ),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("BIOENGINE_ARTIFACT_DOWNLOAD_TOKEN", raising=False)
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)
    return app_dir, env


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


def test_setup_creates_dirs_and_extends_sys_path(replica_env, patched_download) -> None:
    app_dir, env = replica_env

    replica_init.setup_replica_environment()

    source = app_dir / "source"
    assert source.is_dir()
    assert (source / "entry.py").is_file()
    assert (source / "subpkg" / "util.py").is_file()
    assert (source / "config" / "settings.yaml").is_file()
    assert not (source / "manifest.yaml").exists()
    assert not (source / "README.md").exists()
    assert not (source / "frontend").exists()
    assert not (source / "__pycache__").exists()

    assert (app_dir / "home").is_dir()
    assert (app_dir / "tmp").is_dir()
    assert os.environ["HOME"] == str(app_dir / "home")
    assert os.environ["TMPDIR"] == str(app_dir / "tmp")
    assert str(source) in sys.path

    marker = (app_dir / ".version").read_text().strip()
    assert marker == "1.2.0"


def test_second_setup_at_same_version_skips_download(
    replica_env, patched_download, monkeypatch
) -> None:
    app_dir, env = replica_env
    replica_init.setup_replica_environment()
    patched_download.clear()

    sentinel = app_dir / "source" / "marker.txt"
    sentinel.write_text("preserve me")

    download_calls = {"count": 0}
    original = replica_init._download_and_extract

    def counting(url, token, dest, logger):
        download_calls["count"] += 1
        return original(url, token, dest, logger)

    monkeypatch.setattr(replica_init, "_download_and_extract", counting)
    replica_init.setup_replica_environment()

    assert download_calls["count"] == 0
    assert sentinel.read_text() == "preserve me"


def test_version_change_triggers_redownload(
    replica_env, patched_download, monkeypatch
) -> None:
    app_dir, env = replica_env
    replica_init.setup_replica_environment()

    sentinel = app_dir / "source" / "sentinel.txt"
    sentinel.write_text("v1")

    monkeypatch.setenv("BIOENGINE_ARTIFACT_VERSION", "1.3.0")
    monkeypatch.setenv(
        "BIOENGINE_ARTIFACT_DOWNLOAD_URL",
        "https://hypha.aicell.io/bioimage-io/artifacts/model-runner"
        "/create-zip-file?version=1.3.0",
    )

    replica_init.setup_replica_environment()

    assert not sentinel.exists()
    assert (app_dir / "source" / "entry.py").is_file()
    assert (app_dir / ".version").read_text().strip() == "1.3.0"


def test_setup_is_noop_without_app_dir(monkeypatch) -> None:
    monkeypatch.delenv("BIOENGINE_APP_DIR", raising=False)
    replica_init.setup_replica_environment()  # must not raise


def test_setup_raises_when_version_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(tmp_path / "app"))
    monkeypatch.delenv("BIOENGINE_ARTIFACT_VERSION", raising=False)
    with pytest.raises(RuntimeError, match="BIOENGINE_ARTIFACT_VERSION"):
        replica_init.setup_replica_environment()


def test_local_artifact_path_shortcircuits_download(
    replica_env, monkeypatch, tmp_path, fake_artifact_files
) -> None:
    app_dir, env = replica_env
    local_root = tmp_path / "local-dev"
    artifact_subdir = local_root / "model-runner"
    artifact_subdir.mkdir(parents=True)
    for rel, content in fake_artifact_files.items():
        out = artifact_subdir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)

    monkeypatch.setenv("BIOENGINE_LOCAL_ARTIFACT_PATH", str(local_root))

    def boom(*a, **kw):
        raise AssertionError("download must not be called when local path exists")

    monkeypatch.setattr(replica_init, "_download_and_extract", boom)

    replica_init.setup_replica_environment()
    assert (app_dir / "source" / "entry.py").is_file()
    assert (app_dir / ".version").read_text().strip() == "1.2.0"


def test_authed_url_appends_token_via_query_string() -> None:
    base = "https://hypha.aicell.io/ws/artifacts/x/create-zip-file?version=1"
    assert replica_init._authed_url(base, "abc 123").endswith(
        "&token=abc%20123"
    )
    no_query = "https://hypha.aicell.io/ws/artifacts/x/create-zip-file"
    assert replica_init._authed_url(no_query, "abc").endswith("?token=abc")


def test_authed_url_returns_base_when_no_token() -> None:
    base = "https://hypha.aicell.io/ws/artifacts/x/create-zip-file?version=1"
    assert replica_init._authed_url(base, None) == base
    assert replica_init._authed_url(base, "") == base
