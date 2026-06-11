"""Unit tests for AppBuilder's content-hashed file:// runtime_env package
writer (:meth:`AppBuilder._write_pkg_to_runtime_env_dir` and helpers).

The whole point of this code path is to take the ray-client bridge out
of the package upload flow. These tests verify the invariants the deNBI
session and I aligned on:

* Content hash is deterministic across walk order.
* Same input → same output path → no re-write.
* Exclude patterns filter the right files.
* Atomic publish: only the final ``bioengine_pkg_<hash>.zip`` survives
  a successful write.
* Orphan ``*.tmp.<pid>`` files left by a crashed write are swept on
  AppBuilder construction.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from bioengine.apps.builder import AppBuilder


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path / "apps_workdir"


@pytest.fixture
def builder(workdir: Path) -> AppBuilder:
    return AppBuilder(apps_workdir=workdir)


@pytest.fixture
def pkg(tmp_path: Path) -> Path:
    """A representative app package: top-level py + sub-package + the
    kinds of files _PY_MODULES_EXCLUDES is supposed to drop."""
    root = tmp_path / "demo_pkg"
    root.mkdir()
    (root / "manifest.yaml").write_text("name: x\n")
    (root / "README.md").write_text("docs\n")
    (root / "tutorial.ipynb").write_text("{}\n")
    (root / "deployment.py").write_text("VALUE = 1\n")
    (root / "utils.py").write_text("def f(): pass\n")
    (root / "frontend").mkdir()
    (root / "frontend" / "index.html").write_text("<html></html>\n")
    sub = root / "runtimes"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    (sub / "a.py").write_text("VALUE = 2\n")
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "stale.pyc").write_bytes(b"\x00\x01")
    return root


def _zipped_entries(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return sorted(zf.namelist())


# ────────────────────────── exclude matcher ─────────────────────────────


@pytest.mark.parametrize(
    "rel,expected",
    [
        ("manifest.yaml", True),
        ("manifest.yml", True),
        ("README.md", True),
        ("README", True),
        ("notes.md", True),
        ("tutorial.ipynb", True),
        ("logo.png", True),
        ("frontend/index.html", True),
        ("frontend/static/app.js", True),
        ("__pycache__/x.pyc", True),
        (".git/HEAD", True),
        (".github/workflows/ci.yml", True),
        ("deployment.py", False),
        ("utils.py", False),
        ("runtimes/a.py", False),
        ("runtimes/__init__.py", False),
    ],
)
def test_is_excluded(rel: str, expected: bool) -> None:
    assert (
        AppBuilder._is_excluded(rel, AppBuilder._PY_MODULES_EXCLUDES)
        is expected
    )


# ─────────────────────── packaging end-to-end ───────────────────────────


def test_writes_zip_and_returns_file_uri(builder: AppBuilder, pkg: Path) -> None:
    uri = builder._write_pkg_to_runtime_env_dir(pkg)

    assert uri.startswith("file:///")
    out_path = Path(uri[len("file://"):])
    assert out_path.is_file()
    assert out_path.name.startswith("bioengine_pkg_")
    assert out_path.name.endswith(".zip")
    assert out_path.parent.name == "_runtime_env_packages"


def test_zip_excludes_non_python_content(
    builder: AppBuilder, pkg: Path
) -> None:
    uri = builder._write_pkg_to_runtime_env_dir(pkg)
    entries = _zipped_entries(Path(uri[len("file://"):]))

    assert "deployment.py" in entries
    assert "utils.py" in entries
    assert "runtimes/__init__.py" in entries
    assert "runtimes/a.py" in entries

    for excluded in (
        "manifest.yaml",
        "README.md",
        "tutorial.ipynb",
        "frontend/index.html",
        "__pycache__/stale.pyc",
    ):
        assert excluded not in entries, (
            f"{excluded!r} should not be shipped"
        )


def test_is_idempotent_no_rewrite(
    builder: AppBuilder, pkg: Path
) -> None:
    uri1 = builder._write_pkg_to_runtime_env_dir(pkg)
    path = Path(uri1[len("file://"):])
    mtime_before = path.stat().st_mtime_ns

    uri2 = builder._write_pkg_to_runtime_env_dir(pkg)
    assert uri2 == uri1
    assert path.stat().st_mtime_ns == mtime_before, (
        "Second call should detect existing file and skip the write"
    )


def test_different_content_produces_different_hash(
    builder: AppBuilder, pkg: Path
) -> None:
    uri1 = builder._write_pkg_to_runtime_env_dir(pkg)
    (pkg / "deployment.py").write_text("VALUE = 999\n")

    uri2 = builder._write_pkg_to_runtime_env_dir(pkg)
    assert uri2 != uri1
    assert Path(uri1[len("file://"):]).is_file()
    assert Path(uri2[len("file://"):]).is_file()


def test_rename_is_filename_not_path(
    builder: AppBuilder, pkg: Path
) -> None:
    """If the only difference between two materialisations is the
    source directory location, the resulting filename should be
    identical — hash lives in the filename, not in apps_workdir."""
    uri1 = builder._write_pkg_to_runtime_env_dir(pkg)

    moved = pkg.parent / "renamed_pkg"
    pkg.rename(moved)
    uri2 = builder._write_pkg_to_runtime_env_dir(moved)

    assert Path(uri1).name == Path(uri2).name


def test_atomic_publish_no_tmp_leftovers(
    builder: AppBuilder, pkg: Path
) -> None:
    builder._write_pkg_to_runtime_env_dir(pkg)
    pkg_dir = builder.apps_workdir / "_runtime_env_packages"

    leftover = list(pkg_dir.glob("*.tmp.*"))
    assert leftover == [], (
        f"Atomic write should leave no .tmp.<pid> files: found {leftover}"
    )


def test_raises_on_empty_input(
    builder: AppBuilder, tmp_path: Path
) -> None:
    empty = tmp_path / "empty_pkg"
    empty.mkdir()
    (empty / "README.md").write_text("only docs\n")

    with pytest.raises(RuntimeError, match="No Python source files"):
        builder._write_pkg_to_runtime_env_dir(empty)


# ─────────────────────────── startup sweep ──────────────────────────────


def test_sweep_removes_orphan_tmp_files(workdir: Path) -> None:
    pkg_dir = workdir / "_runtime_env_packages"
    pkg_dir.mkdir(parents=True)
    real = pkg_dir / "bioengine_pkg_aaaaaaaaaaaaaaaa.zip"
    real.write_bytes(b"real")
    orphan_a = pkg_dir / "bioengine_pkg_bbbbbbbbbbbbbbbb.zip.tmp.12345"
    orphan_a.write_bytes(b"partial")
    orphan_b = pkg_dir / "bioengine_pkg_cccccccccccccccc.zip.tmp.67890"
    orphan_b.write_bytes(b"partial")
    unrelated = pkg_dir / "some_other_file.txt"
    unrelated.write_bytes(b"not ours")

    AppBuilder(apps_workdir=workdir)

    assert real.is_file()
    assert unrelated.is_file()
    assert not orphan_a.exists()
    assert not orphan_b.exists()


def test_sweep_silent_when_dir_absent(workdir: Path) -> None:
    AppBuilder(apps_workdir=workdir)
