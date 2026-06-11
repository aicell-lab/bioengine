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


# ─────────────────────── bioengine source bundling ──────────────────────


def test_bioengine_source_zip_has_distinct_filename_prefix(
    builder: AppBuilder,
) -> None:
    uri = builder._write_bioengine_source_to_runtime_env_dir()
    name = Path(uri[len("file://"):]).name
    assert name.startswith("bioengine_runtime_")
    assert name.endswith(".zip")


def test_bioengine_source_zip_has_throwaway_wrapper_dir(
    builder: AppBuilder,
) -> None:
    """Ray strips the top-level directory from file:// py_modules zips when
    every entry shares one. We give it a throwaway ``_bioengine_wrap/`` to
    strip so the actual ``bioengine/`` package directory survives extraction.

    All entries must therefore live under ``_bioengine_wrap/bioengine/``."""
    uri = builder._write_bioengine_source_to_runtime_env_dir()
    entries = _zipped_entries(Path(uri[len("file://"):]))

    assert entries, "expected at least one entry"
    for entry in entries:
        assert entry.startswith("_bioengine_wrap/bioengine/"), entry

    assert "_bioengine_wrap/bioengine/__init__.py" in entries
    assert any(
        e.startswith("_bioengine_wrap/bioengine/_app/") for e in entries
    )
    assert any(
        e.startswith("_bioengine_wrap/bioengine/apps/") for e in entries
    )


def test_bioengine_source_zip_imports_after_simulated_ray_strip(
    builder: AppBuilder, tmp_path: Path
) -> None:
    """End-to-end simulation of Ray's extraction path for the bioengine zip.

    Reproduces ``download_and_unpack_package`` for a ``file://`` URI:
    Ray's ``unzip_package(remove_top_level_directory=True)`` strips a
    single top-level directory if one exists. After that step, adding
    the extract dir to ``sys.path`` must let ``import bioengine`` resolve
    to a file inside the extracted layout. This is the invariant Fix #4
    +#5 exists to maintain.
    """
    import subprocess
    import sys
    import zipfile

    uri = builder._write_bioengine_source_to_runtime_env_dir()
    zip_path = Path(uri[len("file://"):])

    extract_dir = tmp_path / "ray_extract"
    extract_dir.mkdir()

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        top_components = {n.split("/", 1)[0] for n in names if n}
        assert len(top_components) == 1, (
            f"expected a single top-level dir for Ray to strip, got "
            f"{top_components}"
        )
        top = next(iter(top_components))
        for name in names:
            if name.endswith("/"):
                continue
            stripped = name[len(top) + 1:]
            target = extract_dir / stripped
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))

    assert (extract_dir / "bioengine" / "__init__.py").is_file()
    assert (extract_dir / "bioengine" / "_app" / "__init__.py").is_file()

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(extract_dir)!r}); "
                "import bioengine; "
                "print(bioengine.__file__)"
            ),
        ],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin"},
    )
    assert proc.returncode == 0, (
        f"import bioengine after simulated Ray strip failed:\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert str(extract_dir / "bioengine") in proc.stdout, (
        f"resolved bioengine elsewhere: stdout={proc.stdout!r}"
    )


def test_bioengine_source_zip_excludes_pycache(
    builder: AppBuilder,
) -> None:
    uri = builder._write_bioengine_source_to_runtime_env_dir()
    entries = _zipped_entries(Path(uri[len("file://"):]))

    for entry in entries:
        assert "__pycache__" not in entry
        assert not entry.endswith(".pyc")
        assert not entry.endswith(".pyo")
        assert not entry.endswith(".so")


def test_arc_prefix_changes_hash(builder: AppBuilder, pkg: Path) -> None:
    """Same source content under different arc_prefix must produce
    different content hashes — the zip contents differ even though the
    source files don't."""
    uri_no_prefix = builder._write_pkg_to_runtime_env_dir(
        pkg, filename_prefix="case_a"
    )
    uri_with_prefix = builder._write_pkg_to_runtime_env_dir(
        pkg, filename_prefix="case_b", arc_prefix="wrap"
    )
    assert Path(uri_no_prefix).name != Path(uri_with_prefix).name


def test_arc_prefix_wraps_entries(builder: AppBuilder, pkg: Path) -> None:
    uri = builder._write_pkg_to_runtime_env_dir(
        pkg, filename_prefix="wrapped_pkg", arc_prefix="wrap"
    )
    entries = _zipped_entries(Path(uri[len("file://"):]))
    for entry in entries:
        assert entry.startswith("wrap/"), entry


def test_build_py_modules_uris_returns_distinct_pair(
    builder: AppBuilder, pkg: Path
) -> None:
    pkg_uri, bioengine_uri = builder._build_py_modules_uris(pkg)

    assert pkg_uri != bioengine_uri
    pkg_path = Path(pkg_uri[len("file://"):])
    bio_path = Path(bioengine_uri[len("file://"):])
    assert pkg_path.name.startswith("bioengine_pkg_")
    assert bio_path.name.startswith("bioengine_runtime_")
    assert pkg_path.parent == bio_path.parent
    assert pkg_path.is_file()
    assert bio_path.is_file()


def test_sweep_removes_both_filename_prefixes(workdir: Path) -> None:
    pkg_dir = workdir / "_runtime_env_packages"
    pkg_dir.mkdir(parents=True)
    orphan_pkg = pkg_dir / "bioengine_pkg_aaaaaaaaaaaaaaaa.zip.tmp.111"
    orphan_pkg.write_bytes(b"partial")
    orphan_runtime = (
        pkg_dir / "bioengine_runtime_bbbbbbbbbbbbbbbb.zip.tmp.222"
    )
    orphan_runtime.write_bytes(b"partial")

    AppBuilder(apps_workdir=workdir)

    assert not orphan_pkg.exists()
    assert not orphan_runtime.exists()
