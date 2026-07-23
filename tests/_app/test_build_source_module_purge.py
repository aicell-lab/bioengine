"""Unit test for ``_purge_stale_source_modules`` — the build-side purge that
keeps a warm Ray build process from serializing a previous version's cached
app-source module instead of the freshly synced on-disk source.

``introspect_app_in_ray_task`` and ``build_and_run_application`` run as Ray
tasks that may reuse a warm worker process. With pickle-by-value, the entry
class's code is captured at build time, so a stale cached module would bake old
code into the deployment (this served stale model-runner tagging after a KTH
worker cycle). The helper drops exactly the app-source modules so the next
import reads the new code, and leaves framework / stdlib modules alone.
"""
from __future__ import annotations

import importlib
import sys

from bioengine._app import bootstrap


def test_purge_drops_stale_source_and_reimports_fresh(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "entry.py").write_text("MARKER = 'v1'\n")

    monkeypatch.syspath_prepend(str(source))
    try:
        m = importlib.import_module("entry")
        assert m.MARKER == "v1"
        assert "entry" in sys.modules

        # Fresh source synced to the same path (as _ensure_source would do).
        import os
        import time

        f = source / "entry.py"
        f.write_text("MARKER = 'v2'\n")
        future = time.time() + 10
        os.utime(f, (future, future))

        bootstrap._purge_stale_source_modules(str(source))

        assert "entry" not in sys.modules
        assert "os" in sys.modules  # stdlib untouched

        m2 = importlib.import_module("entry")
        assert m2.MARKER == "v2"
    finally:
        sys.modules.pop("entry", None)


def test_purge_drops_stale_module_imported_from_a_different_path(tmp_path, monkeypatch):
    """The stale module may have been imported earlier under a DIFFERENT path
    (0.11.29-era Ray py_modules / a prior app_dir), so a path-only match misses
    it. The current source root defines the same top-level name, so the purge
    must drop it by name — otherwise Python keeps serving the cached copy."""
    old = tmp_path / "old_ray_pkg"
    old.mkdir()
    (old / "entry.py").write_text("MARKER = 'stale'\n")
    new_source = tmp_path / "app" / "source"
    new_source.mkdir(parents=True)
    (new_source / "entry.py").write_text("MARKER = 'fresh'\n")

    # Import the STALE copy from the old path (its __file__ is under old/).
    monkeypatch.syspath_prepend(str(old))
    try:
        stale = importlib.import_module("entry")
        assert stale.MARKER == "stale"

        # Build now has the fresh source on sys.path[0]; the stale 'entry' is
        # still cached with an old __file__ that the new source path doesn't
        # cover — a path-only purge would miss it.
        monkeypatch.syspath_prepend(str(new_source))
        bootstrap._purge_stale_source_modules(str(new_source))

        assert "entry" not in sys.modules
        assert importlib.import_module("entry").MARKER == "fresh"
    finally:
        sys.modules.pop("entry", None)


def test_purge_leaves_non_source_modules(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    # A framework module lives in site-packages, not under source/, and its
    # top-level name ('bioengine') is not one the app source root defines.
    assert "bioengine._app.bootstrap" in sys.modules
    bootstrap._purge_stale_source_modules(str(source))
    assert "bioengine._app.bootstrap" in sys.modules
