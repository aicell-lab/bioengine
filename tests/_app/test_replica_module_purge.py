"""Unit test for ``_purge_stale_app_modules`` — the per-replica sys.modules
purge that lets a redeploy on a REUSED Ray worker process pick up freshly
synced source instead of a prior deploy's cached modules.

The end-to-end trigger (a fresh replica landing on a warm worker process) only
occurs when the runtime_env is stable across deploys — e.g. a public artifact
that mints no download token, on a single-slot GPU that Ray keeps warm. That is
hard to reproduce off a real GPU cluster, so we test the purge's contract
directly: given a stale app-source module in ``sys.modules`` and fresh source on
disk, the purge drops exactly the app-source modules so the next import reads
the new code, and leaves framework / stdlib modules untouched.
"""
from __future__ import annotations

import importlib
import logging
import sys

from bioengine._app import mixin


def test_purge_drops_app_source_modules_and_reimports_fresh(tmp_path, monkeypatch):
    app_dir = tmp_path / "app"
    source = app_dir / "source"
    source.mkdir(parents=True)
    (source / "probemod.py").write_text("VALUE = 'v1'\n")

    monkeypatch.setenv("BIOENGINE_APP_DIR", str(app_dir))
    monkeypatch.syspath_prepend(str(source))
    try:
        m = importlib.import_module("probemod")
        assert m.VALUE == "v1"
        assert "probemod" in sys.modules

        # Source changes on disk (as the per-file Hypha sync would do). Advance
        # the mtime so the cached .pyc is invalidated — the real sync writes the
        # changed file with a fresh (later) mtime; without this the sub-second
        # rewrite in the test would keep the v1 bytecode.
        import os
        import time

        probe = source / "probemod.py"
        probe.write_text("VALUE = 'v2'\n")
        future = time.time() + 10
        os.utime(probe, (future, future))

        mixin._purge_stale_app_modules(logging.getLogger("test"))

        # The stale app module is gone; a stdlib module is untouched.
        assert "probemod" not in sys.modules
        assert "os" in sys.modules

        # Re-import now reads the new source.
        m2 = importlib.import_module("probemod")
        assert m2.VALUE == "v2"
    finally:
        sys.modules.pop("probemod", None)


def test_purge_noop_without_app_dir(monkeypatch):
    monkeypatch.delenv("BIOENGINE_APP_DIR", raising=False)
    # Must not raise and must not touch sys.modules.
    before = set(sys.modules)
    mixin._purge_stale_app_modules(logging.getLogger("test"))
    assert set(sys.modules) == before


def test_purge_leaves_non_source_modules(tmp_path, monkeypatch):
    """A module whose file is outside <app_dir>/source is never purged."""
    app_dir = tmp_path / "app"
    (app_dir / "source").mkdir(parents=True)
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(app_dir))
    # bioengine.* modules live in site-packages, not app_dir/source.
    assert "bioengine._app.mixin" in sys.modules
    mixin._purge_stale_app_modules(logging.getLogger("test"))
    assert "bioengine._app.mixin" in sys.modules
