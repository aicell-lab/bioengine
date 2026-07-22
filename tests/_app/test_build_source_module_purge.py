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


def test_purge_leaves_non_source_modules(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    # A framework module lives in site-packages, not under source/.
    assert "bioengine._app.bootstrap" in sys.modules
    bootstrap._purge_stale_source_modules(str(source))
    assert "bioengine._app.bootstrap" in sys.modules
