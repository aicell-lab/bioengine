"""Unit tests for ``@bioengine.health_check(depends_on=[...])``.

Three layers, none needing a live Ray cluster:

* decorator — ``depends_on`` is captured into the lifecycle dict, and the bare
  form keeps working;
* build-time — ``introspect_app`` rejects a ``depends_on`` name that isn't a
  deployment in the app's composition graph;
* runtime — the gate ``_make_check_health`` installs latches on first-ready and
  fails only on a definite post-latch zero, tolerating "unknown".
"""

import asyncio
import textwrap

import pytest

import bioengine
from bioengine._app import mixin
from bioengine._app.bootstrap import introspect_app
from bioengine._app.errors import UnknownDependencyError
from bioengine._app.mixin import _make_check_health


# ─────────────────────────── decorator capture ───────────────────────────


def test_bare_health_check_has_empty_depends_on():
    @bioengine.app()
    class App:
        @bioengine.health_check
        async def liveness(self):
            pass

    lc = App.func_or_class._bioengine_lifecycle
    assert lc["health_check"] == "liveness"
    assert lc["health_check_depends_on"] == []


def test_depends_on_captured_into_lifecycle():
    @bioengine.app()
    class App:
        @bioengine.health_check(depends_on=["RuntimeDeployment"])
        async def liveness(self):
            pass

    lc = App.func_or_class._bioengine_lifecycle
    assert lc["health_check"] == "liveness"
    assert lc["health_check_depends_on"] == ["RuntimeDeployment"]


def test_depends_on_must_be_strings():
    with pytest.raises(TypeError, match="list of deployment-name strings"):

        @bioengine.app()
        class App:
            @bioengine.health_check(depends_on=[object()])
            async def liveness(self):
                pass


# ───────────────────────── build-time validation ─────────────────────────


def _build_depends_pkg(tmp_path, monkeypatch, depends_on: str) -> str:
    """Write a two-deployment app whose entry depends_on ``depends_on`` and
    return its ``module:Class`` entry id."""
    pkg = tmp_path / "dep_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "runtime.py").write_text(
        textwrap.dedent(
            """
            import bioengine
            @bioengine.app(num_cpus=0)
            class Runtime:
                pass
            """
        )
    )
    (pkg / "entry.py").write_text(
        textwrap.dedent(
            f"""
            import bioengine
            from .runtime import Runtime
            @bioengine.app(num_cpus=0)
            class Entry:
                def __init__(self, runtime: Runtime):
                    self.runtime = runtime

                @bioengine.health_check(depends_on={depends_on!r})
                async def liveness(self):
                    pass
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import sys

    for name in ("dep_pkg", "dep_pkg.runtime", "dep_pkg.entry"):
        sys.modules.pop(name, None)
    return "dep_pkg.entry:Entry"


def test_depends_on_matching_deployment_is_accepted(tmp_path, monkeypatch):
    entry_id = _build_depends_pkg(tmp_path, monkeypatch, ["Runtime"])
    spec = introspect_app(entry_id)
    entry = spec["classes"][entry_id]
    assert entry["deployment_name"] == "Entry"
    assert entry["lifecycle_methods"]["health_check_depends_on"] == ["Runtime"]


def test_depends_on_unknown_deployment_rejected(tmp_path, monkeypatch):
    entry_id = _build_depends_pkg(tmp_path, monkeypatch, ["Runtimee"])
    with pytest.raises(UnknownDependencyError, match="Runtimee"):
        introspect_app(entry_id)


# ──────────────────────────── runtime gate ───────────────────────────────


class _FakeReplica:
    """Minimal stand-in carrying the ``_bioengine_*`` state the gate reads."""

    def __init__(self):
        self._bioengine_health_check_lock = asyncio.Lock()
        self._bioengine_replica_initialized = True
        self._bioengine_replica_test_failed = False
        self._bioengine_test_task = None
        self._bioengine_dep_seen_ready = {}
        self._bioengine_app_name = "app"


def _gate(monkeypatch, readings):
    """Install a ``_make_check_health`` gating on one dependency, backed by a
    fake replica-count reader yielding ``readings`` in order."""
    seq = iter(readings)

    async def fake_read(app_name, deployment_name, logger):
        return next(seq)

    monkeypatch.setattr(mixin, "_read_running_replicas", fake_read)
    lifecycle = {
        "async_init": None,
        "smoke_test": None,
        "health_check": None,
        "health_check_depends_on": ["Dep"],
    }
    return _make_check_health(_FakeReplica, lifecycle), _FakeReplica()


def test_unknown_reading_is_tolerated(monkeypatch):
    check_health, replica = _gate(monkeypatch, [None])
    asyncio.run(check_health(replica))
    assert replica._bioengine_dep_seen_ready.get("Dep") is None


def test_zero_before_ever_ready_is_tolerated(monkeypatch):
    check_health, replica = _gate(monkeypatch, [0])
    asyncio.run(check_health(replica))  # not latched → no raise
    assert not replica._bioengine_dep_seen_ready.get("Dep")


def test_zero_after_ready_raises(monkeypatch):
    check_health, replica = _gate(monkeypatch, [2, 0])

    async def drive():
        await check_health(replica)  # sees 2 → latches ready
        assert replica._bioengine_dep_seen_ready["Dep"] is True
        await check_health(replica)  # sees 0 after latch → unhealthy

    with pytest.raises(RuntimeError, match="no running replica"):
        asyncio.run(drive())
