"""A GPU runtime that cannot write its run store says so, and refuses up front.

The live witness: a CellposeRuntime replica whose HOME had never been
redirected failed a training write while every deployment read HEALTHY. The
entry-side probe does not cover it — a runtime replica is a different process
with its own HOME, which is why one replica could write the share while another
in the same image could not.

WHY NO HEALTH CHECK, pinned below as a test rather than left as a comment: a
failing Serve health check RESTARTS the replica, so reporting this through the
health check would crash-loop the GPU runtimes under a cluster-side fault the
app cannot repair — and tonight's lasted hours. Any "fail after N attempts"
threshold is a bet on that duration. A probe that never fails the check cannot
be tuned wrong, and the fault is surfaced instead through `ping`/`store_status`
and refused at the door.

The row's clause 3 is the control, and it is load-bearing: with a writable
store nothing may change, because a check that refused everything would satisfy
clause 1 on its own.
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest

import training


def _runtime_classes():
    """The two runtime classes, unwrapped from their Serve deployments."""
    import runtime
    import runtime_cellpose

    out = []
    for mod, name in ((runtime, "RuntimeApp"), (runtime_cellpose, "CellposeRuntime")):
        obj = getattr(mod, name, None)
        inner = getattr(obj, "func_or_class", None) or getattr(
            obj, "__wrapped_class__", None
        )
        cls = inner if isinstance(inner, type) else obj
        assert isinstance(cls, type), f"{name} is not reachable as a class"
        out.append((name, cls))
    return out


def _bare(cls):
    """An instance with the store-probe state and nothing else constructed."""
    inst = cls.__new__(cls)
    inst._store_ok = None
    inst._store_detail = "not probed yet"
    inst._store_checked_at = 0.0
    return inst


@pytest.fixture()
def writable_store(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(training, "sessions_root", lambda: root)
    return root


@pytest.fixture()
def read_only_store(tmp_path, monkeypatch):
    """A store that EXISTS and refuses writes — the shape that fooled the old
    check, since `mkdir(exist_ok=True)` succeeds on it."""
    root = tmp_path / "sessions"
    root.mkdir()
    root.chmod(0o555)
    monkeypatch.setattr(training, "sessions_root", lambda: root)
    yield root
    root.chmod(0o755)


# ── the runtimes answer for themselves ──────────────────────────────────────


@pytest.mark.parametrize("name,cls", _runtime_classes())
@pytest.mark.asyncio
async def test_a_runtime_REPORTS_an_unwritable_store(name, cls, read_only_store):
    inst = _bare(cls)
    report = await inst.store_status()
    assert report["writable"] is False, (name, report)
    assert str(read_only_store) in report["detail"], report


@pytest.mark.parametrize("name,cls", _runtime_classes())
@pytest.mark.asyncio
async def test_a_runtime_reports_a_WRITABLE_store(name, cls, writable_store):
    """Clause 3's half of the predicate: the probe must not refuse everything."""
    inst = _bare(cls)
    report = await inst.store_status()
    assert report["writable"] is True, (name, report)
    assert str(writable_store) in report["detail"], report


@pytest.mark.parametrize("name,cls", _runtime_classes())
@pytest.mark.asyncio
async def test_the_probe_LEAVES_NOTHING_BEHIND(name, cls, writable_store):
    """It writes to prove the store accepts writes; it must not accumulate."""
    inst = _bare(cls)
    await inst.store_status()
    await inst.store_status(max_age_s=0)
    assert sorted(p.name for p in writable_store.iterdir()) == [], list(
        writable_store.iterdir()
    )


@pytest.mark.parametrize("name,cls", _runtime_classes())
@pytest.mark.asyncio
async def test_the_VERDICT_IS_CACHED_and_reused(name, cls, writable_store):
    """One probe per TTL, not one per caller: the probe writes a file, and the
    entry asks on paths that run per request."""
    inst = _bare(cls)
    calls = []
    real = training.assert_sessions_root_writable

    def counting():
        calls.append(1)
        return real()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(training, "assert_sessions_root_writable", counting)
        await inst.store_status()
        await inst.store_status()
        await inst.store_status()
        assert len(calls) == 1, f"{name} probed {len(calls)} times for three asks"
        # And a caller that explicitly wants a fresh answer gets one.
        await inst.store_status(max_age_s=0)
        assert len(calls) == 2, len(calls)


@pytest.mark.parametrize("name,cls", _runtime_classes())
@pytest.mark.asyncio
async def test_PING_REPORTS_THE_CACHE_AND_NEVER_PROBES(name, cls, writable_store):
    """`ping` is called inside the entry's 2 s readiness timeout. A probe there
    would make a replica wedged on a hung mount hang that check, which is the
    failure the timeout exists for."""
    inst = _bare(cls)
    inst._gpu_lock = asyncio.Lock()
    inst.start_time = 0.0
    inst._loaded_model_type = None
    inst._loaded_key = None
    calls = []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            training,
            "assert_sessions_root_writable",
            lambda: calls.append(1) or writable_store,
        )
        out = await inst.ping()
    assert "store" in out, out
    assert out["store"]["writable"] is None, out["store"]
    assert calls == [], f"{name}'s ping probed the store"


# ── no health check, by construction ───────────────────────────────────────


def _health_check_decorated(module) -> list:
    """Functions in *module* carrying `@bioengine.health_check`, by AST.

    Decorators, not text. The first version of this grepped the source for the
    string and matched a DOCSTRING that says the class "declares no
    ``@bioengine.health_check``" -- so it reported a health check in a module
    whose prose denies having one. That is the same defect I had just repaired
    in the sessions-root guard, which is why it is written down here as well
    as fixed: a scan matches the shape you wrote, not the thing you meant.
    """
    import ast

    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "health_check"
                and isinstance(target.value, ast.Name)
                and target.value.id == "bioengine"
            ):
                found.append(node.name)
    return found


@pytest.mark.parametrize("name,cls", _runtime_classes())
def test_NEITHER_RUNTIME_DECLARES_A_HEALTH_CHECK(name, cls):
    """THE design constraint, as a test rather than a comment.

    A failing Serve health check restarts the replica. Reporting an unwritable
    store through one would crash-loop the GPU runtimes under a fault the app
    cannot repair, on a shared cluster. If someone later adds a health check to
    a runtime, this goes red and they have to argue for it on the row.
    """
    module = __import__(cls.__module__)
    decorated = _health_check_decorated(module)
    assert decorated == [], (
        f"{cls.__module__} now declares a health check on {decorated}. A "
        "failing one RESTARTS the replica, which is what this design avoids."
    )


def test_that_the_health_check_guard_CAN_fail(tmp_path, monkeypatch):
    """The control. A guard that reads a file can silently measure the wrong
    thing, and this one already did once -- it matched the phrase inside a
    docstring. So: a module that really decorates a method must be caught, and
    one that only MENTIONS the decorator in prose must not be."""
    import types

    real = tmp_path / "fake_runtime.py"
    real.write_text(
        "import bioengine\n"
        "class R:\n"
        "    @bioengine.health_check\n"
        "    async def _health(self):\n        pass\n"
    )
    mod = types.SimpleNamespace(__file__=str(real))
    assert _health_check_decorated(mod) == ["_health"]

    prose = tmp_path / "prose_runtime.py"
    prose.write_text(
        "class R:\n"
        '    """This class declares no @bioengine.health_check at all."""\n'
        "    async def ping(self):\n        pass\n"
    )
    mod2 = types.SimpleNamespace(__file__=str(prose))
    assert _health_check_decorated(mod2) == [], "prose was read as a decorator"


# ── the entry refuses at the door ───────────────────────────────────────────


def _entry_class():
    import entry

    for obj in vars(entry).values():
        if isinstance(obj, type) and obj.__name__ == "EntryApp":
            return obj
        inner = getattr(obj, "__wrapped_class__", None) or getattr(
            obj, "func_or_class", None
        )
        if isinstance(inner, type) and inner.__name__ == "EntryApp":
            return inner
    raise AssertionError("EntryApp not found")


class _FakeRuntime:
    def __init__(self, report):
        self._report = report
        self.asked = 0

    async def store_status(self, max_age_s=None):
        self.asked += 1
        if isinstance(self._report, Exception):
            raise self._report
        return self._report


def _entry_with(runtime):
    cls = _entry_class()
    inst = cls.__new__(cls)
    inst._runtime_for = lambda model_type=None: runtime
    return inst


@pytest.mark.asyncio
async def test_the_entry_REFUSES_a_run_when_the_runtime_cannot_write():
    """Clause 1's second half: refused UP FRONT with the reason, instead of
    dying at the runtime's first status write -- which left the user a Ray
    traceback as the explanation."""
    rt = _FakeRuntime(
        {
            "writable": False,
            "detail": "the training-run store at /home/ray/.bioengine/x "
            "exists but refuses writes: [Errno 30] Read-only file system",
        }
    )
    inst = _entry_with(rt)
    with pytest.raises(RuntimeError) as exc:
        await inst._require_runtime_store_writable("cpsam")
    msg = str(exc.value)
    assert "/home/ray/.bioengine/x" in msg, msg
    assert "No run was started" in msg, msg
    assert rt.asked == 1


@pytest.mark.asyncio
async def test_a_WRITABLE_store_changes_nothing_at_the_entry():
    """Clause 3. A check that refused everything would satisfy the test above,
    so the permitting case is asserted too."""
    rt = _FakeRuntime({"writable": True, "detail": "the run store accepts writes"})
    inst = _entry_with(rt)
    await inst._require_runtime_store_writable("cpsam")  # must not raise
    assert rt.asked == 1


@pytest.mark.asyncio
async def test_an_UNREADABLE_verdict_does_NOT_refuse():
    """Deliberate, and the opposite of fail-closed for once. The readiness
    check has already established that the runtime answers, so a missing
    verdict here is this check's own failure -- and taking training offline
    because one status call timed out is a worse trade than letting the run
    proceed and fail at its write, which is today's behaviour anyway."""
    for report in ({}, {"writable": None}, RuntimeError("timeout")):
        inst = _entry_with(_FakeRuntime(report))
        await inst._require_runtime_store_writable("cpsam")


def test_START_TRAINING_ASKS_before_it_accepts_anything():
    """Structural, because the behavioural path needs the whole training stack.

    It asserts the refusal is reached BEFORE the session is created: a check
    placed after `new_session_id` would leave a FAILED run behind for a fault
    the worker already knew about.
    """
    import ast
    import inspect

    cls = _entry_class()
    # textwrap.dedent, because a method's source is indented and `ast.parse`
    # refuses that -- the decorator line keeps `lstrip()` from being enough.
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.start_training)))
    # SOURCE order, by line number. `ast.walk` is breadth-first, so its order
    # is not the order the code runs -- the first version of this test read
    # index 10 before index 4 and reported a real ordering as broken.
    seen = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            seen.append((node.lineno, node.func.attr))
    calls = [name for _line, name in sorted(seen)]
    assert (
        "_require_runtime_store_writable" in calls
    ), "start_training does not ask whether the runtime can write its store"
    assert "new_session_id" in calls, "start_training no longer creates a session?"
    assert calls.index("_require_runtime_store_writable") < calls.index(
        "new_session_id"
    ), "the store check runs AFTER the session is created, so a refused run is "
    "left behind as FAILED"


# ── the STATUS surface, which is what the page reads ───────────────────────
#
# Round 1 of the review: the verdict reached only internal composition
# endpoints, so `get_training_capabilities` -- the status the annotate page
# polls -- still answered `trainable: True, reason: "fits"` for a backend whose
# runtime could not write a single run record. The refusal at `start_training`
# tells a user who has already committed to a run; the page reads this.


class _PingRuntime:
    """A runtime that answers `ping` with a given store verdict."""

    def __init__(self, store, *, fail=False):
        self._store = store
        self._fail = fail
        self.pings = 0
        self.store_status_calls = 0

    async def ping(self):
        self.pings += 1
        if self._fail:
            raise RuntimeError("the runtime did not answer")
        return {"status": "ok", "store": self._store}

    async def store_status(self, max_age_s=None):
        # Counted so a test can assert the STATUS path does not probe.
        self.store_status_calls += 1
        return self._store


def _caps_entry(monkeypatch, store, *, backends=("cellpose",), fail=False):
    cls = _entry_class()
    inst = cls.__new__(cls)
    rt = _PingRuntime(store, fail=fail)
    inst._enabled_backends = set(backends)
    inst._runtime_for = lambda model_type=None: rt
    # The two deploy-time declarations `_gpu_view` reads. Set explicitly
    # because this instance skips __init__: a GPU big enough for every model,
    # so the only thing that can make a model untrainable here is the store.
    inst.declared_gpu_memory_mb = 40000
    inst.declared_gpu_device = "declared-for-test"
    inst._rt = rt
    return inst


@pytest.mark.asyncio
async def test_CAPABILITIES_reports_an_unwritable_store_per_backend(monkeypatch):
    detail = "the training-run store at /home/ray/.bioengine/x refuses writes"
    inst = _caps_entry(monkeypatch, {"writable": False, "detail": detail})

    caps = await inst.get_training_capabilities()

    assert caps["stores"]["cellpose"]["writable"] is False, caps["stores"]
    assert detail in caps["stores"]["cellpose"]["detail"]
    cellpose_models = [m for m in caps["models"] if m["backend"] == "cellpose"]
    assert cellpose_models, caps["models"]
    for m in cellpose_models:
        assert m["trainable"] is False, m
        assert detail in m["reason"], m


@pytest.mark.asyncio
async def test_a_WRITABLE_store_leaves_capabilities_unchanged(monkeypatch):
    """Clause 3 at the status surface. A change that marked everything
    untrainable would satisfy the test above on its own."""
    inst = _caps_entry(
        monkeypatch, {"writable": True, "detail": "the run store accepts writes"}
    )
    caps = await inst.get_training_capabilities()
    assert caps["stores"]["cellpose"]["writable"] is True
    for m in [m for m in caps["models"] if m["backend"] == "cellpose"]:
        assert m["trainable"] is True, m
        assert m["reason"] == "fits", m


@pytest.mark.asyncio
async def test_CAPABILITIES_NEVER_PROBES_the_store(monkeypatch):
    """It is polled by the page. It reads the runtime's CACHED verdict through
    `ping` and must not call `store_status`, which may write and may block on a
    wedged mount."""
    inst = _caps_entry(monkeypatch, {"writable": True, "detail": "ok"})
    await inst.get_training_capabilities()
    assert inst._rt.pings >= 1
    assert inst._rt.store_status_calls == 0, "capabilities probed the store"


@pytest.mark.asyncio
async def test_a_runtime_that_cannot_be_ASKED_is_not_reported_as_either(monkeypatch):
    """`writable: None` is a third answer. Collapsing it into False would make
    an unreachable runtime look like a broken filesystem, and into True would
    hide the fault this row is about."""
    inst = _caps_entry(monkeypatch, {"writable": False, "detail": "x"}, fail=True)
    caps = await inst.get_training_capabilities()
    assert caps["stores"]["cellpose"]["writable"] is None, caps["stores"]
    for m in [m for m in caps["models"] if m["backend"] == "cellpose"]:
        assert m["trainable"] is not False or "store" not in m["reason"], m


# ── the gate is REACHED, not merely present in the source ──────────────────


@pytest.mark.asyncio
async def test_START_TRAINING_actually_awaits_the_store_gate(monkeypatch):
    """Closes the hole round 1 named in my ordering test: that one reads AST
    call names in source order, so it stays GREEN if the gate sits in a dead
    branch or its await is dropped. This one runs the method and requires the
    refusal to come OUT of it, with no session created."""
    import entry

    cls = _entry_class()
    inst = cls.__new__(cls)
    started = []

    inst._enabled_backends = {"cellpose"}
    inst._runtime_for = lambda model_type=None: _PingRuntime(
        {"writable": False, "detail": "store at /home/ray/.bioengine refuses writes"}
    )

    async def _check(model_type=None):
        return None

    monkeypatch.setattr(inst, "_check_runtime_available", _check, raising=False)
    monkeypatch.setattr(
        training, "new_session_id", lambda: started.append(1) or "x", raising=True
    )

    with pytest.raises(RuntimeError) as exc:
        await cls.start_training.__get__(inst)(
            train_images=[], train_labels=[], model_type="cpsam", n_epochs=1
        )
    msg = str(exc.value)
    assert "/home/ray/.bioengine" in msg, msg
    assert "No run was started" in msg, msg
    assert started == [], "a session was created before the store was checked"


# ── the replica that writes answers for itself ─────────────────────────────


@pytest.mark.parametrize("name,cls", _runtime_classes())
@pytest.mark.asyncio
async def test_TRAIN_refuses_in_the_process_that_would_write(
    name, cls, read_only_store
):
    """Round 1's minor (b): the entry's verdict comes from whichever replica
    its handle routed to, which need not be the one that trains. So `train`
    asks for itself, with a FRESH probe, before taking the GPU lock -- a
    refused run must neither hold the GPU nor die mid-epoch."""
    inst = _bare(cls)
    inst._gpu_lock = asyncio.Lock()
    took_lock = []

    class _Watch:
        async def __aenter__(self_inner):
            took_lock.append(1)

        async def __aexit__(self_inner, *a):
            return False

    inst._gpu_lock = _Watch()

    with pytest.raises(RuntimeError) as exc:
        await cls.train.__get__(inst)("2026-11-26-120000-aaaaaaaa", "cpsam", {})
    msg = str(exc.value)
    assert "cannot write" in msg, msg
    assert str(read_only_store) in msg, msg
    assert took_lock == [], "it took the GPU lock before refusing"


@pytest.mark.parametrize("name,cls", _runtime_classes())
@pytest.mark.asyncio
async def test_TRAIN_does_not_refuse_on_a_writable_store(
    name, cls, writable_store, monkeypatch
):
    """The control: it must get past the check. Stopped immediately afterwards
    with a sentinel so the test does not run a real training subprocess."""
    inst = _bare(cls)

    class _Boom:
        async def __aenter__(self_inner):
            raise AssertionError("REACHED THE GPU LOCK")

        async def __aexit__(self_inner, *a):
            return False

    inst._gpu_lock = _Boom()
    with pytest.raises(AssertionError) as exc:
        await cls.train.__get__(inst)("2026-11-26-120000-aaaaaaaa", "cpsam", {})
    assert "REACHED THE GPU LOCK" in str(exc.value), str(exc.value)


@pytest.mark.parametrize("name,cls", _runtime_classes())
@pytest.mark.asyncio
async def test_TRAIN_RE_PROBES_rather_than_trusting_a_cached_verdict(
    name, cls, read_only_store
):
    """Why `train` asks with `max_age_s=0`.

    The entry has already asked, so the cache can hold a verdict up to a
    minute old -- and the store can have changed since, which is precisely
    what happened in production when a replica came up with a HOME that was
    never redirected. This plants a FRESH-looking cached "writable" over a
    store that is actually read-only: with a re-probe the run is refused, and
    with a cached read it would start and die at its first write.

    Without this test the `max_age_s=0` could be dropped silently -- an arm
    that replaced it with a long TTL was GREEN until this existed, because
    the other fixtures leave the cache empty and an empty cache always probes.
    """
    import time as _time

    inst = _bare(cls)
    inst._store_ok = True
    inst._store_detail = "a stale verdict that says the store is fine"
    inst._store_checked_at = _time.time()  # fresh enough for any TTL

    class _Boom:
        async def __aenter__(self_inner):
            raise AssertionError("REACHED THE GPU LOCK")

        async def __aexit__(self_inner, *a):
            return False

    inst._gpu_lock = _Boom()

    with pytest.raises(RuntimeError) as exc:
        await cls.train.__get__(inst)("2026-11-26-120000-aaaaaaaa", "cpsam", {})
    msg = str(exc.value)
    assert "cannot write" in msg, msg
    assert "stale verdict" not in msg, "it reported the cached verdict, not a probe"
    assert str(read_only_store) in msg, msg
