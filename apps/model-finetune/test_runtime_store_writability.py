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
import time

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


def _caps_entry(
    monkeypatch, store, *, backends=("cellpose",), fail=False, runtime=None
):
    cls = _entry_class()
    inst = cls.__new__(cls)
    rt = runtime if runtime is not None else _PingRuntime(store, fail=fail)
    inst._enabled_backends = set(backends)
    inst._runtime_for = lambda model_type=None: rt
    # The two deploy-time declarations `_gpu_view` reads. Set explicitly
    # because this instance skips __init__: a GPU big enough for every model,
    # so the only thing that can make a model untrainable here is the store.
    inst.declared_gpu_memory_mb = 40000
    inst.declared_gpu_device = "declared-for-test"
    # This instance skips __init__, so the state the method uses is set here
    # rather than left to an AttributeError. Empty means "no backend has been
    # found unreachable yet", which is what a fresh replica has.
    inst._store_view_unreachable_at = {}
    inst._rt = rt
    return inst


class _HungRuntime:
    """A runtime whose `ping` never answers.

    The permanent case on a cluster with fewer GPUs than enabled backends: the
    deployment sits UPSCALING and the handle's call never returns. Not an
    exception and not slow -- it does not come back at all, which is why only a
    timeout bounds it.
    """

    def __init__(self):
        self.pings = 0
        self.store_status_calls = 0

    async def ping(self):
        self.pings += 1
        await asyncio.Event().wait()  # forever

    async def store_status(self, max_age_s=None):
        self.store_status_calls += 1
        await asyncio.Event().wait()

    async def gpu_memory_info(self):
        await asyncio.Event().wait()


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


# ── what the status surface COSTS, because the page polls it ───────────────
#
# Review round 2's finding 3, as tests. Adding the store view made this method
# ping every enabled runtime, and the first version did it one await after
# another with a 5 s timeout each -- so with VRAM declared, both backends
# enabled and one hung runtime it took 10.01 s and returned no verdict anyway.
# The declared path exists to answer CPU-side without waking a GPU, so that was
# a regression in the exact case the path was built for.
#
# A BOUND, NOT A STOPWATCH READING. These assert an upper bound comfortably
# below the broken behaviour and comfortably above the fixed one, so they fail
# on the regression returning and do not fail on a loaded machine.


@pytest.mark.asyncio
async def test_CAPABILITIES_IS_BOUNDED_against_a_HUNG_runtime(monkeypatch):
    """The measured regression, as a failing condition.

    Two enabled backends and a runtime that never answers. Serialised at 5 s
    each this was 10 s; gathered and bounded it is one timeout.
    """
    import entry

    hung = _HungRuntime()
    inst = _caps_entry(
        monkeypatch, None, backends=("microsam", "cellpose"), runtime=hung
    )

    started = asyncio.get_running_loop().time()
    caps = await inst.get_training_capabilities()
    elapsed = asyncio.get_running_loop().time() - started

    # AN ABSOLUTE BOUND, not one derived from the timeout constant. Deriving it
    # would make the test rise with the thing it is meant to constrain: a change
    # putting the timeout back to 5 s would also move the bound to 6 s and pass.
    # 3 s is comfortably above the fixed behaviour (one 2 s timeout) and
    # comfortably below the measured regression (10.01 s).
    BOUND_S = 3.0
    assert elapsed < BOUND_S, (
        f"capabilities took {elapsed:.2f}s against a hung runtime, over the "
        f"{BOUND_S:.1f}s bound. Serialised per-backend awaits at 5s each were "
        "measured at 10.01s for this exact case."
    )
    # And the configured timeout is small enough that the bound is not merely
    # met by luck on a fast machine.
    assert entry._STORE_VIEW_PING_TIMEOUT_S <= 2.0, entry._STORE_VIEW_PING_TIMEOUT_S
    # And the answer is the honest one: could not ask, not a verdict. Compared
    # against `None` explicitly rather than falsily -- `False` is also falsy and
    # is the OPPOSITE claim, so a `not ...` test here would accept the one
    # answer this row exists to keep out.
    for backend in ("microsam", "cellpose"):
        assert caps["stores"][backend]["writable"] is None, caps["stores"]


@pytest.mark.asyncio
async def test_the_views_are_GATHERED_not_awaited_one_after_another(monkeypatch):
    """The mechanism, independent of wall-clock.

    A timing bound alone could be met by shortening the timeout while keeping
    the awaits serial, which is only a smaller version of the same bug. This
    asserts both backends were in flight AT ONCE: each ping records how many
    were running when it started, and a serial implementation can never see
    more than one.
    """
    inflight = {"now": 0, "max": 0}

    class _CountingRuntime:
        def __init__(self):
            self.pings = 0
            self.store_status_calls = 0

        async def ping(self):
            self.pings += 1
            inflight["now"] += 1
            inflight["max"] = max(inflight["max"], inflight["now"])
            try:
                await asyncio.sleep(0.05)
                return {"status": "ok", "store": {"writable": True, "detail": "ok"}}
            finally:
                inflight["now"] -= 1

    inst = _caps_entry(
        monkeypatch, None, backends=("microsam", "cellpose"), runtime=_CountingRuntime()
    )
    await inst.get_training_capabilities()
    assert inflight["max"] == 2, (
        f"at most {inflight['max']} store view(s) were in flight at once; the "
        "two backends were awaited one after another"
    )


@pytest.mark.asyncio
async def test_a_backend_that_did_not_answer_IS_NOT_ASKED_AGAIN_immediately(
    monkeypatch,
):
    """The other half, and the one that matters for a POLLED method.

    An under-GPU'd cluster leaves a backend UPSCALING indefinitely, so a bound
    paid once per poll is still a bound paid forever. The first poll pays the
    timeout; the next polls inside the window must not.
    """
    import entry

    hung = _HungRuntime()
    inst = _caps_entry(monkeypatch, None, backends=("cellpose",), runtime=hung)

    await inst.get_training_capabilities()
    assert hung.pings == 1, hung.pings

    started = asyncio.get_running_loop().time()
    await inst.get_training_capabilities()
    second = asyncio.get_running_loop().time() - started

    assert hung.pings == 1, f"the hung runtime was asked again ({hung.pings} pings)"
    assert second < 0.5, f"the second poll still took {second:.2f}s"

    # THE WINDOW IS BOUNDED, asserted as an absolute number. Deriving the age
    # below from `_STORE_VIEW_UNREACHABLE_FOR_S` is what made the first version
    # of this test useless: a mutant widening the window to effectively forever
    # also moved the age, so the backend was still re-asked and the test passed.
    # The same mistake as deriving a timing bound from the timeout it bounds.
    assert entry._STORE_VIEW_UNREACHABLE_FOR_S <= 300.0, (
        f"the unreachable window is {entry._STORE_VIEW_UNREACHABLE_FOR_S}s. A "
        "window longer than a few minutes writes a backend off for practical "
        "purposes, which is what this bound exists to stop."
    )
    # THE CONTROL on that window: once it lapses the backend IS asked again, so
    # a runtime that comes up is not written off permanently. One day of age is
    # past any window the assertion above permits.
    inst._store_view_unreachable_at["cellpose"] = time.time() - 86400.0
    await inst.get_training_capabilities()
    assert hung.pings == 2, (
        "the backend was never asked again after the window lapsed, so an "
        "unreachable runtime would be written off for the life of the replica"
    )


@pytest.mark.asyncio
async def test_a_runtime_that_ANSWERS_is_not_penalised_by_an_earlier_failure(
    monkeypatch,
):
    """A SUCCESSFUL ask clears the mark, driven through the code.

    This is the arm a mutant taught me to write properly. The first version
    cleared the mark itself and then checked the verdict came back, so it never
    exercised the clearing at all: deleting the line that clears it left the
    test green. Here the mark is only ever set and cleared by the method, and
    the final poll is the one that can only pass if the clearing happened.

    Why it matters: without clearing, one blip suppresses the store verdict for
    the whole window -- and reporting that verdict is what this row is for.
    """
    import entry

    class _FailsOnceRuntime:
        def __init__(self):
            self.pings = 0
            self.store_status_calls = 0

        async def ping(self):
            self.pings += 1
            if self.pings == 1:
                raise RuntimeError("the runtime did not answer")
            return {"status": "ok", "store": {"writable": True, "detail": "ok"}}

    rt = _FailsOnceRuntime()
    inst = _caps_entry(monkeypatch, None, runtime=rt)

    # Window of zero while the mark is established and then superseded, so no
    # test-side poking of the cache is needed to get past it.
    monkeypatch.setattr(entry, "_STORE_VIEW_UNREACHABLE_FOR_S", 0.0)

    # Poll 1: the ask fails, so the METHOD records the backend as unreachable.
    caps = await inst.get_training_capabilities()
    assert caps["stores"]["cellpose"]["writable"] is None, caps["stores"]
    assert (
        inst._store_view_unreachable_at.get("cellpose") is not None
    ), "the failure was not recorded, so this test cannot show it being cleared"

    # Poll 2: the ask succeeds, so the METHOD must clear the mark.
    caps = await inst.get_training_capabilities()
    assert caps["stores"]["cellpose"]["writable"] is True, caps["stores"]

    # Poll 3, with a real window restored. This is the assertion the mutant
    # that drops the clearing fails: an uncleared mark suppresses this poll and
    # the verdict reverts to None.
    monkeypatch.setattr(entry, "_STORE_VIEW_UNREACHABLE_FOR_S", 30.0)
    caps = await inst.get_training_capabilities()
    assert caps["stores"]["cellpose"]["writable"] is True, (
        f"the verdict reverted to {caps['stores']['cellpose']} after a "
        "successful ask, so the unreachable mark was never cleared"
    )


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
