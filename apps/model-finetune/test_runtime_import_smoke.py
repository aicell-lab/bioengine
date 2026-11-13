"""Regression tests for #0371 — a replica with no ML stack must not be advertised
as ready.

On the Berzelius worker the ``CellposeRuntime`` replica reported ``HEALTHY`` and its
deployment reported ``RUNNING`` while every call raised ``ModuleNotFoundError: No
module named 'cellpose'``. Measured from the source, nothing on the health path
imports the ML stack: both runtimes declare only ``@bioengine.async_init`` (the busy
log), the framework's ``check_health`` additionally retains a CUDA context through
ctypes on ``libcuda.so.1`` — injected by the NVIDIA container runtime independently
of the pip layer — and neither runtime declares a ``@bioengine.health_check``, so the
periodic branch is a no-op. ``ping()``, which the report blamed, is not the health
probe at all: Ray Serve never calls it. It is an entry-side pre-call gate
(``entry.py``, ``_check_runtime_available``) with a 2 s timeout.

The fix is a ``@bioengine.smoke_test`` on each runtime: the framework runs it once
after ``async_init`` and fails the replica when it raises. It does **not** keep
traffic off while that happens — ``check_health`` creates the task and tests
``.done()`` in the same call, so poll 1 cannot fail, and Serve condemns only after 3
consecutive failed checks, so at ``health_check_period_s=30.0`` a broken replica
serves real traffic for ≈90 s first. That window is what ``_settle_check_health``
below models, and it is the honest limit of this fix. ``ping()`` stays shallow
on purpose and that is pinned here too — importing torch cannot fit a 2 s pre-call
gate, and a per-poll import of a heavy stack is its own defect.

Plain pytest. No GPU, no live cluster, no real torch/cellpose install: the imports are
intercepted, which is the only way to exercise the missing-module arm at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import os
import re

import pytest

import runtime
import runtime_cellpose

# (module, deployment class, the name the error message must carry)
BACKENDS = [
    (runtime, runtime.RuntimeApp, "micro-sam"),
    (runtime_cellpose, runtime_cellpose.CellposeRuntime, "Cellpose"),
]
IDS = ["microsam", "cellpose"]


def _user_class(deployment):
    """The class the ``@bioengine.app`` decorator wrapped.

    Both runtimes are ``ray.serve.deployment.Deployment`` objects once the decorator
    has run, so their methods are not reachable through them.
    """
    return deployment.func_or_class


def _hook(deployment, kind: str):
    """The method on the user class marked with a given bioengine lifecycle kind."""
    cls = _user_class(deployment)
    found = [
        name
        for name in dir(cls)
        if getattr(getattr(cls, name, None), "_bioengine_kind", None) == kind
    ]
    assert len(found) <= 1, f"{cls.__name__} marks more than one {kind}: {found}"
    return getattr(cls, found[0]) if found else None


# ── the hook exists at all ───────────────────────────────────────────────────


@pytest.mark.parametrize("mod,deployment,_label", BACKENDS, ids=IDS)
def test_the_runtime_declares_a_smoke_test(mod, deployment, _label) -> None:
    """The reported defect in one line: before the fix there was no startup gate.

    A presence assertion, deliberately separate from the behaviour tests below —
    those intercept imports, and an intercepted import can be made to raise whether
    or not a hook is wired to Ray Serve's one-shot path.
    """
    assert _hook(deployment, "smoke_test") is not None


@pytest.mark.parametrize("mod,deployment,_label", BACKENDS, ids=IDS)
def test_the_import_list_is_not_empty(mod, deployment, _label) -> None:
    """CONTROL for the raising arm: an empty list makes the loop unable to raise, so
    the passing arm below would pass vacuously."""
    assert len(mod._RUNTIME_IMPORTS) > 0


@pytest.mark.parametrize("mod,deployment,_label", BACKENDS, ids=IDS)
def test_every_gated_module_is_one_the_bodies_import(mod, deployment, _label) -> None:
    """A module nothing imports would fail a replica that can serve every call.

    Drift guard with teeth: the gate list has to be grounded in the module's own
    source, not in the requirements file.
    """
    source = inspect.getsource(mod)
    for name in mod._RUNTIME_IMPORTS:
        pattern = rf"^\s*(?:import {re.escape(name)}\b|from {re.escape(name)}[\s.])"
        assert re.search(pattern, source, re.MULTILINE), (
            f"{name} is gated by {mod.__name__}._RUNTIME_IMPORTS but nothing in "
            f"{mod.__name__} imports it"
        )


# ── behaviour: the arm that fails, and the arm that must not ────────────────


@pytest.fixture
def break_imports(monkeypatch):
    """Make named top-level modules unimportable, leaving every other import real."""

    def _apply(*broken: str):
        real = importlib.import_module

        def fake(name, *args, **kwargs):
            if name.split(".")[0] in broken:
                raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake)

    return _apply


@pytest.mark.parametrize("mod,deployment,label", BACKENDS, ids=IDS)
def test_a_missing_module_fails_the_replica(
    mod, deployment, label, break_imports
) -> None:
    """The defect: with the stack absent the replica used to come up ready."""
    broken = mod._RUNTIME_IMPORTS[0]
    break_imports(broken)
    hook = _hook(deployment, "smoke_test")

    with pytest.raises(RuntimeError) as excinfo:
        import asyncio

        asyncio.run(hook(object.__new__(_user_class(deployment))))

    message = str(excinfo.value)
    assert broken in message, f"the error does not name the missing module: {message}"
    assert label in message, f"the error does not say which backend: {message}"
    # The pip output is in runtime_env_setup-*.log, not in the app's logs — an
    # operator reading get_app_status has to be told where to look.
    assert "runtime_env_setup" in message


@pytest.mark.parametrize("mod,deployment,_label", BACKENDS, ids=IDS)
def test_every_missing_module_is_reported_not_just_the_first(
    mod, deployment, _label, break_imports
) -> None:
    """A half-installed env usually lost more than one module."""
    if len(mod._RUNTIME_IMPORTS) < 2:
        pytest.skip(f"{mod.__name__} gates only one module")
    broken = tuple(mod._RUNTIME_IMPORTS[:2])
    break_imports(*broken)
    hook = _hook(deployment, "smoke_test")

    with pytest.raises(RuntimeError) as excinfo:
        import asyncio

        asyncio.run(hook(object.__new__(_user_class(deployment))))

    for name in broken:
        assert name in str(excinfo.value)


@pytest.mark.parametrize("mod,deployment,_label", BACKENDS, ids=IDS)
def test_an_importable_stack_passes(mod, deployment, _label, monkeypatch) -> None:
    """DISCRIMINATOR for the arm above: a gate that always raises would fail every
    replica, including a healthy one, and would still satisfy every test that only
    checks for a raise.

    The real ML stacks are not installed in this test environment and must not be —
    the two backends' numpy pins are irreconcilable, so no one process can hold both.
    So stub the import: this asserts the hook raises *because* an import failed, not
    unconditionally.
    """
    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: object())
    hook = _hook(deployment, "smoke_test")

    import asyncio

    asyncio.run(hook(object.__new__(_user_class(deployment))))  # must not raise


# ── the decision that ping() stays shallow, pinned ──────────────────────────


@pytest.mark.parametrize("mod,deployment,_label", BACKENDS, ids=IDS)
def test_ping_stays_shallow_and_answers_with_the_stack_absent(
    mod, deployment, _label, break_imports
) -> None:
    """Recorded decision, not an accident: readiness is gated once at startup, and
    the entry's 2 s pre-call probe does no importing.

    If someone later deepens ``ping()``, this fails and they have to re-argue the
    2 s timeout in ``entry._check_runtime_available``.
    """
    break_imports(*mod._RUNTIME_IMPORTS)
    cls = _user_class(deployment)
    instance = object.__new__(cls)
    # ping() reads these three only.
    instance.start_time = 0.0
    instance._gpu_lock = type("L", (), {"locked": lambda self: False})()
    instance._loaded_model_type = None
    instance._loaded_key = None

    import asyncio

    result = asyncio.run(cls.ping(instance))
    assert result["status"] == "ok"


# ── the framework actually runs it: Ray Serve's readiness path, end to end ───


@pytest.fixture
def replica_env(monkeypatch, tmp_path):
    """Let a replica's wrapped ``__init__`` run outside Ray.

    ``_setup_replica`` chdirs into ``BIOENGINE_APP_DIR`` and skips proxy
    registration when ``BIOENGINE_PROXY_ACTOR_NAME`` is unset — which also leaves
    ``_bioengine_has_gpu`` False, so no CUDA context is attempted here. The cwd is
    restored because the rest of the suite reads files relative to it.
    """
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(tmp_path))
    monkeypatch.delenv("BIOENGINE_PROXY_ACTOR_NAME", raising=False)
    monkeypatch.delenv("BIOENGINE_APPLICATION_ID", raising=False)
    cwd = os.getcwd()
    try:
        yield
    finally:
        os.chdir(cwd)


async def _settle_check_health(instance) -> None:
    """Poll ``check_health`` the way Ray Serve does, until the smoke test is read.

    The framework launches the smoke test as a background task and reads its
    exception on a *later* call, so one poll is not enough — and that lag is the
    one honest limitation of this fix. Two periods make it up, not one: poll 1
    creates the task and tests ``.done()`` in the same call, so it cannot fail,
    and Serve condemns a replica only after 3 consecutive failed checks
    (``REPLICA_HEALTH_CHECK_UNHEALTHY_THRESHOLD`` in
    ``ray.serve._private.constants``). So with ``health_check_period_s=30.0`` a
    broken replica reports HEALTHY on poll 1 and is routable across polls 2-4 —
    ≈90 s, roughly 3 periods, not one. Bounded, and the missing modules are named
    in the replica log, unlike reporting HEALTHY forever.

    This function deliberately awaits the task between the two polls rather than
    sleeping, so the test observes the raise without waiting out a real period.
    """
    await instance.check_health()
    if instance._bioengine_test_task is not None:
        with contextlib.suppress(Exception):
            await instance._bioengine_test_task
    await instance.check_health()


@pytest.mark.parametrize("mod,deployment,label", BACKENDS, ids=IDS)
def test_the_framework_marks_the_replica_unhealthy(
    mod, deployment, label, break_imports, replica_env
) -> None:
    """The property that matters, through the real ``check_health`` Ray Serve polls.

    The marker tests above would pass on a hook the framework never calls. This one
    goes through ``@bioengine.app``'s own wiring: the class's scanned lifecycle, the
    wrapped ``__init__``, and the installed ``check_health``.
    """
    assert deployment.func_or_class._bioengine_lifecycle["smoke_test"] is not None
    break_imports(mod._RUNTIME_IMPORTS[0])
    instance = _user_class(deployment)()

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(_settle_check_health(instance))

    assert "Smoke test failed" in str(excinfo.value)
    assert mod._RUNTIME_IMPORTS[0] in str(excinfo.value)


@pytest.mark.parametrize("mod,deployment,_label", BACKENDS, ids=IDS)
def test_the_framework_keeps_a_good_replica_healthy(
    mod, deployment, _label, monkeypatch, replica_env
) -> None:
    """DISCRIMINATOR for the test above: ``check_health`` must not raise just because
    a smoke test is now declared. Without this, wiring a hook that always failed
    would read as a pass."""
    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: object())
    instance = _user_class(deployment)()

    asyncio.run(_settle_check_health(instance))  # must not raise
