"""Pin that a lost Ray Serve controller is actually detected.

``_recover_from_controller_loss`` has exactly one trigger, and on Ray 2.55.1
neither of the two shapes a real controller loss produces reached it — so the
whole recovery path, and every improvement made to it, was unreachable:

* **cached Serve client** — ``serve.status`` raises ``ActorDiedError``, whose
  message carries none of the three substrings the matcher looked for.
* **fresh client, or a genuinely restarted head** — ``serve.status`` is the one
  Serve API that passes ``raise_if_no_controller_running=False``, so it
  *returns* ``ServeStatus(proxies={}, applications={})`` and the ``except``
  branch is never entered at all.

Both shapes were measured in a throwaway container against the published worker
image; the verbatim message below is that measurement, not a paraphrase.

The second shape cannot be decided from the status object: an empty status is
also exactly what a healthy cluster with no applications returns. Asking Ray
whether the controller actor exists is what separates them, which is why a live
controller is pinned here as hard as a dead one — a change that swept on every
empty status would redeploy every app of every idle worker.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from ray.exceptions import ActorDiedError, ActorUnavailableError

from bioengine.apps import manager as manager_mod
from bioengine.apps.manager import (
    _CONTROLLER_RECOVERY_COOLDOWN_SECONDS,
    _MISSING_FROM_STATUS_TOLERANCE,
    AppsManager,
    _is_serve_controller_gone,
    _serve_controller_is_missing,
)

APP_ID = "model-runner"
ENTRY = "ModelRunner"

# Measured verbatim on Ray 2.55.1: ray.kill(SERVE_CONTROLLER_ACTOR) under a
# cached Serve client, sampled at t=0/2/5/10/20/30 s — identical all six times.
MEASURED_ACTOR_DIED = (
    "The actor died unexpectedly before finishing this task.\n"
    "\tclass_name: ServeController\n"
    "\tactor_id: b492f7d06763c8ec4738f37001000000\n"
    "\tpid: 355\n"
    "\tname: SERVE_CONTROLLER_ACTOR\n"
    "\tnamespace: serve\n"
    "\tip: 172.17.0.8\n"
    "The actor is dead because it was killed by `ray.kill`."
)


def _actor_died(message: str) -> ActorDiedError:
    """An ActorDiedError that renders ``message``.

    ``__str__`` returns ``error_msg``, so the constructor's cause-formatting is
    bypassed rather than re-implemented here.
    """
    exc = ActorDiedError()
    exc.error_msg = message
    return exc


def _make_manager(built_app: bool = True) -> AppsManager:
    is_deployed = asyncio.Event()
    is_deployed.set()

    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.controller_loss")
    manager.server = None
    manager._redeploy_backoff = {}
    manager._missing_from_status = {}
    manager._deleted_pending_redeploy = set()
    manager._identity_warned = set()
    manager._last_controller_recovery = 0.0
    manager._deployed_applications = {
        APP_ID: {
            "is_deployed": is_deployed,
            "artifact_id": "bioimage-io/model-runner",
            "version": "2.8.0",
            "auto_redeploy": True,
            "entry_deployment_name": ENTRY,
            "built_app": (
                SimpleNamespace(spec={"entry_id": f"deployment:{ENTRY}", "classes": {}})
                if built_app
                else None
            ),
            "deployment_task": None,
        }
    }

    ray_cluster = MagicMock()
    ray_cluster.check_connection = AsyncMock()
    ray_cluster.call_with_reconnect = AsyncMock(
        return_value=SimpleNamespace(applications={})
    )
    ray_cluster.proxy_actor_handle.get_serve_instance_details.remote = AsyncMock(
        return_value={}
    )
    ray_cluster.proxy_actor_handle.get_replica_identities.remote = AsyncMock(
        return_value={}
    )
    manager.ray_cluster = ray_cluster
    manager._deploy_application = AsyncMock()
    return manager


async def _tick(manager: AppsManager) -> None:
    await manager.monitor_applications()
    await asyncio.sleep(0)


def _controller_lost_lines(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if "Ray Serve controller lost" in r.getMessage()
    ]


# --------------------------------------------------------------------------
# Shape 1: serve.status RAISES — the matcher is what must recognise it
# --------------------------------------------------------------------------


def test_the_measured_actor_died_message_is_recognised() -> None:
    assert _is_serve_controller_gone(_actor_died(MEASURED_ACTOR_DIED)) is True


def test_a_live_but_unreachable_controller_is_not_a_loss() -> None:
    # The discriminator for the shape above. ActorUnavailableError names the
    # same actor with nearly the same words but means the controller is alive
    # and briefly unreachable; re-bootstrapping Serve under it would destroy a
    # working instance. Matching on the message alone would call this a loss.
    exc = ActorUnavailableError(error_message=MEASURED_ACTOR_DIED, actor_id=None)
    assert "serve_controller_actor" in str(exc).lower()
    assert _is_serve_controller_gone(exc) is False


def test_the_death_of_an_unrelated_actor_is_not_a_loss() -> None:
    other = MEASURED_ACTOR_DIED.replace("SERVE_CONTROLLER_ACTOR", "SomeOtherActor")
    other = other.replace("ServeController", "SomeOtherActor")
    assert _is_serve_controller_gone(_actor_died(other)) is False


@pytest.mark.parametrize(
    "message",
    [
        "There is no Serve instance running on this Ray cluster.",
        "Ray Serve: no Serve instance running",
        "The actor handle doesn't have a handle for ServeController",
    ],
)
def test_the_previously_handled_shapes_still_match(message: str) -> None:
    # These three were the whole matcher before. None of them is what Ray 2.55.1
    # was measured to produce, but dropping them would narrow the fix into a
    # swap rather than an extension.
    assert _is_serve_controller_gone(RuntimeError(message)) is True


# --------------------------------------------------------------------------
# Shape 2: serve.status RETURNS an empty status — the actor lookup decides
# --------------------------------------------------------------------------


def test_the_lookup_is_skipped_entirely_when_ray_is_not_connected(monkeypatch) -> None:
    # ray.get_actor auto-starts a local Ray instance when none is connected, so
    # the guard is not a formality: without it a monitoring probe would bring a
    # cluster up and then report its empty Serve namespace as a lost controller.
    calls = []
    monkeypatch.setattr(manager_mod.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(
        manager_mod.ray, "get_actor", lambda *a, **k: calls.append(1) or object()
    )
    assert _serve_controller_is_missing() is False
    assert calls == []


def test_the_actor_lookup_reports_gone_only_on_a_clean_not_found(monkeypatch) -> None:
    monkeypatch.setattr(manager_mod.ray, "is_initialized", lambda: True)

    def found(*_args, **_kwargs):
        return object()

    def not_found(*_args, **_kwargs):
        raise ValueError("Failed to look up actor with name 'SERVE_CONTROLLER_ACTOR'")

    def broken(*_args, **_kwargs):
        raise RuntimeError("Ray has not been initialized")

    monkeypatch.setattr(manager_mod.ray, "get_actor", found)
    assert _serve_controller_is_missing() is False
    monkeypatch.setattr(manager_mod.ray, "get_actor", not_found)
    assert _serve_controller_is_missing() is True
    # An unrelated failure must read as "present". Reporting gone here would
    # make every Ray hiccup redeploy every tracked app.
    monkeypatch.setattr(manager_mod.ray, "get_actor", broken)
    assert _serve_controller_is_missing() is False


@pytest.mark.asyncio
async def test_an_empty_status_with_a_missing_controller_sweeps(
    monkeypatch, caplog
) -> None:
    manager = _make_manager()
    monkeypatch.setattr(manager_mod, "_serve_controller_is_missing", lambda: True)

    with caplog.at_level(logging.INFO, logger="test.controller_loss"):
        await _tick(manager)

    manager._deploy_application.assert_awaited_once()
    assert len(_controller_lost_lines(caplog)) == 1
    # On the very first tick, not after wearing through the per-app tolerance.
    assert manager._missing_from_status == {}


@pytest.mark.asyncio
async def test_an_empty_status_with_a_live_controller_does_not_sweep(
    monkeypatch, caplog
) -> None:
    # The control that makes the test above mean something: an idle worker whose
    # apps are all stopped sees the identical empty status every tick.
    manager = _make_manager()
    monkeypatch.setattr(manager_mod, "_serve_controller_is_missing", lambda: False)

    with caplog.at_level(logging.INFO, logger="test.controller_loss"):
        await _tick(manager)

    assert _controller_lost_lines(caplog) == []
    manager._deploy_application.assert_not_awaited()
    # It went down the per-app path instead, exactly as before this change.
    assert manager._missing_from_status == {APP_ID: 1}


@pytest.mark.asyncio
async def test_a_populated_status_never_consults_the_actor_lookup(
    monkeypatch,
) -> None:
    # The lookup is an RPC. It must stay on the rare branch rather than run on
    # every monitoring tick of every healthy worker.
    manager = _make_manager()
    manager.ray_cluster.call_with_reconnect.return_value = SimpleNamespace(
        applications={APP_ID: SimpleNamespace(status=SimpleNamespace(value="RUNNING"))}
    )
    calls = []
    monkeypatch.setattr(
        manager_mod, "_serve_controller_is_missing", lambda: calls.append(1) or True
    )

    await _tick(manager)

    assert calls == []


# --------------------------------------------------------------------------
# Rate: now that the path can fire, show it is bounded
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_sweep_rate_is_bounded_by_the_cooldown(monkeypatch, caplog) -> None:
    manager = _make_manager()
    monkeypatch.setattr(manager_mod, "_serve_controller_is_missing", lambda: True)

    with caplog.at_level(logging.INFO, logger="test.controller_loss"):
        for _ in range(10):
            await _tick(manager)

    # Ten ticks inside one cooldown window, one sweep. The monitoring loop runs
    # every 10 s by default against a 25 s cooldown, so a persistent loss sweeps
    # at most once per three ticks rather than on every one.
    assert len(_controller_lost_lines(caplog)) == 1


@pytest.mark.asyncio
async def test_a_persistent_loss_still_converges_through_the_per_app_path(
    monkeypatch, caplog
) -> None:
    # The flip side of bounding the sweep: the ticks it suppresses fall through
    # to the per-app path, so an app that the first sweep's redeploy did not
    # bring back is still retried on its own backoff. Bounding the sweep must
    # not become a cap on recovery.
    manager = _make_manager()
    monkeypatch.setattr(manager_mod, "_serve_controller_is_missing", lambda: True)

    with caplog.at_level(logging.WARNING, logger="test.controller_loss"):
        for _ in range(10):
            await _tick(manager)

    assert manager._deploy_application.await_count > 1
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("absent from the Ray Serve status report" in w for w in warnings)


@pytest.mark.asyncio
async def test_the_sweep_runs_again_once_the_cooldown_has_elapsed(
    monkeypatch, caplog
) -> None:
    # Bounded must not mean once-ever: a loss that outlives the first sweep has
    # to keep being acted on.
    manager = _make_manager()
    monkeypatch.setattr(manager_mod, "_serve_controller_is_missing", lambda: True)

    with caplog.at_level(logging.INFO, logger="test.controller_loss"):
        await _tick(manager)
        manager._last_controller_recovery = (
            time.time() - _CONTROLLER_RECOVERY_COOLDOWN_SECONDS - 1
        )
        await _tick(manager)

    assert len(_controller_lost_lines(caplog)) == 2


@pytest.mark.asyncio
async def test_a_cooled_down_sweep_falls_back_to_the_per_app_path(
    monkeypatch, caplog
) -> None:
    # The sweep skips the per-app loop for the tick it runs on, because every
    # app is absent for one reason it just acted on. It must not skip it on the
    # ticks where the cooldown suppressed it, or an app the sweep cannot rebuild
    # would never be reported at all.
    manager = _make_manager(built_app=False)
    monkeypatch.setattr(manager_mod, "_serve_controller_is_missing", lambda: True)
    manager._last_controller_recovery = time.time()  # cooled down from the start

    with caplog.at_level(logging.WARNING, logger="test.controller_loss"):
        for _ in range(_MISSING_FROM_STATUS_TOLERANCE):
            await _tick(manager)

    assert _controller_lost_lines(caplog) == []
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("recovered from a previous worker" in line for line in warnings), (
        "an adopted app must still be reported through the per-app path when "
        "the sweep is cooling down"
    )
