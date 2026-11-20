"""Auto-redeploy stops after a bounded number of attempts, and says it stopped.

A Hypha outage of under a minute can kill an app replica that happens to be
starting, permanently: ``async_init`` runs from inside ``check_health``, Ray
Serve counts an exception out of it as a failed replica *start*, and three
failed starts back to back are terminal. The app lands in ``DEPLOY_FAILED``.

``auto_redeploy`` is the recovery path, and it worked — but only in one
direction. The per-app exponential backoff capped how OFTEN a redeploy fires
(10 s doubling to a 600 s floor) and never capped HOW MANY times, so:

* an app that WOULD recover came back unaided — the good case, and the one the
  existing tests cover;
* an app that can NEVER recover (deleted artifact, GPU gone, a manifest this
  worker refuses) was redeployed every 10 minutes for as long as the worker
  lived. Each attempt is a real ``serve.run``: it builds a runtime_env,
  schedules replicas, and on a GPU app holds the reservation while it fails.

"Bounded rate, unbounded count" is not bounded. Both halves have to hold at
once, which is why the two arms are in one file: a fix that stops retrying
satisfies the first alone, and a fix that retries forever satisfies the second
alone, and each reads like success if you only look at its own arm.

The give-up is also deliberately loud. An app that is quietly no longer being
retried is the same invisible failure as an app that was never retried — which
is how a dead production app went unnoticed for 19 hours while the worker kept
reporting itself ready.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import (
    _REDEPLOY_BACKOFF_INITIAL_SECONDS,
    _REDEPLOY_BACKOFF_MAX_SECONDS,
    _REDEPLOY_MAX_ATTEMPTS,
    AppsManager,
)

APP_ID = "model-runner"
OTHER_APP_ID = "model-finetune"
ENTRY = "ModelRunner"
VERSION = "2.8.0"


def _serve_status(state: str, *, also: dict | None = None):
    apps = {APP_ID: SimpleNamespace(status=SimpleNamespace(value=state))}
    for app_id, app_state in (also or {}).items():
        apps[app_id] = SimpleNamespace(status=SimpleNamespace(value=app_state))
    return SimpleNamespace(applications=apps)


def _app_entry(entry: str = ENTRY) -> dict:
    is_deployed = asyncio.Event()
    is_deployed.set()
    return {
        "is_deployed": is_deployed,
        "artifact_id": "bioimage-io/model-runner",
        "version": VERSION,
        "auto_redeploy": True,
        "entry_deployment_name": entry,
        "built_app": SimpleNamespace(
            spec={"entry_id": f"deployment:{entry}", "classes": {}}
        ),
        "deployment_task": None,
    }


def _make_manager(*, extra_app: bool = False) -> AppsManager:
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.giveup")
    manager.server = None
    manager._redeploy_backoff = {}
    manager._missing_from_status = {}
    manager._deleted_pending_redeploy = set()
    manager._identity_warned = set()
    manager._deployed_applications = {APP_ID: _app_entry()}
    if extra_app:
        manager._deployed_applications[OTHER_APP_ID] = _app_entry("FineTune")

    ray_cluster = MagicMock()
    ray_cluster.check_connection = AsyncMock()
    ray_cluster.call_with_reconnect = AsyncMock()
    ray_cluster.proxy_actor_handle.get_serve_instance_details.remote = AsyncMock(
        return_value={}
    )
    ray_cluster.proxy_actor_handle.get_replica_identities.remote = AsyncMock(
        return_value={}
    )
    manager.ray_cluster = ray_cluster
    manager._deploy_application = AsyncMock()
    return manager


class _Clock:
    """A clock the test advances, so the backoff deadline is honoured for real.

    Freezing time instead (or zeroing ``next_attempt_at`` between ticks) would
    let the cap be "reached" by a route the production code never takes, and the
    arms below count attempts.
    """

    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance_past_the_backoff(self) -> None:
        # Jump the whole cap, so every tick is past its deadline whatever the
        # doubling has reached.
        self.now += _REDEPLOY_BACKOFF_MAX_SECONDS + 1.0


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    import bioengine.apps.manager as manager_module

    c = _Clock()
    monkeypatch.setattr(manager_module.time, "time", c)
    return c


async def _tick(manager: AppsManager, state: str, **kwargs) -> None:
    manager.ray_cluster.call_with_reconnect.return_value = _serve_status(
        state, **kwargs
    )
    await manager.monitor_applications()
    await asyncio.sleep(0)


def _errors(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def _status_field(manager: AppsManager, app_id: str = APP_ID):
    """Read the give-up exactly where _get_app_status reads it.

    Deliberately NOT the monitor's backoff dict: the published field is what a
    caller sees, and reading the internal bookkeeping instead would pass against
    a version that records the give-up and never reports it.
    """
    return (manager._deployed_applications.get(app_id) or {}).get(
        "auto_redeploy_exhausted_at"
    )


# ── the constant itself ──────────────────────────────────────────────────────


def test_the_cap_is_finite_and_leaves_room_for_a_real_outage() -> None:
    """Both bounds, asserted directly, because every arm below counts off it.

    At 1 the worker would give up on the first transient failure, which is the
    19-hour outage described above. Unbounded is the defect. And the
    schedule has to outlast a platform restart: with the doubling capped at
    `_REDEPLOY_BACKOFF_MAX_SECONDS`, the cap must buy at least half an hour.
    """
    assert 1 < _REDEPLOY_MAX_ATTEMPTS < 1000

    total = sum(
        min(
            _REDEPLOY_BACKOFF_MAX_SECONDS,
            _REDEPLOY_BACKOFF_INITIAL_SECONDS * (2 ** (n - 1)),
        )
        for n in range(1, _REDEPLOY_MAX_ATTEMPTS)
    )
    assert total >= 1800, (
        f"the cap allows only {total:.0f}s of retrying, which is shorter than a "
        f"node drain or an image pull — a recoverable app would be abandoned"
    )


# ── ARM 1: an app that never recovers is given up on, loudly ─────────────────


@pytest.mark.asyncio
async def test_a_never_recovering_app_stops_being_redeployed(clock, caplog) -> None:
    manager = _make_manager()

    with caplog.at_level(logging.ERROR, logger="test.giveup"):
        # Enough ticks to blow well past the cap if nothing stops it.
        for _ in range(_REDEPLOY_MAX_ATTEMPTS * 3):
            await _tick(manager, "DEPLOY_FAILED")
            clock.advance_past_the_backoff()

    fired = manager._deploy_application.await_count
    assert fired == _REDEPLOY_MAX_ATTEMPTS, (
        f"fired {fired} redeploys against a cap of {_REDEPLOY_MAX_ATTEMPTS}; "
        f"{'the cap is not enforced' if fired > _REDEPLOY_MAX_ATTEMPTS else 'it gave up early'}"
    )

    # CONTROL on the arm itself: the cap must have been REACHED, not merely
    # never exceeded. A manager that fired nothing at all also satisfies
    # "<= cap", and that is the opposite defect.
    assert fired > 1, "nothing was retried, so this arm proves nothing about a cap"

    assert _status_field(manager) is not None, (
        "the worker stopped retrying without recording that it had stopped — "
        "DEPLOY_FAILED reads the same whether a redeploy is due or will never come"
    )

    errors = _errors(caplog)
    assert len(errors) == 1, (
        f"expected exactly ONE give-up line, got {len(errors)}: a line per tick "
        f"buries the log it is supposed to make readable"
    )
    assert "Giving up" in errors[0] and APP_ID in errors[0]
    assert str(_REDEPLOY_MAX_ATTEMPTS) in errors[0]


@pytest.mark.asyncio
async def test_giving_up_on_one_app_does_not_touch_another(clock) -> None:
    """The cap is per-app. A shared counter would abandon a healthy neighbour."""
    manager = _make_manager(extra_app=True)

    for _ in range(_REDEPLOY_MAX_ATTEMPTS * 2):
        await _tick(manager, "DEPLOY_FAILED", also={OTHER_APP_ID: "RUNNING"})
        clock.advance_past_the_backoff()

    assert _status_field(manager, APP_ID) is not None
    assert _status_field(manager, OTHER_APP_ID) is None, (
        "a RUNNING app acquired a give-up marker from its neighbour's failures"
    )


# ── ARM 2: an app that does recover still comes back, unaided ────────────────


@pytest.mark.asyncio
async def test_an_app_that_recovers_comes_back_without_a_human(clock) -> None:
    """The half a cap can silently destroy.

    A fix that simply stops retrying passes arm 1 and fails here, which is why
    the two live together.
    """
    manager = _make_manager()

    # Fail for a while — but inside the cap — then recover.
    failures = _REDEPLOY_MAX_ATTEMPTS - 1
    for _ in range(failures):
        await _tick(manager, "DEPLOY_FAILED")
        clock.advance_past_the_backoff()

    assert manager._deploy_application.await_count == failures, (
        "the app was not being retried at all before it recovered"
    )
    assert _status_field(manager) is None, "gave up while still inside the cap"

    await _tick(manager, "RUNNING")
    assert _status_field(manager) is None, "recovery left a stale give-up marker"
    assert APP_ID not in manager._redeploy_backoff, (
        "recovery did not clear the backoff, so the next unrelated failure "
        "would start from an already-exhausted counter"
    )


@pytest.mark.asyncio
async def test_a_flapping_app_never_accumulates_its_way_to_a_give_up(clock) -> None:
    """The counter is about CONSECUTIVE failures.

    An app that recovers between failures must never reach the cap, however many
    failures it has had in total — otherwise a long-lived worker eventually
    abandons an app that has always come back.
    """
    manager = _make_manager()

    for _ in range(_REDEPLOY_MAX_ATTEMPTS * 2):
        await _tick(manager, "DEPLOY_FAILED")
        clock.advance_past_the_backoff()
        await _tick(manager, "RUNNING")
        clock.advance_past_the_backoff()

    assert _status_field(manager) is None, (
        "an app that recovered after every single failure was given up on"
    )


# ── the give-up is readable from the status surface, not only the log ────────


@pytest.mark.asyncio
async def test_the_give_up_is_reported_on_the_apps_status(clock) -> None:
    """A log line is not an API. A watcher reads status, not stdout.

    `auto_redeploy_exhausted_at` is the only thing that distinguishes "failed,
    retry due in nine minutes" from "failed, nothing will ever retry this
    again" — and conflating those is exactly what hid a dead app for 19 hours.
    """
    manager = _make_manager()
    for _ in range(_REDEPLOY_MAX_ATTEMPTS * 2):
        await _tick(manager, "DEPLOY_FAILED")
        clock.advance_past_the_backoff()

    gave_up_at = _status_field(manager)
    assert gave_up_at is not None

    import inspect

    src = inspect.getsource(AppsManager._get_app_status)
    assert "auto_redeploy_exhausted_at" in src, (
        "the give-up timestamp exists internally but _get_app_status does not "
        "publish it, so no caller can see it"
    )
    # Scoped to that one method: a module-wide grep would pass against a version
    # that mentions the key anywhere else and never reports it.
