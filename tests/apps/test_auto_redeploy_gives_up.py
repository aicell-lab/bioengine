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

import bioengine.apps.manager as _manager_module
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


class _LoopIsUnbounded(BaseException):
    """Escapes the retry loop's `except Exception`, which an AssertionError does not.

    The loop catches `Exception` around each `deploy_app` call and logs a warning,
    so a ceiling that raises `AssertionError` is swallowed and the unbounded
    mutant keeps spinning. Measured: with an AssertionError ceiling the mutant
    still ran past a 180 s wall-clock timeout with no output at all. Deriving
    from `BaseException` is what makes the ceiling reach the test runner.
    """


def _serve_status(state: str, *, also: dict | None = None):
    apps = {APP_ID: SimpleNamespace(status=SimpleNamespace(value=state))}
    for app_id, app_state in (also or {}).items():
        apps[app_id] = SimpleNamespace(status=SimpleNamespace(value=app_state))
    return SimpleNamespace(applications=apps)


def _app_entry(entry: str = ENTRY, last_updated_at: float = 1.0) -> dict:
    is_deployed = asyncio.Event()
    is_deployed.set()
    return {
        "is_deployed": is_deployed,
        # What the retry budget is keyed on: deploy_app bumps this on every
        # (re)deploy of an application_id.
        "last_updated_at": last_updated_at,
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

    # The monitor fires on the first unhealthy observation and then waits
    # between fires, so N attempts span N-1 waits.
    monitor_total = sum(
        min(
            _REDEPLOY_BACKOFF_MAX_SECONDS,
            _REDEPLOY_BACKOFF_INITIAL_SECONDS * (2 ** (n - 1)),
        )
        for n in range(1, _REDEPLOY_MAX_ATTEMPTS)
    )
    # The startup retry loop sleeps BEFORE each attempt, so N attempts span N
    # waits — a different figure from the same constants, and conflating the two
    # is how the comment on _REDEPLOY_MAX_ATTEMPTS came to claim a span about
    # double the real one.
    delay, startup_total = _REDEPLOY_BACKOFF_INITIAL_SECONDS, 0.0
    for _ in range(_REDEPLOY_MAX_ATTEMPTS):
        startup_total += delay
        delay = min(delay * 2, _REDEPLOY_BACKOFF_MAX_SECONDS)

    for label, total in (("monitor", monitor_total), ("startup", startup_total)):
        assert total >= 1800, (
            f"the {label} cap allows only {total:.0f}s of retrying, which is "
            f"shorter than a node drain or an image pull — a recoverable app "
            f"would be abandoned"
        )

    # The exact figures, so the comment documenting them cannot drift unnoticed.
    # Asserted rather than described: the previous comment's "~85 minutes" was
    # wrong by about a factor of two and nothing caught it.
    assert (monitor_total, startup_total) == (2430.0, 3030.0), (
        f"the schedule changed: monitor={monitor_total:.0f}s "
        f"startup={startup_total:.0f}s. Update the comment on "
        f"_REDEPLOY_MAX_ATTEMPTS to match before changing this assertion."
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


# ── the startup retry loop's bound ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_startup_retry_loop_is_bounded_too(monkeypatch) -> None:
    """`_retry_startup_applications` loops on its own schedule, not the monitor's.

    It retries apps that failed inside `deploy_app` and therefore have no entry
    in `_deployed_applications`, so `monitor_applications` cannot see them and
    none of the arms above touch this path. Its loop was `while pending:` with no
    attempt limit — the same unbounded defect in a second place, and reverting
    the bound left every other test in the tree green.

    Two details are load-bearing, and both were arrived at the hard way:

    * The SCHEDULE is shrunk rather than the clock stubbed. Monkeypatching
      `asyncio.sleep` globally replaces machinery pytest-asyncio's own event loop
      uses, and it turns the unbounded mutant into an infinite BUSY loop — a
      container running that mutant escaped a stopped harness and spun for four
      hours. These two constants are read from module globals on every call, so
      patching them is a data change with no effect on the loop.
    * The ceiling raises a `BaseException` subclass. The loop wraps each
      `deploy_app` call in `except Exception`, so an `AssertionError` ceiling is
      swallowed and changes nothing.
    """
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.giveup")

    attempts = {"n": 0}
    # Just above the cap: "it made more attempts than the cap allows" is the
    # whole claim, and a larger ceiling only makes the mutant slower to refuse.
    CEILING = _REDEPLOY_MAX_ATTEMPTS + 5

    async def _always_fails(**_kwargs):
        attempts["n"] += 1
        if attempts["n"] > CEILING:
            raise _LoopIsUnbounded(
                f"still retrying after {CEILING} attempts, so the loop is not "
                f"bounded by _REDEPLOY_MAX_ATTEMPTS={_REDEPLOY_MAX_ATTEMPTS}"
            )
        raise RuntimeError("artifact is gone")

    manager.deploy_app = _always_fails
    monkeypatch.setattr(_manager_module, "_REDEPLOY_BACKOFF_INITIAL_SECONDS", 0.001)
    monkeypatch.setattr(_manager_module, "_REDEPLOY_BACKOFF_MAX_SECONDS", 0.001)

    await manager._retry_startup_applications(
        [{"artifact_id": "bioimage-io/never-coming-back"}]
    )

    assert attempts["n"] == _REDEPLOY_MAX_ATTEMPTS, (
        f"made {attempts['n']} attempts against a cap of {_REDEPLOY_MAX_ATTEMPTS}"
    )
    # CONTROL: the loop must have RUN. Returning immediately also satisfies
    # "<= cap", and a startup app never retried once is the opposite defect.
    assert attempts["n"] > 1, "nothing was retried, so this proves nothing about a cap"


@pytest.mark.asyncio
async def test_the_startup_give_up_is_announced(monkeypatch, caplog) -> None:
    """A startup app that never deployed has no status row, so the log is all there is.

    `get_app_status` reads `_deployed_applications`, and an app that failed inside
    `deploy_app` has no entry there. `auto_redeploy_exhausted_at` therefore cannot
    carry this give-up, which makes the error line the only surface — and an
    unannounced give-up is the same invisible failure as an unbounded retry.
    """
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.giveup")

    seen = {"n": 0}

    async def _always_fails(**_kwargs):
        seen["n"] += 1
        if seen["n"] > _REDEPLOY_MAX_ATTEMPTS + 5:
            raise _LoopIsUnbounded("the retry loop is not bounded")
        raise RuntimeError("artifact is gone")

    manager.deploy_app = _always_fails
    monkeypatch.setattr(_manager_module, "_REDEPLOY_BACKOFF_INITIAL_SECONDS", 0.001)
    monkeypatch.setattr(_manager_module, "_REDEPLOY_BACKOFF_MAX_SECONDS", 0.001)

    with caplog.at_level(logging.ERROR, logger="test.giveup"):
        await manager._retry_startup_applications(
            [{"artifact_id": "bioimage-io/never-coming-back"}]
        )

    errors = _errors(caplog)
    assert len(errors) == 1, f"expected one give-up line, got {len(errors)}"
    assert "Giving up" in errors[0]
    assert "never-coming-back" in errors[0], (
        "the line does not name the application, so an operator cannot act on it"
    )


# ── an explicit redeploy after a give-up gets its retries back ──────────────


@pytest.mark.asyncio
async def test_an_explicit_redeploy_after_a_give_up_is_retried_again(clock) -> None:
    """The regression the attempt cap introduced, and the worst possible one.

    The give-up message tells the operator to "redeploy it explicitly once the
    cause is fixed". `deploy_app` replaces the app's tracking entry, which clears
    `auto_redeploy_exhausted_at` — but the monitor's `_redeploy_backoff` entry is
    popped only on undeploy and on RUNNING, so `consecutive_failures` stayed at
    the cap. The first unhealthy observation after that redeploy then went
    straight to the give-up branch: **zero** automatic retries, plus a second
    "10 consecutive auto-redeploys all failed" line that is false for the new
    deploy.

    Before the cap existed a stale counter only cost a longer delay. With it, it
    means never — so the fix made the documented recovery path strictly worse
    than the defect it was fixing, for anyone who followed the instruction.
    """
    manager = _make_manager()

    for _ in range(_REDEPLOY_MAX_ATTEMPTS * 2):
        await _tick(manager, "DEPLOY_FAILED")
        clock.advance_past_the_backoff()

    assert _status_field(manager) is not None, "the arm needs a give-up to have happened"
    manager._deploy_application.reset_mock()

    # Replace the tracking entry the way deploy_app does — a NEW
    # `last_updated_at`, everything else the same — and do NOT touch
    # `_redeploy_backoff`. Clearing it here would make this arm a tautology: it
    # would pass against a worker that never resets the budget, which is the
    # defect. The exhausted entry is deliberately left in place so the monitor
    # has to notice for itself that it belongs to a previous deploy.
    stale = dict(manager._redeploy_backoff.get(APP_ID) or {})
    assert stale.get("consecutive_failures") == _REDEPLOY_MAX_ATTEMPTS, (
        f"the arm needs an exhausted budget to still be present, got {stale}"
    )
    manager._deployed_applications[APP_ID] = _app_entry(last_updated_at=2.0)

    await _tick(manager, "DEPLOY_FAILED")

    assert manager._deploy_application.await_count > 0, (
        "the redeploy the give-up message asks for got no automatic retries at "
        "all — the stale attempt counter sent the first failure straight to the "
        "give-up branch"
    )
    assert _status_field(manager) is None, (
        "the fresh deploy is already marked exhausted"
    )
