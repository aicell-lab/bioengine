"""Pin that absence from the Serve status report is unknown, not unhealthy.

``monitor_applications`` used to compute::

    unhealthy = application is None or state in ("DEPLOY_FAILED", "UNHEALTHY")

which reads "Ray did not mention this app" as if it were "Ray says this app is
broken". They are not the same claim: the first can simply mean the status
report was incomplete for a tick, and acting on it tears down a healthy app and
destroys whatever it was in the middle of. The first such observation fired a
redeploy immediately — the backoff only starts *after* the first fire.

The counterpart constraint is that unknown must still converge: never
redeploying a genuinely dead app is its own outage, so absence is tolerated for
a finite number of consecutive ticks and then let through.

Both fires also used to produce the same log line, so an "absent from the
report" restart and a genuine UNHEALTHY restart were indistinguishable after
the fact — which is why nobody could say how often the first already happened.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import _MISSING_FROM_STATUS_TOLERANCE, AppsManager

APP_ID = "model-runner"
ENTRY = "ModelRunner"
VERSION = "2.8.0"


def _serve_status(state: str | None):
    """A serve.status() stand-in; ``None`` means the app is absent from it."""
    if state is None:
        return SimpleNamespace(applications={})
    return SimpleNamespace(
        applications={APP_ID: SimpleNamespace(status=SimpleNamespace(value=state))}
    )


def _make_manager() -> AppsManager:
    is_deployed = asyncio.Event()
    is_deployed.set()

    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.monitor")
    manager._redeploy_backoff = {}
    manager._missing_from_status = {}
    manager._deleted_pending_redeploy = set()
    manager._deployed_applications = {
        APP_ID: {
            "is_deployed": is_deployed,
            "artifact_id": "bioimage-io/model-runner",
            "version": VERSION,
            "auto_redeploy": True,
            "built_app": SimpleNamespace(
                spec={"entry_id": "cid0", "classes": {"cid0": {"qualname": ENTRY}}}
            ),
            "deployment_task": None,
        }
    }

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
    manager._identity_warned = set()

    # Let the real _fire_redeploy run so its log line is under test; only the
    # deploy coroutine it schedules is stubbed out. _verify_running_identities
    # runs for real off the two stubs above — mocking its return value would
    # pin this file to that method's tuple width, which a sibling change to the
    # same monitor widens.
    manager._deploy_application = AsyncMock()

    return manager


def _replica_running(version: str) -> tuple:
    """Serve details + baked identities for one RUNNING entry replica."""
    details = {
        "applications": {
            APP_ID: {
                "deployments": {
                    ENTRY: {"replicas": [{"replica_id": "r1", "state": "RUNNING"}]}
                }
            }
        }
    }
    return details, {ENTRY: {"r1": {"version": version}}}


async def _tick(manager: AppsManager, state: str | None) -> None:
    manager.ray_cluster.call_with_reconnect.return_value = _serve_status(state)
    await manager.monitor_applications()
    # Let any task _fire_redeploy scheduled actually start.
    await asyncio.sleep(0)


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    ]


def test_the_tolerance_is_finite_and_above_one() -> None:
    # The two constraints the rest of this file rests on, asserted directly
    # because every other test counts its ticks off this constant: at 1 they
    # would silently degenerate into the old fire-on-first-absence behaviour
    # while still passing, and unbounded would make unknown a permanent veto.
    assert 1 < _MISSING_FROM_STATUS_TOLERANCE < 100


@pytest.mark.asyncio
async def test_absence_below_the_tolerance_does_not_redeploy(caplog) -> None:
    manager = _make_manager()

    with caplog.at_level(logging.INFO, logger="test.monitor"):
        for _ in range(_MISSING_FROM_STATUS_TOLERANCE - 1):
            await _tick(manager, None)

    manager._deploy_application.assert_not_awaited()
    assert _warnings(caplog) == [], (
        "An app merely missing from the status report must not be restarted — "
        "the report may simply have been incomplete for a tick."
    )
    assert any("treating as unknown" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_absence_converges_and_eventually_redeploys(caplog) -> None:
    # Unknown may not be a permanent veto: an app that is genuinely gone has to
    # be recovered, or tolerance turns into its own outage.
    manager = _make_manager()

    with caplog.at_level(logging.WARNING, logger="test.monitor"):
        for _ in range(_MISSING_FROM_STATUS_TOLERANCE):
            await _tick(manager, None)

    manager._deploy_application.assert_awaited_once()
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "absent from the Ray Serve status report" in warnings[0]
    assert str(_MISSING_FROM_STATUS_TOLERANCE) in warnings[0]


@pytest.mark.asyncio
async def test_reappearing_resets_the_tolerance() -> None:
    # The counter is about *consecutive* absences. A flapping report that never
    # misses the app twice running must never accumulate its way to a restart.
    manager = _make_manager()

    for _ in range(3):
        for _ in range(_MISSING_FROM_STATUS_TOLERANCE - 1):
            await _tick(manager, None)
        await _tick(manager, "RUNNING")

    manager._deploy_application.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_transitional_state_never_redeploys() -> None:
    manager = _make_manager()

    for state in ("NOT_STARTED", "DEPLOYING", "DELETING"):
        for _ in range(_MISSING_FROM_STATUS_TOLERANCE + 2):
            await _tick(manager, state)

    manager._deploy_application.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_explicit_unhealthy_verdict_still_fires_at_once(caplog) -> None:
    # The tolerance applies to absence only. An explicit verdict is a fact, and
    # delaying recovery on it would be a regression.
    manager = _make_manager()

    with caplog.at_level(logging.WARNING, logger="test.monitor"):
        await _tick(manager, "UNHEALTHY")

    manager._deploy_application.assert_awaited_once()
    assert "Ray Serve reports status UNHEALTHY" in _warnings(caplog)[0]


@pytest.mark.asyncio
async def test_deploy_failed_fires_at_once(caplog) -> None:
    manager = _make_manager()

    with caplog.at_level(logging.WARNING, logger="test.monitor"):
        await _tick(manager, "DEPLOY_FAILED")

    manager._deploy_application.assert_awaited_once()
    assert "Ray Serve reports status DEPLOY_FAILED" in _warnings(caplog)[0]


@pytest.mark.asyncio
async def test_a_deliberate_delete_skips_the_tolerance(caplog) -> None:
    # The stale-replica branch deletes the app expecting the *next* tick to
    # redeploy it. That absence is known, not unknown, so making it wait out the
    # tolerance would triple the recovery latency of a path that is already
    # degraded.
    manager = _make_manager()
    details, identities = _replica_running("2.7.0")
    manager.ray_cluster.proxy_actor_handle.get_serve_instance_details.remote.return_value = (
        details
    )
    manager.ray_cluster.proxy_actor_handle.get_replica_identities.remote.return_value = (
        identities
    )

    with caplog.at_level(logging.WARNING, logger="test.monitor"):
        await _tick(manager, "RUNNING")
        manager._deploy_application.assert_not_awaited()
        await _tick(manager, None)

    manager._deploy_application.assert_awaited_once()
    assert any("version mismatch" in line for line in _warnings(caplog))
    assert manager._missing_from_status == {}


@pytest.mark.asyncio
async def test_a_verified_running_app_clears_a_stale_delete_marker() -> None:
    # If the app comes back healthy on its own the marker must go, or a much
    # later genuine status gap would bypass the tolerance it is entitled to.
    manager = _make_manager()
    manager._deleted_pending_redeploy.add(APP_ID)

    await _tick(manager, "RUNNING")
    for _ in range(_MISSING_FROM_STATUS_TOLERANCE - 1):
        await _tick(manager, None)

    manager._deploy_application.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_two_causes_are_separable_in_the_log(caplog) -> None:
    # The whole point of the reason string: after an incident you must be able
    # to tell a status-report gap from a real failure verdict.
    absent_manager = _make_manager()
    verdict_manager = _make_manager()

    with caplog.at_level(logging.WARNING, logger="test.monitor"):
        for _ in range(_MISSING_FROM_STATUS_TOLERANCE):
            await _tick(absent_manager, None)
        await _tick(verdict_manager, "UNHEALTHY")

    absent_line, verdict_line = _warnings(caplog)
    assert absent_line != verdict_line
    assert "UNHEALTHY" not in absent_line
    assert "absent" not in verdict_line
