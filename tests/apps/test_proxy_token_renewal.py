"""A proxy's Hypha token is renewed before its 30-day TTL runs out.

Without this, a deploy was the only thing that ever minted one. An app in the
worker's startup list is redeployed on every worker restart and so quietly got
a fresh token; an **adopted** app kept the one from its original deploy and died
at exactly 30 days with ``HTTP Error 401: Unauthorized`` — surfaced as
``ProxyDeployment=UNHEALTHY``, which points at the app rather than at the
credential. smart-microscopy-assistant failed that way at KTH on 2026-10-18 at
32.1 days; cellpose3-runner was measured at 16.3 days with 13.7 left.

**The proxy cannot renew its own token, and an earlier version of this file
asserted that it could.** Hypha's ``generate_token`` requires admin on the
target workspace; the proxy is given ``read_write`` and gets
``PermissionError: Only admin can generate token.`` — measured against
hypha.aicell.io on 2026-10-27. Every test here used to feed the proxy a fake
client whose ``generate_token`` always succeeded, so the suite was green on a
mechanism the server refuses. The renewal is therefore split: the worker holds
the admin credential and mints, the proxy actor holds the result, and the
replica collects it.

The clock is moved rather than waited on, so every threshold case below is a
real call with a synthetic age.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps import manager as manager_module
from bioengine.apps import proxy_deployment as pd_module
from bioengine.apps.manager import _TOKEN_RENEW_AFTER_FRACTION, AppsManager
from tests.apps._proxy_double import ProxyDouble

TTL = 3600 * 24 * 30  # the production value, builder.py:738
DAY = 86400
APP_ID = "cellpose3-runner"
RENEWED = "RENEWED-NOT-A-CREDENTIAL"
ORIGINAL = "ORIGINAL-NOT-A-CREDENTIAL"


# ===== the proxy must not try to mint: the server refuses it =====


def test_the_proxy_never_calls_generate_token() -> None:
    """The regression guard for the measured fact.

    A re-introduced self-mint fails in the worst possible way: renewal is on the
    advisory path, so Hypha's refusal is one warning per tick and the app dies
    of expiry 30 days later with a 401 that points at the app. Asserted over the
    AST rather than by grepping, because a call split across adjacent string
    literals is invisible to a text search.
    """
    tree = ast.parse(Path(pd_module.__file__).read_text())
    calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate_token"
    ]
    assert calls == [], (
        f"proxy_deployment.py calls generate_token at line(s) {calls}; the proxy "
        f"holds a read_write token and Hypha requires admin, so this cannot work"
    )


def test_the_worker_is_the_one_that_mints() -> None:
    """The positive control for the test above: the call did not just vanish."""
    tree = ast.parse(Path(manager_module.__file__).read_text())
    calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate_token"
    ]
    assert calls, "nothing in the apps manager mints a token any more"


# ===== the worker side: when it mints, and what it does with the result =====


class _Server:
    """Minimal stand-in for the worker's admin Hypha client."""

    def __init__(self, token: str = RENEWED, fail: Exception | None = None):
        self.token = token
        self.fail = fail
        self.calls: list[dict] = []
        self.config = SimpleNamespace(workspace="bioimage-io")

    async def generate_token(self, config):
        self.calls.append(config)
        if self.fail is not None:
            raise self.fail
        return self.token


def _manager(
    age_days: float,
    *,
    server: _Server | None = None,
    ttl: float = TTL,
    auto_redeploy: bool = False,
    pending: bool = False,
    reported_issued_at: float | None = None,
) -> AppsManager:
    is_deployed = asyncio.Event()
    is_deployed.set()

    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.renewal")
    manager.server = _Server() if server is None else server
    manager._redeploy_backoff = {}
    manager._missing_from_status = {}
    manager._deleted_pending_redeploy = set()
    manager._identity_warned = set()
    manager._deploy_application = AsyncMock()
    manager._deployed_applications = {
        APP_ID: {
            "is_deployed": is_deployed,
            "artifact_id": "bioimage-io/cellpose3-runner",
            "version": "1.0.0",
            # Default False on purpose: the app that hit this in production was
            # adopted, so the flag was unset and the auto-redeploy filter would
            # have skipped it.
            "auto_redeploy": auto_redeploy,
            "entry_deployment_name": "Entry",
            "built_app": None,
            "deployment_task": None,
            "proxy_service_token_issued_at": time.time() - age_days * DAY,
            "proxy_service_token_ttl_seconds": float(ttl),
        }
    }

    ray_cluster = MagicMock()
    ray_cluster.check_connection = AsyncMock()
    ray_cluster.call_with_reconnect = AsyncMock(
        return_value=SimpleNamespace(applications={})
    )
    handle = ray_cluster.proxy_actor_handle
    handle.get_reported_token_issued_at.remote = AsyncMock(
        return_value=reported_issued_at
    )
    handle.has_pending_token.remote = AsyncMock(return_value=pending)
    handle.set_pending_token.remote = AsyncMock(return_value=None)
    handle.get_serve_instance_details.remote = AsyncMock(return_value={})
    handle.get_replica_identities.remote = AsyncMock(return_value={})
    manager.ray_cluster = ray_cluster

    return manager


def _deposits(manager: AppsManager) -> list:
    return manager.ray_cluster.proxy_actor_handle.set_pending_token.remote.await_args_list


@pytest.mark.asyncio
async def test_a_fresh_token_is_left_alone() -> None:
    manager = _manager(age_days=1)
    await manager._renew_proxy_tokens()
    assert manager.server.calls == []
    assert _deposits(manager) == []


@pytest.mark.asyncio
async def test_renewal_happens_at_two_thirds_of_the_ttl() -> None:
    """Early enough to matter: an expired token cannot be exchanged for a fresh
    one, so the margin is the point. Two thirds of 30 days leaves ten."""
    just_before = _manager(age_days=19.9)
    await just_before._renew_proxy_tokens()
    assert just_before.server.calls == [], "renewed too early"

    just_after = _manager(age_days=20.1)
    await just_after._renew_proxy_tokens()
    assert len(just_after.server.calls) == 1, "did not renew at the threshold"


@pytest.mark.asyncio
async def test_the_adopted_app_case_that_broke_production() -> None:
    """16.3 days is cellpose3-runner as measured; 25 is past the threshold."""
    survives = _manager(age_days=16.3)
    await survives._renew_proxy_tokens()
    assert survives.server.calls == []

    renews = _manager(age_days=25)
    await renews._renew_proxy_tokens()
    assert len(_deposits(renews)) == 1
    assert _deposits(renews)[0].kwargs["token"] == RENEWED


@pytest.mark.asyncio
async def test_it_asks_for_the_same_scope_the_proxy_was_given() -> None:
    """A narrower renewal would fail later and elsewhere, when the proxy next
    needs a permission the original token had."""
    manager = _manager(age_days=25)
    await manager._renew_proxy_tokens()
    assert manager.server.calls[0] == {
        "workspace": "bioimage-io",
        "permission": "read_write",
        "expires_in": TTL,
    }


@pytest.mark.asyncio
async def test_the_deposit_carries_the_ttl_so_the_proxy_can_renew_again() -> None:
    """The replica adopts issue time and TTL from the deposit. Without the TTL a
    collected token would inherit whatever the replica last had, and a proxy
    started from a deploy-time value would renew on the wrong schedule."""
    manager = _manager(age_days=25)
    before = time.time()
    await manager._renew_proxy_tokens()
    kwargs = _deposits(manager)[0].kwargs
    assert kwargs["application_id"] == APP_ID
    assert kwargs["ttl_seconds"] == TTL
    assert before <= kwargs["issued_at"] <= time.time()


@pytest.mark.asyncio
async def test_an_uncollected_token_is_not_minted_twice() -> None:
    """Each mint is a live credential. Minting on every tick while a replica is
    slow to collect would leave a trail of valid tokens nothing accounts for."""
    manager = _manager(age_days=25, pending=True)
    await manager._renew_proxy_tokens()
    assert manager.server.calls == []
    assert _deposits(manager) == []


@pytest.mark.asyncio
async def test_the_replicas_own_report_wins_over_the_deploy_time_value() -> None:
    """After one renewal the deploy-time value is 25 days stale. Reading it
    instead of the replica's report would re-mint on every tick forever."""
    manager = _manager(age_days=25, reported_issued_at=time.time() - DAY)
    await manager._renew_proxy_tokens()
    assert manager.server.calls == [], "renewed against a superseded issue time"


@pytest.mark.asyncio
async def test_a_zero_ttl_disables_renewal() -> None:
    manager = _manager(age_days=25, ttl=0)
    await manager._renew_proxy_tokens()
    assert manager.server.calls == []


@pytest.mark.asyncio
async def test_no_hypha_client_yet_is_not_an_error() -> None:
    manager = _manager(age_days=25)
    manager.server = None
    await manager._renew_proxy_tokens()  # must not raise


@pytest.mark.asyncio
async def test_a_failed_mint_is_advisory(caplog) -> None:
    """Days of margin remain, so the next tick can retry. Failing the caller's
    whole pass over a retryable mint would be worse than the expiry it
    prevents."""
    manager = _manager(age_days=25, server=_Server(fail=RuntimeError("hypha down")))
    with caplog.at_level(logging.WARNING, logger="test.renewal"):
        await manager._renew_proxy_tokens()
    assert _deposits(manager) == []
    assert any("hypha down" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_failure_reading_the_issue_time_is_advisory() -> None:
    """The warning interpolates the age, which is unknown on this path — an
    unbound local here would turn an advisory failure into a raise."""
    manager = _manager(age_days=25)
    manager.ray_cluster.proxy_actor_handle.get_reported_token_issued_at.remote = (
        AsyncMock(side_effect=RuntimeError("actor gone"))
    )
    await manager._renew_proxy_tokens()  # must not raise


@pytest.mark.asyncio
async def test_one_apps_failure_does_not_stop_the_next() -> None:
    manager = _manager(age_days=25)
    second = dict(manager._deployed_applications[APP_ID])
    manager._deployed_applications["other-app"] = second
    calls = {"n": 0}

    async def flaky(config):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return RENEWED

    manager.server.generate_token = flaky
    await manager._renew_proxy_tokens()
    assert len(_deposits(manager)) == 1, "a failure on the first app skipped the rest"


# ===== it must not sit behind the auto-redeploy filter =====


@pytest.mark.asyncio
async def test_the_monitor_renews_an_app_that_opted_out_of_auto_redeploy() -> None:
    """``monitor_applications`` returns early when no app has ``auto_redeploy``.
    A token expires either way, and the app that hit this in production was an
    adopted one with the flag unset — so a renewal placed after that filter
    would reach nobody.
    """
    manager = _manager(age_days=25, auto_redeploy=False)
    await manager.monitor_applications()
    assert len(_deposits(manager)) == 1, (
        "monitor_applications skipped renewal for an app with auto_redeploy off"
    )


@pytest.mark.asyncio
async def test_the_monitor_still_renews_when_auto_redeploy_is_on() -> None:
    manager = _manager(age_days=25, auto_redeploy=True)
    await manager.monitor_applications()
    assert len(_deposits(manager)) == 1


# ===== the proxy side: collecting the deposit =====


def _proxy(pending=None, *, handle: bool = True, fail: Exception | None = None):
    inst = object.__new__(ProxyDouble)
    inst.application_id = APP_ID
    inst.workspace = "bioimage-io"
    inst.entry_deployment_ready = True
    inst.proxy_service_token = ORIGINAL
    inst._token_issued_at = time.time() - 25 * DAY
    inst._token_ttl_seconds = float(TTL)
    inst._connection_lost = False
    inst.server = SimpleNamespace()

    if handle:
        actor = MagicMock()
        actor.take_pending_token.remote = AsyncMock(
            return_value=pending, side_effect=fail
        )
        inst._proxy_actor_handle = actor
    else:
        inst._proxy_actor_handle = None
    return inst


def _deposit(issued_at: float | None = None, ttl: float = TTL) -> dict:
    return {
        "token": RENEWED,
        "issued_at": time.time() if issued_at is None else issued_at,
        "ttl_seconds": ttl,
    }


@pytest.mark.asyncio
async def test_nothing_waiting_is_a_no_op() -> None:
    inst = _proxy(pending=None)
    await inst._collect_renewed_token()
    assert inst.proxy_service_token == ORIGINAL
    assert inst._connection_lost is False


@pytest.mark.asyncio
async def test_a_waiting_token_is_adopted_with_its_clock() -> None:
    issued = time.time() - 60
    inst = _proxy(pending=_deposit(issued_at=issued, ttl=TTL / 2))
    await inst._collect_renewed_token()
    assert inst.proxy_service_token == RENEWED
    assert inst._token_issued_at == issued
    assert inst._token_ttl_seconds == TTL / 2


@pytest.mark.asyncio
async def test_collection_forces_a_client_rebuild() -> None:
    """The live client still holds the old token — nothing re-authenticates an
    established connection. Without the rebuild the renewal is recorded and
    never used, which looks exactly like success.
    """
    inst = _proxy(pending=_deposit())
    await inst._collect_renewed_token()
    assert inst._connection_lost is True, (
        "collected without forcing a rebuild, so the stale token stays in use"
    )


@pytest.mark.asyncio
async def test_the_token_value_is_never_logged(caplog) -> None:
    """These logs are readable through ``get_app_status``."""
    inst = _proxy(pending=_deposit())
    with caplog.at_level(logging.INFO, logger=pd_module.logger.name):
        await inst._collect_renewed_token()
    assert any("Collected a renewed" in r.getMessage() for r in caplog.records)
    assert all(RENEWED not in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_an_unreachable_actor_keeps_the_working_token() -> None:
    inst = _proxy(fail=RuntimeError("actor gone"))
    await inst._collect_renewed_token()
    assert inst.proxy_service_token == ORIGINAL
    assert inst._connection_lost is False, "a failed collection must not force a rebuild"


@pytest.mark.asyncio
async def test_no_actor_handle_is_not_an_error() -> None:
    """A replica that could not resolve the actor logs that at construction; the
    maintenance loop must still run."""
    inst = _proxy(handle=False)
    await inst._collect_renewed_token()  # must not raise


# ===== the maintenance loop drives it =====


@pytest.mark.asyncio
async def test_the_maintenance_tick_collects() -> None:
    """A collection nothing invokes is the original bug in a new place."""
    inst = _proxy(pending=_deposit())
    inst._flush_usage = lambda: asyncio.sleep(0)
    inst.websocket_service_id = "ws-id"
    inst._next_register_at = 0.0
    inst._registration_lock = asyncio.Lock()
    inst._register_services = lambda: asyncio.sleep(0)
    inst._report_service_registration = lambda ok: None
    inst._probe_due_at = 0.0
    inst._registration_failure = None
    inst._is_permanent_registration_error = lambda e: False

    await inst._maintenance_tick()
    assert inst.proxy_service_token == RENEWED, (
        "_maintenance_tick did not collect; the token would expire untouched"
    )


@pytest.mark.asyncio
async def test_the_maintenance_tick_does_not_raise_on_a_bare_replica() -> None:
    """``_maintenance_tick`` documents that it never raises. The deleted
    renewal constant left its callee raising ``NameError`` on every tick, which
    the loop swallowed into one warning while silently dropping re-registration,
    the reachability probe and usage reporting for the rest of the replica's
    life.
    """
    inst = _proxy(handle=False)
    inst._flush_usage = lambda: asyncio.sleep(0)
    inst.entry_deployment_ready = False
    await inst._maintenance_tick()


# ===== the constant =====


def test_the_renewal_fraction_leaves_real_margin() -> None:
    """Guards the constant itself. Renewing with only hours left would make the
    fix depend on the worker being up at the right moment."""
    margin_days = TTL * (1 - _TOKEN_RENEW_AFTER_FRACTION) / DAY
    assert margin_days >= 7, f"only {margin_days:.1f} days of margin"
