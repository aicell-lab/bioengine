"""The proxy renews its own Hypha token before the 30-day TTL runs out.

Without this, a deploy was the only thing that ever minted one. An app in the
worker's startup list is redeployed on every worker restart and so quietly got
a fresh token; an **adopted** app kept the one from its original deploy and died
at exactly 30 days with ``HTTP Error 401: Unauthorized`` — surfaced as
``ProxyDeployment=UNHEALTHY``, which points at the app rather than at the
credential. smart-microscopy-assistant failed that way at KTH on 2026-10-18 at
32.1 days; cellpose3-runner was measured at 16.3 days with 13.7 left.

The clock is moved rather than waited on, so every case below is a real call
through ``_renew_token_if_due`` with a synthetic age.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from bioengine.apps import proxy_deployment as pd_module
from tests.apps._proxy_double import ProxyDouble

TTL = 3600 * 24 * 30  # the production value, builder.py:738
DAY = 86400


class _Server:
    """Minimal stand-in for the Hypha client the proxy holds."""

    def __init__(self, token="RENEWED-NOT-A-CREDENTIAL", fail=None):
        self.token = token
        self.fail = fail
        self.calls = []

    async def generate_token(self, config):
        self.calls.append(config)
        if self.fail is not None:
            raise self.fail
        return self.token


def _proxy(age_days, server=None, ttl=TTL):
    inst = object.__new__(ProxyDouble)
    inst.application_id = "app"
    inst.workspace = "ws"
    inst.entry_deployment_ready = True
    inst.proxy_service_token = "ORIGINAL-NOT-A-CREDENTIAL"
    inst._token_issued_at = time.time() - age_days * DAY
    inst._token_ttl_seconds = float(ttl)
    inst._connection_lost = False
    inst.server = server if server is not None else _Server()
    return inst


# ===== when it renews =====


@pytest.mark.asyncio
async def test_a_fresh_token_is_left_alone() -> None:
    inst = _proxy(age_days=1)
    await inst._renew_token_if_due()
    assert inst.server.calls == []
    assert inst.proxy_service_token == "ORIGINAL-NOT-A-CREDENTIAL"
    assert inst._connection_lost is False


@pytest.mark.asyncio
async def test_renewal_happens_at_two_thirds_of_the_ttl() -> None:
    """Early enough to matter: an expired token cannot mint its replacement, so
    the margin is the point. Two thirds of 30 days leaves ten."""
    just_before = _proxy(age_days=19.9)
    await just_before._renew_token_if_due()
    assert just_before.server.calls == [], "renewed too early"

    just_after = _proxy(age_days=20.1)
    await just_after._renew_token_if_due()
    assert len(just_after.server.calls) == 1, "did not renew at the threshold"


@pytest.mark.asyncio
async def test_the_adopted_app_case_that_broke_production() -> None:
    """16.3 days is cellpose3-runner as measured; 25 is past the threshold."""
    survives = _proxy(age_days=16.3)
    await survives._renew_token_if_due()
    assert survives.server.calls == []

    renews = _proxy(age_days=25)
    await renews._renew_token_if_due()
    assert renews.proxy_service_token == "RENEWED-NOT-A-CREDENTIAL"


# ===== that the new token is actually used =====


@pytest.mark.asyncio
async def test_renewal_forces_a_client_rebuild() -> None:
    """The live client still holds the old token — nothing re-authenticates an
    established connection. Without the rebuild the renewal is recorded and
    never used, which looks exactly like success.
    """
    inst = _proxy(age_days=25)
    await inst._renew_token_if_due()
    assert inst._connection_lost is True, (
        "renewed without forcing a rebuild, so the stale token stays in use"
    )


@pytest.mark.asyncio
async def test_the_issue_time_is_reset_so_it_does_not_renew_every_tick() -> None:
    inst = _proxy(age_days=25)
    await inst._renew_token_if_due()
    await inst._renew_token_if_due()
    assert len(inst.server.calls) == 1, "renewed twice; the clock was not reset"


@pytest.mark.asyncio
async def test_it_asks_for_the_same_scope_it_was_given() -> None:
    """A narrower renewal would fail later and elsewhere, when the proxy next
    needs a permission the original token had."""
    inst = _proxy(age_days=25)
    await inst._renew_token_if_due()
    assert inst.server.calls[0] == {
        "workspace": "ws",
        "permission": "read_write",
        "expires_in": TTL,
    }


# ===== failure is advisory =====


@pytest.mark.asyncio
async def test_a_failed_renewal_keeps_the_working_token() -> None:
    """Days of margin remain, so the next tick can retry. Failing the replica
    over a retryable renewal would be worse than the expiry it prevents."""
    inst = _proxy(age_days=25, server=_Server(fail=RuntimeError("hypha down")))
    await inst._renew_token_if_due()
    assert inst.proxy_service_token == "ORIGINAL-NOT-A-CREDENTIAL"
    assert inst._connection_lost is False, "a failed renewal must not force a rebuild"


@pytest.mark.asyncio
async def test_no_server_yet_is_not_an_error() -> None:
    """Before the first registration there is no client to mint through."""
    inst = _proxy(age_days=25, server=None)
    inst.server = None
    await inst._renew_token_if_due()  # must not raise


@pytest.mark.asyncio
async def test_a_zero_ttl_disables_renewal() -> None:
    inst = _proxy(age_days=25, ttl=0)
    await inst._renew_token_if_due()
    assert inst.server.calls == []


# ===== the maintenance loop drives it =====


@pytest.mark.asyncio
async def test_the_maintenance_tick_calls_it() -> None:
    """A renewal nothing invokes is the original bug in a new place."""
    inst = _proxy(age_days=25)
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
    assert len(inst.server.calls) == 1, (
        "_maintenance_tick did not renew; the token would expire untouched"
    )


def test_the_renewal_fraction_leaves_real_margin() -> None:
    """Guards the constant itself. Renewing with only hours left would make the
    fix depend on the worker being up at the right moment."""
    margin_days = TTL * (1 - pd_module._TOKEN_RENEW_AFTER_FRACTION) / DAY
    assert margin_days >= 7, f"only {margin_days:.1f} days of margin"
