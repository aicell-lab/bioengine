"""A worker must not deadlock when its minted token expires while disconnected (#0632).

MEASURED FAILURE THIS PINS, deNBI 2026-11-18. A 6.5-hour network outage outlasted the 3-hour
token the worker mints for itself. When the path returned the reconnect failed on
"Authentication error: The token has expired", and the refresh step then CRASHED:

    Monitoring step 'token_expiry' failed: 'NoneType' object is not callable
    worker.py:1216 in _check_token_expiry

304 consecutive monitoring failures and climbing, on a pod that was 1/1 Running with
RESTARTS=0. A restart cured it instantly by re-reading HYPHA_TOKEN from the environment -- a
credential valid for a further 177 days -- which is what proved the worker was holding a good
token the whole time and had thrown it away at worker.py:1244.

THE THREE ARMS AND WHY EACH IS NEEDED

  deadlock   connection dead + minted token expiring -> must NOT raise, and must restore the
             bootstrap token so the next reconnect has a usable credential.
  control    connection healthy -> must STILL mint on the normal schedule. Without this, a
             "fix" that simply disabled refreshing would pass the other two.
  no-op      bootstrap token already in use -> must not raise and must not thrash.

`self.server` is deliberately a stand-in whose `generate_token` attribute is None rather than
absent, because that is the measured shape: after an auth failure the object survives and its
RPC methods do not, so `hasattr` is true, the attribute is None, and CALLING it is what raises.
A test using a plain `None` server would pass against the unfixed code and prove nothing.
"""
import time

import pytest

from bioengine.worker.worker import BioEngineWorker


class _DeadServer:
    """Post-auth-failure shape: the object is alive, its RPC methods are gone."""

    generate_token = None
    parse_token = None


class _LiveServer:
    def __init__(self):
        self.minted = 0

    async def generate_token(self, _config=None):
        self.minted += 1
        return "freshly-minted-token"

    async def parse_token(self, _token):
        class _Info:
            expires_at = time.time() + 3600 * 3
        return _Info()


def _worker(server, token, bootstrap, expires_at):
    w = BioEngineWorker.__new__(BioEngineWorker)
    w.server = server
    w._token = token
    w._bootstrap_token = bootstrap
    w._token_expires_at = expires_at
    w.workspace = "ws"
    w.client_id = "cid"
    import logging
    w.logger = logging.getLogger("test_0632")
    return w


@pytest.mark.asyncio
async def test_dead_connection_falls_back_to_the_bootstrap_token_without_raising():
    w = _worker(_DeadServer(), "expired-minted-token", "long-lived-bootstrap", time.time() - 1)
    await w._check_token_expiry()          # must not raise
    assert w._token == "long-lived-bootstrap", (
        "the worker did not restore the credential it started with, so the next reconnect "
        "would reuse the expired minted token and the deadlock survives"
    )


@pytest.mark.asyncio
async def test_healthy_connection_still_mints_on_schedule(caplog):
    """CONTROL: the guard must not have been bought by disabling refresh altogether."""
    live = _LiveServer()
    w = _worker(live, "old-minted", "long-lived-bootstrap", time.time() - 1)
    await w._check_token_expiry()
    assert live.minted == 1, "a healthy worker stopped refreshing its token"
    assert w._token == "freshly-minted-token"
    assert w._token_expires_at > time.time() + 3600


@pytest.mark.asyncio
async def test_dead_connection_with_bootstrap_already_in_use_is_a_quiet_no_op():
    w = _worker(_DeadServer(), "long-lived-bootstrap", "long-lived-bootstrap", time.time() - 1)
    await w._check_token_expiry()          # must not raise
    assert w._token == "long-lived-bootstrap"


@pytest.mark.asyncio
async def test_a_token_far_from_expiry_is_left_alone():
    """Not every tick should touch the token; only one near expiry."""
    live = _LiveServer()
    w = _worker(live, "current", "boot", time.time() + 3600 * 10)
    await w._check_token_expiry()
    assert live.minted == 0
    assert w._token == "current"
