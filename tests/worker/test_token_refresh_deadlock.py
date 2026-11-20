"""A worker must not deadlock when its minted token expires while disconnected.

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
    """Post-auth-failure shape, MEASURED rather than assumed.

    The first version of this fake set `generate_token = None`, which is NOT what happens:
    measured 2026-11-18 against a real disconnected Hypha client, `generate_token` is still a
    live `function` and `callable()` returns True -- the None is deeper inside hypha_rpc and
    surfaces only when the call is MADE. That fake let a liveness-guard fix pass the unit arms
    and then fail the live reproduction, so it is written here the way the system really behaves.
    """

    async def generate_token(self, _config=None):
        raise TypeError("'NoneType' object is not callable")

    async def parse_token(self, _token):
        raise TypeError("'NoneType' object is not callable")


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


def _init_source():
    import inspect

    return inspect.getsource(BioEngineWorker.__init__)


def test_the_bootstrap_copy_is_refreshed_after_an_interactive_login():
    """A worker that logs in interactively must not keep the pre-login bootstrap copy.

    __init__ captures the bootstrap credential from the constructor argument and the
    environment. On the interactive-login path both are absent, so that capture is None and
    the real credential arrives later, from login(). Without a second assignment the fallback
    in _check_token_expiry has nothing to restore and the deadlock survives for exactly the
    workers that never set HYPHA_TOKEN.

    Read from __init__'s OWN source rather than the module's, so an assignment elsewhere in
    worker.py cannot satisfy this.
    """
    src = _init_source()
    i_login = src.find("self._token = login(")
    assert i_login != -1, "the interactive-login assignment moved; re-anchor this arm"
    assert "self._bootstrap_token = self._token" in src[i_login:], (
        "the bootstrap copy is never refreshed after login(), so a worker started without a "
        "token holds None and cannot fall back when its minted token expires offline"
    )


def test_the_pre_login_bootstrap_capture_is_still_present():
    """CONTROL: the fix must ADD an assignment, not relocate the only one.

    A worker given a token never enters the login branch, so the early capture is the one that
    protects it. Had the assignment merely moved, the arm above would pass and this one fails.
    """
    src = _init_source()
    i_login = src.find("self._token = login(")
    assert i_login != -1
    assert "self._bootstrap_token = self._token" in src[:i_login], (
        "the capture that protects a worker started WITH a token was removed"
    )


@pytest.mark.asyncio
async def test_a_login_issued_credential_survives_a_mint_and_a_dead_connection():
    """The end state the login path has to reach, driven through the real refresh step.

    This arm does not by itself distinguish fixed from unfixed -- it is handed the post-fix
    bootstrap value, and the source-order arm above is what discriminates. It is here because
    the order assertion alone would not show that the restored credential is the useful one.
    """
    live = _LiveServer()
    w = _worker(live, "login-issued-token", "login-issued-token", time.time() - 1)
    await w._check_token_expiry()
    assert w._token == "freshly-minted-token", "a healthy worker stopped minting"

    w.server = _DeadServer()
    w._token_expires_at = time.time() - 1
    await w._check_token_expiry()
    assert w._token == "login-issued-token", (
        "fell back to nothing and kept the expired minted token, so the next reconnect fails"
    )
