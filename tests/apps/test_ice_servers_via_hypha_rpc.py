"""Pin the ICE-server fetch path: authenticated Hypha RPC, not raw HTTP.

The Hypha TURN-server endpoint requires auth. The old code used a raw
``httpx.AsyncClient`` without any Authorization header, so every
ProxyDeployment startup emitted a loud

    ❌ HTTP error fetching ICE servers for <app>: Client error
       '403 Forbidden' for url 'https://hypha.aicell.io/turn-server/services/coturn/get_rtc_ice_servers'

then fell back to aiortc's built-in STUN defaults — WebRTC still worked
for peers on friendly networks, but restrictive NATs lost the TURN
relay. Chiron/Tabula spawns a new trainer replica every few minutes so
the noise was hard to miss.

The fix is to call the same TURN service through ``self.server``
(hypha_rpc), which carries the proxy's own authenticated token. These
tests pin the code shape so a future refactor can't silently regress
back to raw HTTP.
"""
from __future__ import annotations

import inspect

from bioengine.apps import proxy_deployment as pd_module


def _fn_source() -> str:
    return inspect.getsource(pd_module.ProxyDeployment.func_or_class._fetch_ice_servers)


def test_fetch_ice_servers_uses_hypha_rpc_not_raw_http() -> None:
    src = _fn_source()
    # The fix must call get_service through the authenticated Hypha
    # connection, then invoke get_rtc_ice_servers on the handle.
    assert "self.server.get_service" in src, (
        "expected _fetch_ice_servers to go through the authenticated "
        "self.server connection, not a raw HTTP client."
    )
    assert "get_rtc_ice_servers" in src, (
        "expected _fetch_ice_servers to call get_rtc_ice_servers on "
        "the coturn service handle."
    )


def test_fetch_ice_servers_no_longer_uses_raw_httpx_client() -> None:
    """Guard against a future contributor re-introducing the raw HTTP path.
    ``AsyncClient`` in httpx doesn't send Hypha's auth header and would
    silently 403 again."""
    src = _fn_source()
    assert "AsyncClient" not in src, (
        "_fetch_ice_servers must not create an httpx AsyncClient — that "
        "code path 403'd every time (Chiron field report)."
    )
    assert "ICE_SERVERS_URL" not in src, (
        "the hard-coded HTTP URL was removed; use the Hypha service id "
        "constant instead."
    )


def test_fallback_still_returns_none_on_failure() -> None:
    """If the RPC fails, we must return None so aiortc's built-in STUN
    defaults kick in. Raising here would make Ray Serve mark the replica
    UNHEALTHY on the first coturn hiccup — worse than degraded WebRTC."""
    src = _fn_source()
    # There must be a return None inside the except block.
    assert "except Exception" in src
    except_idx = src.find("except Exception")
    tail = src[except_idx:]
    assert "return None" in tail, (
        "the exception path must return None (fall back to aiortc "
        "defaults), not re-raise."
    )


def test_no_traceback_dumped_on_expected_failure() -> None:
    """The old code logged at ERROR with the full exception. Downgrade to
    a one-line WARNING — the fallback is well-defined and operators
    interpret the ❌ ERROR as a broken deployment."""
    src = _fn_source()
    # The exception branch must use logger.warning, not logger.error.
    except_idx = src.find("except Exception")
    tail = src[except_idx:]
    assert "logger.warning" in tail, (
        "the exception path must log at WARNING, not ERROR — "
        "the fallback path is defined and not a broken deployment."
    )
    # And must not use exc_info / re-raise / logger.exception (which
    # would dump the traceback back into the log).
    assert "exc_info" not in tail
    assert "logger.exception" not in tail
