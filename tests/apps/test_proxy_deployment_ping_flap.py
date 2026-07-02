"""Pin the ping-flap fix for ProxyDeployment.check_health.

Field report (Chiron/Tabula, bioengine 0.11.17): personal-workspace Hypha
bridges show ping tail latency above the Ray Serve health_check_timeout
of 5s. The old code treated a single ``echo("ping")`` failure as fatal —
nulled ``self.server`` without disconnecting, then the next tick tried to
reconnect under the same client_id and Hypha refused with 'Client
already exists and is active' until its own stale-client TTL swept the
lingering registration (30–60s), producing a flap cycle at ~40s cadence.

The fix (folded into PR #135) has two parts:

1. ``check_health`` tolerates up to ``_MAX_CONSECUTIVE_PING_FAILURES - 1``
   consecutive ping failures without failing the health check or resetting
   the connection. Only after the Nth consecutive failure do we release
   the client_id and hand a failure to Ray Serve.

2. ``_register_services`` always calls ``_reset_server_connection`` first
   so any lingering client under our client_id is freed before
   ``connect_to_server`` runs. Belt-and-braces against the 'already
   exists' collision — even if a code path forgot to null ``self.server``
   properly, the reconnect still succeeds cleanly.

These tests read the source (rather than exercising Ray Serve's health
loop live) because ``check_health`` is 90% Ray-Serve-specific glue that
is impractical to unit-test standalone; the contract we're pinning is
that the *shape* of the code hasn't regressed.
"""
from __future__ import annotations

import inspect
import re

from bioengine.apps import proxy_deployment as pd_module


def test_max_consecutive_ping_failures_is_three() -> None:
    """The threshold is a module-level constant so tests + future call
    sites can reference the same source of truth."""
    assert pd_module._MAX_CONSECUTIVE_PING_FAILURES == 3


def test_counter_initialised_in_init() -> None:
    src = inspect.getsource(pd_module.ProxyDeployment.func_or_class.__init__)
    assert "_consecutive_ping_failures = 0" in src


def test_check_health_uses_counter_threshold() -> None:
    src = inspect.getsource(pd_module.ProxyDeployment.func_or_class.check_health)
    # Increments then compares to the module constant — the two lines
    # together enforce "swallow the first N-1 failures".
    assert "_consecutive_ping_failures += 1" in src
    assert "_MAX_CONSECUTIVE_PING_FAILURES" in src


def test_check_health_returns_early_on_transient_failure() -> None:
    """The transient branch must ``return`` (health check succeeds)
    rather than ``raise`` — otherwise Ray Serve still marks the replica
    unhealthy on the very first ping failure, defeating the whole fix."""
    src = inspect.getsource(pd_module.ProxyDeployment.func_or_class.check_health)
    # There is a bare ``return`` inside the ping-except block, before the
    # final ``raise RuntimeError("Hypha server connection failed")``.
    transient_return = re.search(r"transient.*?\n.*?return\b", src, re.DOTALL | re.IGNORECASE)
    assert transient_return is not None, (
        "check_health must return early on transient ping failures, "
        "not raise — Ray Serve would flap otherwise."
    )


def test_check_health_resets_connection_on_terminal_failure() -> None:
    src = inspect.getsource(pd_module.ProxyDeployment.func_or_class.check_health)
    # After N-th consecutive failure the connection must be reset before
    # raising, otherwise the reconnect still hits 'Client already exists'.
    assert "_reset_server_connection" in src


def test_register_services_calls_reset_before_connect() -> None:
    """_register_services must free the client_id before connect_to_server —
    otherwise a lingering registration causes 'Client already exists and is
    active' and the replica flaps until Hypha's stale-client TTL fires."""
    src = inspect.getsource(pd_module.ProxyDeployment.func_or_class._register_services)
    # Look at the actual call, not any mention in a comment / docstring.
    reset_call = re.search(r"await self\._reset_server_connection\(\)", src)
    connect_call = re.search(r"connect_to_server\(", src)
    assert reset_call is not None, "expected an actual _reset_server_connection() call"
    assert connect_call is not None, "expected a connect_to_server(...) call"
    assert reset_call.start() < connect_call.start(), (
        "_reset_server_connection() must be awaited BEFORE connect_to_server()"
    )


def test_reset_server_connection_is_idempotent_and_bounded() -> None:
    src = inspect.getsource(pd_module.ProxyDeployment.func_or_class._reset_server_connection)
    # Idempotent: handles self.server is None
    assert "self.server is not None" in src or "if self.server" in src
    # Bounded: disconnect() wrapped in a timeout so a wedged transport
    # can't stall the caller (health-check reset, __del__, etc.).
    assert "wait_for" in src
    # Always clears state — sets server to None regardless of disconnect outcome.
    assert "self.server = None" in src
