"""``EntryApp._health`` must fail on a DEAD Hypha connection, not just a missing one.

The check used to be ``if self._s3 is None: raise``. That is an identity test against
``None``, and this file's subject already knew it was insufficient: the comment in
``_connect`` explaining why ``_s3`` is wrapped in ``reconnecting_service`` observed that a
proxy pinned to a client id the server later forgets would hang every call to the full
method timeout, and that ``_health`` tested ``is None``, *"which a dead handle passes"*.
So the hazard was written down at the site that avoided it while the health check that
would surface it could not. (That sentence in ``_connect`` has since been updated to
describe the probe, so it no longer reads as a live defect; it is quoted here as the
record of why this row existed.)

Why it matters: a replica in that state passes its health check, stays in the Serve
routing table, and converts every call into a full-method-timeout hang instead of a fast
failure. That is what "the app is up but nothing works" looks like from outside, and it is
the shape annotation-broker hit for real on 2026-08-24 (``broker.py`` ``_health`` carries
the same remedy and the same reasoning).

Plain pytest, no Ray, no network, no GPU: ``_health`` is called as an unbound coroutine on
a bare instance with only the two attributes it reads, and the Hypha handle is a stub whose
``echo`` is told what to do per arm.

THE ARM THAT MAKES THE REST NON-VACUOUS is ``test_health_actually_calls_echo``. Arms that
only assert "healthy does not raise" and "dead raises" are both satisfiable by a ``_health``
that never exercises the connection at all, as the previous version proved for a year — so
one arm asserts the call was really made.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import entry


def _health_of(instance):
    """The real coroutine off the Serve-wrapped class.

    ``entry.EntryApp`` is a ``ray.serve.deployment.Deployment`` once ``@bioengine.app``
    has run, so its methods are not reachable through it. Same accessor as
    ``test_preflight.py``.
    """
    return getattr(entry.EntryApp.func_or_class, "_health").__get__(instance)


class _Hypha:
    """A Hypha server handle whose ``echo`` behaves as the arm requires."""

    def __init__(self, behaviour="ok", delay=0.0):
        self._behaviour = behaviour
        self._delay = delay
        self.echo_calls = 0

    async def echo(self, payload):
        self.echo_calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._behaviour == "send-failure":
            # The real shape: hypha_rpc fails at SEND time against a client instance the
            # server has forgotten. The connection object is alive and the websocket
            # underneath has happily reconnected; only the pinned client id is gone.
            raise Exception("Failed to send message to client: client not found")
        if self._behaviour == "raise":
            raise RuntimeError("connection closed")
        return payload


def _entry(hypha, s3="a-live-handle"):
    """A bare EntryApp carrying only what ``_health`` reads."""
    inst = object.__new__(entry.EntryApp.func_or_class)
    inst._s3 = s3
    inst._hypha = hypha
    return inst


async def test_a_live_connection_is_healthy() -> None:
    """Baseline. Without this, an always-raising ``_health`` would satisfy every arm below."""
    hypha = _Hypha("ok")
    await _health_of(_entry(hypha))()  # must not raise


async def test_a_dead_handle_is_UNHEALTHY() -> None:
    """THE arm this row exists for.

    ``_s3`` is non-None, so the old ``is None`` check passed here and the replica stayed
    in the routing table. The send failure is what every real call would have hit.
    """
    hypha = _Hypha("send-failure")
    with pytest.raises(RuntimeError, match="Hypha server connection unusable"):
        await _health_of(_entry(hypha))()


async def test_a_HUNG_connection_is_unhealthy_within_the_bound(monkeypatch) -> None:
    """A hang is the failure mode the row names, and it must not hang the CHECK.

    The bound has to be well under the deployment's ``health_check_timeout_s=30.0`` so
    Serve is told UNHEALTHY by this check rather than timing the check itself out, which
    it would report less specifically. Shortened here so the test is fast; the assertion
    is that the bound is enforced at all.
    """
    monkeypatch.setattr(entry, "_HEALTH_CALL_TIMEOUT_S", 0.05)
    hypha = _Hypha("ok", delay=5.0)
    with pytest.raises(RuntimeError, match="Hypha server connection unusable"):
        await asyncio.wait_for(_health_of(_entry(hypha))(), timeout=3.0)


async def test_a_missing_s3_handle_is_still_unhealthy() -> None:
    """The original contract, kept. The new exercise is in ADDITION to it, not instead:
    ``_s3 is None`` means ``_connect`` never finished, which is a different fault from a
    connection that completed and later died."""
    hypha = _Hypha("ok")
    with pytest.raises(RuntimeError, match="S3 storage service not connected"):
        await _health_of(_entry(hypha, s3=None))()


async def test_health_actually_calls_echo() -> None:
    """ANTI-VACUITY CONTROL — the arm that gives the others their meaning.

    A ``_health`` that dropped the liveness exercise would still pass
    ``test_a_live_connection_is_healthy`` and ``test_a_missing_s3_handle_is_still_unhealthy``,
    and that is precisely the state this file was written to end. Assert the dependency was
    exercised, so coverage cannot be removed while the suite stays green.
    """
    hypha = _Hypha("ok")
    await _health_of(_entry(hypha))()
    assert hypha.echo_calls == 1, (
        "_health returned without exercising the Hypha connection, so a dead handle "
        "would pass it again"
    )


async def test_the_bound_is_read_from_the_module_constant() -> None:
    """The timeout must be a named constant rather than a literal buried in the call, so
    it can be read against ``health_check_timeout_s`` without re-reading the body."""
    assert isinstance(entry._HEALTH_CALL_TIMEOUT_S, float)
    cfg = entry.EntryApp._deployment_config
    assert entry._HEALTH_CALL_TIMEOUT_S < cfg.health_check_timeout_s, (
        "the liveness call must be bounded BELOW the deployment's health-check timeout, "
        "or Serve times out the check instead of being told the connection is dead"
    )


async def test_the_OLD_check_passed_this_exact_handle() -> None:
    """DISCRIMINATION CONTROL, and what makes ``test_a_dead_handle_is_UNHEALTHY``
    evidence rather than an assertion.

    ``test_a_live_connection_is_healthy`` shows the new check does not reject
    everything. This shows the complementary half: that the OLD body *accepted* the very
    handle the new one rejects. Without it, the row's premise — that a dead connection
    used to pass — is taken on trust, and a reader cannot tell a real regression test
    from one written against a bug that never existed.

    The body is reconstructed rather than imported because the point is what the previous
    revision DID; pulling it from git history would make this test depend on the SHA.
    """
    hypha = _Hypha("send-failure")
    inst = _entry(hypha)

    async def _old_health(self) -> None:
        if self._s3 is None:
            raise RuntimeError("S3 storage service not connected")

    assert await _old_health(inst) is None, "the old body did not accept a dead handle"
    assert hypha.echo_calls == 0, "the reconstructed old body is not the old body"


async def test_the_cheap_precondition_is_reached_BEFORE_the_wire() -> None:
    """Order matters, and ``test_a_missing_s3_handle_is_still_unhealthy`` does not pin it.

    That arm asserts the right exception; it would also pass if the probe ran first and
    happened to succeed. With no ``_s3`` there is nothing to serve and nothing to probe,
    so probing anyway would spend the bound on every check of a replica whose
    ``_connect`` never finished, and would report a CONNECTION fault for what is a
    missing dependency — the wrong remedy, since replacing that replica does not help.
    """
    hypha = _Hypha("ok")
    with pytest.raises(RuntimeError, match="S3 storage service not connected"):
        await _health_of(_entry(hypha, s3=None))()
    assert hypha.echo_calls == 0, (
        "the liveness probe ran before the `_s3 is None` precondition, so a replica "
        "that never connected is reported as a dead connection"
    )


def test_the_decision_not_to_wrap_the_connection_is_recorded_at_the_site() -> None:
    """``self._s3`` is wrapped in ``reconnecting_service`` and ``self._hypha`` is not.

    That asymmetry reads as an oversight unless the reasoning is written where the
    wrapping happens, and an "oversight" invites a fix that cannot work: the wrapper
    re-resolves a service proxy THROUGH a connection, so it has nothing to offer the
    connection itself. Asserted off the source because the claim is about what a
    maintainer sees, not about behaviour.
    """
    src = (Path(__file__).with_name("entry.py")).read_text(encoding="utf-8")

    # CONTROLS first: the file must have been read, and a phrase that is not in it must
    # be absent — otherwise a mis-resolved path or an empty read passes every assertion.
    assert len(src) > 10_000, f"entry.py read back implausibly short ({len(src)} chars)"
    assert "ZZZ_PHRASE_THAT_IS_NOT_IN_ENTRY_PY" not in src

    assert "deliberately NOT wrapped" in src, (
        "the decision not to wrap `self._hypha` is not stated where `self._s3` is wrapped"
    )
    assert "different object from the connection" in src, (
        "the REASON is not stated: that reconnecting_service re-resolves a service "
        "proxy, which is not the connection it was resolved through"
    )
