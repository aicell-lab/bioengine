"""Pins the window in which a replica with a failed smoke test is routable.

``@bioengine.smoke_test`` used to be documented as a traffic gate. It is not:
``_make_check_health`` launches it as a background task and tests ``.done()``
in the same call, so the first poll cannot see it fail, and Ray Serve then
needs ``REPLICA_HEALTH_CHECK_UNHEALTHY_THRESHOLD`` consecutive failures to
condemn the replica. These tests fail if either half of that bound moves —
the first poll starting to report the failure, a later poll stopping, or Ray
changing the threshold the docstring quotes.

No Ray cluster: the replica is a plain object carrying the ``_bioengine_*``
attributes ``_setup_replica`` installs.
"""

import asyncio

import pytest

from bioengine._app.mixin import _make_check_health


class _Replica:
    """Minimal stand-in for an initialised BioEngine replica."""

    def __init__(self, smoke_raises: bool) -> None:
        self._bioengine_replica_initialized = False
        self._bioengine_replica_test_failed = False
        self._bioengine_test_task = None
        self._bioengine_health_check_lock = asyncio.Lock()
        self._bioengine_has_gpu = False
        self._bioengine_node_id = None
        self._bioengine_proxy_handle = None
        self._bioengine_gpu_sampler = None
        self._bioengine_gpu_init_failed = False
        self._smoke_raises = smoke_raises
        self.calls: list = []

    async def my_async_init(self) -> None:
        self.calls.append("async_init")

    async def my_smoke(self) -> None:
        self.calls.append("smoke")
        if self._smoke_raises:
            raise RuntimeError("boom")


def _check_health(replica: _Replica):
    return _make_check_health(
        type(replica),
        {"async_init": "my_async_init", "smoke_test": "my_smoke"},
    ).__get__(replica, type(replica))


async def _drain() -> None:
    """Let the background smoke-test task run to completion."""
    for _ in range(10):
        await asyncio.sleep(0)


async def test_first_poll_cannot_report_a_smoke_test_failure() -> None:
    replica = _Replica(smoke_raises=True)
    check_health = _check_health(replica)

    await check_health()

    assert replica.calls == ["async_init"]
    assert replica._bioengine_test_task is not None
    assert replica._bioengine_test_task.done() is False
    assert replica._bioengine_replica_test_failed is False

    # Retrieving the exception pins *why* the first poll missed it, and keeps
    # asyncio from logging it as never retrieved: the failure is already there
    # one event-loop turn later, so the miss is the absent await and not a
    # smoke test that was merely slow.
    await _drain()
    assert str(replica._bioengine_test_task.exception()) == "boom"


async def test_every_later_poll_reports_it() -> None:
    """Polls 2-4 must all fail, or Serve never reaches its threshold of 3."""
    replica = _Replica(smoke_raises=True)
    check_health = _check_health(replica)

    await check_health()
    await _drain()
    assert replica.calls == ["async_init", "smoke"]

    for _ in range(3):
        with pytest.raises(RuntimeError, match="Smoke test failed: boom"):
            await check_health()

    assert replica._bioengine_replica_test_failed is True
    assert replica.calls == ["async_init", "smoke"]


async def test_a_passing_smoke_test_keeps_the_replica_healthy() -> None:
    replica = _Replica(smoke_raises=False)
    check_health = _check_health(replica)

    await check_health()
    await _drain()
    await check_health()

    assert replica.calls == ["async_init", "smoke"]
    assert replica._bioengine_replica_test_failed is False


def test_serve_condemns_only_after_three_consecutive_failures() -> None:
    """The "three consecutive failed checks" both docstrings quote is Ray's."""
    from ray.serve._private.constants import (
        REPLICA_HEALTH_CHECK_UNHEALTHY_THRESHOLD,
    )

    assert REPLICA_HEALTH_CHECK_UNHEALTHY_THRESHOLD == 3


def test_the_smoke_test_docstring_does_not_promise_a_traffic_gate() -> None:
    """The claim the tests above refute must not drift back into the docs."""
    from bioengine._app.decorators import async_init, smoke_test

    # Collapsed so the assertions survive re-wrapping: these phrases are
    # claims about the documented bound, not about where the lines break.
    doc = " ".join((smoke_test.__doc__ or "").split())
    assert "never admits traffic" not in doc
    assert "does not gate traffic" in doc
    # The figure derived from Ray's constants, not the "~3 x" it replaced.
    assert "two ``health_check_period_s``" in doc
    # Condemnation, not admission, is what the status change tracks.
    assert "still reads ``HEALTHY`` for that window" in doc
    # That window bounds one replica, not the exposure. Serve's circuit
    # breaker is gated on ``not self._replica_has_started``, and that flag is
    # cleared only by a new ``deploy()`` — so a version whose replicas start
    # fine and then fail their smoke test never trips it, and condemned
    # replicas are replaced indefinitely. Dropping this sentence restores the
    # "bounded window" reading the row was filed to remove.
    assert "restart loop" in doc

    # ``async_init`` is the hook that really does gate traffic: Serve awaits
    # its first ``check_health`` inside replica initialisation, so the
    # contrast the smoke-test docstring draws has to stay true of both.
    assert "before traffic is admitted" in " ".join(
        (async_init.__doc__ or "").split()
    )
