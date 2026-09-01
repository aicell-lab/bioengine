"""Pin that ProxyDeployment gates its Hypha service off the data plane.

The proxy no longer probes the entry in-band via ``check_health.remote()`` —
that call was routed through the entry's router and counted against its
``max_ongoing_requests``, so a saturated-but-healthy entry head-of-line-blocked
the probe. Instead the proxy reads every sibling deployment's RUNNING replica
count out-of-band from the Serve controller (``serve.status()``), which a
saturated app can neither block nor fail.

The gate:

* registers only once every sibling deployment has a RUNNING replica;
* tolerates an "unknown" reading (``None``) — a controller mid-restart never
  deregisters a healthy app;
* deregisters when a sibling that was seen RUNNING drops to zero (the outage
  stays visible in the app status via the down deployment itself, so the proxy
  does not fail its own health);
* saturation is a non-event: a busy replica is still ``RUNNING`` in
  ``serve.status()``, so the count is unaffected.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from bioengine.apps import proxy_deployment as pd_module

_ProxyCls = pd_module.ProxyDeployment.func_or_class


class _WsService:
    async def get_load(self):
        return 0

    async def get_num_pcs(self):
        return 0


class _Server:
    async def echo(self, msg):
        return msg

    async def get_service_info(self, service_id):
        return {"id": service_id}

    async def get_service(self, service_id):
        return _WsService()


class _Recorder:
    def __init__(self):
        self.called = False

    async def __call__(self, *args, **kwargs):
        self.called = True


def _bare_proxy(**attrs):
    inst = object.__new__(_ProxyCls)
    inst.application_id = "app"
    inst._own_deployment_name = "ProxyDeployment"
    inst.entry_deployment_ready = False
    inst._dep_seen_ready = {}
    inst.server = None
    inst.websocket_service_id = None
    inst._registration_lock = asyncio.Lock()
    inst._registration_failure = None
    inst._connection_lost = False
    inst._probe_due_at = 0.0
    inst._next_register_at = 0.0
    inst._maintenance_task = None
    # The maintenance loop is exercised in test_proxy_hypha_decoupled_health;
    # here it would only spawn a background task the gate tests never await.
    inst._ensure_maintenance_task = lambda: None
    inst._deregister_services = _Recorder()
    for key, value in attrs.items():
        setattr(inst, key, value)
    return inst


def _stub_counts(value):
    async def _counts(self=None):
        return value

    return _counts


async def _run(inst):
    await inst.check_health()


@pytest.mark.asyncio
async def test_saturation_does_not_deregister() -> None:
    """A busy-but-RUNNING sibling keeps the service registered."""
    inst = _bare_proxy(
        entry_deployment_ready=True,
        _dep_seen_ready={"EntryDeployment": True},
        server=_Server(),
        websocket_service_id="ws",
    )
    inst._sibling_running_counts = _stub_counts({"EntryDeployment": 1})
    await _run(inst)
    assert inst._deregister_services.called is False


@pytest.mark.asyncio
async def test_unknown_status_does_not_deregister() -> None:
    """A ``None`` reading (controller mid-restart) never deregisters."""
    inst = _bare_proxy(
        entry_deployment_ready=True,
        _dep_seen_ready={"EntryDeployment": True},
        server=_Server(),
        websocket_service_id="ws",
    )
    inst._sibling_running_counts = _stub_counts(None)
    await _run(inst)
    assert inst._deregister_services.called is False


@pytest.mark.asyncio
async def test_sibling_drop_deregisters() -> None:
    """A sibling seen RUNNING that drops to zero deregisters the service."""
    inst = _bare_proxy(
        entry_deployment_ready=True,
        _dep_seen_ready={"RuntimeDeployment": True},
        server=_Server(),
        websocket_service_id="ws",
    )
    inst._sibling_running_counts = _stub_counts({"RuntimeDeployment": 0})
    await _run(inst)
    assert inst._deregister_services.called is True


@pytest.mark.asyncio
async def test_initial_gate_waits_for_all_running() -> None:
    """The service registers only once every sibling is RUNNING.

    check_health owns the gate; the background maintenance task does the
    registering, and must stay idle until the gate opens.
    """
    registered = _Recorder()
    inst = _bare_proxy()
    inst._register_services = registered

    inst._sibling_running_counts = _stub_counts(
        {"EntryDeployment": 1, "RuntimeDeployment": 0}
    )
    await _run(inst)
    assert inst.entry_deployment_ready is False
    await inst._maintenance_tick()
    assert registered.called is False

    inst._sibling_running_counts = _stub_counts(
        {"EntryDeployment": 1, "RuntimeDeployment": 1}
    )
    await _run(inst)
    assert inst.entry_deployment_ready is True
    await inst._maintenance_tick()
    assert registered.called is True


def test_probe_is_off_the_data_plane() -> None:
    """check_health must read sibling status out-of-band, never in-band."""
    src = inspect.getsource(_ProxyCls.check_health)
    assert "entry_deployment_handle.check_health.remote" not in src, (
        "the entry health probe must not be issued in-band — it counts against "
        "the entry's max_ongoing_requests and blocks under load."
    )
    assert "_sibling_running_counts" in src
