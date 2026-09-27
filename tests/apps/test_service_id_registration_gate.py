"""Pin that a service address is advertised only once it actually resolves.

``get_app_status`` used to derive ``service_ids`` from the worker's client id
and gate only on "a ProxyDeployment replica is alive". A live proxy replica is
a *precondition* for Hypha registration, not evidence of it: the replica
reports healthy to Ray while it waits for its siblings to come up, and only
then registers with Hypha. So the worker handed out an id — and a
``static_site_url`` built from it — tens of seconds before anything answered
at that address, with ``status`` already reading ``RUNNING``.

The contract now:

* The proxy pushes its registration state to the ``BioEngineProxyActor`` (a
  claim at replica init, which opens the record as unregistered, True after
  ``_register_services`` succeeds, False on deregistration), and the worker
  reads it before advertising.
* A ``False`` report withholds both ids and the ``static_site_url``.
* *Never reported* is not ``False``. An app whose proxy started before this
  actor existed — a newer worker, or an actor recreated after eviction —
  serves perfectly well and must keep its id, so the worker falls back to the
  replica-alive gate.
* The record is self-correcting, not write-once: a replica whose reachability
  probe succeeds re-asserts ``True`` every probe interval. Otherwise a
  successor that claims the record and then never registers withholds a
  working address for as long as the predecessor keeps serving it.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps import proxy_deployment as pd_module
from bioengine.apps.manager import AppsManager
from bioengine.cluster.proxy_actor import BioEngineProxyActor
from tests.apps._proxy_double import PROXY_CLS as _ProxyCls
from tests.apps._proxy_double import ProxyDouble

_ActorCls = BioEngineProxyActor.__ray_actor_class__

APP_ID = "nuclei-seg"
WORKER_CLIENT_ID = "worker-abc"


def _make_app_info() -> dict:
    deployed = asyncio.Event()
    deployed.set()
    return {
        "is_deployed": deployed,
        "display_name": "Nuclei Segmentation",
        "description": "Segment nuclei.",
        "artifact_id": "bioimage-io/nuclei-seg",
        "version": "1.0.1",
        "recovered_app": False,
        "application_kwargs": {},
        "application_env_vars": {},
        "disable_gpu": False,
        "application_resources": {},
        "authorized_users": ["*"],
        "available_methods": ["segment"],
        "max_ongoing_requests": 1,
        "scaling": {},
        "static_site_url": "https://static.example/nuclei-seg/",
        "started_at": 1_700_000_000.0,
        "last_updated_at": 1_700_000_000.0,
        "last_updated_by": "user@example.com",
        "auto_redeploy": False,
        "deployed_by_worker_client_id": WORKER_CLIENT_ID,
        "proxy_service_token_issued_at": None,
        "proxy_service_token_ttl_seconds": None,
    }


def _make_manager(*, replicas: dict, registered) -> AppsManager:
    """An AppsManager wired with only what the service-id path touches.

    ``registered`` is the proxy actor's answer: ``True``/``False``/``None``, or
    an exception instance to raise from the actor call.
    """
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test")
    manager._deployed_applications = {APP_ID: _make_app_info()}
    # Not a startup application; the status path consults this list.
    manager.startup_applications = []

    server = MagicMock()
    server.config.workspace = "bioimage-io"
    server.config.client_id = WORKER_CLIENT_ID
    server.config.public_base_url = "https://hypha.example"
    manager.server = server

    ray_cluster = MagicMock()
    handle = ray_cluster.proxy_actor_handle
    handle.get_deployment_replicas.remote = AsyncMock(return_value=replicas)
    if isinstance(registered, BaseException):
        handle.get_service_registration.remote = AsyncMock(side_effect=registered)
    else:
        handle.get_service_registration.remote = AsyncMock(return_value=registered)
    manager.ray_cluster = ray_cluster
    return manager


async def _status(manager: AppsManager) -> dict:
    return await manager._get_app_status(
        application_id=APP_ID,
        instance_details={
            "applications": {APP_ID: {"status": "RUNNING", "deployments": {}}}
        },
        n_previous_replica=0,
        logs_tail=30,
    )


# ===== the worker only advertises what the proxy has registered =====


@pytest.mark.asyncio
async def test_a_live_but_unregistered_proxy_advertises_no_service_id() -> None:
    # The reported bug verbatim: the ProxyDeployment replica is alive, so the
    # old gate passed, but its Hypha services do not exist yet.
    manager = _make_manager(replicas={"r0": "actor0"}, registered=False)

    ids, registered = await manager._get_application_service_ids(APP_ID)

    assert ids == {"websocket_service_id": None, "webrtc_service_id": None}
    assert registered is False


@pytest.mark.asyncio
async def test_a_registered_proxy_advertises_the_service_id() -> None:
    manager = _make_manager(replicas={"r0": "actor0"}, registered=True)

    ids, registered = await manager._get_application_service_ids(APP_ID)

    assert registered is True
    assert ids["websocket_service_id"].endswith(f":{APP_ID}")
    assert ids["webrtc_service_id"].endswith(f":{APP_ID}-rtc")


@pytest.mark.asyncio
async def test_no_replica_still_means_no_service_id() -> None:
    # The pre-existing gate must survive: no proxy replica, no address.
    manager = _make_manager(replicas={}, registered=True)

    ids, registered = await manager._get_application_service_ids(APP_ID)

    assert ids == {"websocket_service_id": None, "webrtc_service_id": None}
    assert registered is None


# ===== absence of a report is not a negative report =====


@pytest.mark.asyncio
async def test_an_app_that_never_reported_keeps_its_service_id() -> None:
    # A recovered app, or one whose proxy predates this actor, serves without
    # ever reporting. Hiding its id would break a working deployment.
    manager = _make_manager(replicas={"r0": "actor0"}, registered=None)

    ids, registered = await manager._get_application_service_ids(APP_ID)

    assert registered is None
    assert ids["websocket_service_id"] is not None


@pytest.mark.asyncio
async def test_an_unreadable_registration_record_does_not_hide_a_live_app() -> None:
    manager = _make_manager(
        replicas={"r0": "actor0"}, registered=RuntimeError("actor unreachable")
    )

    ids, registered = await manager._get_application_service_ids(APP_ID)

    assert registered is None
    assert ids["websocket_service_id"] is not None


# ===== what the status payload says =====


@pytest.mark.asyncio
async def test_status_withholds_the_static_site_url_until_registration() -> None:
    # The static site is handed the service id as a query parameter, so an
    # unregistered app must not produce a URL either.
    manager = _make_manager(replicas={"r0": "actor0"}, registered=False)

    status = await _status(manager)

    assert status["status"] == "RUNNING"
    assert status["static_site_url"] is None
    assert status["service_registered"] is False


@pytest.mark.asyncio
async def test_status_builds_the_static_site_url_once_registered() -> None:
    manager = _make_manager(replicas={"r0": "actor0"}, registered=True)

    status = await _status(manager)

    assert status["service_registered"] is True
    assert status["static_site_url"].startswith("https://static.example/nuclei-seg/?")
    assert status["service_ids"]["websocket_service_id"] in status["static_site_url"]


# ===== the proxy actor's side of the record =====


def _bare_actor() -> object:
    actor = object.__new__(_ActorCls)
    actor._last_called_at = 0.0
    actor.application_replicas = {}
    actor.replica_identities = {}
    actor.service_registrations = {}
    return actor


def test_the_actor_reports_none_before_anything_is_pushed() -> None:
    actor = _bare_actor()
    assert actor.get_service_registration(APP_ID) is None


def test_the_actor_round_trips_both_registration_states() -> None:
    actor = _bare_actor()
    actor.report_service_registration(APP_ID, False)
    assert actor.get_service_registration(APP_ID) is False
    actor.report_service_registration(APP_ID, True)
    assert actor.get_service_registration(APP_ID) is True


def test_clearing_an_application_forgets_its_registration() -> None:
    # Called from undeploy_app only, so this is about an app_id that is later
    # deployed afresh — not about redeploy, which is covered by the init claim.
    actor = _bare_actor()
    actor.report_service_registration(APP_ID, True)
    actor.clear_application_replicas(APP_ID)
    assert actor.get_service_registration(APP_ID) is None


# ===== the proxy pushes the state it is in =====


class _Handle:
    """Stands in for the Ray actor handle; records the ``.remote`` payloads."""

    def __init__(self) -> None:
        self.reports = []
        self.claims = []
        self.report_service_registration = MagicMock()
        self.report_service_registration.remote = self._record
        self.claim_service_registration = MagicMock()
        self.claim_service_registration.remote = self._record_claim
        self.register_serve_replica = MagicMock()

    def _record(
        self, application_id: str, registered: bool, replica_id=None
    ) -> None:
        self.reports.append((application_id, registered))

    def _record_claim(self, application_id: str, replica_id=None) -> None:
        self.claims.append((application_id, replica_id))


def _bare_proxy(**attrs):
    inst = object.__new__(ProxyDouble)
    inst.application_id = APP_ID
    inst._replica_id = "replica-0"
    inst.entry_deployment_ready = True
    inst.server = None
    inst.websocket_service_id = None
    inst.rtc_service_id = None
    inst.mcp_service_id = None
    inst._rtc_config = None
    inst._ice_expires_at = None
    inst._registration_lock = asyncio.Lock()
    inst._maintenance_task = None
    inst._connection_lost = False
    inst._registration_failure = None
    inst._usage_ledger = None
    inst._probe_due_at = 0.0
    inst._next_register_at = 0.0
    inst._proxy_actor_handle = _Handle()
    for key, value in attrs.items():
        setattr(inst, key, value)
    return inst


@pytest.mark.asyncio
async def test_a_successful_registration_is_reported() -> None:
    inst = _bare_proxy()

    async def _register():
        inst.websocket_service_id = "ws"

    inst._register_services = _register

    await inst._maintenance_tick()

    assert inst._proxy_actor_handle.reports == [(APP_ID, True)]


@pytest.mark.asyncio
async def test_a_failed_registration_is_not_reported_as_registered() -> None:
    inst = _bare_proxy()

    async def _register():
        raise RuntimeError("hypha down")

    inst._register_services = _register

    await inst._maintenance_tick()

    assert inst._proxy_actor_handle.reports == []


@pytest.mark.asyncio
async def test_deregistering_reports_the_service_as_gone() -> None:
    inst = _bare_proxy()

    await inst._deregister_services()

    assert inst._proxy_actor_handle.reports == [(APP_ID, False)]


def _construct_proxy(monkeypatch, handle: _Handle, replica_tag: str = "replica-0"):
    """Build a real ProxyDeployment, with only Ray's two lookups stubbed.

    ``ProxyDouble`` is that class minus Ray Serve's async destructor, which no
    test has a replica to await; nothing on the claim path differs.
    """
    monkeypatch.setattr(pd_module.ray, "get_actor", lambda name, namespace: handle)
    monkeypatch.setattr(
        pd_module,
        "get_replica_context",
        lambda: MagicMock(
            deployment="ProxyDeployment", replica_tag=replica_tag, app_name=APP_ID
        ),
    )
    return ProxyDouble(
        application_id=APP_ID,
        application_name="Nuclei Segmentation",
        application_description="Segment nuclei.",
        app_data={},
        entry_deployment_handle=MagicMock(),
        method_schemas=[],
        max_ongoing_requests=1,
        server_url="https://hypha.example",
        workspace="bioimage-io",
        proxy_service_token="tok",
        worker_client_id=WORKER_CLIENT_ID,
        authorized_users={"*": ["*"]},
        proxy_actor_name="proxy-actor",
        debug=False,
    )


def test_the_proxy_claims_the_record_as_unregistered_at_init(monkeypatch) -> None:
    # Without the claim the worker sees "never reported" on a brand-new app and
    # falls back to the replica-alive gate — i.e. the original bug. Constructing
    # the real class is the point: a claim placed where the actor handle is not
    # set yet would silently no-op and still read fine in the source.
    handle = _Handle()

    _construct_proxy(monkeypatch, handle)

    assert handle.claims == [(APP_ID, "replica-0")]
    assert handle.reports == []


def test_the_claim_carries_the_reporting_replica_tag(monkeypatch) -> None:
    handle = _Handle()

    _construct_proxy(monkeypatch, handle, replica_tag="replica-7")

    assert handle.claims == [(APP_ID, "replica-7")]


def test_a_missing_actor_handle_never_breaks_the_replica() -> None:
    inst = _bare_proxy(_proxy_actor_handle=None)
    inst._report_service_registration(True)  # must not raise
    inst._claim_service_registration()  # must not raise


def test_reporting_survives_a_part_built_replica() -> None:
    # __del__ -> _deregister_services -> here, reachable before __init__ has
    # assigned the handle at all.
    inst = object.__new__(ProxyDouble)
    inst._report_service_registration(False)  # must not raise AttributeError


# ===== a departing replica cannot overwrite its successor's verdict =====


def test_a_late_deregistration_from_the_old_replica_is_ignored() -> None:
    # Rolling update: the incoming replica claims the record, registers and
    # reports True; the outgoing replica's __del__ deregisters afterwards.
    # Taking that last write leaves a healthy app reading False until the
    # serving replica's next reachability probe re-asserts True — a whole probe
    # interval of a working address being withheld. The guard avoids the window
    # rather than relying on the re-assert to close it.
    actor = _bare_actor()
    actor.report_service_registration(APP_ID, True, replica_id="old")
    actor.claim_service_registration(APP_ID, replica_id="new")
    # Asserted here and not only at the end: the claim has to displace the
    # predecessor's record for the gate to bite at all, and a final-state-only
    # assertion passes just as happily when it was silently dropped.
    assert actor.get_service_registration(APP_ID) is False
    actor.report_service_registration(APP_ID, True, replica_id="new")

    actor.report_service_registration(APP_ID, False, replica_id="old")

    assert actor.get_service_registration(APP_ID) is True


def test_the_current_replica_can_still_deregister() -> None:
    actor = _bare_actor()
    actor.report_service_registration(APP_ID, True, replica_id="new")

    actor.report_service_registration(APP_ID, False, replica_id="new")

    assert actor.get_service_registration(APP_ID) is False


def test_the_old_replica_leaving_first_still_leaves_the_app_registered() -> None:
    # The other interleaving of the same update: __del__ lands before the
    # incoming replica gets as far as registering.
    actor = _bare_actor()
    actor.report_service_registration(APP_ID, True, replica_id="old")

    actor.report_service_registration(APP_ID, False, replica_id="old")
    actor.claim_service_registration(APP_ID, replica_id="new")
    actor.report_service_registration(APP_ID, True, replica_id="new")

    assert actor.get_service_registration(APP_ID) is True


class _ActorHandle:
    """Routes the proxy's ``.remote(...)`` pushes into a real proxy actor.

    ``register_serve_replica`` resolves an actor id through Ray's state API, so
    it is stubbed; every other push lands on the real actor method.
    """

    def __init__(self, actor: object) -> None:
        self._actor = actor
        self.register_serve_replica = MagicMock()

    def __getattr__(self, name: str) -> MagicMock:
        method = getattr(self._actor, name)
        return MagicMock(remote=lambda *args, **kwargs: method(*args, **kwargs))


def test_a_replica_that_died_without_deregistering_loses_the_record(
    monkeypatch,
) -> None:
    # SIGKILL, node loss, health-driven restart: Ray never runs __del__, so no
    # False ever arrives from the dead replica and the record stays True under
    # its tag. The successor's init is then the only signal left, and a plain
    # tagged False from it is exactly what the guard drops — leaving the worker
    # advertising an address that answers nothing for the whole startup window,
    # which is the reported bug. Driven through the real ProxyDeployment so it
    # pins the outcome, not the call shape that produces it.
    actor = _bare_actor()
    actor.report_service_registration(APP_ID, True, replica_id="old")

    _construct_proxy(monkeypatch, _ActorHandle(actor), replica_tag="new")

    assert actor.get_service_registration(APP_ID) is False


class _LiveServer:
    """A Hypha connection whose reachability probe answers."""

    def __init__(self) -> None:
        self.probes = 0

    async def get_service_info(self, _sid):
        self.probes += 1

    async def unregister_service(self, _sid):
        return None


@pytest.mark.asyncio
async def test_a_successor_that_never_registers_cannot_pin_a_serving_app_at_false(
    monkeypatch,
) -> None:
    # Redeploy: Ray Serve builds the new proxy replica while the old one is
    # still serving, and both want the same Hypha client_id, so the successor's
    # registration is refused for as long as the predecessor holds it. Its
    # init-time claim has already replaced the record with False. Nothing then
    # re-reported True — a registered replica skips the registration branch of
    # _maintenance_tick entirely — so the worker withheld a working address
    # indefinitely, which is the mirror image of the bug the claim fixed.
    actor = _bare_actor()

    old = _bare_proxy(_replica_id="old", _proxy_actor_handle=_ActorHandle(actor))
    server = _LiveServer()

    async def _register():
        old.server = server
        old.websocket_service_id = "ws"

    old._register_services = _register
    await old._maintenance_tick()
    assert actor.get_service_registration(APP_ID) is True

    _construct_proxy(monkeypatch, _ActorHandle(actor), replica_tag="new")
    # Asserted mid-way: without the claim actually landing there is no
    # divergence left to heal, and the final assertion would pass vacuously.
    assert actor.get_service_registration(APP_ID) is False

    # The predecessor is still serving, so its periodic probe is the only
    # signal left that the app is reachable.
    old._probe_due_at = 0.0
    await old._maintenance_tick()

    assert server.probes == 1
    assert actor.get_service_registration(APP_ID) is True


@pytest.mark.asyncio
async def test_the_probe_does_not_re_register_an_app_that_deregistered_itself() -> None:
    # A sibling deployment going down closes the gate via _deregister_services.
    # The self-healing re-assert must not undo that on the next tick: this
    # replica reports False about itself and stays that way until it re-registers.
    actor = _bare_actor()
    inst = _bare_proxy(
        _replica_id="r0",
        _proxy_actor_handle=_ActorHandle(actor),
        server=_LiveServer(),
        websocket_service_id="ws",
    )
    actor.report_service_registration(APP_ID, True, replica_id="r0")

    await inst._deregister_services()
    inst._probe_due_at = 0.0
    await inst._maintenance_tick()
    await inst._maintenance_tick()

    assert actor.get_service_registration(APP_ID) is False


@pytest.mark.asyncio
async def test_a_failing_probe_does_not_assert_the_app_as_registered() -> None:
    # The probe is what establishes the address still resolves. When it fails
    # the replica rebuilds its client instead, and must not claim reachability
    # it has just failed to confirm.
    class _AmnesiacServer:
        async def get_service_info(self, sid):
            raise KeyError(f"Service not found: {sid}@*")

    actor = _bare_actor()
    inst = _bare_proxy(
        _replica_id="r0",
        _proxy_actor_handle=_ActorHandle(actor),
        server=_AmnesiacServer(),
        websocket_service_id="ws",
    )
    actor.claim_service_registration(APP_ID, replica_id="newer")

    await inst._maintenance_tick()

    assert inst._connection_lost is True
    assert actor.get_service_registration(APP_ID) is False


@pytest.mark.asyncio
async def test_a_probe_answered_after_a_deregistration_does_not_re_register() -> None:
    # check_health can deregister this replica while the probe is suspended.
    # Reporting True on the answer to a question asked before that would leave
    # a deregistered app advertised with no way back, since every later tick
    # returns at the readiness gate.
    entered = asyncio.Event()
    release = asyncio.Event()

    class _SlowServer:
        async def get_service_info(self, _sid):
            entered.set()
            await release.wait()

        async def unregister_service(self, _sid):
            return None

    actor = _bare_actor()
    inst = _bare_proxy(
        _replica_id="r0",
        _proxy_actor_handle=_ActorHandle(actor),
        server=_SlowServer(),
        websocket_service_id="ws",
    )
    actor.report_service_registration(APP_ID, True, replica_id="r0")

    tick = asyncio.create_task(inst._maintenance_tick())
    await asyncio.wait_for(entered.wait(), timeout=5)
    await inst._deregister_services()
    release.set()
    await asyncio.wait_for(tick, timeout=5)

    assert actor.get_service_registration(APP_ID) is False


def test_an_untagged_report_still_works() -> None:
    # A replica that could not read its Ray replica context reports None, which
    # must degrade to the plain last-writer behaviour rather than being dropped.
    actor = _bare_actor()
    actor.report_service_registration(APP_ID, True)

    actor.report_service_registration(APP_ID, False)

    assert actor.get_service_registration(APP_ID) is False


# ===== the deployment log no longer claims completion =====


@pytest.mark.asyncio
async def test_deploy_does_not_claim_completion_before_the_replicas_run(
    caplog,
) -> None:
    # serve.run is non-blocking here — the replicas are only starting, and the
    # Hypha service does not exist until the proxy registers it.
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.deploy_log")
    manager._deployed_applications = {APP_ID: _make_app_info()}
    manager._deployed_applications[APP_ID]["is_deployed"] = asyncio.Event()
    manager._deployed_applications[APP_ID]["built_app"] = MagicMock()
    manager.app_builder = MagicMock()
    manager.app_builder.submit = AsyncMock()

    with caplog.at_level(logging.INFO, logger="test.deploy_log"):
        task = asyncio.create_task(manager._deploy_application(APP_ID))
        await asyncio.wait_for(
            manager._deployed_applications[APP_ID]["is_deployed"].wait(), timeout=5
        )
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    manager.app_builder.submit.assert_awaited_once()
    emitted = [
        r.getMessage()
        for r in caplog.records
        if r.name == "test.deploy_log" and r.levelno == logging.INFO
    ]
    submitted = [m for m in emitted if APP_ID in m and "1.0.1" in m]
    assert submitted, emitted
    assert not any("completed deployment" in m.lower() for m in emitted)
