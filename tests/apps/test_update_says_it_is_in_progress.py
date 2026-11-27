"""RED arm for the update path, written as a standalone pytest file so it can be dropped in.

These arms assert the DEFECT rather than a mechanism, so they were written and shown RED
before the fix existed and did not need rewriting when its shape was ruled on.

WHAT IT ASSERTS, and why these two statements are design-independent:

  A  During an update's build -- the window where the caller's RPC can time out -- a caller
     must be able to tell that an update is IN PROGRESS. Today it reads the OLD version's
     Serve status (typically RUNNING at the old version) with nothing distinguishing "my
     update is still building" from "my update never started". Any fix that makes the window
     visible satisfies this; none of them is assumed here.

  B  In the window after the build and before re-registration, the status must not claim the
     application "has not been deployed yet". It IS deployed -- at the old version -- and is
     being updated. That sentence is false regardless of how the fix is shaped.

Both are driven through the REAL ``deploy_app`` with the build (or the resource check) held
open by a fake builder. Never by hand-building tracked state: that is how the first attempt
at the new-application fix went green without touching its incident.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager

WORKSPACE = "bioimage-io"
ARTIFACT_ID = f"{WORKSPACE}/nuclei-seg"
APP_ID = "nuclei-seg"
CONTEXT = {"user": {"id": "u-1", "email": "u@lab.test"}}
VERSIONS = [
    {"version": "1.0.0", "created_at": 1},
    {"version": "2.0.0", "created_at": 2},
]
OLD, NEW = "1.0.0", "2.0.0"
NOT_DEPLOYED_PHRASE = "has not been deployed yet"


def _fake_app(version=NEW):
    return SimpleNamespace(
        metadata={
            "name": "Nuclei Seg",
            "description": "Segment nuclei.",
            "version": version,
            "application_kwargs": {},
            "application_env_vars": {},
            "resources": {},
            "authorized_users": {"*": ["*"]},
            "available_methods": ["segment"],
            "deployed_by_worker_client_id": "client-1",
            "proxy_service_token_issued_at": 0.0,
            "proxy_service_token_ttl_seconds": 3600,
        },
        spec={"classes": {}},
        scaling={},
    )


def _running_entry():
    ev = asyncio.Event()
    ev.set()
    return {
        "display_name": "Nuclei Seg",
        "description": "Segment nuclei.",
        "artifact_id": ARTIFACT_ID,
        "version": OLD,
        "source_signature": None,
        "application_kwargs": {},
        "application_env_vars": {},
        "redacted_env_var_keys": {},
        "hypha_token": "tok",
        "disable_gpu": False,
        "max_ongoing_requests": 10,
        "proxy_memory_in_gb": 0.5,
        "scaling": {},
        "application_resources": {},
        "resource_shortfall": None,
        "authorized_users": {"*": ["*"]},
        "available_methods": ["segment"],
        "static_site_url": None,
        "started_at": 0.0,
        "last_updated_at": 0.0,
        "last_updated_by": "u-1",
        "deployed_by_worker_client_id": "client-1",
        "proxy_service_token_issued_at": 0.0,
        "proxy_service_token_ttl_seconds": 3600,
        "entry_deployment_name": "EntryDeployment",
        "built_app": None,
        "auto_redeploy": False,
        "ice_servers": None,
        "debug": False,
        "deployment_task": None,
        "is_deployed": ev,
        "undeployment_task": None,
        "recovered_app": False,
    }


class _Harness:
    """Drives a real UPDATE through deploy_app, gated where the arm needs it."""

    def __init__(self, *, gate: str):
        m = object.__new__(AppsManager)
        m.logger = logging.getLogger("test.update_path")
        m.admin_users = ["*"]
        m._deployment_lock = asyncio.Lock()
        m._check_initialized = lambda: None
        m.server = MagicMock()
        m.server.config.workspace = WORKSPACE
        m.server.config.public_base_url = "https://example.invalid"
        m.artifact_manager = MagicMock()
        m.artifact_manager.read = AsyncMock(return_value={"versions": VERSIONS})
        m.ray_cluster = MagicMock()
        m.ray_cluster.check_connection = AsyncMock()
        m.ray_cluster.call_with_reconnect = AsyncMock()
        m._cancel_deployment_process = AsyncMock()
        m._deploy_application = AsyncMock()
        m._generate_application_id = AsyncMock(return_value=APP_ID)
        m._get_deployment_status = AsyncMock(return_value={})
        m._get_application_service_ids = AsyncMock(return_value=({}, False))
        m._reported_token_issued_at = AsyncMock(return_value=0.0)
        m._get_startup_version_pin = AsyncMock(return_value=None)
        m._filter_secret_env_vars = lambda env: dict(env or {})
        # The application is ALREADY deployed: this is an update, not a new deploy.
        m._deployed_applications = {APP_ID: _running_entry()}

        self.m = m
        self.reached = asyncio.Event()
        self.release = asyncio.Event()

        async def _build(**_kw):
            if gate == "build":
                self.reached.set()
                await self.release.wait()
            return _fake_app()

        async def _check_resources(**_kw):
            if gate == "resources":
                self.reached.set()
                await self.release.wait()
            return None

        m.app_builder = SimpleNamespace(build=_build)
        m._check_resources = _check_resources

    #: What ``start()`` asks deploy_app for. An arm sets this to None to make a
    #: version-less call -- round 2 of this change had an arm that set it while ``start()``
    #: still hard-coded NEW, so the arm never made the call it claimed to and pinned
    #: nothing. The attribute and its use must stay together.
    requested_version = NEW

    def start(self):
        return asyncio.ensure_future(
            self.m.deploy_app(
                artifact_id=ARTIFACT_ID,
                application_id=APP_ID,
                version=self.requested_version,
                context=CONTEXT,
            )
        )

    async def status(self, instance_details):
        return await asyncio.wait_for(
            self.m._get_app_status(
                application_id=APP_ID,
                instance_details=instance_details,
                n_previous_replica=0,
                logs_tail=0,
            ),
            timeout=5,
        )

    async def finish(self, fut):
        self.release.set()
        try:
            await asyncio.wait_for(fut, timeout=5)
        except Exception:
            pass


SERVE_RUNNING_OLD = {
    "applications": {APP_ID: {"status": "RUNNING", "message": "", "deployments": {}}}
}
SERVE_EMPTY = {"applications": {}}


@pytest.mark.asyncio
async def test_A_an_update_in_flight_is_named_in_the_message() -> None:
    """RED on main. The message must name BOTH versions.

    The ruling: keep Serve's status, and say the update explicitly -- "update to <version>
    in progress; the running version is <old>". Naming only one of the two would leave the
    caller unable to tell which is which.
    """
    h = _Harness(gate="build")
    fut = h.start()
    try:
        await asyncio.wait_for(h.reached.wait(), timeout=5)
        assert not fut.done(), "the RPC returned; this arm must read during the build"

        st = await h.status(SERVE_RUNNING_OLD)
        msg = str(st.get("message", ""))

        assert NEW in msg, (
            f"the message does not name the version being deployed ({NEW}), so a caller "
            f"cannot tell WHICH update is in flight: {msg!r}"
        )
        assert OLD in msg, (
            f"the message does not name the version still running ({OLD}), so a caller "
            f"cannot tell what is serving right now: {msg!r}"
        )
    finally:
        await h.finish(fut)


@pytest.mark.asyncio
async def test_A2_serve_status_is_NOT_overridden_while_the_old_version_serves() -> None:
    """GREEN both ways, and it must stay green.

    The ruling is explicit: do NOT report DEPLOYING over a running application, because a
    caller would read it as "not serving". Serve says RUNNING and that is TRUE -- the old
    version really is up. This arm stops the fix being made by overriding the status.
    """
    h = _Harness(gate="build")
    fut = h.start()
    try:
        await asyncio.wait_for(h.reached.wait(), timeout=5)
        st = await h.status(SERVE_RUNNING_OLD)
        assert st.get("status") == "RUNNING", (
            f"Serve reported RUNNING for the old version and the status was overridden to "
            f"{st.get('status')!r}; a caller would read that as not serving"
        )
    finally:
        await h.finish(fut)


@pytest.mark.asyncio
async def test_A3_a_field_carries_the_version_the_update_is_heading_for() -> None:
    """RED on main. The caller cannot derive this.

    ``version`` in the response is what is RUNNING, which during an update is still the old
    one, so the target has to be carried separately or it is unavailable.
    """
    h = _Harness(gate="build")
    fut = h.start()
    try:
        await asyncio.wait_for(h.reached.wait(), timeout=5)
        st = await h.status(SERVE_RUNNING_OLD)
        assert st.get("updating_to_version") == NEW, (
            f"no field carries the version being deployed: updating_to_version="
            f"{st.get('updating_to_version')!r}, while version={st.get('version')!r} is "
            f"the running one"
        )
    finally:
        await h.finish(fut)


@pytest.mark.asyncio
async def test_B_a_deployed_application_being_updated_is_not_called_never_deployed() -> (
    None
):
    """RED today. The sentence is false whatever the fix's shape."""
    h = _Harness(gate="resources")
    fut = h.start()
    try:
        await asyncio.wait_for(h.reached.wait(), timeout=5)
        assert (
            APP_ID in h.m._deployed_applications
        ), "the application left tracking, so this arm is not in the window it claims"

        st = await h.status(SERVE_EMPTY)
        assert st.get("status") == "DEPLOYING", (
            f"in the delete-to-redeploy window the status reads {st.get('status')!r}; the "
            f"ruling requires DEPLOYING for an id the worker is actively building"
        )
        assert NOT_DEPLOYED_PHRASE not in str(st.get("message", "")), (
            f"an application deployed at {OLD} and being updated to {NEW} is reported as "
            f'"{NOT_DEPLOYED_PHRASE}": status={st.get("status")!r} '
            f"message={str(st.get('message'))[:120]!r}"
        )
    finally:
        await h.finish(fut)


@pytest.mark.asyncio
async def test_C_the_new_application_window_is_unaffected() -> None:
    """GREEN, and must stay green: the fix for updates must not regress new deploys.

    A brand-new application still in its build reports DEPLOYING rather than the
    NOT_RUNNING invitation. That is the behaviour the new-application change established.
    """
    h = _Harness(gate="build")
    h.m._deployed_applications = {}  # a NEW application, nothing tracked
    fut = h.start()
    try:
        await asyncio.wait_for(h.reached.wait(), timeout=5)
        st = await h.status(SERVE_EMPTY)
        assert st.get("status") == "DEPLOYING", (
            f"a NEW application in its build must still report DEPLOYING, got "
            f"{st.get('status')!r}"
        )
        assert "To deploy this application" not in str(
            st.get("message", "")
        ), "the NOT_RUNNING retry invitation came back for a new application"
    finally:
        await h.finish(fut)


# ── the regression the first round shipped ─────────────────────────────────────────────


def _init_value_for_the_marker():
    """Run ``__init__``'s OWN assignment for the marker and return what it produced.

    Deliberately executing the real line rather than hard-coding a value. Round 1 of this
    change broke EVERY deploy_app -- new or update -- because the marker became a mapping
    while ``__init__`` still built a set, so ``mark()`` did item assignment on a set and
    raised TypeError before the build. Nothing caught it: every arm builds the manager with
    ``object.__new__``, and the one test line that DID mirror ``__init__`` was edited to
    match the new type instead of the code being fixed, which removed the last signal.

    Reading the line from the source couples this arm to ``__init__``: if the two ever
    disagree again, this goes red instead of production.
    """
    import re
    from pathlib import Path

    import bioengine.apps.manager as manager_module

    src = Path(manager_module.__file__).read_text()
    m = re.search(r"^\s*(self\._deploying_application_ids = .+)$", src, re.M)
    assert (
        m
    ), "__init__ no longer initialises _deploying_application_ids; this arm is blind"
    holder = SimpleNamespace()
    exec(m.group(1).strip(), {}, {"self": holder})
    return holder._deploying_application_ids


@pytest.mark.asyncio
async def test_deploy_app_works_on_a_marker_built_the_way_init_builds_it() -> None:
    """RED if __init__ and the marker's use ever disagree again.

    Drives a real deploy_app with the marker holding exactly what __init__ produces.
    """
    h = _Harness(gate="build")
    h.m._deploying_application_ids = _init_value_for_the_marker()
    fut = h.start()
    try:
        await asyncio.wait_for(h.reached.wait(), timeout=5)
        assert not fut.done(), (
            "deploy_app returned or raised before reaching the build, with the marker "
            "initialised the way __init__ initialises it"
        )
    finally:
        await h.finish(fut)
        if fut.done() and fut.exception() is not None:
            raise AssertionError(
                f"deploy_app raised with __init__'s marker value: {fut.exception()!r}"
            )


@pytest.mark.asyncio
async def test_the_update_is_named_even_when_serve_has_gone_quiet() -> None:
    """RED on main. The window the issue's own table lists and arm A did not cover.

    ``is_deployed`` is set (the old application was deployed) and Ray Serve reports nothing,
    so the status is UNHEALTHY. That remains honest, but without naming the update a caller
    reads a fault where there is a deployment in progress.
    """
    h = _Harness(gate="build")
    fut = h.start()
    try:
        await asyncio.wait_for(h.reached.wait(), timeout=5)
        st = await h.status(SERVE_EMPTY)
        msg = str(st.get("message", ""))
        assert NEW in msg, (
            f"Serve went quiet during the update and the message says nothing about it: "
            f"status={st.get('status')!r} message={msg!r}"
        )
    finally:
        await h.finish(fut)


@pytest.mark.asyncio
async def test_a_version_less_update_reports_the_inherited_version() -> None:
    """RED on main. What omitting the version actually does, measured not assumed.

    ``deploy_app`` inherits the running version when the caller names none, and that happens
    BEFORE the id is marked -- so an update never carries a null target. The field therefore
    reports the OLD version here, which is correct: that IS what this deploy is heading for.

    Round 2 of this change asserted "not None" instead, through a harness that never made a
    version-less call, so it passed whatever the code did.
    """
    h = _Harness(gate="build")
    h.requested_version = None
    fut = h.start()
    try:
        await asyncio.wait_for(h.reached.wait(), timeout=5)
        st = await h.status(SERVE_RUNNING_OLD)
        assert st.get("updating_to_version") == OLD, (
            f"a version-less update should report the inherited version {OLD!r}, got "
            f"{st.get('updating_to_version')!r}"
        )
        assert NOT_DEPLOYED_PHRASE not in str(st.get("message", ""))
    finally:
        await h.finish(fut)


@pytest.mark.asyncio
async def test_the_latest_default_is_reachable_and_pinned() -> None:
    """Makes the ``or "latest"`` default LIVE rather than dead code.

    The default guarantees the field is never null while a deploy is in flight. With a
    tracked entry whose own version is None -- which an adopted application can have -- the
    inherit above yields None too, and the default is the only thing standing between the
    caller and a null that is indistinguishable from "nothing in flight".

    Round 2 found the previous version-less arm dead: removing the default passed every arm.
    This one fails without it.
    """
    h = _Harness(gate="build")
    h.m._deployed_applications[APP_ID]["version"] = None
    h.requested_version = None
    fut = h.start()
    try:
        await asyncio.wait_for(h.reached.wait(), timeout=5)
        st = await h.status(SERVE_RUNNING_OLD)
        assert st.get("updating_to_version") == "latest", (
            f"with nothing to inherit the field must fall back to 'latest' rather than a "
            f"null a caller cannot tell from 'no deploy in flight', got "
            f"{st.get('updating_to_version')!r}"
        )
    finally:
        await h.finish(fut)
