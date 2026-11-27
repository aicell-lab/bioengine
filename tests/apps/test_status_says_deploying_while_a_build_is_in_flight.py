"""A caller whose ``deploy_app`` RPC timed out must not be told to call it again.

THE WINDOW THAT MATTERS, and the one round 1 of this change missed. ``deploy_app`` does all
the slow work before it registers anything::

    await self.app_builder.build(...)          # artifact sync + the introspect Ray task
    await self._check_resources(...)
    self._deployed_applications[application_id] = {...}     # registration, AFTER the build
    deployment_task = asyncio.create_task(...)
    return {...}                                            # returns WITHOUT awaiting

So the only window in which a caller's RPC can time out is the BUILD window — and there the
application is in ``_deployed_applications`` nowhere, so ``_get_app_status`` took its early
return and said ``NOT_RUNNING``, "To deploy this application, call deploy_app(...)". That is
the retry invitation, and a retry starts a second build of the same application.

The first attempt at this change keyed on ``deployment_task`` being pending. That state is
real but unreachable by a timed-out caller, because the task exists only after registration,
i.e. after the RPC has already returned. The arms went green against a hand-built
``application_info`` and never touched the incident.

WHICH IS WHY EVERY ARM HERE DRIVES ``deploy_app`` ITSELF, with the build held open by a fake
builder, and reads status while the RPC is genuinely still pending. Nothing is hand-built
into ``_deployed_applications``.
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
VERSIONS = [{"version": "1.0.0", "created_at": 1}]
# The INVITATION is the distinctive phrase of the NOT_RUNNING message, not any mention of
# deploy_app. The first draft of this arm matched the bare substring "call deploy_app" and
# failed against the fix -- because the DEPLOYING message names deploy_app precisely in order
# to FORBID calling it ("Do NOT call deploy_app again"). Matching the shape rather than the
# thing made a correct message look like the defect.
INVITATION = "To deploy this application"


def _fake_app():
    """The minimum of ``AppBuilder.build``'s result that ``deploy_app`` reads."""
    return SimpleNamespace(
        metadata={
            "name": "Nuclei Seg",
            "description": "Segment nuclei.",
            "version": "1.0.0",
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


class _Harness:
    """An ``AppsManager`` driven through the real ``deploy_app``, with a gated build."""

    def __init__(self, *, build_raises: bool = False):
        m = object.__new__(AppsManager)
        m.logger = logging.getLogger("test.deploying_status")
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
        m._check_resources = AsyncMock(return_value=None)
        m._cancel_deployment_process = AsyncMock()
        m._deploy_application = AsyncMock()
        m._generate_application_id = AsyncMock(return_value=APP_ID)
        m._deployed_applications = {}
        # The field under test. Initialised the way __init__ does, so the arms cannot pass by
        # the attribute merely existing on the instance.
        m._deploying_application_ids = set()

        self.manager = m
        self.in_build = asyncio.Event()
        self.release_build = asyncio.Event()
        self.builds = 0
        self._build_raises = build_raises

        async def _build(**_kwargs):
            self.builds += 1
            self.in_build.set()
            await self.release_build.wait()
            if self._build_raises:
                raise RuntimeError(
                    "build failed (arm: the marker must still be cleared)"
                )
            return _fake_app()

        m.app_builder = SimpleNamespace(build=_build)

    def start_deploy(self):
        return asyncio.ensure_future(
            self.manager.deploy_app(
                artifact_id=ARTIFACT_ID, application_id=APP_ID, context=CONTEXT
            )
        )

    async def status(self, timeout: float = 2.0):
        """Read status the way the public method does, with a bound.

        The bound is the point of one of the arms: ``deploy_app`` holds
        ``_deployment_lock`` across the whole build, so a status read that acquired that
        lock would not return until the deploy finished. ``asyncio.wait_for`` turns that
        into a failure instead of a hang.
        """
        return await asyncio.wait_for(
            self.manager._get_app_status(
                application_id=APP_ID,
                instance_details={"applications": {}},
                n_previous_replica=0,
                logs_tail=0,
            ),
            timeout=timeout,
        )

    async def finish(self, fut):
        self.release_build.set()
        try:
            await asyncio.wait_for(fut, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_status_during_the_held_build_does_not_say_NOT_RUNNING() -> None:
    """RED on main: the timed-out caller is told to call deploy_app again."""
    h = _Harness()
    fut = h.start_deploy()
    try:
        await asyncio.wait_for(h.in_build.wait(), timeout=5)
        assert (
            not fut.done()
        ), "the RPC already returned; this arm must read during the build"

        st = await h.status()

        assert st.get("status") != "NOT_RUNNING", (
            "while the build is still running, get_app_status reports NOT_RUNNING — the "
            "state a caller whose deploy_app RPC timed out actually sees"
        )
        assert INVITATION not in str(st.get("message", "")), (
            f"the message invites the caller to {INVITATION!r} while a build of that same "
            f"application is in flight; acting on it starts a second build: "
            f"{st.get('message')!r}"
        )
    finally:
        await h.finish(fut)


@pytest.mark.asyncio
async def test_status_during_the_held_build_is_DEPLOYING() -> None:
    """RED on main. The value is Ray Serve's own DEPLOYING, not a new one."""
    h = _Harness()
    fut = h.start_deploy()
    try:
        await asyncio.wait_for(h.in_build.wait(), timeout=5)
        st = await h.status()
        assert (
            st.get("status") == "DEPLOYING"
        ), f"a build in flight should report DEPLOYING, got {st.get('status')!r}"
    finally:
        await h.finish(fut)


@pytest.mark.asyncio
async def test_a_status_read_during_the_build_RETURNS_rather_than_blocking() -> None:
    """The lock must NOT be acquired by the status read.

    ``deploy_app`` holds ``_deployment_lock`` for its whole body, build included. An
    implementation that consulted the in-flight set *under* that lock would make every
    status call wait for the deploy — turning an honest answer into a hang, which is worse
    than the wrong answer it replaced. GREEN both before and after the change; it exists to
    stop the fix being made that way.
    """
    h = _Harness()
    fut = h.start_deploy()
    try:
        await asyncio.wait_for(h.in_build.wait(), timeout=5)
        assert (
            h.manager._deployment_lock.locked()
        ), "the deploy is not holding the lock, so this arm is not testing contention"
        await h.status(timeout=2.0)  # a hang fails here via wait_for
    finally:
        await h.finish(fut)


@pytest.mark.asyncio
async def test_a_second_deploy_during_the_held_build_cannot_start_a_second_build() -> (
    None
):
    """What a retry does TODAY, pinned so this change does not alter it by accident.

    Measured: the second call blocks on ``_deployment_lock``, which the first holds across
    its build, so it does not reach ``app_builder.build`` while the first is in flight. That
    is the behaviour being preserved — and it is why the retry invitation mattered: the
    caller's second call does not fail fast, it waits, and then builds again.
    """
    h = _Harness()
    first = h.start_deploy()
    try:
        await asyncio.wait_for(h.in_build.wait(), timeout=5)
        assert h.builds == 1

        second = h.start_deploy()
        await asyncio.sleep(0.2)  # give it every chance to proceed

        assert h.builds == 1, (
            f"a second deploy_app reached the builder while the first build was still "
            f"running: {h.builds} builds"
        )
        assert not second.done(), "the second deploy_app returned without waiting"

        h.release_build.set()
        await asyncio.wait_for(
            asyncio.gather(first, second, return_exceptions=True), timeout=10
        )
        assert (
            h.builds == 2
        ), f"after the first build was released the second should have built, saw {h.builds}"
    finally:
        await h.finish(first)


@pytest.mark.asyncio
async def test_the_marker_is_cleared_once_the_deploy_returns() -> None:
    """A marker left behind would report DEPLOYING for ever."""
    h = _Harness()
    fut = h.start_deploy()
    await asyncio.wait_for(h.in_build.wait(), timeout=5)
    assert (
        APP_ID in h.manager._deploying_application_ids
    ), "the id was not marked during the build, so the status branch cannot fire"
    await h.finish(fut)
    assert APP_ID not in h.manager._deploying_application_ids, (
        "the id is still marked after deploy_app returned; status would report DEPLOYING "
        "for ever"
    )


@pytest.mark.asyncio
async def test_the_marker_is_cleared_when_the_BUILD_RAISES() -> None:
    """The failure path, which is the one a leaked marker would lie about.

    A build that raises must leave nothing behind: otherwise a deploy that FAILED reports
    DEPLOYING indefinitely — a worse falsehood than the NOT_RUNNING this replaces. This is
    what the ``finally`` in the marker is for, and dropping it leaves this arm RED.
    """
    h = _Harness(build_raises=True)
    fut = h.start_deploy()
    await asyncio.wait_for(h.in_build.wait(), timeout=5)
    assert APP_ID in h.manager._deploying_application_ids

    h.release_build.set()
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(fut, timeout=5)

    assert APP_ID not in h.manager._deploying_application_ids, (
        "a build that raised left the id marked, so get_app_status would report DEPLOYING "
        "for an application whose deploy failed"
    )
    st = await h.status()
    assert (
        st.get("status") == "NOT_RUNNING"
    ), f"after a failed build the application really is not deployed, got {st.get('status')!r}"


@pytest.mark.asyncio
async def test_an_unrelated_application_is_unaffected_by_a_build_in_flight() -> None:
    """The marker is per application id, not a global 'something is deploying' flag."""
    h = _Harness()
    fut = h.start_deploy()
    try:
        await asyncio.wait_for(h.in_build.wait(), timeout=5)
        st = await asyncio.wait_for(
            h.manager._get_app_status(
                application_id="some-other-app",
                instance_details={"applications": {}},
                n_previous_replica=0,
                logs_tail=0,
            ),
            timeout=2.0,
        )
        assert st.get("status") == "NOT_RUNNING", (
            f"an application nobody is deploying must still read NOT_RUNNING, got "
            f"{st.get('status')!r}"
        )
    finally:
        await h.finish(fut)
