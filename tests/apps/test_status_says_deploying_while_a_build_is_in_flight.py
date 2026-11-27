"""While a deploy is still building, ``get_app_status`` must say so.

THE DEFECT. In ``_get_app_status``, when Ray Serve has no details for an application yet,
there are exactly two cases::

    if application_info["is_deployed"].is_set():
        status = "UNHEALTHY"      # tracked, marked deployed, absent from Serve
    else:
        status = "NOT_STARTED"
        message = f"Application '{application_id}' has not been deployed yet."

Mid-deploy all three of these hold at once: the application IS in ``_deployed_applications``;
Serve has no ``application_details`` for it yet; and ``is_deployed`` is CLEAR, because it is
cleared when the deploy starts and set only on completion. So a caller is told the
application "has not been deployed yet" while a build task is running and about to succeed.

That is the dishonesty behind the incident this comes from: ``deploy_app``'s RPC timed out
although the deploy SUCCEEDED, and nothing the caller could read afterwards said so.

It is worse than silence. "NOT_STARTED ... has not been deployed yet" invites a retry, and a
retry while a build is in flight is the very race that had to be fixed separately: a second
deploy racing the first. So today's message steers a timed-out caller into that bug.

``DEPLOYING`` is already in this method's own documented vocabulary -- its docstring reads
``Application States: [NOT_STARTED, DEPLOYING, DEPLOY_FAILED, RUNNING, UNHEALTHY, DELETING]``
-- and it is emitted nowhere in the module. The surface already promises the distinction the
caller needs; the code never makes it.

WHAT THESE ARMS DRIVE. The real ``_get_app_status`` on a real ``AppsManager``, with only the
collaborators that reach outside it stubbed, and with an ``application_info`` built to mirror
the shape the deploy path actually registers. A fake smaller than the surface reports its own
gap as the defect, which has happened repeatedly on this lane.
"""

from __future__ import annotations

import asyncio

import pytest

from bioengine.apps.manager import AppsManager

APP = "an-app-being-built"
NOT_DEPLOYED_PHRASE = "has not been deployed yet"


def _application_info(*, is_deployed: bool, deployment_task) -> dict:
    """Mirror of the dict the deploy path registers in ``_deployed_applications``.

    Keyed off the real registration site so the arms exercise the real reads. Every key
    ``_get_app_status`` touches is present; the values are inert.
    """
    ev = asyncio.Event()
    if is_deployed:
        ev.set()
    return {
        "display_name": "An App",
        "description": "an application used by these arms",
        "artifact_id": "ws/an-app",
        "version": "1.0.0",
        "source_signature": None,
        "application_kwargs": {},
        "application_env_vars": {},
        "redacted_env_var_keys": {},
        "hypha_token": None,
        "disable_gpu": False,
        "max_ongoing_requests": 10,
        "proxy_memory_in_gb": 1,
        "scaling": {},
        "application_resources": {
            "num_cpus": 1,
            "num_gpus": 0.0,
            "memory": 1,
            "VRAM_MB": 0,
        },
        "resource_shortfall": None,
        "authorized_users": {"*": ["*"]},
        "available_methods": [],
        "static_site_url": None,
        "started_at": 0.0,
        "last_updated_at": 0.0,
        "last_updated_by": "tester",
        "deployed_by_worker_client_id": "worker-1",
        "proxy_service_token_issued_at": None,
        "proxy_service_token_ttl_seconds": None,
        "entry_deployment_name": "EntryDeployment",
        "built_app": None,
        "auto_redeploy": True,
        "ice_servers": None,
        "debug": False,
        "deployment_task": deployment_task,
        "is_deployed": ev,
        "undeployment_task": None,
        "recovered_app": False,
    }


class _Stub:
    """A real-enough AppsManager for ``_get_app_status``.

    Only the three collaborators that reach outside the method are replaced. Everything the
    method computes, it computes for real.
    """

    def __init__(self, info):
        self._deployed_applications = {APP: info} if info is not None else {}
        self.logger = _Logger()

    async def _get_deployment_status(self, **kwargs):
        return {}

    async def _get_application_service_ids(self, application_id):
        return {}, False

    async def _reported_token_issued_at(self, application_id, stored):
        return stored

    async def _get_startup_version_pin(self, *a, **k):
        return None

    def _filter_secret_env_vars(self, env):
        return dict(env or {})


class _Logger:
    def __init__(self):
        self.lines = []

    def _rec(self, msg):
        self.lines.append(str(msg))

    info = warning = error = debug = _rec


async def _status(*, is_deployed: bool, deployment_task, tracked: bool = True):
    info = (
        _application_info(is_deployed=is_deployed, deployment_task=deployment_task)
        if tracked
        else None
    )
    mgr = _Stub(info)
    fn = getattr(AppsManager, "_get_app_status", None)
    if fn is None:
        pytest.fail(
            "AppsManager._get_app_status does not exist, so nothing reports app status"
        )
    return await fn(
        mgr,
        application_id=APP,
        instance_details={"applications": {}},
        n_previous_replica=0,
        logs_tail=0,
    )


async def _a_live_task():
    """A task that is pending until the test lets it go -- a build still in flight."""
    gate = asyncio.Event()

    async def _blocked():
        await gate.wait()

    t = asyncio.create_task(_blocked())
    await asyncio.sleep(0)  # let it start
    return t, gate


async def _a_finished_task():
    async def _done():
        return None

    t = asyncio.create_task(_done())
    await t
    return t


@pytest.mark.asyncio
async def test_a_build_in_flight_is_reported_as_DEPLOYING() -> None:
    """A1, RED today: reports NOT_STARTED while a build task is running."""
    task, gate = await _a_live_task()
    try:
        st = await _status(is_deployed=False, deployment_task=task)
        assert st.get("status") == "DEPLOYING", (
            f"a deploy is in flight but status reads {st.get('status')!r}; a caller whose "
            f"deploy_app RPC timed out cannot tell a running build from one that never started"
        )
    finally:
        gate.set()
        await task


@pytest.mark.asyncio
async def test_the_message_does_not_invite_a_retry_while_a_build_is_in_flight() -> None:
    """A2, RED today, and separate from A1 on purpose.

    Getting the status right while leaving a message that says "has not been deployed yet"
    would fix the label and not the harm: that message invites a second deploy, and a second
    deploy racing the first is the race that had to be fixed separately.
    """
    task, gate = await _a_live_task()
    try:
        st = await _status(is_deployed=False, deployment_task=task)
        msg = str(st.get("message", ""))
        assert NOT_DEPLOYED_PHRASE not in msg, (
            f"the message still says {NOT_DEPLOYED_PHRASE!r} while a build is in flight, "
            f"which invites the retry that races the running deploy: {msg!r}"
        )
        assert msg.strip(), "a deploy in flight reported no message at all"
    finally:
        gate.set()
        await task


@pytest.mark.asyncio
async def test_a_finished_deployment_task_is_not_reported_as_DEPLOYING() -> None:
    """A3, green today. Guards the fix that tests the KEY rather than the task's state."""
    task = await _a_finished_task()
    st = await _status(is_deployed=False, deployment_task=task)
    assert st.get("status") != "DEPLOYING", (
        "a deployment task that has already finished is being reported as DEPLOYING, so the "
        "check is on the key's presence rather than on whether anything is running"
    )


@pytest.mark.asyncio
async def test_marked_deployed_but_absent_from_serve_is_still_UNHEALTHY() -> None:
    """A4, green today. The neighbour the DEPLOYING case must not swallow."""
    st = await _status(is_deployed=True, deployment_task=None)
    assert st.get("status") == "UNHEALTHY", (
        f"an application marked deployed but absent from Serve must stay UNHEALTHY, got "
        f"{st.get('status')!r}"
    )


@pytest.mark.asyncio
async def test_nothing_running_and_never_deployed_is_still_NOT_STARTED() -> None:
    """A5, green today. The other neighbour."""
    st = await _status(is_deployed=False, deployment_task=None)
    assert st.get("status") == "NOT_STARTED", (
        f"an application with no deployment task and no completion must stay NOT_STARTED, "
        f"got {st.get('status')!r}"
    )
    assert NOT_DEPLOYED_PHRASE in str(
        st.get("message", "")
    ), "the NOT_STARTED message changed; this arm exists so the honest case keeps its wording"


@pytest.mark.asyncio
async def test_an_untracked_application_is_still_NOT_RUNNING() -> None:
    """A6, green today. The early return above the branch being changed must stay untouched."""
    st = await _status(is_deployed=False, deployment_task=None, tracked=False)
    assert st.get("status") == "NOT_RUNNING", (
        f"an application this worker does not track must still report NOT_RUNNING, got "
        f"{st.get('status')!r}"
    )


def test_DEPLOYING_is_in_the_documented_vocabulary() -> None:
    """The value is not invented here: the method's own docstring already lists it.

    If someone removes it from the docstring, the status this change emits would no longer be
    documented, and a caller switching on `status` would have no way to know it exists.
    """
    doc = (getattr(AppsManager, "_get_app_status", None).__doc__) or ""
    assert "DEPLOYING" in doc, (
        "the method's docstring no longer lists DEPLOYING among its application states, so "
        "the value this reports is undocumented"
    )
