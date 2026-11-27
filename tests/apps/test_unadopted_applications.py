"""A Ray Serve app the worker declines to adopt must stay VISIBLE, not vanish.

Recovery walks ``serve.status()`` and skips anything it will not take on. Nothing on that
path deletes the Serve app, so a skipped app keeps running and keeps its GPUs while being
absent from ``_deployed_applications`` — and from ``get_app_status``. A throwaway held 2 of
one cluster's 3 GPUs that way with no BioEngine surface naming it.

Deleting the Serve app is the wrong remedy: the sibling-workspace skip is another worker's
app. Reporting it is the fix, so these assert it is REPORTED.

Every test here fails before the change: ``_unadopted_applications`` does not exist, so the
attribute lookup raises.
"""

import asyncio
import types

import pytest

from bioengine.apps.manager import AppsManager


class _FakeDeployments(dict):
    """Serve's ``application.deployments`` — a mapping keyed by deployment name."""


def _app(deployments=("ProxyDeployment",)):
    return types.SimpleNamespace(
        deployments=_FakeDeployments({d: None for d in deployments})
    )


def _manager_with(serve_apps, app_data_by_id, workspace="ws-under-test"):
    """An AppsManager wired to a fake Serve status, without constructing the real thing."""
    mgr = AppsManager.__new__(AppsManager)  # no __init__: it needs a live cluster
    mgr._deployed_applications = {}
    mgr._unadopted_applications = {}
    mgr.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    mgr.server = types.SimpleNamespace(
        config=types.SimpleNamespace(workspace=workspace)
    )

    status = types.SimpleNamespace(applications=serve_apps)

    async def _call_with_reconnect(fn, *args, **kwargs):
        return status

    mgr.ray_cluster = types.SimpleNamespace(call_with_reconnect=_call_with_reconnect)

    class _Handle:
        def __init__(self, app_id):
            self._app_id = app_id

        class _Remote:
            def __init__(self, value):
                self._value = value

            def remote(self):
                fut = asyncio.get_event_loop().create_future()
                fut.set_result(self._value)
                return fut

        @property
        def get_app_data(self):
            return self._Remote(app_data_by_id.get(self._app_id, {}))

    import bioengine.apps.manager as mod

    return mgr, mod, _Handle


@pytest.mark.asyncio
async def test_app_without_proxy_deployment_is_reported_not_silently_dropped(
    monkeypatch,
):
    """The quietest path: no ProxyDeployment logged NOTHING at all."""
    mgr, mod, _Handle = _manager_with({"stray-app": _app(deployments=("Runtime",))}, {})
    await mgr.recover_deployed_applications()

    assert (
        "stray-app" not in mgr._deployed_applications
    ), "precondition: the app must not be adopted, or this test proves nothing"
    assert "stray-app" in mgr._unadopted_applications, (
        "a Serve app the worker skipped must be reported, not dropped in silence — "
        "it keeps running and keeps its GPUs"
    )
    assert "ProxyDeployment" in mgr._unadopted_applications["stray-app"]


@pytest.mark.asyncio
async def test_sibling_workspace_app_is_reported(monkeypatch):
    """The second quiet path: a sibling workspace's app logged only at DEBUG."""
    mgr, mod, _Handle = _manager_with(
        {"siblings-app": _app()},
        {
            "siblings-app": {
                "format_version": "0.6.0",
                "worker_workspace": "some-other-ws",
            }
        },
    )
    monkeypatch.setattr(
        mod.serve, "get_app_handle", lambda app_id: _Handle(app_id), raising=False
    )

    await mgr.recover_deployed_applications()

    assert "siblings-app" not in mgr._deployed_applications
    assert "siblings-app" in mgr._unadopted_applications, (
        "an app belonging to a sibling workspace is legitimately not ours to delete, "
        "which is exactly why it must be visible instead"
    )


@pytest.mark.asyncio
async def test_an_adopted_app_is_not_reported_as_unadopted(monkeypatch):
    """The discriminating case: this must NOT fire for an app the worker owns.

    Without it, a check that reports every app would pass the two tests above while
    saying nothing true.
    """
    mgr, mod, _Handle = _manager_with({"mine": _app()}, {})
    mgr._deployed_applications["mine"] = {"display_name": "mine"}

    await mgr.recover_deployed_applications()

    assert (
        "mine" not in mgr._unadopted_applications
    ), "an already-adopted app is not an orphan"


@pytest.mark.asyncio
async def test_the_report_is_rebuilt_each_pass(monkeypatch):
    """An app that became adopted (or disappeared) must stop being reported."""
    mgr, mod, _Handle = _manager_with({"stray-app": _app(deployments=("Runtime",))}, {})
    await mgr.recover_deployed_applications()
    assert "stray-app" in mgr._unadopted_applications

    # it is adopted by the time of the next pass
    mgr._deployed_applications["stray-app"] = {"display_name": "stray"}
    await mgr.recover_deployed_applications()

    assert (
        "stray-app" not in mgr._unadopted_applications
    ), "the report is a snapshot of the current pass, not an append-only log"


def test_the_report_is_withheld_from_the_anonymous_status_view():
    """Declaring a field 'knowingly withheld' in a test literal is not a guarantee.

    `unadopted_applications` names Ray Serve applications, and on a shared Ray cluster
    that includes a SIBLING workspace's apps. `get_status` answers unauthenticated
    probes, so the anonymous projection must actually drop it — asserted here against
    the real projection rather than trusted because the field was left off a list.
    """
    from bioengine.worker.worker import (
        PUBLIC_WORKER_STATUS_FIELDS,
        public_worker_status,
    )

    full = {
        "is_ready": True,
        "admin_users": ["admin@example.invalid"],
        "unadopted_applications": {"someone-elses-app": "belongs to a sibling worker"},
    }
    public = public_worker_status(full)

    assert "unadopted_applications" not in public, (
        "the anonymous health surface must not enumerate applications, including "
        "another workspace's"
    )
    assert "unadopted_applications" not in PUBLIC_WORKER_STATUS_FIELDS
    # positive control: the projection does let the public field through, so the
    # absence above is the projection working and not an empty result.
    assert public.get("is_ready") is True
