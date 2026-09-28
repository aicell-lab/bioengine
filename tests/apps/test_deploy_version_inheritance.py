"""Pin that a stale inherited version stops being silent.

``deploy_app`` with an ``application_id`` and no ``version`` inherits the
version already running (``manager.py``, the ``is_update`` branch). That rule is
deliberate and documented, but nothing ever flagged it as a *mistake*: a caller
who had just uploaded newer code got a deploy that reported success while
redeploying the code it had replaced, and every status field agreed. Reported
twice independently — over the API (svamp #0023) and from the CLI
(aicell-lab/bioengine#157).

Two pins here:

- The worker *warns* when the inherited version is not the artifact's newest —
  the case that is almost always a mistake. Pinning to an older version on
  purpose stays legal, it just announces itself. (The resolved version itself
  was never actually silent: the update branch logs it at INFO regardless.)
- ``bioengine apps deploy`` passes the version it just uploaded, so pointing it
  at a running ``--app-id`` actually rolls that app forward instead of
  redeploying what was already there.
- ``bioengine apps run``, which has no version to pin, reports the inheritance
  when it happens instead of printing an id and nothing else.

What ``deploy_app`` itself reports back is pinned in
``test_deploy_app_return.py``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager

ARTIFACT_ID = "bioimage-io/nuclei-seg"
APP_ID = "nuclei-seg"


def _make_manager(*, versions) -> AppsManager:
    """An AppsManager wired with only what the version report touches."""
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.deploy")

    artifact_manager = MagicMock()
    if versions is None:
        artifact_manager.read = AsyncMock(side_effect=RuntimeError("artifact gone"))
    else:
        artifact_manager.read = AsyncMock(return_value={"versions": versions})
    manager.artifact_manager = artifact_manager
    return manager


def _versions(*pairs):
    return [{"version": v, "created_at": t} for v, t in pairs]


@pytest.mark.asyncio
async def test_a_stale_inherited_version_warns_and_names_both(caplog) -> None:
    # rich-mole's case verbatim: 1.0.1 was uploaded, 1.0.0 is what runs.
    manager = _make_manager(versions=_versions(("1.0.0", 1), ("1.0.1", 2)))

    with caplog.at_level(logging.WARNING, logger="test.deploy"):
        await manager._warn_if_inherited_version_is_stale(APP_ID, ARTIFACT_ID, "1.0.0")

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    # Both numbers must be present — one of them alone is not actionable.
    assert "'1.0.0'" in warnings[0]
    assert "'1.0.1'" in warnings[0]
    assert "version='1.0.1'" in warnings[0], (
        "The warning has to name the way out, or it just restates the symptom."
    )


@pytest.mark.asyncio
async def test_inheriting_the_newest_version_does_not_warn(caplog) -> None:
    # Redeploying the running version *is* the request when nothing newer
    # exists; warning here would train people to ignore the warning.
    manager = _make_manager(versions=_versions(("1.0.0", 1), ("1.0.1", 2)))

    with caplog.at_level(logging.INFO, logger="test.deploy"):
        await manager._warn_if_inherited_version_is_stale(APP_ID, ARTIFACT_ID, "1.0.1")

    assert caplog.records == []


@pytest.mark.asyncio
async def test_an_unpinned_running_app_is_not_reported_as_stale(caplog) -> None:
    # If the running app carries no version it was deployed as "latest", so the
    # update resolves latest again and genuinely does roll forward. Warning
    # would be false.
    manager = _make_manager(versions=_versions(("1.0.0", 1), ("1.0.1", 2)))

    with caplog.at_level(logging.INFO, logger="test.deploy"):
        await manager._warn_if_inherited_version_is_stale(APP_ID, ARTIFACT_ID, None)

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    manager.artifact_manager.read.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_artifact_never_blocks_the_deploy(caplog) -> None:
    # The report is observability. If the version lookup fails it must stay
    # quiet and let the deploy proceed, not raise into the caller.
    manager = _make_manager(versions=None)

    with caplog.at_level(logging.INFO, logger="test.deploy"):
        await manager._warn_if_inherited_version_is_stale(APP_ID, ARTIFACT_ID, "1.0.0")

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.asyncio
async def test_no_committed_versions_is_not_a_mismatch(caplog) -> None:
    manager = _make_manager(versions=[])

    with caplog.at_level(logging.INFO, logger="test.deploy"):
        await manager._warn_if_inherited_version_is_stale(APP_ID, ARTIFACT_ID, "1.0.0")

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.asyncio
async def test_a_version_entry_without_created_at_never_blocks_the_deploy() -> None:
    # "Never blocks the deploy" has to cover the ranking too, not just the read
    # — a malformed entry raising out of max() would fail the whole deploy.
    manager = _make_manager(versions=[{"version": "1.0.0"}])

    assert await manager._get_latest_artifact_version(ARTIFACT_ID) is None


# ── The update path has to actually call it ───────────────────────────────────


class _StopBeforeRedeploy(Exception):
    """Sentinel: deploy_app got past the version report, stop it going further."""


def _make_deploy_manager(*, versions, running_version: str) -> AppsManager:
    """An AppsManager that can be driven through deploy_app's update branch.

    Everything the branch touches before the report is real; the first step
    *after* it raises, so the test observes the real call site rather than a
    string in the source.
    """
    manager = _make_manager(versions=versions)
    manager.server = MagicMock()
    manager.admin_users = ["*"]
    manager._deployment_lock = asyncio.Lock()
    manager._check_initialized = lambda: None
    manager.ray_cluster = MagicMock()
    manager.ray_cluster.check_connection = AsyncMock()
    manager._deployed_applications = {
        APP_ID: {
            "started_at": 0.0,
            "version": running_version,
            "artifact_id": ARTIFACT_ID,
            "application_kwargs": {},
            "application_env_vars": {},
            "hypha_token": "tok",
            "disable_gpu": False,
            "max_ongoing_requests": 10,
            "proxy_memory_in_gb": 0.5,
            "auto_redeploy": False,
            "debug": False,
            "scaling": {},
        }
    }

    async def _stop(**_kwargs):
        raise _StopBeforeRedeploy

    manager._cancel_deployment_process = _stop
    return manager


CONTEXT = {"user": {"id": "u-1", "email": "u@lab.test"}}


@pytest.mark.asyncio
async def test_deploy_app_warns_when_the_update_inherits_a_stale_version(
    caplog,
) -> None:
    manager = _make_deploy_manager(
        versions=_versions(("1.0.0", 1), ("1.0.1", 2)), running_version="1.0.0"
    )

    with caplog.at_level(logging.WARNING, logger="test.deploy"):
        with pytest.raises(_StopBeforeRedeploy):
            await manager.deploy_app(
                artifact_id=ARTIFACT_ID, application_id=APP_ID, context=CONTEXT
            )

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    # The resolved version, not the argument that was omitted.
    assert "'1.0.0'" in warnings[0]
    assert "version='1.0.1'" in warnings[0]


@pytest.mark.asyncio
async def test_deploy_app_stays_quiet_when_the_caller_named_the_version(
    caplog,
) -> None:
    # An explicit version is the caller's decision, stale or not. Warning here
    # would fire on every deliberate pin.
    manager = _make_deploy_manager(
        versions=_versions(("1.0.0", 1), ("1.0.1", 2)), running_version="1.0.0"
    )

    with caplog.at_level(logging.WARNING, logger="test.deploy"):
        with pytest.raises(_StopBeforeRedeploy):
            await manager.deploy_app(
                artifact_id=ARTIFACT_ID,
                application_id=APP_ID,
                version="1.0.0",
                context=CONTEXT,
            )

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    manager.artifact_manager.read.assert_not_awaited()


# ── CLI: `bioengine apps deploy` must deploy what it just uploaded ────────────

def _manifest(version: str | None) -> str:
    # `version` is an optional manifest field, so a version-less manifest is a
    # real input, not a malformed one.
    version_line = f"version: {version}\n" if version is not None else ""
    return (
        "format_version: 0.6.0\n"
        "name: Nuclei Seg\n"
        "id: nuclei-seg\n"
        'id_emoji: "🔬"\n'
        "description: Segment nuclei.\n"
        "type: ray-serve\n"
        f"{version_line}"
        "entry: nuclei_seg.deployment:NucleiSeg\n"
    )


MANIFEST = _manifest("1.0.1")


def _invoke_deploy(
    monkeypatch,
    tmp_path: Path,
    *,
    manifest: str = MANIFEST,
    deployed_version=None,
    version_source: str = "requested",
):
    """Run ``apps deploy`` against a stub worker; return (kwargs, click result).

    ``deployed_version`` defaults to echoing back whatever version the command
    passed — the honest stub for the pinned path. Pass it explicitly to model a
    worker that deployed something *other* than what the manifest named.
    """
    from click.testing import CliRunner

    from bioengine.cli import apps as apps_cli

    app_dir = tmp_path / "nuclei-seg"
    app_dir.mkdir()
    (app_dir / "manifest.yaml").write_text(manifest)
    (app_dir / "deployment.py").write_text("class NucleiSeg:\n    pass\n")

    recorded: dict = {}

    worker = MagicMock()
    worker.upload_app = AsyncMock(return_value=ARTIFACT_ID)

    async def _deploy_app(**kwargs):
        recorded.update(kwargs)
        return {
            "application_id": APP_ID,
            "artifact_id": ARTIFACT_ID,
            "version": (
                kwargs["version"] if deployed_version is None else deployed_version
            ),
            "version_source": version_source,
        }

    worker.deploy_app = _deploy_app

    monkeypatch.setattr(
        apps_cli, "require_worker", lambda *a: ("https://hypha.test", "ws/w", "tok")
    )
    monkeypatch.setattr(apps_cli, "connect_worker", AsyncMock(return_value=worker))

    result = CliRunner().invoke(
        apps_cli.apps_group, ["deploy", str(app_dir), "--app-id", APP_ID]
    )
    return recorded, result


@pytest.fixture
def deploy_cli(monkeypatch, tmp_path: Path):
    recorded, result = _invoke_deploy(monkeypatch, tmp_path)
    assert result.exit_code == 0, result.output
    return recorded, result.output


def test_apps_deploy_pins_the_version_it_uploaded(deploy_cli) -> None:
    # Without this the command uploads 1.0.1, targets a running app, inherits
    # that app's 1.0.0 and reports success having deployed nothing new.
    recorded, _ = deploy_cli
    assert recorded["version"] == "1.0.1"
    assert recorded["application_id"] == APP_ID


def test_apps_deploy_tells_the_user_which_version(deploy_cli) -> None:
    _, output = deploy_cli
    assert "1.0.1" in output


def test_apps_deploy_fails_when_the_deploy_inherited_a_version(
    monkeypatch, tmp_path: Path
) -> None:
    # A version-less manifest passes version=None, so targeting a running app
    # inherits its version. This command's whole purpose is to roll forward, so
    # that is an error — and the pre-existing failure mode is the nastiest kind:
    # it reported the inherited version as if it were the uploaded one.
    _, result = _invoke_deploy(
        monkeypatch,
        tmp_path,
        manifest=_manifest(None),
        deployed_version="0.0.9",
        version_source="inherited",
    )

    assert result.exit_code != 0, result.output
    assert "0.0.9" in result.output
    assert "NOT deployed" in result.output, (
        "The whole point is that the upload succeeded and the deploy did not "
        "ship it — saying only 'inherited' buries that."
    )
    assert "in flight" in result.output, (
        "deploy_app has already started the redeploy before returning, so "
        "exiting here does not stop it."
    )
    assert "manifest.yaml" in result.output


def test_apps_deploy_reports_the_workers_version_not_the_manifests(
    monkeypatch, tmp_path: Path
) -> None:
    # A version-less manifest on a *fresh* application_id resolves to the
    # artifact's newest version: nothing is inherited, so this succeeds — and it
    # is the one success path where the deployed version is not the string the
    # manifest carried. Echoing manifest_version here printed "(version None)".
    _, result = _invoke_deploy(
        monkeypatch,
        tmp_path,
        manifest=_manifest(None),
        deployed_version="1.0.2",
        version_source="latest",
    )

    assert result.exit_code == 0, result.output
    assert "Application ID: nuclei-seg (version 1.0.2)" in result.output
    assert "(version None)" not in result.output


# ── CLI: `bioengine apps run` must say when it inherited a version ────────────


def _run_cli(monkeypatch, version_source: str):
    """Run ``apps run`` against a stub worker reporting ``version_source``."""
    from click.testing import CliRunner

    from bioengine.cli import apps as apps_cli

    worker = MagicMock()
    worker.deploy_app = AsyncMock(
        return_value={
            "application_id": APP_ID,
            "artifact_id": ARTIFACT_ID,
            "version": "1.0.0",
            "version_source": version_source,
        }
    )

    monkeypatch.setattr(
        apps_cli, "require_worker", lambda *a: ("https://hypha.test", "ws/w", "tok")
    )
    monkeypatch.setattr(apps_cli, "connect_worker", AsyncMock(return_value=worker))

    result = CliRunner().invoke(
        apps_cli.apps_group, ["run", ARTIFACT_ID, "--app-id", APP_ID]
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_apps_run_reports_an_inherited_version(monkeypatch) -> None:
    # `apps run --app-id <running>` with no --version redeploys what is already
    # there. The command has to say so; otherwise it looks like a roll-forward.
    output = _run_cli(monkeypatch, "inherited")

    assert "1.0.0" in output
    assert "redeployed" in output
    assert "--version" in output, (
        "Naming the way out is the point — without it the note just restates "
        "that something happened."
    )


def test_apps_run_stays_quiet_when_the_version_was_not_inherited(monkeypatch) -> None:
    # A note on every deploy is a note nobody reads. Asserted on the note's own
    # wording, not on the word "inherited" — which the note never uses, so that
    # assertion could never have failed.
    output = _run_cli(monkeypatch, "latest")

    assert "1.0.0" in output
    assert "redeployed" not in output
    assert "--version" not in output
