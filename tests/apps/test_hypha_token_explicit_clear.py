"""Pin that ``--hypha-token ''`` really deploys without a token.

Both CLI deploy commands advertise an empty ``--hypha-token`` as the way to run
an app with no credential. It only ever did that on a *fresh* application. On a
redeploy the CLI collapsed the empty string to ``None`` before the RPC, and
``deploy_app``'s update branch reads ``None`` as "caller said nothing" and
re-injects the token the instance was already holding — the opposite of what
was asked for. A third state hid it further: an app the worker *adopted* stores
no token, so there is nothing to inherit and the flag looks correct again.

The empty string is now carried through to ``deploy_app`` intact, which makes
the two intentions distinguishable at the one place that has to tell them
apart. An omitted argument and an explicit ``None`` are the same thing once the
call crosses RPC, so a value is the only thing that survives the trip.

Everything here is driven from the CLI, because that is the surface the claim
is made on, and through the real ``deploy_app`` update branch and the real
``_build_env_vars``, because "no token in the replica environment" is a fact
about the end of that chain rather than about any one link.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine._app.mixin import _unmask_secret_env_vars
from bioengine.apps.builder import AppBuilder
from bioengine.apps.manager import AppsManager

ARTIFACT_ID = "bioimage-io/nuclei-seg"
APP_ID = "nuclei-seg"
STORED_TOKEN = "stored-placeholder-token"
AUTH_TOKEN = "cli-auth-placeholder-token"
CONTEXT = {"user": {"id": "u-1", "email": "u@lab.test"}}


class _StopAfterBuild(Exception):
    """Sentinel: the env vars are composed, nothing past this point is wired."""


def _env_var_builder() -> AppBuilder:
    """A real AppBuilder carrying only what ``_build_env_vars`` reads."""
    builder = object.__new__(AppBuilder)
    builder.server = MagicMock()
    builder.server.config.workspace = "bioimage-io"
    builder.server_url = "https://hypha.test"
    builder.apps_workdir = Path("/tmp/bioengine-apps")
    builder.worker_service_id = "bioimage-io/worker"
    builder.proxy_actor_name = None
    builder.data_server_url = None
    return builder


def _running_app(*, hypha_token: Optional[str]) -> Dict[str, Any]:
    return {
        "started_at": 0.0,
        "version": "1.0.0",
        "artifact_id": ARTIFACT_ID,
        "application_kwargs": {},
        "application_env_vars": {},
        "redacted_env_var_keys": {},
        "hypha_token": hypha_token,
        "disable_gpu": False,
        "max_ongoing_requests": 10,
        "proxy_memory_in_gb": 0.5,
        "auto_redeploy": False,
        "debug": False,
        "scaling": {},
    }


# The three states the flag has to mean the same thing in. "running" is the one
# that was broken; "adopted" is the one that was accidentally right.
STATES = {
    "fresh": None,
    "running": _running_app(hypha_token=STORED_TOKEN),
    "adopted": _running_app(hypha_token=None),
}


def _make_manager(state: str, recorder: dict) -> AppsManager:
    """An AppsManager driven through the real ``deploy_app`` up to the build.

    ``app_builder.build`` is where the resolved token leaves ``deploy_app``, so
    the stub composes the replica env vars with the real ``_build_env_vars``
    and stops there.
    """
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.hypha_token")
    manager.server = MagicMock()
    manager.admin_users = ["*"]
    manager._deployment_lock = asyncio.Lock()
    manager._check_initialized = lambda: None
    manager.ray_cluster = MagicMock()
    manager.ray_cluster.check_connection = AsyncMock()
    manager._cancel_deployment_process = AsyncMock()
    manager.artifact_manager = MagicMock()
    manager.artifact_manager.read = AsyncMock(return_value={"versions": []})

    existing = STATES[state]
    manager._deployed_applications = {} if existing is None else {APP_ID: dict(existing)}

    env_builder = _env_var_builder()

    async def _build(**kwargs):
        recorder["hypha_token"] = kwargs["hypha_token"]
        recorder["env_vars"] = env_builder._build_env_vars(
            kwargs["application_id"],
            kwargs["artifact_id"],
            kwargs["version"],
            {
                k: v
                for env in (kwargs["application_env_vars"] or {}).values()
                for k, v in env.items()
                if not k.startswith("_")
            },
            {
                k[1:]: v
                for env in (kwargs["application_env_vars"] or {}).values()
                for k, v in env.items()
                if k.startswith("_")
            },
            kwargs["hypha_token"],
            None,
        )
        raise _StopAfterBuild

    manager.app_builder = MagicMock()
    manager.app_builder.build = _build
    return manager


def _invoke(monkeypatch, state: str, extra_args: list) -> dict:
    """Run ``bioengine apps run`` against a worker backed by a real manager."""
    from click.testing import CliRunner

    from bioengine.cli import apps as apps_cli

    recorder: dict = {}
    manager = _make_manager(state, recorder)

    async def _deploy_app(**kwargs):
        recorder["rpc_kwargs"] = dict(kwargs)
        try:
            await manager.deploy_app(context=CONTEXT, **kwargs)
        except _StopAfterBuild:
            pass
        return {
            "application_id": APP_ID,
            "artifact_id": ARTIFACT_ID,
            "version": "1.0.0",
            "version_source": "requested",
        }

    worker = MagicMock()
    worker.deploy_app = _deploy_app

    monkeypatch.setattr(
        apps_cli,
        "require_worker",
        lambda *a: ("https://hypha.test", "bioimage-io/worker", AUTH_TOKEN),
    )
    monkeypatch.setattr(apps_cli, "connect_worker", AsyncMock(return_value=worker))

    result = CliRunner().invoke(
        apps_cli.apps_group,
        ["run", ARTIFACT_ID, "--app-id", APP_ID, "--version", "1.0.0", *extra_args],
    )
    assert result.exit_code == 0, result.output
    recorder["output"] = result.output
    return recorder


def _replica_env(env_vars: Dict[str, str], monkeypatch) -> Dict[str, str]:
    """What the replica's ``os.environ`` holds after the setup hook unmasks."""
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("HYPHA_TOKEN", raising=False)
    _unmask_secret_env_vars()
    import os

    return dict(os.environ)


# ── The empty string survives the trip to deploy_app ──────────────────────────


@pytest.mark.parametrize("state", list(STATES))
def test_an_empty_hypha_token_is_not_collapsed_to_none(monkeypatch, state) -> None:
    # The whole defect in one assertion: the CLI used to send None here, which
    # deploy_app cannot tell apart from an omitted argument.
    recorder = _invoke(monkeypatch, state, ["--hypha-token", ""])

    assert recorder["rpc_kwargs"]["hypha_token"] == ""
    assert recorder["rpc_kwargs"]["hypha_token"] is not None


# ── All three states end with no token in the replica environment ─────────────


@pytest.mark.parametrize("state", list(STATES))
def test_an_empty_hypha_token_leaves_no_token_in_the_replica(
    monkeypatch, state
) -> None:
    # The claim the flag makes, asserted at the end of the chain and nowhere
    # else. Of the three states only "running" ever broke it, so a version of
    # this test that covered "fresh" alone would have passed throughout.
    recorder = _invoke(monkeypatch, state, ["--hypha-token", ""])

    env_vars = recorder["env_vars"]
    assert "_BIOENGINE_SECRET_HYPHA_TOKEN" not in env_vars
    assert STORED_TOKEN not in env_vars.values()

    replica_env = _replica_env(env_vars, monkeypatch)
    # Absent, not present-and-empty: an app testing `"HYPHA_TOKEN" in environ`
    # or indexing it must see no token, same as one testing falsiness.
    assert "HYPHA_TOKEN" not in replica_env


@pytest.mark.parametrize("state", list(STATES))
def test_an_empty_hypha_token_reaches_the_builder_as_an_empty_token(
    monkeypatch, state
) -> None:
    # The mechanism behind the test above: the empty string is threaded all the
    # way to the builder rather than being turned into None somewhere in the
    # middle, so the inherit in deploy_app's update branch never fires.
    recorder = _invoke(monkeypatch, state, ["--hypha-token", ""])

    assert recorder["hypha_token"] == "", (
        "The update branch must not inherit over an explicitly empty token; "
        f"state {state!r} resolved to {recorder['hypha_token']!r}."
    )


# ── The two behaviours that must not change ───────────────────────────────────


@pytest.mark.parametrize("state", list(STATES))
def test_an_explicit_token_is_injected_unchanged(monkeypatch, state) -> None:
    recorder = _invoke(monkeypatch, state, ["--hypha-token", "explicit-placeholder"])

    assert recorder["hypha_token"] == "explicit-placeholder"
    replica_env = _replica_env(recorder["env_vars"], monkeypatch)
    assert replica_env["HYPHA_TOKEN"] == "explicit-placeholder"


@pytest.mark.parametrize(
    "state, expected",
    [("fresh", AUTH_TOKEN), ("running", AUTH_TOKEN), ("adopted", AUTH_TOKEN)],
)
def test_omitting_the_flag_still_defaults_to_the_auth_token(
    monkeypatch, state, expected
) -> None:
    # Omission is unchanged: the CLI substitutes --token, so deploy_app never
    # reaches the inherit for this route.
    recorder = _invoke(monkeypatch, state, [])

    assert recorder["rpc_kwargs"]["hypha_token"] == expected
    replica_env = _replica_env(recorder["env_vars"], monkeypatch)
    assert replica_env["HYPHA_TOKEN"] == expected


def test_an_omitted_token_over_rpc_still_inherits_on_a_running_app() -> None:
    # The CLI always sends something, so the inherit is reachable only from a
    # direct RPC call. It has to keep working — clearing a token on every
    # version bump that omitted the parameter would be a worse bug than this one.
    recorder: dict = {}
    manager = _make_manager("running", recorder)

    async def _go():
        with pytest.raises(_StopAfterBuild):
            await manager.deploy_app(
                artifact_id=ARTIFACT_ID,
                application_id=APP_ID,
                version="1.0.0",
                context=CONTEXT,
            )

    asyncio.run(_go())

    assert recorder["hypha_token"] == STORED_TOKEN
    assert recorder["env_vars"]["_BIOENGINE_SECRET_HYPHA_TOKEN"] == STORED_TOKEN


def test_a_cleared_token_is_what_a_later_omitted_redeploy_inherits() -> None:
    # Clearing has to stick: the state the worker stores after an empty token is
    # what the next omitted redeploy reads back, so "" must not read as "unset".
    recorder: dict = {}
    manager = _make_manager("running", recorder)
    manager._deployed_applications[APP_ID]["hypha_token"] = ""

    async def _go():
        with pytest.raises(_StopAfterBuild):
            await manager.deploy_app(
                artifact_id=ARTIFACT_ID,
                application_id=APP_ID,
                version="1.0.0",
                context=CONTEXT,
            )

    asyncio.run(_go())

    assert recorder["hypha_token"] == ""
    assert "_BIOENGINE_SECRET_HYPHA_TOKEN" not in recorder["env_vars"]


# ── The help screens have to describe what the flag now does ──────────────────


def _help(command: str) -> str:
    from click.testing import CliRunner

    from bioengine.cli import apps as apps_cli

    result = CliRunner().invoke(apps_cli.apps_group, [command, "--help"])
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


@pytest.mark.parametrize("command", ["run", "deploy"])
def test_the_help_describes_the_empty_token_truthfully(command) -> None:
    text = _help(command)

    assert "--hypha-token '' to deploy without one" in text
    assert "no token is injected" in text
    assert "clears the token it currently holds" in text
    # The sentence this replaces was true on a fresh app and false on a redeploy.
    assert "explicitly deploy without a token" not in text


@pytest.mark.parametrize("command", ["run", "deploy"])
def test_the_help_does_not_contradict_the_env_note(command) -> None:
    text = _help(command)

    # --env's note says a plain HYPHA_TOKEN is overwritten by --hypha-token.
    assert "--env HYPHA_TOKEN=... is overwritten by --hypha-token" in text
    # So "deploy without a token" is only true if the flag also says what
    # happens to a plain --env HYPHA_TOKEN — otherwise the two notes disagree
    # about the same deployment on the same screen.
    assert "A plain --env HYPHA_TOKEN=... does reach the deployment" in text


def test_both_commands_document_the_flag_identically() -> None:
    # One claim, two screens. They drifted apart before; the defect was in both
    # only because the sentence was copied.
    marker = "Pass --hypha-token ''"
    run_help, deploy_help = _help("run"), _help("deploy")
    assert run_help[run_help.index(marker) :].startswith(
        deploy_help[deploy_help.index(marker) : deploy_help.index(marker) + 120]
    )
