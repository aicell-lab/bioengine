"""``bioengine apps logs`` has to print what the replicas printed.

It never did. The command read ``logs`` off each entry of the projected replica
list, and that list carries seven keys — none of them ``logs``. The log text
lives one level up, at ``deployments.<name>.logs``, keyed by replica id. The
wrong lookup returned ``None`` every time and the command exited having printed
nothing, which reads exactly like a deployment that has not logged yet.

So the assertions here are on the log *text* reaching stdout. A test that only
checked "did not crash", or "printed something", would have passed against the
broken command.

The payload is built by the real producers rather than typed out to match the
reader: the deployment half comes from ``AppsManager._get_deployment_status``
driven with a proxy stub returning the shape ``proxy_actor.get_deployment_logs``
documents, and the unauthorized variant is the real ``public_app_status``
projection of that same payload. A fixture invented here could agree with the
CLI and disagree with the worker.

Three states have to stay distinguishable in the output, because collapsing
them is how the defect survived: logs present, no logs yet, and logs withheld
from a caller who is not on the application's roster.

One case is pinned here specifically to stop a plausible tidy-up: the log map
keys recently dead replicas too, and the deployment's ``replicas`` list does
not, so iterating that list instead would drop the output of exactly the
crashed replica an operator came looking for.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

# Imported by full path: the ``bioengine`` package resolves its authoring
# symbols through a PEP 562 ``__getattr__``, so a submodule has to be named.
import bioengine.cli.apps as apps_cli
from bioengine.apps.manager import AppsManager, public_app_status

APP_ID = "cellpose-finetuning"
DEPLOYMENT = "CellposeApp"
REPLICA_ID = "CellposeApp#hgTdrR"
WORKER = "example-workspace/bioengine-worker"
TOKEN = "FAKE-TOKEN-NOT-A-CREDENTIAL"

STDOUT_LINE = "PLANTED-STDOUT-LINE"
STDERR_LINE = "PLANTED-STDERR-LINE"


def _serve_details(replicas: list) -> dict:
    """The Ray Serve controller's dump of one application, as the manager reads it."""
    return {
        "status": "RUNNING",
        "message": "",
        "deployments": {
            DEPLOYMENT: {
                "status": "HEALTHY",
                "message": "",
                "replicas": replicas,
            }
        },
    }


LIVE_REPLICA = {
    "replica_id": REPLICA_ID,
    "node_id": "node-1",
    "node_ip": "10.0.0.1",
    "node_instance_id": "pod-1",
    "state": "RUNNING",
    "pid": 4321,
    "start_time_s": 1_700_000_000.0,
    "log_file_path": "/tmp/replica.log",
    "actor_id": "actor-1",
}

# What ``proxy_actor.get_deployment_logs`` returns: one entry per replica id,
# each carrying its stdout and stderr already split into lines.
PROXY_LOGS = {
    REPLICA_ID: {
        "creation_timestamp": 1_700_000_000.0,
        "timezone": "Europe/Stockholm",
        "stdout": ["booting…", STDOUT_LINE],
        "stderr": [STDERR_LINE],
    }
}

EMPTY_PROXY_LOGS: dict = {}

# A replica that has already died. ``get_deployment_logs`` merges the n most
# recent dead replicas into the same map, but the deployment's ``replicas``
# list is the Serve controller's *live* set and never mentions it — which is
# why the log map, not that list, is what the CLI iterates.
DEAD_REPLICA_ID = "CellposeApp#0zXk9q"
CRASH_LINE = "PLANTED-CRASHED-REPLICA-LINE"
PROXY_LOGS_WITH_DEAD_REPLICA = {
    **PROXY_LOGS,
    DEAD_REPLICA_ID: {
        "creation_timestamp": 1_699_999_000.0,
        "timezone": "Europe/Stockholm",
        "stdout": ["booting…"],
        "stderr": [CRASH_LINE],
    },
}

# Twenty-five lines, as a worker asked for a tail of thirty would return them.
LONG_STDOUT = [f"L{n}" for n in range(25)]
LONG_PROXY_LOGS = {
    REPLICA_ID: {"stdout": LONG_STDOUT, "stderr": []},
}


def _deployments(proxy_logs: dict, replicas: list) -> dict:
    """``deployments`` as the worker really builds it, for the given proxy logs."""
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test")
    ray_cluster = MagicMock()
    ray_cluster.proxy_actor_handle.get_deployment_logs.remote = AsyncMock(
        return_value=proxy_logs
    )
    manager.ray_cluster = ray_cluster
    return asyncio.run(
        manager._get_deployment_status(
            application_id=APP_ID,
            application_details=_serve_details(replicas),
            n_previous_replica=0,
            logs_tail=100,
        )
    )


def _status(proxy_logs: dict = PROXY_LOGS, replicas: list = None) -> dict:
    """An entitled caller's ``get_app_status`` entry for one running app."""
    return {
        "display_name": "Cellpose Finetuning",
        "artifact_id": "bioimage-io/cellpose-finetuning",
        "version": "1.0.1",
        "status": "RUNNING",
        "message": "",
        "deployments": _deployments(
            proxy_logs, [LIVE_REPLICA] if replicas is None else replicas
        ),
        "authorized_users": {"*": ["member@example.invalid"]},
        "service_ids": {},
    }


@pytest.fixture
def invoke(monkeypatch):
    def _invoke(command: list, status: dict):
        worker = MagicMock()
        worker.get_app_status = AsyncMock(return_value={APP_ID: status})
        monkeypatch.setattr(
            apps_cli, "connect_worker", AsyncMock(return_value=worker)
        )
        result = CliRunner().invoke(
            apps_cli.apps_group,
            [*command, "--worker", WORKER, "--token", TOKEN],
            env={"HYPHA_TOKEN": "", "BIOENGINE_TOKEN": "", "BIOENGINE_SERVER_URL": ""},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        _invoke.worker = worker
        return result.output

    return _invoke


# ── The payload the CLI has to read from ──────────────────────────────────────


def test_the_projected_replica_entries_carry_no_logs_key():
    """The premise of the defect, pinned against the real projection: reading
    ``logs`` off a replica entry can only ever return nothing."""
    deployments = _deployments(PROXY_LOGS, [LIVE_REPLICA])
    for replica in deployments[DEPLOYMENT]["replicas"]:
        assert "logs" not in replica
    assert deployments[DEPLOYMENT]["logs"] == PROXY_LOGS


# ── State 1: entitled, logs present ───────────────────────────────────────────


def test_apps_logs_prints_the_log_text(invoke):
    output = invoke(["logs", APP_ID], _status())
    assert STDOUT_LINE in output
    assert STDERR_LINE in output
    assert REPLICA_ID in output


def test_apps_status_prints_the_log_text(invoke):
    """The sibling reader had the same wrong-level lookup, one level higher:
    it read ``logs`` off the application, where the field never lives either."""
    output = invoke(["status", APP_ID], _status())
    assert STDOUT_LINE in output


def test_a_dead_replicas_logs_are_printed_though_it_is_not_in_the_replica_list(invoke):
    """The whole reason the log map is what gets iterated.

    ``get_deployment_logs`` merges the most recent dead replicas into the map it
    returns; the deployment's ``replicas`` list is the live set and contains
    none of them. Iterating that list instead would look tidier and would drop
    exactly the output of the crashed replica an operator is looking for — this
    defect returning by another route.
    """
    status = _status(proxy_logs=PROXY_LOGS_WITH_DEAD_REPLICA)
    live_ids = [r["replica_id"] for r in status["deployments"][DEPLOYMENT]["replicas"]]
    assert DEAD_REPLICA_ID not in live_ids
    assert DEAD_REPLICA_ID in status["deployments"][DEPLOYMENT]["logs"]

    output = invoke(["logs", APP_ID], status)
    assert CRASH_LINE in output
    assert DEAD_REPLICA_ID in output


# ── The line count the caller asked for is the line count printed ─────────────


def test_both_commands_print_every_line_the_worker_returned(invoke):
    """``--tail``/``--logs`` is applied worker-side, per replica and per stream,
    so whatever comes back is already the requested tail. A second cap here
    would silently show fewer lines than the number the user typed."""
    status = _status(proxy_logs=LONG_PROXY_LOGS)

    logs_output = invoke(["logs", APP_ID, "--tail", "30"], status)
    status_output = invoke(["status", APP_ID, "--logs", "30"], status)

    for line in LONG_STDOUT:
        assert f"  {line}\n" in logs_output
        assert f"  {line}\n" in status_output


def test_the_requested_line_count_reaches_the_worker(invoke):
    """The other half of the same promise: the flag has to be forwarded, not
    just respected on the way out."""
    invoke(["logs", APP_ID, "--tail", "7"], _status())
    assert invoke.worker.get_app_status.await_args.kwargs["logs_tail"] == 7

    invoke(["status", APP_ID, "--logs", "7"], _status())
    assert invoke.worker.get_app_status.await_args.kwargs["logs_tail"] == 7


# ── State 2: entitled, nothing logged yet ─────────────────────────────────────


def test_a_deployment_with_no_logs_says_so(invoke):
    output = invoke(["logs", APP_ID], _status(proxy_logs=EMPTY_PROXY_LOGS))
    assert STDOUT_LINE not in output
    assert "No logs recorded yet" in output


# ── State 3: not entitled, the field is withheld ──────────────────────────────


def test_a_caller_off_the_roster_is_told_the_logs_were_withheld(invoke):
    """``public_app_status`` drops ``logs`` from the deployment rather than
    emptying it, so the CLI can tell "withheld" from "nothing logged yet" —
    and has to, or an unauthorized caller sees the same silence as the bug."""
    withheld = public_app_status(_status())
    assert "logs" not in withheld["deployments"][DEPLOYMENT]

    output = invoke(["logs", APP_ID], withheld)
    assert "withheld" in output.lower()
    assert "No logs recorded yet" not in output


def test_the_three_states_print_three_different_things(invoke):
    present = invoke(["logs", APP_ID], _status())
    empty = invoke(["logs", APP_ID], _status(proxy_logs=EMPTY_PROXY_LOGS))
    withheld = invoke(["logs", APP_ID], public_app_status(_status()))
    assert len({present, empty, withheld}) == 3


# ── The other shapes the field takes ──────────────────────────────────────────


def test_a_log_retrieval_error_is_reported_as_one(invoke):
    """The manager substitutes ``{"error": ...}`` for the replica map when the
    proxy call raises; that is not a replica id and must not print as one."""
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test")
    ray_cluster = MagicMock()
    ray_cluster.proxy_actor_handle.get_deployment_logs.remote = AsyncMock(
        side_effect=RuntimeError("gcs unreachable")
    )
    manager.ray_cluster = ray_cluster
    deployments = asyncio.run(
        manager._get_deployment_status(
            application_id=APP_ID,
            application_details=_serve_details([LIVE_REPLICA]),
            n_previous_replica=0,
            logs_tail=100,
        )
    )
    status = _status()
    status["deployments"] = deployments

    output = invoke(["logs", APP_ID], status)
    assert "Log retrieval failed" in output
    assert "gcs unreachable" in output


def test_an_app_that_is_not_running_reports_why(invoke):
    """``get_app_status`` answers a non-deployed app with status and message
    only — no ``deployments`` key at all."""
    output = invoke(
        ["logs", APP_ID],
        {
            "status": "NOT_RUNNING",
            "message": f"Application '{APP_ID}' is not currently deployed.",
        },
    )
    assert "not currently deployed" in output
