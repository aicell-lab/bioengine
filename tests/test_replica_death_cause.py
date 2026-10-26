"""A dead replica's cause reaches ``get_app_status``, with Ray's own wording.

A ``CellposeRuntime`` replica on deNBI was killed mid-merge. Its own logs just
stopped — last line a healthy ``CALL ping OK 1.1ms``, no traceback — and the
Serve controller recorded only "crashed, marking it unhealthy immediately", so
the operator-facing status carried no reason at all. Ray's GCS held one the
whole time, and it discriminates: the same replica class died four different
ways that night (OOM, ``ray.kill``, node death, bare worker exit) and the GCS
names each. Nothing in the worker read it.

The fixtures below are the real shapes captured from that cluster's GCS, so a
Ray upgrade that moves the fields fails here rather than silently returning
None forever. ``_MISSING`` is the discrimination control: an unmatched actor
must yield None, otherwise a helper that returned a constant would pass.
"""

from types import SimpleNamespace

from bioengine.cluster.proxy_actor import BioEngineProxyActor

_PLAIN_CLASS = BioEngineProxyActor.__ray_metadata__.modified_class

#: Captured from deNBI's GCS — the replica the incident was filed about.
_WORKER_DIED = {
    "actor_died_error_context": {
        "error_message": (
            "The actor is dead because its worker process has died. Worker "
            "exit type: SYSTEM_ERROR Worker exit detail: Worker unexpectedly "
            "exits with a connection error code 2. End of file."
        ),
        "pid": 21954,
        "node_ip_address": "172.25.0.81",
        "reason": "WORKER_DIED",
        "never_started": False,
    }
}
_RAY_KILL = {
    "actor_died_error_context": {
        "error_message": "The actor is dead because it was killed by `ray.kill`.",
        "reason": "RAY_KILL",
        "never_started": False,
    }
}
_NODE_DIED = {
    "actor_died_error_context": {
        "error_message": "The actor died because its node has died.",
        "reason": "NODE_DIED",
        "node_death_info": {
            "reason": "EXPECTED_TERMINATION",
            "reason_message": "received SIGTERM",
        },
        "never_started": False,
    }
}
#: OOM is the one shape that carries no ``reason`` of its own.
_OOM = {
    "oom_context": {
        "error_message": (
            "1 worker(s) were killed due to the node running low on memory. "
            "Memory on the node (IP: 172.25.0.179) was 28.42GB / 29.80GB"
        ),
        "fail_immediately": False,
    }
}
_MISSING = object()


def _actor(death_causes):
    """Proxy actor whose GCS lookup answers from ``{actor_id: death_cause}``."""
    actor = object.__new__(_PLAIN_CLASS)
    actor._last_called_at = 0.0

    def _list(resource, options, raise_on_missing_output):
        (_, _, actor_id), = options.filters
        cause = death_causes.get(actor_id, _MISSING)
        if cause is _MISSING:
            return []
        return [SimpleNamespace(actor_id=actor_id, death_cause=cause)]

    actor.state_api_client = SimpleNamespace(list=_list)
    return actor


def test_each_gcs_shape_maps_to_its_own_reason():
    actor = _actor(
        {
            "a_worker": _WORKER_DIED,
            "a_kill": _RAY_KILL,
            "a_node": _NODE_DIED,
            "a_oom": _OOM,
        }
    )
    assert actor._get_actor_death_cause("a_worker")["reason"] == "WORKER_DIED"
    assert actor._get_actor_death_cause("a_kill")["reason"] == "RAY_KILL"
    assert actor._get_actor_death_cause("a_node")["reason"] == "NODE_DIED"
    assert actor._get_actor_death_cause("a_oom")["reason"] == "OOM"

    # The incident's own message has to survive, not just its label — this is
    # the only text that says SYSTEM_ERROR rather than a clean exit.
    assert "SYSTEM_ERROR" in actor._get_actor_death_cause("a_worker")["message"]
    # A node death's cause is in the nested record, not the top-level message.
    assert "received SIGTERM" in actor._get_actor_death_cause("a_node")["message"]


def test_unknown_or_live_actor_yields_no_cause():
    actor = _actor({"a_worker": _WORKER_DIED})
    assert actor._get_actor_death_cause("evicted_from_gcs") is None
    assert _actor({"alive": None})._get_actor_death_cause("alive") is None


def test_a_failed_gcs_lookup_does_not_break_the_status_payload():
    actor = object.__new__(_PLAIN_CLASS)
    actor._last_called_at = 0.0

    def _raise(resource, options, raise_on_missing_output):
        raise RuntimeError("GCS unreachable")

    actor.state_api_client = SimpleNamespace(list=_raise)
    assert actor._get_actor_death_cause("a_worker") is None


def _deployment_actor(death_causes, alive, registered):
    actor = _actor(death_causes)
    actor.application_replicas = {"soup": {"CellposeRuntime": registered}}
    actor.get_deployment_replicas = lambda application_id, deployment_name: alive
    actor.get_actor_logs = lambda actor_id, tail, timezone, creation_timestamp: {
        "stdout": [],
        "stderr": [f"logs for {actor_id}"],
    }
    return actor


def test_dead_replica_logs_carry_the_cause_and_live_ones_do_not():
    actor = _deployment_actor(
        death_causes={"a_dead": _WORKER_DIED},
        alive={"live1": "a_live"},
        registered={
            "live1": ("a_live", 1.0, "UTC"),
            "dd12spcw": ("a_dead", 2.0, "UTC"),
        },
    )
    logs = actor.get_deployment_logs(
        application_id="soup",
        deployment_name="CellposeRuntime",
        n_previous_replica=-1,
    )
    assert set(logs) == {"live1", "dd12spcw"}
    assert logs["dd12spcw"]["death_cause"]["reason"] == "WORKER_DIED"
    # A live replica has not died, so claiming a cause for it would be a lie.
    assert "death_cause" not in logs["live1"]


def test_cause_survives_a_replica_whose_logs_cannot_be_read():
    """The case that motivated this: the logs are the part that says nothing."""
    actor = _deployment_actor(
        death_causes={"a_dead": _WORKER_DIED},
        alive={},
        registered={"dd12spcw": ("a_dead", 2.0, "UTC")},
    )

    def _boom(actor_id, tail, timezone, creation_timestamp):
        raise RuntimeError("log file already rotated away")

    actor.get_actor_logs = _boom
    logs = actor.get_deployment_logs(
        application_id="soup",
        deployment_name="CellposeRuntime",
        n_previous_replica=-1,
    )
    assert "rotated away" in logs["dd12spcw"]["stderr"][0]
    assert logs["dd12spcw"]["death_cause"]["reason"] == "WORKER_DIED"


def test_the_default_n_previous_replica_hides_the_cause_entirely():
    """Why the CLI has to override it: at the default the dead replica is gone.

    ``get_deployment_logs`` returns early when ``n_previous_replica == 0``, and
    both the dead replica's logs and its ``death_cause`` are attached after
    that return. Every other test here passes ``-1``, so this is the one that
    pins the behaviour a caller gets by default.
    """
    actor = _deployment_actor(
        death_causes={"a_dead": _WORKER_DIED},
        alive={"live1": "a_live"},
        registered={
            "live1": ("a_live", 1.0, "UTC"),
            "dd12spcw": ("a_dead", 2.0, "UTC"),
        },
    )
    logs = actor.get_deployment_logs(
        application_id="soup",
        deployment_name="CellposeRuntime",
    )
    assert set(logs) == {"live1"}
    assert not any("death_cause" in streams for streams in logs.values())


def test_apps_logs_asks_for_the_dead_replica():
    """The seam the rendering depends on: the request has to reach past 0.

    Rendering is covered below with hand-built maps, which cannot catch a
    caller that never asks for a dead replica — the field would simply never
    be in the payload and every render test would still pass.
    """
    from unittest.mock import patch

    from click.testing import CliRunner

    from bioengine.cli import apps as apps_cli

    seen = {}

    async def _get_app_status(**kwargs):
        seen.update(kwargs)
        return {
            "deployments": {
                "CellposeRuntime": {
                    "logs": {
                        "dd12spcw": {
                            "stdout": ["CALL ping OK 1.1ms"],
                            "stderr": [],
                            "death_cause": {
                                "reason": "WORKER_DIED",
                                "message": "Worker exit type: SYSTEM_ERROR",
                            },
                        }
                    }
                }
            }
        }

    async def _connect(*args, **kwargs):
        return SimpleNamespace(get_app_status=_get_app_status)

    with patch.object(
        apps_cli, "require_worker", lambda w, t, s: ("http://srv", "ws/worker", "tok")
    ), patch.object(apps_cli, "connect_worker", _connect):
        result = CliRunner().invoke(apps_cli.logs, ["soup"])

    assert result.exit_code == 0, result.output
    assert seen.get("n_previous_replica", 0) >= 1
    assert "died: WORKER_DIED" in result.output


def _render(logs):
    """``bioengine apps logs`` output for one deployment's replica log map."""
    from click.testing import CliRunner

    import click

    from bioengine.cli.apps import _echo_deployment_logs

    @click.command()
    def cmd():
        _echo_deployment_logs({"logs": logs})

    result = CliRunner().invoke(cmd, [])
    assert result.exit_code == 0, result.output
    return result.output


def test_the_cli_shows_the_cause_a_dead_replica_s_own_logs_do_not_carry():
    out = _render(
        {
            "dd12spcw": {
                "stdout": ["CALL ping OK 1.1ms"],
                "stderr": [],
                "death_cause": {
                    "reason": "WORKER_DIED",
                    "message": "Worker exit type: SYSTEM_ERROR\nsecond line",
                },
            }
        }
    )
    assert "died: WORKER_DIED" in out
    assert "SYSTEM_ERROR" in out
    # Ray's OOM text runs to kilobytes, so only the first line is rendered.
    assert "second line" not in out
    # The healthy-looking tail is still shown — it is why the cause is needed.
    assert "CALL ping OK" in out


def test_a_cause_with_no_log_lines_is_not_reported_as_nothing_logged():
    out = _render(
        {
            "dd12spcw": {
                "stdout": [],
                "stderr": [],
                "death_cause": {"reason": "RAY_KILL", "message": ""},
            }
        }
    )
    assert "died: RAY_KILL" in out
    assert "No logs recorded yet" not in out


def test_a_live_replica_renders_exactly_as_before():
    out = _render({"live1": {"stdout": ["hello"], "stderr": []}})
    assert "died:" not in out
    assert out.splitlines() == ["[live1] stdout", "  hello"]
