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
        actor.gcs_calls += 1
        (field, _, value), = options.filters
        if field == "state":
            # The BULK query: every dead actor, which the caller intersects with
            # its own replicas. Modelled rather than special-cased, because the
            # cost claim ("one query however many replicas") is only true if the
            # code really asks this way.
            assert value == "DEAD", f"unexpected state filter {value!r}"
            return [
                SimpleNamespace(actor_id=aid, death_cause=cause)
                for aid, cause in death_causes.items()
                if cause is not _MISSING
            ]
        cause = death_causes.get(value, _MISSING)
        if cause is _MISSING:
            return []
        return [SimpleNamespace(actor_id=value, death_cause=cause)]

    actor.gcs_calls = 0
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


def test_the_default_now_REPORTS_the_cause_without_the_logs():
    """REPLACES a test that pinned the opposite, and the swap is deliberate.

    The previous version of this test asserted that at the default the dead
    replica is absent and no ``death_cause`` appears anywhere — it pinned the
    blind spot as intended behaviour, and explained that this is why the CLI
    overrides the default. That blind spot was then decided to be a defect: a
    caller using defaults saw no dead replica and concluded nothing had died,
    which is this module's own failure mode one level up.

    So the default now reports THAT a replica died and why, with no log lines.
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

    assert set(logs) == {"live1", "dd12spcw"}
    assert logs["dd12spcw"]["death_cause"]["reason"] == "WORKER_DIED"
    # the cheap half only: no log lines are fetched at the default
    assert logs["dd12spcw"]["stdout"] == []
    assert logs["dd12spcw"]["stderr"] == []
    assert logs["dd12spcw"]["logs_omitted"] is True
    # and the live replica is untouched by any of this
    assert "death_cause" not in logs["live1"]


def test_a_deployment_with_NO_dead_replica_gains_nothing_and_asks_the_gcs_nothing():
    """CONTROL. The default path must stay free when nothing has died."""
    actor = _deployment_actor(
        death_causes={},
        alive={"live1": "a_live"},
        registered={"live1": ("a_live", 1.0, "UTC")},
    )
    calls = []
    inner = actor._get_actor_death_cause
    actor._get_actor_death_cause = lambda actor_id: (calls.append(actor_id), inner(actor_id))[1]

    logs = actor.get_deployment_logs(
        application_id="soup",
        deployment_name="CellposeRuntime",
    )
    assert set(logs) == {"live1"}
    assert calls == [], f"asked the GCS {len(calls)} time(s) with nothing dead"


def test_the_default_costs_exactly_ONE_gcs_QUERY_however_many_have_died():
    """CONTROL on the cost, which is what made the original defect tempting.

    The first implementation held the cost down by describing a single dead
    replica — one GCS lookup instead of one per replica, with 94 dead replicas
    measured on a live worker. Review showed what that bought: the chosen one
    was the last REGISTERED, so a deliberate stop could hide an earlier crash.

    The cost argument is answered a different way now. Every dead actor is
    fetched in ONE query and intersected locally, so the lookup count does not
    grow with the number of dead replicas AND no death has to be dropped.
    Without this assertion the obvious rewrite — a lookup per replica — passes
    every other test here.
    """
    registered = {"live1": ("a_live", 1.0, "UTC")}
    for i in range(40):
        registered[f"dead{i:02d}"] = (f"a_dead{i:02d}", 2.0 + i, "UTC")

    actor = _deployment_actor(
        death_causes={f"a_dead{i:02d}": _WORKER_DIED for i in range(40)},
        alive={"live1": "a_live"},
        registered=registered,
    )
    logs = actor.get_deployment_logs(
        application_id="soup",
        deployment_name="CellposeRuntime",
    )
    assert actor.gcs_calls == 1, (
        f"{actor.gcs_calls} GCS queries at the default with 40 dead replicas"
    )

    # ...and all 40 are reported, because every one of them died abnormally.
    described = set(logs) - {"live1", "_deliberately_stopped_replicas"}
    assert described == {f"dead{i:02d}" for i in range(40)}, (
        f"described {len(described)} of 40 abnormal deaths"
    )


def test_many_deliberate_stops_are_summarised_not_listed():
    """The payload bound. A long-lived deployment's dead replicas are mostly
    routine scale-downs; listing all of them would make the default status
    grow without telling an operator anything. They are counted instead, and
    the one crash among them is still described in full."""
    registered = {"live1": ("a_live", 1.0, "UTC")}
    causes = {}
    for i in range(30):
        registered[f"stop{i:02d}"] = (f"a_stop{i:02d}", 2.0 + i, "UTC")
        causes[f"a_stop{i:02d}"] = _RAY_KILL
    registered["crashed"] = ("a_oom", 99.0, "UTC")
    causes["a_oom"] = _OOM

    logs = _deployment_actor(
        death_causes=causes, alive={"live1": "a_live"}, registered=registered
    ).get_deployment_logs(
        application_id="soup",
        deployment_name="CellposeRuntime",
    )
    assert logs["_deliberately_stopped_replicas"] == 30
    assert set(logs) - {"live1", "_deliberately_stopped_replicas"} == {"crashed"}
    assert logs["crashed"]["death_cause"]["reason"] == "OOM"


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


def test_a_routine_scale_down_cannot_hide_a_crash_at_the_default():
    """THE REGRESSION THIS ISSUE IS ABOUT, one level up.

    Reported by review of this change. Timeline, in REGISTRATION order:

      A registers, serves, and later dies of OOM
      B is added by autoscaling, then deliberately scaled down (RAY_KILL)

    Picking the last-REGISTERED dead replica reports B's RAY_KILL and drops
    A's OOM, so the operator reads "something was deliberately stopped" and
    concludes nothing crashed — the exact wrong-layer diagnosis the
    ``death_cause`` field exists to prevent.

    Ray's ``ActorState`` carries NO death timestamp (measured against ray
    2.55: the dataclass has no end-time field at all), so "the most recently
    dead replica" is not derivable here and ordering cannot be repaired.
    Choosing ONE dead replica is therefore the defect. An abnormal death is
    never dropped.
    """
    actor = _deployment_actor(
        death_causes={"a_oom": _OOM, "b_kill": _RAY_KILL},
        alive={"c_live": "a_c"},
        registered={
            "a_crashed": ("a_oom", 1.0, "UTC"),   # registered FIRST, died of OOM
            "b_stopped": ("b_kill", 2.0, "UTC"),  # registered LAST, deliberate stop
            "c_live": ("a_c", 3.0, "UTC"),
        },
    )
    logs = actor.get_deployment_logs(
        application_id="soup",
        deployment_name="CellposeRuntime",
    )
    assert "a_crashed" in logs, (
        "the OOM crash is absent at the default: a deliberate stop registered "
        f"later hid it. got {sorted(logs)}"
    )
    assert logs["a_crashed"]["death_cause"]["reason"] == "OOM"


def test_clause_2_the_exception_is_carried_by_the_LOGS_not_by_the_cause():
    """CLAUSE 2, and an honest statement of which path meets it.

    The clause asks that the surfaced cause name the server-side exception.
    Ray's GCS does NOT carry it: for a replica that raised and exited, the
    death cause reads ``WORKER_DIED`` with Ray's own wording about
    ``SYSTEM_ERROR``, and the exception's name appears nowhere in it. The
    filed incident's ``IncompleteRead`` is in the replica's STDERR.

    So the exception is reachable at ``n_previous_replica >= 1``, where the
    logs are fetched, and NOT at the default, which omits them by design.
    This test pins both halves so the limit is a measured property rather
    than something a reader discovers against a live worker.
    """
    actor = _deployment_actor(
        death_causes={"a_dead": _WORKER_DIED},
        alive={},
        registered={"crashed": ("a_dead", 2.0, "UTC")},
    )
    actor.get_actor_logs = lambda actor_id, tail, timezone, creation_timestamp: {
        "stdout": [],
        "stderr": [
            "Traceback (most recent call last):",
            "  File 'entry.py', line 1, in _score",
            "http.client.IncompleteRead: IncompleteRead(0 bytes read)",
        ],
    }

    with_logs = actor.get_deployment_logs(
        application_id="soup",
        deployment_name="CellposeRuntime",
        n_previous_replica=1,
    )
    assert any("IncompleteRead" in line for line in with_logs["crashed"]["stderr"]), (
        "the exception is not reachable even with logs requested"
    )
    assert with_logs["crashed"]["death_cause"]["reason"] == "WORKER_DIED"
    # ...and the cause ALONE does not name it — which is why the clause needs
    # the logs path, not the default.
    assert "IncompleteRead" not in with_logs["crashed"]["death_cause"]["message"]

    at_default = actor.get_deployment_logs(
        application_id="soup",
        deployment_name="CellposeRuntime",
    )
    assert at_default["crashed"]["logs_omitted"] is True
    assert at_default["crashed"]["stderr"] == []
