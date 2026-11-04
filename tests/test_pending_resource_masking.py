"""Unit tests for masking injected secrets out of the pending-resources payload.

A SLURM-mode worker reports pending actors/tasks/jobs through ``get_status()``.
Ray's detailed state records carry each pending item's whole runtime env,
``env_vars`` included, so an injected ``_BIOENGINE_SECRET_*`` value — in
practice a long-lived workspace-admin ``HYPHA_TOKEN`` — reached every caller
of ``get_status()`` in cleartext. ``get_app_status`` was never affected: it
routes env vars through ``ApplicationManager._filter_secret_env_vars``, which
is why the leak survived unnoticed — the convention existed and this path
simply did not pass through it.

Measured on a Berzelius ``--mode slurm`` worker (BioEngine 0.16.52): one
``get_status()`` returned a decodable JWT at
``ray_cluster.cluster.pending_resources.actors[0].serialized_runtime_env``
whose claims were ``scope='ws:<workspace>#a'`` — workspace admin — with ``exp``
in 2027.

Each record type names the runtime env differently, so the builder-level tests
below use the real Ray record classes rather than one stand-in: a fix keyed on
the actor's field name alone leaves the task and job paths still leaking.
"""

import json
from unittest.mock import MagicMock

import pytest
from ray.util.state.common import ActorState, JobState, StateResource, TaskState

from bioengine.cluster.proxy_actor import (
    BioEngineProxyActor,
    _mask_secret_env_vars,
    _pending_records,
)

_ProxyActor = BioEngineProxyActor.__ray_metadata__.modified_class

SECRET = "eyJhbGciOiJIUzI1NiJ9.header.payload"
ENV_VARS = {"_BIOENGINE_SECRET_HYPHA_TOKEN": SECRET, "BIOENGINE_APP_DIR": "/d"}


def test_masks_an_actors_serialized_runtime_env():
    """Actors carry the runtime env as a JSON *string*, not a dict."""
    record = {"serialized_runtime_env": json.dumps({"env_vars": dict(ENV_VARS)})}
    masked = _mask_secret_env_vars(record)
    env = json.loads(masked["serialized_runtime_env"])["env_vars"]
    assert env["_BIOENGINE_SECRET_HYPHA_TOKEN"] == "*****"
    assert SECRET not in json.dumps(masked)


def test_masks_a_tasks_runtime_env_info():
    """Tasks use a different key, so matching on the actor key alone fails."""
    record = {"runtime_env_info": {"runtime_env": {"env_vars": dict(ENV_VARS)}}}
    masked = _mask_secret_env_vars(record)
    assert SECRET not in json.dumps(masked)
    assert masked["runtime_env_info"]["runtime_env"]["env_vars"][
        "_BIOENGINE_SECRET_HYPHA_TOKEN"
    ] == "*****"


def test_masks_a_jobs_runtime_env():
    """Jobs use a third key, again different from the other two."""
    record = {"runtime_env": {"env_vars": dict(ENV_VARS)}}
    masked = _mask_secret_env_vars(record)
    assert SECRET not in json.dumps(masked)


def test_masks_every_value_not_only_underscore_prefixed_ones():
    """A caller can put a secret in any env var, under any name."""
    masked = _mask_secret_env_vars({"runtime_env": {"env_vars": dict(ENV_VARS)}})
    assert masked["runtime_env"]["env_vars"] == {
        "_BIOENGINE_SECRET_HYPHA_TOKEN": "*****",
        "BIOENGINE_APP_DIR": "*****",
    }


def test_leaves_the_rest_of_the_record_intact():
    """``required_resources`` is what the SLURM scale-up loop reads."""
    record = {
        "actor_id": "abc",
        "required_resources": {"CPU": 1.0, "GPU": 1.0},
        "serialized_runtime_env": json.dumps({"env_vars": dict(ENV_VARS)}),
    }
    masked = _mask_secret_env_vars(record)
    assert masked["actor_id"] == "abc"
    assert masked["required_resources"] == {"CPU": 1.0, "GPU": 1.0}


def test_masks_a_secret_nested_in_a_list():
    record = {"items": [{"runtime_env": {"env_vars": dict(ENV_VARS)}}]}
    assert SECRET not in json.dumps(_mask_secret_env_vars(record))


def test_masks_an_unparseable_env_vars_string_whole():
    """``serialized_runtime_env`` is only *usually* JSON.

    Never raise on it — and never print it either, since a blob that cannot
    be parsed cannot be shown to be free of secrets.
    """
    record = {"serialized_runtime_env": "env_vars={not json}"}
    assert _mask_secret_env_vars(record) == {"serialized_runtime_env": "*****"}


def test_record_with_no_runtime_env_is_unchanged():
    record = {"actor_id": "abc", "state": "PENDING_CREATION"}
    assert _mask_secret_env_vars(record) == record


def _actor_state():
    return ActorState(
        actor_id="a1",
        class_name="ServeReplica:app:Runtime",
        state="PENDING_CREATION",
        job_id="01000000",
        name="r",
        node_id=None,
        pid=None,
        ray_namespace="serve",
        serialized_runtime_env=json.dumps({"env_vars": dict(ENV_VARS)}),
        required_resources={"CPU": 1.0, "GPU": 1.0},
    )


def _task_state():
    return TaskState(
        task_id="t1",
        attempt_number=0,
        name="t",
        state="PENDING_NODE_ASSIGNMENT",
        job_id="01000000",
        actor_id=None,
        type="NORMAL_TASK",
        func_or_class_name="train",
        parent_task_id="ffffffffffffffff",
        node_id=None,
        worker_id=None,
        worker_pid=None,
        error_type=None,
        runtime_env_info={"runtime_env": {"env_vars": dict(ENV_VARS)}},
    )


@pytest.mark.parametrize("build", [_actor_state, _task_state], ids=["actor", "task"])
def test_pending_records_masks_real_ray_dataclasses(build):
    assert SECRET not in json.dumps(_pending_records([build()]), default=str)


def _job_state():
    """``JobState`` is a pydantic model, not a dataclass.

    ``dataclasses.asdict`` raises on it, so the jobs collector could not
    convert its own records at all — the masking has to survive that too.
    """
    return JobState(
        type="SUBMISSION",
        status="PENDING",
        entrypoint="python train.py",
        runtime_env={"env_vars": dict(ENV_VARS)},
    )


def test_pending_records_converts_a_job_without_raising():
    converted = _pending_records([_job_state()])
    assert len(converted) == 1
    assert SECRET not in json.dumps(converted, default=str)


def _proxy_actor_listing(records):
    """A bare proxy actor whose state API returns ``records`` per resource."""
    actor = object.__new__(_ProxyActor)
    client = MagicMock()
    client.list.side_effect = lambda resource, options, raise_on_missing_output: [
        records[resource]()
    ]
    actor.state_api_client = client
    return actor


BUILDERS = {
    StateResource.ACTORS: _actor_state,
    StateResource.TASKS: _task_state,
    StateResource.JOBS: _job_state,
}


@pytest.mark.parametrize(
    "method", ["_get_pending_jobs", "_get_pending_actors", "_get_pending_tasks"]
)
def test_no_pending_builder_leaks_the_secret(method):
    out = getattr(_proxy_actor_listing(BUILDERS), method)()
    assert out, f"{method} returned nothing; the assertions below would be vacuous"
    blob = json.dumps(out, default=str)
    assert SECRET not in blob, f"{method} leaked the injected secret"
    assert "*****" in blob, f"{method} masked nothing"


def test_an_unmasked_record_really_would_leak():
    """Positive control: the secret is reachable before masking."""
    assert SECRET in _actor_state().serialized_runtime_env
    assert SECRET in json.dumps(_task_state().runtime_env_info)
    assert SECRET in json.dumps(_job_state().runtime_env)


def test_the_zero_resource_actor_filter_still_applies():
    """Masking must not disturb the filtering ``_get_pending_actors`` does,
    or a security fix becomes a SLURM scale-up bug."""

    def idle_actor():
        actor = _actor_state()
        actor.required_resources = {"CPU": 0, "GPU": 0}
        return actor

    actor = _proxy_actor_listing({**BUILDERS, StateResource.ACTORS: idle_actor})
    assert actor._get_pending_actors() == []
