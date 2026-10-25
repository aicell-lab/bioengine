"""Unit tests for masking injected secrets out of the pending-resources payload.

A SLURM-mode worker reports pending actors/tasks/jobs through ``get_status()``.
Ray's detailed state records carry each pending item's whole runtime env,
``env_vars`` included, so an injected ``_BIOENGINE_SECRET_*`` value — in
practice a long-lived workspace-admin ``HYPHA_TOKEN`` — reached every caller
of ``get_status()`` in cleartext. These tests pin the masking and the record
conversion it depends on.
"""

import json

import pytest
from ray.util.state.common import ActorState, JobState, TaskState

from bioengine.cluster.proxy_actor import _mask_secret_env_vars, _pending_records

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


def test_leaves_non_secret_env_vars_readable():
    """Masking everything would be useless for debugging a stuck deploy."""
    masked = _mask_secret_env_vars({"runtime_env": {"env_vars": dict(ENV_VARS)}})
    assert masked["runtime_env"]["env_vars"]["BIOENGINE_APP_DIR"] == "/d"


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


def test_passes_through_a_string_that_is_not_json():
    """``serialized_runtime_env`` is only *usually* JSON; never raise on it."""
    record = {"call_site": "env_vars are set in foo.py"}
    assert _mask_secret_env_vars(record) == record


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


def test_pending_records_converts_a_job_without_raising():
    """``JobState`` is a pydantic model, not a dataclass.

    ``dataclasses.asdict`` raises on it, so the jobs collector could not
    convert its own records at all — the masking has to survive that too.
    """
    job = JobState(
        type="SUBMISSION",
        status="PENDING",
        entrypoint="python train.py",
        runtime_env={"env_vars": dict(ENV_VARS)},
    )
    converted = _pending_records([job])
    assert len(converted) == 1
    assert SECRET not in json.dumps(converted, default=str)
