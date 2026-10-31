"""Pin that pending-resource records never carry a deployment's secrets.

``_get_pending_jobs`` / ``_get_pending_actors`` / ``_get_pending_tasks`` query
Ray's state API with ``detail=True`` and ``asdict()`` the records straight into
``cluster.pending_resources``. A detailed record carries
``serialized_runtime_env`` verbatim, and for a BioEngine deployment that string
contains the injected ``_BIOENGINE_SECRET_HYPHA_TOKEN``.

``get_status`` returns ``pending_resources``, so before this was masked a single
status call handed the caller a full workspace credential. ``get_app_status``
was never affected — it routes env vars through
``AppsManager._filter_secret_env_vars`` — which is why the leak survived: the
convention existed and this path simply did not pass through it.

Measured on a Berzelius ``--mode slurm`` worker (BioEngine 0.16.52): one
``get_status()`` returned a decodable JWT at
``ray_cluster.cluster.pending_resources.actors[0].serialized_runtime_env``
whose claims were ``scope='ws:<workspace>#a'`` — workspace admin — with ``exp``
in 2027.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

from ray.util.state.common import StateResource

from bioengine.cluster.proxy_actor import (
    BioEngineProxyActor,
    _mask_runtime_env_secrets,
)

_ProxyActor = BioEngineProxyActor.__ray_metadata__.modified_class

SECRET = "eyJhbGciOiJIUzI1NiJ9.PRETEND-THIS-IS-A-REAL-JWT.signature"


def _runtime_env(**extra: Any) -> str:
    payload: Dict[str, Any] = {
        "env_vars": {
            "_BIOENGINE_SECRET_HYPHA_TOKEN": SECRET,
            "BIOENGINE_APP_ID": "model-finetune-e2e",
        },
        "pip": {"packages": ["torch==2.4.0"]},
        "working_dir": "gcs://_ray_pkg_abc123.zip",
    }
    payload.update(extra)
    return json.dumps(payload)


@dataclass
class _Record:
    """Stand-in for a Ray state-API record; only the read fields matter."""

    serialized_runtime_env: Optional[str] = None
    required_resources: Dict[str, float] = field(
        default_factory=lambda: {"CPU": 1.0, "GPU": 1.0}
    )


def _actor_with(client_records) -> Any:
    actor = object.__new__(_ProxyActor)
    client = MagicMock()

    def _list(resource, options, raise_on_missing_output):
        return client_records[resource]

    client.list.side_effect = _list
    actor.state_api_client = client
    return actor


def _records_for(record: _Record) -> Dict[StateResource, list]:
    return {
        StateResource.JOBS: [record],
        StateResource.ACTORS: [record],
        StateResource.TASKS: [record],
    }


def test_the_secret_is_absent_from_every_pending_builder() -> None:
    for build in ("_get_pending_jobs", "_get_pending_actors", "_get_pending_tasks"):
        actor = _actor_with(_records_for(_Record(serialized_runtime_env=_runtime_env())))
        out = getattr(actor, build)()
        assert out, f"{build} returned nothing; the test would vacuously pass"
        blob = json.dumps(out)
        assert SECRET not in blob, f"{build} leaked the injected secret"
        assert "*****" in blob, f"{build} did not mask anything"


def test_an_unmasked_record_really_would_leak() -> None:
    # The positive control. Without it, a builder that returned [] or dropped
    # serialized_runtime_env for an unrelated reason would pass the test above
    # while proving nothing about masking.
    raw = _runtime_env()
    assert SECRET in raw
    assert SECRET in json.dumps({"serialized_runtime_env": raw})


def test_masking_keeps_the_keys_and_the_rest_of_the_runtime_env() -> None:
    # Dropping the field outright would make a stuck deploy harder to diagnose
    # in exactly the window an operator calls get_status.
    masked = _mask_runtime_env_secrets(
        {"serialized_runtime_env": _runtime_env()}
    )
    env = json.loads(masked["serialized_runtime_env"])

    assert env["env_vars"] == {
        "_BIOENGINE_SECRET_HYPHA_TOKEN": "*****",
        "BIOENGINE_APP_ID": "*****",
    }
    assert env["pip"] == {"packages": ["torch==2.4.0"]}
    assert env["working_dir"] == "gcs://_ray_pkg_abc123.zip"


def test_every_value_is_masked_not_only_underscore_prefixed_ones() -> None:
    # AppsManager._filter_secret_env_vars keys off a leading underscore. Relying
    # on that convention here would make this disclosure boundary depend on a
    # naming choice made in another module, so a secret injected under a plain
    # name must still be masked.
    masked = _mask_runtime_env_secrets(
        {"serialized_runtime_env": json.dumps({"env_vars": {"HYPHA_TOKEN": SECRET}})}
    )
    assert SECRET not in masked["serialized_runtime_env"]


def test_an_unparseable_runtime_env_is_masked_whole() -> None:
    masked = _mask_runtime_env_secrets(
        {"serialized_runtime_env": "not json " + SECRET}
    )
    assert masked["serialized_runtime_env"] == "*****"


def test_records_without_a_runtime_env_are_untouched() -> None:
    for value in (None, "", {"env_vars": {}}):
        record = {"serialized_runtime_env": value, "required_resources": {"CPU": 1.0}}
        assert _mask_runtime_env_secrets(dict(record)) == record


def test_an_empty_env_vars_block_is_left_alone() -> None:
    raw = json.dumps({"env_vars": {}, "pip": None})
    assert _mask_runtime_env_secrets({"serialized_runtime_env": raw})[
        "serialized_runtime_env"
    ] == raw


def test_the_zero_resource_actor_filter_still_applies() -> None:
    # The mask must not disturb the filtering _get_pending_actors does, or a
    # security fix becomes a SLURM scale-up bug.
    idle = _Record(
        serialized_runtime_env=_runtime_env(), required_resources={"CPU": 0, "GPU": 0}
    )
    actor = _actor_with(_records_for(idle))
    assert actor._get_pending_actors() == []
