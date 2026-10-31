"""Unit tests for deploy_app's ``scaling`` map validation.

These exercise the pure helper ``_validate_scaling_map`` directly, so they need no
Ray and no cluster. That is the point of the boundary: ``num_replicas=0`` used to
clear BioEngine's check and then be refused by Ray from inside the deploy, where
the error names neither the ``scaling`` parameter nor the entry that was wrong.
Ray's own bound, measured on ray 2.55.1 in ``bioengine-worker:0.16.52``, is
``num_replicas is expected to larger than 0`` for 0, accepted for 1, and a pydantic
ValidationError for -1 — so 0 was the only value this validator let through that
Ray then rejected.
"""

import pytest

from bioengine.apps.manager import _validate_scaling_map

DEPLOYMENTS = ("CellposeRuntime", "EntryApp")


def test_zero_replicas_is_rejected_and_the_message_names_the_entry():
    """The defect. Asserted on the message, not merely that something raised.

    Ray raises for 0 too, so a test that only checks "it raised" would pass
    against the unfixed validator once Ray is in the picture.
    """
    with pytest.raises(ValueError) as excinfo:
        _validate_scaling_map(
            {"CellposeRuntime": {"num_replicas": 0}}, DEPLOYMENTS, "my-app"
        )

    message = str(excinfo.value)
    assert "scaling entry for 'CellposeRuntime'" in message
    assert "num_replicas=0" in message
    assert ">= 1" in message
    assert ">= 0" not in message, "the stated bound still contradicts what is enforced"


def test_one_replica_is_accepted():
    """The boundary is off-by-one in one direction only."""
    _validate_scaling_map(
        {"CellposeRuntime": {"num_replicas": 1}}, DEPLOYMENTS, "my-app"
    )


def test_negative_replicas_is_still_rejected():
    with pytest.raises(ValueError, match="num_replicas=-1"):
        _validate_scaling_map(
            {"CellposeRuntime": {"num_replicas": -1}}, DEPLOYMENTS, "my-app"
        )


def test_an_unknown_deployment_name_is_rejected():
    with pytest.raises(ValueError, match="is not a user deployment"):
        _validate_scaling_map(
            {"NoSuchClass": {"num_replicas": 2}}, DEPLOYMENTS, "my-app"
        )


def test_a_non_dict_entry_is_rejected():
    with pytest.raises(ValueError, match="must be a dict"):
        _validate_scaling_map({"EntryApp": 3}, DEPLOYMENTS, "my-app")


def test_neither_key_is_rejected():
    with pytest.raises(ValueError, match="got neither"):
        _validate_scaling_map({"EntryApp": {}}, DEPLOYMENTS, "my-app")


def test_both_keys_are_rejected():
    with pytest.raises(ValueError, match="got both"):
        _validate_scaling_map(
            {
                "EntryApp": {
                    "num_replicas": 2,
                    "autoscaling_config": {"min_replicas": 1, "max_replicas": 3},
                }
            },
            DEPLOYMENTS,
            "my-app",
        )


def test_autoscaling_config_alone_is_accepted():
    _validate_scaling_map(
        {"CellposeRuntime": {"autoscaling_config": {"min_replicas": 1, "max_replicas": 3}}},
        DEPLOYMENTS,
        "my-app",
    )


@pytest.mark.parametrize("empty", [None, {}])
def test_an_empty_map_passes(empty):
    """An absent entry is legal — it means "no override", not "one replica"."""
    _validate_scaling_map(empty, DEPLOYMENTS, "my-app")
