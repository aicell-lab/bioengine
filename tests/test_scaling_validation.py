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

from ray.serve.config import AutoscalingConfig

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


def test_a_truthy_non_dict_map_is_rejected():
    """The map itself, not its entries — the same hole #0406 found in deployment_logs.

    ``(scaling or {}).items()`` raised a bare ``AttributeError: 'bool' object has no
    attribute 'items'`` here, from inside the validator whose entire purpose is naming
    what the caller got wrong.
    """
    with pytest.raises(ValueError) as excinfo:
        _validate_scaling_map(True, DEPLOYMENTS, "my-app")

    message = str(excinfo.value)
    assert "scaling must be a dict" in message
    assert "got bool" in message


def test_a_falsy_non_dict_map_is_rejected():
    """The half that never raised at all, and so is the one worth pinning.

    ``scaling=0`` became ``{}`` and cleared every check below it, so the deploy ran on
    Serve's defaults while the caller believed they had set a replica count. A test
    trying only ``True`` would report this boundary fixed while it was still open.
    """
    with pytest.raises(ValueError, match="got int"):
        _validate_scaling_map(0, DEPLOYMENTS, "my-app")


def _a_field_info():
    """Any real FieldInfo — the validator only reads the KEYS, never the values."""
    return next(iter(AutoscalingConfig.model_fields.values()))


def test_a_key_ray_adds_later_is_accepted_without_touching_this_file(monkeypatch):
    """Site 6 of #0212, arm 1 of 2: ``valid_keys`` is derived, not enumerated.

    The whole point of ``set(AutoscalingConfig.model_fields)`` is that a key Ray adds in
    a future release is accepted the day it ships. A hardcoded list would reject it and
    turn a Serve upgrade into a BioEngine bug, which is the opposite of what the guard
    is for — it exists to catch keys Serve would DROP, not to freeze Serve's schema.

    Pinning a forward-compat property normally means waiting for the future. It does not
    here: ``model_fields`` is a plain dict, so a synthetic key can be injected and the
    future simulated. Measured in the shipped image at Ray 2.55.1 / pydantic 2.12.0.
    """
    monkeypatch.setitem(AutoscalingConfig.model_fields, "zz_future_knob", _a_field_info())

    _validate_scaling_map(
        {"CellposeRuntime": {"autoscaling_config": {"zz_future_knob": 7}}},
        DEPLOYMENTS,
        "my-app",
    )


def test_a_key_ray_removes_stops_being_accepted(monkeypatch):
    """Site 6 of #0212, arm 2 of 2 — and the arm that makes arm 1 non-vacuous.

    Arm 1 passes against a mutant that unions a hardcoded set WITH ``model_fields``:
    the synthetic key is in model_fields, so such a mutant accepts it too. Only removal
    separates them. With a real key deleted, the derived form rejects it while any form
    carrying its own hardcoded copy still accepts — so the two arms together pin
    ``valid_keys`` to exactly ``set(model_fields)``, which neither does alone.

    ``aggregation_function`` is a real Ray key; monkeypatch restores it afterwards.
    """
    monkeypatch.delitem(AutoscalingConfig.model_fields, "aggregation_function")

    with pytest.raises(ValueError) as excinfo:
        _validate_scaling_map(
            {"CellposeRuntime": {"autoscaling_config": {"aggregation_function": "mean"}}},
            DEPLOYMENTS,
            "my-app",
        )
    # Named, not merely raised: the message is what tells a user which key was wrong.
    assert "aggregation_function" in str(excinfo.value)
