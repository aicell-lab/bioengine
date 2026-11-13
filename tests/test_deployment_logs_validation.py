"""Unit tests for ``get_app_status``'s ``deployment_logs`` map validation (#0406).

These exercise the pure helper ``_validate_deployment_logs_overrides`` directly, so
they need no Ray and no cluster — the same shape as ``tests/test_scaling_validation.py``.

The defect: ``get_app_status(deployment_logs=True)`` is the natural way to ask for logs
and it crashed with ``AttributeError: 'bool' object has no attribute 'get'`` from
``_get_deployment_status``, naming neither the parameter nor the type. The declared
annotation is ``Optional[Dict[str, Dict[str, int]]]`` under ``@schema_method``, which is
why a validator looks redundant and is not: measured inside
``bioengine-worker:0.16.98`` (hypha_rpc 0.21.40), ``schema_function_pydantic``'s
``process_arguments`` coerces an argument only when its annotation is a ``BaseModel``
subclass and carries the line ``# TODO: Validate the input, for primitive types and
Pydantic models``. A ``Dict[...]`` annotation is a typing generic, so nothing checks it.

Both the map and its entries are covered. A validator that checked only the map would
leave ``{'RuntimeApp': 'nonsense'}`` crashing identically one frame later, at
``override.get("logs_tail")`` — measured in that same image before the fix.
"""

import pytest

from bioengine.apps.manager import _validate_deployment_logs_overrides

A_REAL_OVERRIDE = {"RuntimeApp": {"logs_tail": 200, "n_previous_replica": 3}}


def test_a_bool_is_rejected_and_the_message_names_the_parameter_and_the_type():
    """The reported defect, asserted on the message rather than on "it raised".

    The unfixed code also raised on this input — an AttributeError from a private
    helper. A test asserting only that something raised would pass against it.
    """
    with pytest.raises(ValueError) as excinfo:
        _validate_deployment_logs_overrides(True)

    message = str(excinfo.value)
    assert "deployment_logs must be a dict" in message
    assert "got bool" in message
    assert "logs_tail" in message, "the message should say what the parameter is for"


def test_a_falsy_non_dict_is_rejected_too():
    """The quieter half: ``0`` never crashed, it silently did nothing.

    ``overrides = deployment_logs or {}`` turns any falsy value into an empty map, so
    ``deployment_logs=0`` cleared every check and returned default logs as though no
    override had been asked for. A suite that only tried ``True`` would miss this.
    """
    with pytest.raises(ValueError, match="got int"):
        _validate_deployment_logs_overrides(0)


def test_a_string_is_rejected():
    with pytest.raises(ValueError, match="got str"):
        _validate_deployment_logs_overrides("RuntimeApp")


@pytest.mark.parametrize("empty", [None, {}])
def test_an_empty_map_passes(empty):
    """An absent map means "no override", not a value — it must stay legal."""
    _validate_deployment_logs_overrides(empty)


def test_a_real_override_passes():
    """The positive arm. Without it, a validator that rejected everything would pass."""
    _validate_deployment_logs_overrides(A_REAL_OVERRIDE)


def test_either_key_alone_passes():
    """The docstring promises each entry may set either or both keys."""
    _validate_deployment_logs_overrides({"RuntimeApp": {"logs_tail": 0}})
    _validate_deployment_logs_overrides({"RuntimeApp": {"n_previous_replica": 1}})


def test_the_documented_all_sentinel_passes():
    """``-1`` means "everything" for both keys, per ``get_deployment_logs``'s docstring.

    Pinned because the obvious over-reach when writing this validator is a ``>= 0``
    range check, which would reject the one value the API documents for "all".
    """
    _validate_deployment_logs_overrides(
        {"RuntimeApp": {"logs_tail": -1, "n_previous_replica": -1}}
    )


def test_a_non_dict_entry_is_rejected_and_names_the_entry():
    """The second crash site, one frame past the first."""
    with pytest.raises(ValueError) as excinfo:
        _validate_deployment_logs_overrides({"RuntimeApp": "nonsense"})

    message = str(excinfo.value)
    assert "entry for 'RuntimeApp'" in message
    assert "got str" in message


def test_an_unknown_key_is_rejected_and_names_the_key():
    """A typo'd key is the silent-drop case, and the reason this is not over-reach.

    ``override.get("logs_tail", logs_tail)`` falls back to the top-level default, so
    ``log_tail=200`` returns 30 lines while reading exactly like a request for 200.
    Same reasoning as ``_validate_scaling_map`` rejecting unknown
    ``autoscaling_config`` keys because "Serve would drop them silently".
    """
    with pytest.raises(ValueError) as excinfo:
        _validate_deployment_logs_overrides({"RuntimeApp": {"log_tail": 200}})

    message = str(excinfo.value)
    assert "log_tail" in message
    assert "ignored silently" in message


def test_a_bool_inner_value_is_rejected():
    """``bool`` is an ``int`` subclass, so a plain isinstance check would accept this.

    ``logs_tail=True`` would reach Ray as ``tail=1`` and return a single line — a wrong
    answer rather than an error, which is the failure mode this whole row is about.
    """
    with pytest.raises(ValueError) as excinfo:
        _validate_deployment_logs_overrides({"RuntimeApp": {"logs_tail": True}})

    assert "logs_tail=True" in str(excinfo.value)


def test_a_non_int_inner_value_is_rejected():
    with pytest.raises(ValueError, match="logs_tail='200'"):
        _validate_deployment_logs_overrides({"RuntimeApp": {"logs_tail": "200"}})
