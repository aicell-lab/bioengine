"""Unit tests for ``get_app_status``'s ``deployment_logs`` map validation (#0406).

These exercise the pure helpers directly, so they need no Ray and no cluster —
the same shape as ``tests/test_scaling_validation.py``.

The defect: ``deployment_logs`` was normalised with ``overrides or {}`` and then
indexed with ``.get()``. ``or {}`` rescues only the FALSY wrong types, so
``deployment_logs=True`` survived the normalisation and died three frames deeper
with ``AttributeError: 'bool' object has no attribute 'get'`` — which names
neither the parameter nor the value. MEASURED on 0.16.98 before the fix, driving
the real ``_get_deployment_status`` with a stubbed proxy handle:

    deployment_logs=True            -> AttributeError: 'bool' ... has no attribute 'get'
    deployment_logs={"Dep": 3}      -> AttributeError: 'int' ... has no attribute 'get'
    deployment_logs=False           -> no error, tail=30   (override silently DISCARDED)
    deployment_logs={"Dep": {"logs_tails": 5}}  -> no error, tail=30  (typo discarded)

Both halves matter. The crash is the reported symptom; the silent discard is the
same defect failing the other way, and it is the one a caller cannot notice.
"""

import pytest

from bioengine.apps.manager import (
    _require_deployment_map,
    _validate_deployment_logs_map,
    _validate_scaling_map,
)

DEP = "RuntimeApp"


# --------------------------------------------------------------------------
# The reported defect: a non-dict map
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad, type_name",
    [
        (True, "bool"),
        (1, "int"),
        ("RuntimeApp", "str"),
        (["RuntimeApp"], "list"),
        (0.5, "float"),
    ],
)
def test_a_non_dict_map_is_rejected_and_the_message_names_the_parameter(bad, type_name):
    """Asserted on the message, not merely that something raised.

    The unfixed code also "raises" — with an AttributeError from three frames
    down. A test that only checked ``pytest.raises(Exception)`` would pass
    against the defect it is here to pin.
    """
    with pytest.raises(ValueError) as excinfo:
        _validate_deployment_logs_map(bad)

    message = str(excinfo.value)
    assert "deployment_logs must be a dict" in message
    assert type_name in message, message


@pytest.mark.parametrize("falsy", [False, 0, ""])
def test_a_falsy_non_dict_is_rejected_too(falsy):
    """The half ``or {}`` used to swallow.

    Before the fix these became ``{}`` and the call returned a normal-looking
    status with the overrides silently dropped. Rejecting only the truthy ones
    would fix the traceback and leave the silent failure in place.
    """
    with pytest.raises(ValueError, match="deployment_logs must be a dict"):
        _validate_deployment_logs_map(falsy)


# --------------------------------------------------------------------------
# One level down: entries, keys, values
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [3, "abc", True, ["logs_tail"]])
def test_a_non_dict_entry_is_rejected(bad):
    """``overrides.get(name) or {}`` has the identical hole one level down."""
    with pytest.raises(ValueError) as excinfo:
        _validate_deployment_logs_map({DEP: bad})

    message = str(excinfo.value)
    assert f"deployment_logs entry for '{DEP}' must be a dict" in message
    assert type(bad).__name__ in message, message


def test_an_unknown_key_is_rejected_because_it_would_be_dropped_silently():
    """The ``logs_tails`` typo. MEASURED before the fix: tail stayed 30, so the
    caller asked for 5 lines, got 30, and nothing said the key was ignored.
    Same failure mode as an unknown ``autoscaling_config`` key, which the
    scaling validator already refuses for the same stated reason."""
    with pytest.raises(ValueError) as excinfo:
        _validate_deployment_logs_map({DEP: {"logs_tails": 5}})

    message = str(excinfo.value)
    assert "unknown key(s) ['logs_tails']" in message
    assert "logs_tail" in message, "the message must name the valid keys"


def test_unknown_keys_are_all_named_not_just_the_first():
    with pytest.raises(ValueError) as excinfo:
        _validate_deployment_logs_map({DEP: {"tail": 5, "zz": 1, "logs_tail": 2}})

    message = str(excinfo.value)
    assert "['tail', 'zz']" in message, message


@pytest.mark.parametrize("key", ["logs_tail", "n_previous_replica"])
@pytest.mark.parametrize("bad", ["abc", 1.5, None, [3]])
def test_a_non_int_value_is_rejected(key, bad):
    """``tail='abc'`` reached the proxy actor unexamined before the fix."""
    with pytest.raises(ValueError) as excinfo:
        _validate_deployment_logs_map({DEP: {key: bad}})

    assert f"has {key}=" in str(excinfo.value)


@pytest.mark.parametrize("key", ["logs_tail", "n_previous_replica"])
def test_a_bool_value_is_rejected_even_though_bool_is_an_int(key):
    """``isinstance(True, int)`` is True, so the bool exclusion is load-bearing
    and is not implied by the int check. The file already draws this distinction
    at ``_per_replica_ongoing_limit``."""
    with pytest.raises(ValueError, match=f"has {key}=True"):
        _validate_deployment_logs_map({DEP: {key: True}})


# --------------------------------------------------------------------------
# Controls. Without these every case above also passes against a validator
# that rejects everything.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("empty", [None, {}])
def test_an_absent_map_is_accepted(empty):
    """``deployment_logs`` is optional and its default is None."""
    _validate_deployment_logs_map(empty)


@pytest.mark.parametrize(
    "good",
    [
        {DEP: {"logs_tail": 5}},
        {DEP: {"n_previous_replica": 3}},
        {DEP: {"logs_tail": 200, "n_previous_replica": 3}},
        {DEP: {}},
        {"RuntimeApp": {"logs_tail": 200}, "ProxyDeployment": {"logs_tail": 0}},
    ],
)
def test_the_documented_shapes_are_accepted(good):
    """The last two are lifted from the parameter's own ``examples``."""
    _validate_deployment_logs_map(good)


@pytest.mark.parametrize("key", ["logs_tail", "n_previous_replica"])
def test_minus_one_is_accepted_because_it_is_a_documented_feature(key):
    """NOT a lower bound. ``proxy_actor.get_deployment_logs`` documents -1 as
    "the entire log" for ``tail`` and "all dead replicas" for
    ``n_previous_replica``. The scaling validator next door enforces
    ``num_replicas >= 1``, so mirroring it without reading the proxy would have
    refused a working feature — this pins that it does not."""
    _validate_deployment_logs_map({DEP: {key: -1}})


def test_zero_logs_tail_is_accepted():
    """``{'ProxyDeployment': {'logs_tail': 0}}`` is one of the parameter's own
    documented examples — "no logs for this deployment"."""
    _validate_deployment_logs_map({DEP: {"logs_tail": 0}})


def test_an_unknown_deployment_name_is_accepted():
    """Deliberately NOT rejected, and the one place this validator departs from
    the scaling validator, which DOES refuse an unknown deployment name.

    Two measured reasons. The parameter documents it: "Overrides naming a
    deployment that doesn't exist in any queried app are ignored silently — the
    dashboard may carry stale overrides while the user mutates the app's class
    set." And the map is applied per-application in a loop, so a key belonging
    to app B is legitimately absent from app A.
    """
    _validate_deployment_logs_map({"NoSuchDeployment": {"logs_tail": 5}})


# --------------------------------------------------------------------------
# The sibling hole the report did not name: deploy_app(scaling=...)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad, type_name", [(True, "bool"), (1, "int"), ("x", "str")])
def test_a_non_dict_scaling_map_is_rejected(bad, type_name):
    """``_validate_scaling_map`` — the precedent #0406 cites as the shape to
    copy — had the identical defect in its own first line,
    ``(scaling or {}).items()``. MEASURED before the fix:
    ``scaling=True`` -> ``AttributeError: 'bool' object has no attribute 'items'``,
    reachable from the public ``deploy_app``. Copying the precedent without
    fixing it would have copied the bug."""
    with pytest.raises(ValueError) as excinfo:
        _validate_scaling_map(bad, ("EntryApp",), "demo")

    message = str(excinfo.value)
    assert "scaling must be a dict" in message
    assert type_name in message, message


@pytest.mark.parametrize("falsy", [False, 0])
def test_a_falsy_scaling_map_is_rejected_too(falsy):
    with pytest.raises(ValueError, match="scaling must be a dict"):
        _validate_scaling_map(falsy, ("EntryApp",), "demo")


@pytest.mark.parametrize("empty", [None, {}])
def test_an_absent_scaling_map_is_still_accepted(empty):
    """CONTROL — the regression this fix could plausibly cause. ``scaling``
    defaults to None on every deploy, so refusing None would break all of them."""
    _validate_scaling_map(empty, ("EntryApp",), "demo")


def test_a_valid_scaling_map_is_still_accepted():
    """CONTROL — the top-level check must not shadow the existing per-entry
    validation this helper already did."""
    _validate_scaling_map({"EntryApp": {"num_replicas": 2}}, ("EntryApp",), "demo")


def test_the_existing_scaling_entry_checks_still_fire():
    """CONTROL — pins that adding the top-level guard did not displace the
    checks below it. ``num_replicas=0`` is the defect ``_validate_scaling_map``
    was originally written for."""
    with pytest.raises(ValueError, match="num_replicas=0"):
        _validate_scaling_map({"EntryApp": {"num_replicas": 0}}, ("EntryApp",), "demo")


# --------------------------------------------------------------------------
# The shared accessor
# --------------------------------------------------------------------------


def test_the_shared_accessor_names_whichever_parameter_it_was_given():
    """Both call sites must report their OWN parameter name. A hardcoded name in
    the shared helper would send a ``deploy_app`` caller looking at
    ``deployment_logs``."""
    with pytest.raises(ValueError, match="^some_param must be a dict"):
        _require_deployment_map(True, "some_param")


def test_the_shared_accessor_returns_a_usable_map():
    """None becomes {} so callers keep the ``or {}`` behaviour for the legal
    absent case; a real dict is returned unchanged rather than copied."""
    assert _require_deployment_map(None, "p") == {}
    given = {DEP: {"logs_tail": 1}}
    assert _require_deployment_map(given, "p") is given
