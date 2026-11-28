"""#1103 — an input that cannot fit the model is refused BEFORE it is queued.

``infer`` used to queue any array it could decode. The shape mismatch surfaced
only after the model download and the queue wait: measured at 334 s for a 2-D
``.npy`` sent to a 4-axis model, and ``fearless-crab`` returned a bare
``InvalidArgumentError()`` for a step violation. A screening batch is spent
learning something the RDF already states.

The declared sizes are in the RDF, a small YAML beside the weights, so the
answer is available before anything is enqueued.

TWO SPEC FORMATS, and reading one through the other's shape is exactly how
#0958 made six Zoo models unreachable, so both are covered here:

* 0.4 — a STRING ``axes`` ("bcyx") plus a separate ``shape``, either a fixed
  list or ``{"min": [...], "step": [...]}``, where a step of 0 means fixed.
* 0.5 — ``axes`` as a LIST of objects, each with its own ``size``: an int, a
  ``{"min":, "step":}`` pair, or a reference to another tensor's axis.

THE FAIL-OPEN CASES ARE NOT PADDING. A pre-check that refuses on a guess
rejects work the model would have accepted, which is worse than the slow
failure it replaces, so every unreadable shape must answer "queue it".
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

import entry


def _preflight():
    """The helper under test, or a clean failure saying it is absent.

    A bare ``entry._input_fit_error`` raises AttributeError on a tree without
    the fix, which reports as an ERROR rather than a failed assertion — so the
    RED demonstration would go red for the wrong reason.
    """
    fn = getattr(entry, "_input_fit_error", None)
    assert fn is not None, (
        "entry._input_fit_error is absent: infer still queues an input that "
        "cannot fit the model and fails it only after the queue wait (#1103)"
    )
    return fn


def _infer_source() -> str:
    """Source of ``infer``, whether or not ``@bioengine.app`` wrapped the class.

    The decorator may hand back a Deployment rather than the class, in which
    case the class is under ``func_or_class`` — reading the wrong one raises
    AttributeError and the arm would go red for the wrong reason.
    """
    obj = entry.EntryDeployment
    cls = getattr(obj, "func_or_class", obj)
    return inspect.getsource(cls.infer)


def _rdf_05(axes: List[Dict[str, Any]], input_id: str = "input0") -> Dict[str, Any]:
    return {"inputs": [{"id": input_id, "axes": axes}]}


def _rdf_04(axes: str, shape: Any, name: str = "input0") -> Dict[str, Any]:
    return {"inputs": [{"name": name, "axes": axes, "shape": shape}]}


# A 4-axis 0.5 model: batch, channel, then y/x on a 64 + n*16 grid.
_FOUR_AXIS_05 = _rdf_05(
    [
        {"type": "batch", "id": "b"},
        {"type": "channel", "id": "c", "size": 1},
        {"type": "space", "id": "y", "size": {"min": 64, "step": 16}},
        {"type": "space", "id": "x", "size": {"min": 64, "step": 16}},
    ]
)


# --------------------------------------------------------------------------
# Clause 1a: a 2-D input to a 4-axis model. RED today.
# --------------------------------------------------------------------------


def test_a_2d_input_to_a_4_axis_model_is_refused() -> None:
    err = _preflight()(_FOUR_AXIS_05, {entry.SINGLE_INPUT_KEY: np.zeros((256, 256))})
    assert err is not None, "a 2-D array cannot fit a 4-axis model"
    assert "2 axes" in err and "4" in err, err
    # Names what to do, not just that it is wrong.
    assert "batch" in err or "singleton" in err, err


def test_the_axis_count_message_names_the_declared_axes() -> None:
    err = _preflight()(_FOUR_AXIS_05, {entry.SINGLE_INPUT_KEY: np.zeros((256, 256))})
    assert "b,c,y,x" in err, err


# --------------------------------------------------------------------------
# Clause 1b: an off-grid spatial size. RED today.
# --------------------------------------------------------------------------


def test_an_off_grid_size_is_refused_naming_the_axis_and_nearest_valid() -> None:
    # y = 100 is not 64 + n*16 (96 and 112 are the neighbours).
    err = _preflight()(
        _FOUR_AXIS_05, {entry.SINGLE_INPUT_KEY: np.zeros((1, 1, 100, 128))}
    )
    assert err is not None, "100 is not on a 64 + n*16 grid"
    assert "'y'" in err, err
    assert "96" in err and "112" in err, err
    assert "min 64" in err and "step 16" in err, err


def test_a_size_below_the_minimum_is_refused() -> None:
    err = _preflight()(
        _FOUR_AXIS_05, {entry.SINGLE_INPUT_KEY: np.zeros((1, 1, 32, 64))}
    )
    assert err is not None
    assert "'y'" in err and "64" in err, err


def test_an_off_grid_size_is_refused_for_a_0_4_model_too() -> None:
    """0.4 spells the grid as parallel min/step LISTS beside a string axes."""
    rdf = _rdf_04("bcyx", {"min": [1, 1, 64, 64], "step": [0, 0, 16, 16]})
    err = _preflight()(rdf, {entry.SINGLE_INPUT_KEY: np.zeros((1, 1, 100, 64))})
    assert err is not None
    assert "'y'" in err and "96" in err and "112" in err, err


def test_a_0_4_fixed_shape_mismatch_names_the_fixed_size() -> None:
    rdf = _rdf_04("bcyx", [1, 1, 512, 512])
    err = _preflight()(rdf, {entry.SINGLE_INPUT_KEY: np.zeros((1, 1, 256, 512))})
    assert err is not None
    assert "fixed" in err and "512" in err, err


def test_a_0_4_step_of_zero_means_fixed_not_a_free_axis() -> None:
    """step 0 is how 0.4 says "this axis is fixed at min". Treating it as a
    grid would divide by zero; treating it as free would accept anything."""
    rdf = _rdf_04("bcyx", {"min": [1, 3, 64, 64], "step": [0, 0, 16, 16]})
    err = _preflight()(rdf, {entry.SINGLE_INPUT_KEY: np.zeros((1, 1, 64, 64))})
    assert err is not None, "channel is fixed at 3, so 1 cannot fit"
    assert "'c'" in err and "3" in err, err


# --------------------------------------------------------------------------
# Clause 2: a valid input is untouched. Must pass before AND after.
# --------------------------------------------------------------------------


def test_a_valid_0_5_input_is_not_refused() -> None:
    assert (
        _preflight()(
            _FOUR_AXIS_05, {entry.SINGLE_INPUT_KEY: np.zeros((1, 1, 128, 256))}
        )
        is None
    )


def test_a_valid_0_4_input_is_not_refused() -> None:
    rdf = _rdf_04("bcyx", {"min": [1, 1, 64, 64], "step": [0, 0, 16, 16]})
    assert (
        _preflight()(rdf, {entry.SINGLE_INPUT_KEY: np.zeros((1, 1, 96, 112))}) is None
    )


def test_a_valid_0_4_fixed_input_is_not_refused() -> None:
    rdf = _rdf_04("bcyx", [1, 1, 512, 512])
    assert (
        _preflight()(rdf, {entry.SINGLE_INPUT_KEY: np.zeros((1, 1, 512, 512))}) is None
    )


def test_the_minimum_itself_is_valid() -> None:
    assert (
        _preflight()(_FOUR_AXIS_05, {entry.SINGLE_INPUT_KEY: np.zeros((1, 1, 64, 64))})
        is None
    )


# --------------------------------------------------------------------------
# Fail-open. Each of these is a shape the check cannot read, and each must
# answer "queue it" rather than refuse.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rdf, why",
    [
        ({}, "no inputs key at all"),
        ({"inputs": []}, "an empty inputs list"),
        ({"inputs": "nonsense"}, "inputs that is not a list"),
        ({"inputs": [{"id": "input0"}]}, "an input with no axes"),
        ({"inputs": [{"id": "input0", "axes": 7}]}, "axes of an unknown type"),
        (
            _rdf_05(
                [
                    {
                        "type": "space",
                        "id": "y",
                        "size": {"axis_id": "x", "tensor_id": "input0"},
                    },
                ]
            ),
            "a size that REFERENCES another tensor's axis",
        ),
        (
            _rdf_04("bcyx", {"min": [1, 1, 64], "step": [0, 0, 16]}),
            "a 0.4 min list whose length disagrees with the axes string",
        ),
        (_rdf_04("bcyx", [1, 1, 512]), "a 0.4 fixed shape of the wrong length"),
    ],
)
def test_an_unreadable_declaration_queues_rather_than_refuses(rdf, why) -> None:
    # 1-D, so a declaration with ONE axis still matches on count and the only
    # thing left unreadable is the SIZE. With a 2-D array here the axis-count
    # check fires first — it is answerable — and the case would pass for the
    # wrong reason or fail for one.
    got = _preflight()(rdf, {entry.SINGLE_INPUT_KEY: np.zeros((17,))})
    assert got is None, f"must fail open for {why}, got: {got}"


def test_an_input_key_that_matches_no_declared_input_is_left_alone() -> None:
    """Two declared inputs and a key naming neither: this cannot tell which
    spec applies, so it must not invent one."""
    rdf = {
        "inputs": [
            {"id": "raw", "axes": [{"type": "space", "id": "y", "size": 64}]},
            {"id": "mask", "axes": [{"type": "space", "id": "y", "size": 64}]},
        ]
    }
    assert _preflight()(rdf, {"something-else": np.zeros((17,))}) is None


def test_a_named_multi_input_model_is_checked_per_key() -> None:
    rdf = {
        "inputs": [
            {"id": "raw", "axes": [{"type": "space", "id": "y", "size": 64}]},
            {"id": "mask", "axes": [{"type": "space", "id": "y", "size": 32}]},
        ]
    }
    fn = _preflight()
    assert fn(rdf, {"raw": np.zeros((64,)), "mask": np.zeros((32,))}) is None
    err = fn(rdf, {"raw": np.zeros((64,)), "mask": np.zeros((64,))})
    assert err is not None and "'mask'" in err, err


# --------------------------------------------------------------------------
# The wiring: the check must run BEFORE the job is created, and must not be
# able to fail the submit by itself.
# --------------------------------------------------------------------------


def test_the_check_runs_before_the_job_is_queued() -> None:
    src = _infer_source()
    assert "_input_fit_error" in src, "infer does not pre-check the input fit"
    check_at = src.index("_input_fit_error(")
    queue_at = src.index("self._new_infer_job(")
    assert check_at < queue_at, (
        "the fit check must precede _new_infer_job, or the queue wait the row "
        "is about still happens before the refusal"
    )


def test_an_unreachable_rdf_does_not_fail_the_submit() -> None:
    """The RDF fetch is a network call. If it throws, the job must still queue —
    otherwise a transient artifact-store blip becomes a refused inference."""
    src = _infer_source()
    head = src[: src.index("self._new_infer_job(")]
    assert "get_model_rdf" in head
    assert "except Exception" in head, "the RDF fetch must be guarded"
    assert (
        "model_rdf = None" in head
    ), "a failed fetch must disable the check, not raise"
