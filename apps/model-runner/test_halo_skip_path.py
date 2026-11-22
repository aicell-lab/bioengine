"""An input that already fits the model must not be halo-padded.

The runtime used to gate the skip-padding branch on ``cellpose and
_fits_whole(...)``. Every *other* model whose input already fits fell through to
the final ``else``, which calls ``predict_sample_without_blocking`` WITHOUT the
skip flags and therefore still pads the outer boundary by the declared halo. For
a model with **fixed** space extents that is fatal rather than merely wasteful: a
256x256 input to a fixed-256 model becomes 320x320 under a halo of 32 and the
model refuses the tensor outright, so six Zoo models were unreachable from every
client — no crop, no pad and no declared-size input could get in.

The fix drops the ``cellpose`` conjunct so ``_fits_whole`` gates that branch on
its own.

## Why these cases are structural rather than behavioural

The dispatch is **not importable code**. It lives inside
``runtime._WARM_CHILD_SCRIPT`` — a ~10.8 KB source string that is written out and
run as a separate process — and inside that script it sits inline in a ``for``
loop rather than in a function, so there is no callable to drive. The script's
own imports (``bioimageio.core``, ``bioimageio.spec``) are also absent from the
CI image, which is why importing it is not an option either.

So the branch itself is pinned by parsing the shipped source and asserting
*about the one site*, via ``ast.get_source_segment``. A whole-file grep cannot do
this job: it cannot tell "the guard at this site is ungated" from "the words
``cellpose`` and ``_fits_whole`` both appear somewhere in this file", and both
readings are true of the pre-fix and post-fix text alike.

``_fits_whole`` *is* reachable — it is a self-contained function in that script —
so it is exec'd out of the real shipped source (never a copy pasted into this
file) and driven with fakes. That half is behavioural.

## The discrimination control

``test_restoring_the_pre_fix_conjunction_turns_the_guard_RED`` re-introduces
``cellpose and`` into a copy of the shipped source and asserts the guard
assertion fires. Without it, every structural case here would also pass against
a file in which the dispatch had been deleted outright, and "no assertion
failed" would be indistinguishable from "nothing was checked". It asserts the
substitution actually changed the text first, so the control cannot pass
vacuously.

Plain pytest: no Ray, no network, no GPU, no bioimageio.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

import pytest

import runtime

SCRIPT: str = runtime._WARM_CHILD_SCRIPT


# --- locating the one dispatch site -------------------------------------------


def _fits_whole_branch(script: str) -> ast.If:
    """The single ``If`` in the child script whose test calls ``_fits_whole``.

    Asserting there is exactly one is part of the guard, not a convenience: a
    second site would make "the guard is ungated" ambiguous about which guard.
    """
    tree = ast.parse(script)
    hits: List[ast.If] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            segment = ast.get_source_segment(script, node.test) or ""
            if "_fits_whole" in segment:
                hits.append(node)
    assert len(hits) == 1, (
        "expected exactly one dispatch site whose test calls _fits_whole, "
        "found %d" % len(hits)
    )
    return hits[0]


def _elif_chain(node: ast.If) -> List[ast.If]:
    """``[node, <elif>, <elif>, ...]`` following the ``orelse`` spine."""
    chain = [node]
    current = node
    while len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
        current = current.orelse[0]
        chain.append(current)
    return chain


def _keywords_of_call(body: List[ast.stmt], func_name: str) -> Dict[str, Any]:
    """Keyword names -> constant value (or ``...``) for a call in ``body``."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call):
            attr = node.func
            name = getattr(attr, "attr", None) or getattr(attr, "id", None)
            if name == func_name:
                out: Dict[str, Any] = {}
                for kw in node.keywords:
                    if kw.arg is None:
                        out["**"] = ...
                    elif isinstance(kw.value, ast.Constant):
                        out[kw.arg] = kw.value.value
                    else:
                        out[kw.arg] = ...
                return out
    raise AssertionError("no call to %s() in this branch" % func_name)


def _names_in(node: ast.AST) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


# --- the structural guard -----------------------------------------------------


def test_exactly_one_dispatch_site_tests_fits_whole() -> None:
    assert isinstance(_fits_whole_branch(SCRIPT), ast.If)


def test_the_fits_whole_guard_is_not_conjoined_with_cellpose() -> None:
    """The regression itself: the guard must be the predicate alone."""
    branch = _fits_whole_branch(SCRIPT)
    assert not isinstance(branch.test, ast.BoolOp), (
        "the _fits_whole guard is a boolean combination (%s), so some other "
        "condition can still send a fits-whole input down a padding path"
        % (ast.get_source_segment(SCRIPT, branch.test),)
    )
    assert isinstance(branch.test, ast.Call)
    assert "cellpose" not in _names_in(branch.test), (
        "the _fits_whole guard references `cellpose`, which is the pre-fix "
        "defect: non-Cellpose models that already fit get padded anyway"
    )


def test_the_fits_whole_branch_is_reached_before_the_cellpose_branch() -> None:
    """Order is load-bearing, and it is what makes clause 2 hold.

    A Cellpose model whose input fits whole must still take the skip path. It
    does so *because* this branch precedes ``elif cellpose:`` — if the two were
    swapped, every Cellpose model would be blocked-and-padded again and the
    clause would regress while every other case here still passed.
    """
    chain = _elif_chain(_fits_whole_branch(SCRIPT))
    tests = [ast.get_source_segment(SCRIPT, link.test) or "" for link in chain]
    assert any(t.strip() == "cellpose" for t in tests[1:]), (
        "expected a later `elif cellpose:` in the same chain; chain is %r" % tests
    )


def test_the_fits_whole_branch_requests_both_skips() -> None:
    """Both flags, or the branch still pads (or crops a halo never added)."""
    branch = _fits_whole_branch(SCRIPT)
    kwargs = _keywords_of_call(branch.body, "predict_sample_without_blocking")
    assert kwargs.get("skip_input_padding") is True, kwargs
    assert kwargs.get("skip_output_cropping") is True, kwargs


def test_the_cellpose_blocking_branch_still_passes_computed_block_sizes() -> None:
    """The control: the fix must not disturb a Cellpose model that does NOT fit.

    Without this, deleting the ``elif cellpose:`` branch entirely would leave
    every other case in this module green.
    """
    chain = _elif_chain(_fits_whole_branch(SCRIPT))
    cellpose_links = [
        link
        for link in chain[1:]
        if (ast.get_source_segment(SCRIPT, link.test) or "").strip() == "cellpose"
    ]
    assert cellpose_links, "no `elif cellpose:` branch left in the chain"
    kwargs = _keywords_of_call(cellpose_links[0].body, "predict_sample_with_blocking")
    assert "ns" in kwargs, (
        "the Cellpose blocking branch no longer passes computed block sizes: %r"
        % kwargs
    )


def test_restoring_the_pre_fix_conjunction_turns_the_guard_RED() -> None:
    """Discrimination control — see the module docstring."""
    needle = "elif _fits_whole("
    assert needle in SCRIPT, "the shipped source no longer matches %r" % needle
    pre_fix = SCRIPT.replace(needle, "elif cellpose and _fits_whole(", 1)
    assert pre_fix != SCRIPT, "the mutation did not change the source"

    branch = _fits_whole_branch(pre_fix)
    assert isinstance(branch.test, ast.BoolOp), "mutation did not take effect"
    assert "cellpose" in _names_in(branch.test)


# --- the behavioural half: _fits_whole itself ---------------------------------


class _FakeParameterizedSize:
    def __init__(self, min_: int, step: int) -> None:
        self.min = min_
        self.step = step


def _load_fits_whole(dino_block_size: int = 384):
    """Exec ``_fits_whole`` out of the REAL shipped source, with stubs.

    Deliberately not a copy of the function: a copy would keep passing after the
    shipped predicate changed underneath it.
    """
    tree = ast.parse(SCRIPT)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_fits_whole":
            source = ast.get_source_segment(SCRIPT, node)
            break
    else:  # pragma: no cover - the guard below is the real assertion
        raise AssertionError("_fits_whole is not a top-level def in the child script")

    namespace: Dict[str, Any] = {
        "ParameterizedSize": _FakeParameterizedSize,
        "DINO_BLOCK_SIZE": dino_block_size,
    }
    exec(compile(source, "<_WARM_CHILD_SCRIPT>", "exec"), namespace)
    return namespace["_fits_whole"]


class _Axis:
    def __init__(self, axis_id: str, size: Any, axis_type: str = "space") -> None:
        self.id = axis_id
        self.size = size
        self.type = axis_type


class _Input:
    def __init__(self, input_id: str, axes: List[_Axis]) -> None:
        self.id = input_id
        self.axes = axes


class _Descr:
    def __init__(self, inputs: List[_Input]) -> None:
        self.inputs = inputs


class _Member:
    def __init__(self, sizes: Dict[str, int]) -> None:
        self.sizes = sizes


class _Sample:
    def __init__(self, members: Dict[str, _Member]) -> None:
        self.members = members


def _fixed_256_model() -> _Descr:
    return _Descr([_Input("input0", [_Axis("y", 256), _Axis("x", 256)])])


def _sample(y: int, x: int, input_id: str = "input0") -> _Sample:
    return _Sample({input_id: _Member({"y": y, "x": x})})


def test_an_exactly_sized_input_to_a_fixed_extent_model_fits_whole() -> None:
    """The six unreachable Zoo models, at the predicate level."""
    fits_whole = _load_fits_whole()
    assert fits_whole(_fixed_256_model(), _sample(256, 256)) is True


def test_a_mismatched_input_to_a_fixed_extent_model_does_not_fit_whole() -> None:
    """The padded 320x320 tensor must NOT be treated as fitting."""
    fits_whole = _load_fits_whole()
    assert fits_whole(_fixed_256_model(), _sample(320, 320)) is False
    assert fits_whole(_fixed_256_model(), _sample(128, 128)) is False


def test_fits_whole_does_not_consult_cellpose_at_all() -> None:
    """Clause 2's other half, stated where it can be measured.

    ``_fits_whole`` takes only a description and a sample, so the predicate is
    the same for a Cellpose model and a non-Cellpose one with the same axes.
    Pinned because the fix's correctness rests on the predicate being general.
    """
    source_tree = ast.parse(SCRIPT)
    for node in source_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_fits_whole":
            assert [a.arg for a in node.args.args] == ["descr", "sample"]
            assert "cellpose" not in _names_in(node)
            break
    else:
        raise AssertionError("_fits_whole is not a top-level def in the child script")


def test_a_missing_member_or_unknown_axis_size_keeps_the_blocked_path() -> None:
    """Anything unreasonable-about must read as 'does not fit'."""
    fits_whole = _load_fits_whole()
    assert fits_whole(_fixed_256_model(), _Sample({})) is False
    assert fits_whole(_fixed_256_model(), _sample(256, 256, "other")) is False

    weird = _Descr([_Input("input0", [_Axis("y", "256"), _Axis("x", "256")])])
    assert fits_whole(weird, _sample(256, 256)) is False

    no_extent = _Sample({"input0": _Member({"y": 256})})
    assert fits_whole(_fixed_256_model(), no_extent) is False


def test_a_parameterized_axis_fits_only_on_the_declared_grid_and_under_the_block(
) -> None:
    fits_whole = _load_fits_whole(dino_block_size=384)
    param = _Descr(
        [
            _Input(
                "input0",
                [
                    _Axis("y", _FakeParameterizedSize(64, 16)),
                    _Axis("x", _FakeParameterizedSize(64, 16)),
                ],
            )
        ]
    )
    assert fits_whole(param, _sample(320, 320)) is True  # on-grid, under block
    assert fits_whole(param, _sample(400, 400)) is False  # over DINO_BLOCK_SIZE
    assert fits_whole(param, _sample(32, 32)) is False  # below min
    assert fits_whole(param, _sample(72, 72)) is False  # off the step grid


def test_a_non_space_axis_is_ignored_by_the_predicate() -> None:
    fits_whole = _load_fits_whole()
    with_channel = _Descr(
        [
            _Input(
                "input0",
                [
                    _Axis("c", 3, axis_type="channel"),
                    _Axis("y", 256),
                    _Axis("x", 256),
                ],
            )
        ]
    )
    assert fits_whole(with_channel, _sample(256, 256)) is True
