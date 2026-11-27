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
CI image, so importing it is not an option either.

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
import pathlib
from typing import Any, Dict, List, Optional

import pytest


def _warm_child_script() -> str:
    """``runtime._WARM_CHILD_SCRIPT`` read as TEXT, without importing runtime.

    This module's contract above is "no Ray, no network, no GPU, no bioimageio".
    ``import runtime`` broke that: runtime.py imports bioengine, ray and
    bioimageio at module scope, so collecting this file ERRORED wherever those
    are absent — which is every plain checkout, and which reads as the subject
    failing rather than the harness being short. Parsing the assignment out of
    the source keeps the stated contract true and keeps the string the SHIPPED
    one rather than a copy.
    """
    source = (pathlib.Path(__file__).resolve().parent / "runtime.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "_WARM_CHILD_SCRIPT" for t in node.targets
        ):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
    raise AssertionError("_WARM_CHILD_SCRIPT is not a top-level string assignment")


SCRIPT: str = _warm_child_script()


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


def test_a_fixed_extent_model_is_not_gated_behind_cellpose() -> None:
    """The regression itself — stated as a REQUIREMENT, not as a syntax shape.

    The pre-fix guard was ``cellpose and _fits_whole(...)``, which sent every
    non-Cellpose model that already fitted down a padding path. What must be
    true is that a model with FIXED space extents reaches the skip pair without
    being Cellpose.

    Asserting ``not isinstance(branch.test, ast.BoolOp)`` — no conjunct at all —
    is wrong twice over: it pins the syntax rather than the behaviour, and it
    goes RED on the CORRECT narrower guard, which must exclude parameterized
    non-Cellpose models and therefore needs a conjunct. So: assert what the
    guard must ADMIT and what it must EXCLUDE, and let its shape be whatever
    expresses that.
    """
    branch = _fits_whole_branch(SCRIPT)
    src = ast.get_source_segment(SCRIPT, branch.test) or ""

    # It must still consult _fits_whole ...
    assert "_fits_whole" in src, src
    # ... and must NOT be reachable only for Cellpose. A bare `cellpose and`
    # conjunction over the whole test is the defect; `cellpose or <fixed>` is
    # the admitted narrowing.
    if isinstance(branch.test, ast.BoolOp):
        assert isinstance(branch.test.op, ast.And), src
        # Every conjunct that mentions cellpose must itself be a disjunction,
        # so cellpose can only ever WIDEN the guard, never be required.
        for part in branch.test.values:
            if "cellpose" in _names_in(part):
                assert isinstance(part, ast.BoolOp) and isinstance(part.op, ast.Or), (
                    "`cellpose` appears as a required conjunct (%s), which is the "
                    "pre-fix defect: a non-Cellpose model that already fits would "
                    "be padded anyway" % (ast.get_source_segment(SCRIPT, part),)
                )
    else:
        assert "cellpose" not in _names_in(branch.test), src


def test_the_guard_excludes_parameterized_non_cellpose_models() -> None:
    """The narrowing is REQUIRED, not optional.

    ``_fits_whole`` also returns True for a ParameterizedSize axis that is
    on-grid and <= DINO_BLOCK_SIZE. For such a model the padding is the spec's
    own behaviour — core pads by the halo, the model sees that context, and the
    halo is cropped back off the output. Skipping both flags there returns
    uncropped border pixels the spec marks invalid, silently changing results
    for models that already worked.

    So the guard must consult something beyond ``_fits_whole`` that is false for
    a parameterized model. Goes RED on the over-broad ``elif _fits_whole(...)``
    alone.
    """
    branch = _fits_whole_branch(SCRIPT)
    src = ast.get_source_segment(SCRIPT, branch.test) or ""
    assert "_all_space_axes_fixed" in src or "cellpose" in src, (
        "the guard is `%s`: nothing in it can distinguish a FIXED extent from a "
        "parameterized one, so a parameterized non-Cellpose model that fits "
        "whole would lose its halo context and return invalid border pixels" % src
    )
    # And the helper it leans on must actually reject a non-int extent.
    assert "def _all_space_axes_fixed" in SCRIPT, "helper is missing from the script"


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


#: Helpers the loaded predicates call. Exec'd from the SHIPPED source into the
#: same namespace rather than stubbed, so a test cannot pass against a guard
#: that only exists in this file.
_SHIPPED_HELPERS = ("_axes_are_objects",)


def _source_of(name: str, required: bool = True) -> Optional[str]:
    """The source of one top-level def, out of the REAL shipped script.

    ``required=False`` returns None instead of raising. That exists so a HELPER
    which is absent from the script under test is SKIPPED rather than turned
    into an AssertionError: run these tests against a build that predates the
    guard and the predicate must then fail with its own real defect — the
    AttributeError on ``ipt.id`` — not with "_axes_are_objects is not a
    top-level def". A red for the wrong reason is indistinguishable from a red
    for the right one, and it is the wrong reason that would disappear first.
    """
    tree = ast.parse(SCRIPT)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(SCRIPT, node) or ""
    if required:
        raise AssertionError("%s is not a top-level def in the child script" % name)
    return None


def _load_fits_whole(dino_block_size: int = 384):
    """Exec ``_fits_whole`` out of the REAL shipped source, with stubs.

    Deliberately not a copy of the function: a copy would keep passing after the
    shipped predicate changed underneath it.
    """
    namespace: Dict[str, Any] = {
        "ParameterizedSize": _FakeParameterizedSize,
        "DINO_BLOCK_SIZE": dino_block_size,
    }
    for helper in _SHIPPED_HELPERS:
        src = _source_of(helper, required=False)
        if src:
            exec(compile(src, "<_WARM_CHILD_SCRIPT>", "exec"), namespace)
    exec(compile(_source_of("_fits_whole"), "<_WARM_CHILD_SCRIPT>", "exec"), namespace)
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


def test_a_parameterized_axis_fits_only_on_the_declared_grid_and_under_the_block() -> (
    None
):
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


# --- the three cases a broad skip-padding guard gets wrong -------------------


def _load_fn(name: str, dino_block_size: int = 384):
    """Exec one named top-level def out of the REAL shipped source, with stubs."""
    namespace: Dict[str, Any] = {
        "ParameterizedSize": _FakeParameterizedSize,
        "DINO_BLOCK_SIZE": dino_block_size,
        "Optional": Any,
    }
    for helper in _SHIPPED_HELPERS:
        if helper != name:
            src = _source_of(helper, required=False)
            if src:
                exec(compile(src, "<_WARM_CHILD_SCRIPT>", "exec"), namespace)
    exec(compile(_source_of(name), "<_WARM_CHILD_SCRIPT>", "exec"), namespace)
    return namespace[name]


def _fixed_descr(size: int = 256) -> _Descr:
    return _Descr([_Input("input0", [_Axis("y", size), _Axis("x", size)])])


def _param_descr(min_: int = 64, step: int = 16) -> _Descr:
    return _Descr(
        [
            _Input(
                "input0",
                [
                    _Axis("y", _FakeParameterizedSize(min_, step)),
                    _Axis("x", _FakeParameterizedSize(min_, step)),
                ],
            )
        ]
    )


# round 1 finding 2 — the predicate that narrows the skip guard


def test_all_space_axes_fixed_is_true_for_a_fixed_extent_model() -> None:
    assert _load_fn("_all_space_axes_fixed")(_fixed_descr(256)) is True


def test_all_space_axes_fixed_is_FALSE_for_a_parameterized_model() -> None:
    """A parameterized model must keep the padding path: there the halo is
    context the model is meant to see and is cropped back off the output."""
    assert _load_fn("_all_space_axes_fixed")(_param_descr()) is False


def test_all_space_axes_fixed_ignores_non_space_axes() -> None:
    descr = _Descr(
        [
            _Input(
                "input0",
                [
                    _Axis("batch", None, axis_type="batch"),
                    _Axis("channel", None, axis_type="channel"),
                    _Axis("y", 256),
                    _Axis("x", 256),
                ],
            )
        ]
    )
    assert _load_fn("_all_space_axes_fixed")(descr) is True


def test_a_mixed_model_with_one_parameterized_axis_is_not_all_fixed() -> None:
    descr = _Descr(
        [
            _Input(
                "input0",
                [
                    _Axis("y", 256),
                    _Axis("x", _FakeParameterizedSize(64, 16)),
                ],
            )
        ]
    )
    assert _load_fn("_all_space_axes_fixed")(descr) is False


# round 2 findings A and B — LARGER tiles, only SMALLER is refused


def test_an_input_LARGER_than_a_fixed_extent_is_NOT_refused() -> None:
    """Round 2 finding A, the regression that mattered most.

    Refusing any mismatch made Cellpose-SAM — fixed y=256, x=256 — reject every
    image over 256 px, because the refusal ran before the blocking branch that
    had always tiled it. And finding B showed the stated reason was false:
    bioimageio.core carves the halo OUT of the block (block_meta.py:
    inner_size = block_shape - sum(halo)), so a fixed-256 model is fed 256-px
    tiles. Larger inputs tile; they must never be refused.
    """
    fn = _load_fn("_fixed_extent_too_small")
    assert fn(_fixed_descr(256), _sample(512, 512)) is None
    assert fn(_fixed_descr(256), _sample(1024, 1024)) is None


def test_an_input_SMALLER_than_a_fixed_extent_IS_refused_with_the_remedy() -> None:
    """The one direction that genuinely cannot be rescued: nothing to tile up to."""
    msg = _load_fn("_fixed_extent_too_small")(_fixed_descr(256), _sample(128, 128))
    assert msg, "a 128x128 input to a fixed-256 model was not refused"
    assert "256" in msg and "128" in msg, msg
    assert "576" not in msg, "name what the CALLER supplied, not a padded figure"


def test_the_refusal_message_no_longer_claims_a_fixed_model_cannot_be_tiled() -> None:
    """Round 2 finding B: that sentence was untrue, and acting on it caused A."""
    msg = _load_fn("_fixed_extent_too_small")(_fixed_descr(256), _sample(128, 128))
    assert "cannot be tiled" not in msg, msg
    assert "tiled automatically" in msg or "larger input is tiled" in msg, (
        "the message should say a larger input IS tiled, so the caller learns "
        "the real rule rather than a false one. Got: %s" % msg
    )


def test_an_exactly_sized_input_is_NOT_refused() -> None:
    assert (
        _load_fn("_fixed_extent_too_small")(_fixed_descr(256), _sample(256, 256))
        is None
    )


def test_a_parameterized_model_is_never_refused_by_the_too_small_check() -> None:
    fn = _load_fn("_fixed_extent_too_small")
    assert fn(_param_descr(), _sample(512, 512)) is None
    assert fn(_param_descr(), _sample(16, 16)) is None


# round 2 finding C — the refusal must RAISE, and must sit AFTER the cellpose branch


def test_the_refusal_is_reached_only_AFTER_the_cellpose_branch() -> None:
    """The refusal sits after the Cellpose branch, pinned structurally.

    The refusal must NOT run before the dispatch: a Cellpose fixed-extent model
    with a large input has to reach ``elif cellpose:`` and tile. Goes RED if the
    raise is hoisted above ``if default_blocksize_parameter:``.
    """
    # Anchor on the CALL, not the def: the bare name also matches
    # `def _fixed_extent_too_small(...)` itself, which sits above the dispatch,
    # so an assertion keyed on it would hold even with the call deleted.
    CALL = "_fixed_extent_too_small(pipeline.model_description"
    chain_src = SCRIPT[SCRIPT.index("if default_blocksize_parameter:") :]
    assert CALL in chain_src, "the too-small check is never called in the dispatch"
    refusal_at = chain_src.index(CALL)
    cellpose_at = chain_src.index("elif cellpose:")
    assert refusal_at > cellpose_at, (
        "the too-small refusal is evaluated before `elif cellpose:`, so a "
        "fixed-extent Cellpose model with a large input would be refused "
        "instead of tiled"
    )
    # and nothing may refuse before the dispatch at all
    before = SCRIPT[: SCRIPT.index("if default_blocksize_parameter:")]
    assert (
        CALL not in before
    ), "a refusal before the dispatch bypasses every tiling branch"


def test_the_refusal_actually_RAISES() -> None:
    """Round 2 finding C, mutation 1: keeping the call and deleting the `raise`
    passed 21/21, because the old test only checked where the call sat. Pin the
    raise itself, in the same branch as the call."""
    tree = ast.parse(SCRIPT)
    raises: List[ast.Raise] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        src = ast.get_source_segment(SCRIPT, node) or ""
        if "_fixed_extent_too_small(" in src:
            raises += [n for n in ast.walk(node) if isinstance(n, ast.Raise)]
    assert raises, (
        "no `raise` anywhere in the branch that computes the too-small message: "
        "the refusal is computed and discarded"
    )


def test_a_fixed_non_cellpose_model_that_does_not_fit_goes_to_BLOCKING() -> None:
    """Finding A's positive half: larger-than-declared must TILE, not pad.

    Without this branch the input fell to the final `else`, which pads and lets
    the model refuse the padded size — the original 576 defect, one size up.
    """
    # Scope to the ONE chain link whose body holds the refusal, and read only
    # that link's BODY. Walking every ``ast.If`` and taking the first whose
    # SOURCE contains the call is not enough: the outermost If of an if/elif
    # chain spans the whole chain, so it also sees
    # `predict_sample_with_blocking` from the blocksize and cellpose branches
    # and holds even if THIS branch stops tiling — a mutation swapping this
    # branch to predict_sample_without_blocking stays green against it.
    link = None
    for node in ast.walk(ast.parse(SCRIPT)):
        if not isinstance(node, ast.If):
            continue
        body_src = "\n".join(
            (ast.get_source_segment(SCRIPT, stmt) or "") for stmt in node.body
        )
        if "_fixed_extent_too_small(" in body_src:
            link = body_src
            break
    assert link is not None, "no chain link's BODY computes _fixed_extent_too_small"
    assert "predict_sample_with_blocking" in link, (
        "the fixed-extent branch does not tile; it must, or a larger input is "
        "padded and the model refuses the padded size. Branch body:\n%s" % link[:500]
    )
    assert "predict_sample_without_blocking" not in link, (
        "the fixed-extent branch still calls the padding path. Branch body:\n%s"
        % link[:500]
    )


# --- round 3 finding D: pin the chain's CONDITIONS, not just its bodies ------
#
# Rounds 1, 2 and 3 each refused a version of the same gap: the tests pinned
# where code sat rather than what it decides. Finding D is that gap one level
# over -- the new fixed-extent branch was located by its BODY (the
# _fixed_extent_too_small call) and its TEST was never read, so all three of
#     elif False:
#     elif not _all_space_axes_fixed(...):
#     elif cellpose and _all_space_axes_fixed(...):
# were GREEN 25/25 while a 512x512 input to a fixed-256 model fell to the final
# else, got padded to 576 and was refused -- the outcome body DONE WHEN 2
# forbids.
#
# So stop asserting shapes. Evaluate the chain's real conditions against
# synthetic states and assert WHICH BRANCH each state reaches.


def _dispatch_chain():
    """The (condition_source, branch_label) pairs of the real dispatch chain."""
    target = None
    for node in ast.walk(ast.parse(SCRIPT)):
        if isinstance(node, ast.If):
            src = ast.get_source_segment(SCRIPT, node) or ""
            if "default_blocksize_parameter" in src and "predict_sample" in src:
                target = node
                break
    assert target is not None, "could not find the dispatch chain"

    def label(stmts) -> str:
        body = "\n".join((ast.get_source_segment(SCRIPT, s) or "") for s in stmts)
        if "skip_input_padding" in body:
            return "skip-padding"
        if "_block_size_ns" in body:
            return "cellpose-blocking"
        if "_fixed_extent_too_small" in body:
            return "fixed-blocking"
        if "predict_sample_with_blocking" in body:
            return "blocksize-blocking"
        if "predict_sample_without_blocking" in body:
            return "final-else"
        return "unknown"

    links: List[Any] = []
    node = target
    while True:
        links.append(
            (ast.get_source_segment(SCRIPT, node.test) or "", label(node.body))
        )
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
            continue
        if node.orelse:
            links.append((None, label(node.orelse)))
        break
    return links


def _selected_branch(
    cellpose: bool, fits: bool, all_fixed: bool, blocksize: Any = None
) -> str:
    """Evaluate the REAL conditions in order; return the branch that fires."""
    env: Dict[str, Any] = {
        "cellpose": cellpose,
        "default_blocksize_parameter": blocksize,
        "_fits_whole": lambda *a, **k: fits,
        "_all_space_axes_fixed": lambda *a, **k: all_fixed,
        "pipeline": type("P", (), {"model_description": object()})(),
        "sample": object(),
    }
    for cond, branch in _dispatch_chain():
        if cond is None:  # the final else
            return branch
        if eval(cond, dict(env)):  # noqa: S307 - the condition is our own source
            return branch
    return "none"


def test_the_dispatch_chain_is_the_shape_we_think() -> None:
    """Control: without this, a chain that lost a branch would still 'pass' the
    table below by routing everything somewhere plausible."""
    labels = [b for _, b in _dispatch_chain()]
    assert labels[0] == "blocksize-blocking", labels
    assert "skip-padding" in labels, labels
    assert "cellpose-blocking" in labels, labels
    assert "fixed-blocking" in labels, labels
    assert labels[-1] == "final-else", labels
    assert "unknown" not in labels, labels


def test_a_fixed_non_cellpose_model_at_512_reaches_BLOCKING() -> None:
    """Body DONE WHEN 2, as a routing requirement.

    RED under all three of finding D's mutations: `elif False`,
    `elif not _all_space_axes_fixed(...)` and
    `elif cellpose and _all_space_axes_fixed(...)` each send this state to the
    final else, where it is padded to 576 and refused.
    """
    assert (
        _selected_branch(cellpose=False, fits=False, all_fixed=True) == "fixed-blocking"
    )


def test_a_parameterized_non_cellpose_model_at_512_keeps_the_PADDING_path() -> None:
    """Round 1 finding 2, as a routing requirement: this state must NOT be
    pulled into the fixed branch. RED under `elif not _all_space_axes_fixed(...)`."""
    assert _selected_branch(cellpose=False, fits=False, all_fixed=False) == "final-else"


def test_a_fixed_CELLPOSE_model_at_512_still_tiles_on_the_cellpose_branch() -> None:
    """Round 2 finding A: Cellpose-SAM at 512 must reach `elif cellpose:`."""
    assert (
        _selected_branch(cellpose=True, fits=False, all_fixed=True)
        == "cellpose-blocking"
    )


def test_an_exact_fit_skips_padding_for_both_fixed_and_cellpose() -> None:
    """The original fix, as a routing requirement."""
    assert _selected_branch(cellpose=False, fits=True, all_fixed=True) == "skip-padding"
    assert _selected_branch(cellpose=True, fits=True, all_fixed=False) == "skip-padding"


def test_a_parameterized_model_that_fits_does_NOT_skip_padding() -> None:
    """Round 1 finding 2's core: fits-whole alone must not reach the skip."""
    assert _selected_branch(cellpose=False, fits=True, all_fixed=False) == "final-else"


def test_an_explicit_blocksize_wins_over_everything() -> None:
    """The first link must still short-circuit, whatever the other flags say."""
    assert (
        _selected_branch(cellpose=True, fits=True, all_fixed=True, blocksize=16)
        == "blocksize-blocking"
    )


# --- spec 0.4 descriptors, which every predicate here must DECLINE -----------
#
# A 0.4 input is shaped differently from a 0.5 one, and the difference is not
# cosmetic: the predicates in this module read ``ipt.id`` and iterate
# ``ipt.axes`` as axis OBJECTS. A 0.4 input has ``name`` instead, its ``axes``
# is a STRING, and its extents live in a separate ``shape``. Measured on the
# real zoo model conscientious-seashell (format_version 0.4.10):
#     name='input0'   axes="bcyx"   shape=[1, 3, 512, 512]   and no `id`
# against playful-scorpion (0.5.12), which has `id` and a list of axis objects.


class _V04Input:
    def __init__(
        self,
        name: str = "input0",
        axes: str = "bcyx",
        shape: Any = (1, 3, 512, 512),
    ) -> None:
        self.name = name
        self.axes = axes  # a STRING, exactly as 0.4 declares it
        self.shape = list(shape)
        # No `id` attribute, deliberately: that absence IS the defect's trigger.


def _v04_descr() -> _Descr:
    return _Descr([_V04Input()])


def test_the_spec_04_fake_really_lacks_what_the_predicates_read() -> None:
    """Control for the three below: assert the fake is actually 0.4-shaped.

    Without this, a fake that accidentally grew an ``id`` would make the three
    tests pass for the wrong reason — they would be exercising a 0.5 input.
    """
    ipt = _V04Input()
    assert not hasattr(ipt, "id"), "the fake must not carry id; 0.4 inputs do not"
    assert isinstance(ipt.axes, str), "0.4 axes is a string such as 'bcyx'"
    assert ipt.name == "input0"


def test_fits_whole_declines_a_spec_04_descriptor() -> None:
    """RED before the guard: raises AttributeError instead of answering.

    ``_fits_whole`` dereferences ``ipt.id`` on the first line of its loop. Once
    the dispatch guard stopped short-circuiting it behind ``cellpose``, every
    non-Cellpose model reached it, and a 0.4 model crashed there with
    ``AttributeError: 'InputTensorDescr' object has no attribute 'id'``.
    False is the answer the function's own comment prescribes for a shape it
    cannot reason about: keep the blocked path.
    """
    fits_whole = _load_fits_whole()
    assert fits_whole(_v04_descr(), _sample(512, 512)) is False


def test_all_space_axes_fixed_declines_a_spec_04_descriptor() -> None:
    """RED before the guard, and silently so — the worse of the two.

    This loop does not raise on a 0.4 descriptor: it iterates the string
    "bcyx" as CHARACTERS, none of which has ``.type``, skips every one, and
    returns True — claiming every space axis is fixed for a model whose axes it
    never read. That routes a 0.4 model into the fixed-extent branch.
    """
    all_fixed = _load_fn("_all_space_axes_fixed")
    assert all_fixed(_v04_descr()) is False


def test_the_too_small_message_declines_a_spec_04_descriptor() -> None:
    """There is no actionable sentence to offer about axes we cannot read."""
    too_small = _load_fn("_fixed_extent_too_small")
    assert too_small(_v04_descr(), _sample(512, 512)) is None


def _branch_with_real_predicates(
    descr: Any, sample: Any, cellpose: bool = False
) -> str:
    """Evaluate the REAL chain conditions with the REAL predicates loaded from the script.

    ``_selected_branch`` injects booleans, which pins the chain's SHAPE but cannot show what a
    particular descriptor does to it. This drives the shipped predicates themselves.
    """
    env: Dict[str, Any] = {
        "cellpose": cellpose,
        "default_blocksize_parameter": None,
        "_fits_whole": _load_fits_whole(),
        "_all_space_axes_fixed": _load_fn("_all_space_axes_fixed"),
        "pipeline": type("P", (), {"model_description": descr})(),
        "sample": sample,
    }
    for cond, branch in _dispatch_chain():
        if cond is None:
            return branch
        if eval(cond, dict(env)):  # noqa: S307 - the condition is our own source
            return branch
    return "none"


def test_a_spec_04_descriptor_reaches_the_final_else_through_the_REAL_chain() -> None:
    """The restored path, driven rather than reasoned about.

    The other 0.4 tests assert each predicate in isolation; this one feeds a 0.4 descriptor to the
    real predicates through the real chain conditions and names the branch it lands on. That is the
    branch an unguarded build cannot reach, because it raises before deciding.
    """
    assert _branch_with_real_predicates(_v04_descr(), _sample(512, 512)) == "final-else"


def test_a_real_05_descriptor_still_reaches_skip_padding_through_the_REAL_chain() -> (
    None
):
    """The paired control: the guard must not quietly disable the skip path for 0.5.

    Without this, a guard that declined EVERYTHING would pass every 0.4 test above.
    """
    assert (
        _branch_with_real_predicates(_fixed_256_model(), _sample(256, 256))
        == "skip-padding"
    )


class _IdButStringAxes:
    """Has ``id``, but ``axes`` is a string. Isolates the str check in the guard."""

    def __init__(self) -> None:
        self.id = "input0"
        self.axes = "bcyx"


class _IdAndListButAxisHasNoType:
    """Has ``id`` and a LIST of axes, but the axis objects carry no ``.type``."""

    def __init__(self) -> None:
        self.id = "input0"
        self.axes = [object(), object()]


def test_the_guard_rejects_string_axes_even_when_id_is_present() -> None:
    """A non-empty string-axes input is rejected even when ``id`` is present.

    This pins the OUTCOME, not one arm. Measured by mutation: deleting the
    ``isinstance(axes, str)`` arm leaves THIS case green, because iterating "bcyx" yields
    characters and the ``hasattr(axis, "type")`` arm below rejects them. The arm that this case
    cannot pin is pinned by ``test_the_guard_rejects_EMPTY_string_axes`` below, which is the input
    that does separate them.
    """
    guard = _load_fn("_axes_are_objects")
    assert guard(_Descr([_IdButStringAxes()])) is False


def test_the_guard_rejects_axis_objects_without_a_type_attribute() -> None:
    """Isolates the last check, for the same reason as the one above."""
    guard = _load_fn("_axes_are_objects")
    assert guard(_Descr([_IdAndListButAxisHasNoType()])) is False


def test_the_guard_accepts_a_real_05_shaped_descriptor() -> None:
    """The positive arm: a guard that rejected everything would pass all four tests above."""
    guard = _load_fn("_axes_are_objects")
    assert guard(_fixed_256_model()) is True


class _IdButEmptyStringAxes:
    """Has ``id``, and ``axes`` is the EMPTY string. The one input that separates the guard's arms."""

    def __init__(self) -> None:
        self.id = "input0"
        self.axes = ""


def test_the_guard_rejects_EMPTY_string_axes() -> None:
    """The empty string is why the ``isinstance(axes, str)`` arm is load-bearing.

    For "bcyx" the ``hasattr(axis, "type")`` loop below would also reject. For "" it would not:
    there are no characters, the loop body never runs, and without the str arm the guard answers
    True — reporting that a 0.4 input carries 0.5 axis objects. A vacuous truth, inside a predicate
    written to stop exactly that class of mistake.

    Goes RED with the str arm deleted; "bcyx" alone does not.
    """
    guard = _load_fn("_axes_are_objects")
    assert guard(_Descr([_IdButEmptyStringAxes()])) is False
