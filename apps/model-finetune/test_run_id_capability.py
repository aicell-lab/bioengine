"""Which calls a run identifier alone may make, pinned so it cannot drift.

A run identifier is a timestamp plus 32 random bits. It is not guessable, but it
IS a bearer capability: anyone it has been shared with keeps whatever it grants,
for as long as the run exists, with no way to withdraw it. So the question
"which calls does holding an identifier let you make" has to have one written
answer, and this file is where that answer is enforced rather than described.

The line is NOT "touches the weights". It is whether the call HANDS OUT or
CREATES weights:

* **Open, gated by the run identifier alone** -- status, progress, logs,
  `infer` and `compute_embedding`. These return OUTPUTS computed with the
  model. The caller never receives the checkpoint and no new model derives from
  it. A shared notebook session is built for several people to try one run's
  model, and gating inference would break that.
* **Owner or administrator** -- `export_model` hands out the weights,
  `contribute` publishes them into the shared pool, resuming creates a new
  model derived from them, and stopping acts on somebody else's run.

The tests below read `entry.py` itself rather than a list maintained by hand, so
a method added later that takes a run identifier and decides neither way fails
`test_EVERY_run_keyed_method_is_on_exactly_one_side`. That is the point: the
cost of this file is that a new run-keyed method forces a decision.
"""

import ast
import pathlib

import pytest

ENTRY = pathlib.Path(__file__).with_name("entry.py")
SOURCE = ENTRY.read_text()
TREE = ast.parse(SOURCE)

GATE = "require_caller_may_mutate"

# The parameter names by which a call is keyed to somebody's run. `export_id` is
# included because an export handle is minted by `export_model`, which is itself
# owner-or-admin, so holding the handle is already a granted capability.
RUN_KEY_PARAMS = frozenset({"session_id", "resume_session_id", "export_id"})

OPEN_TO_ANY_HOLDER = frozenset(
    {
        "get_training_status",  # what was trained, and how far it got
        "infer",  # an OUTPUT of the model, not the model
        "compute_embedding",  # likewise
    }
)

# NOT a third opinion, a third STATE: ungated today, and nobody has ruled that
# it should be.
#
# An earlier version of this file put these two in OPEN_TO_ANY_HOLDER and
# described an export handle as safe to pass around. That was wrong on the
# facts. `get_export_status` returns the whole export record, and a READY record
# carries `download_url` -- a six-hour presigned link to the packaged weights --
# plus `resume_checkpoint_file`. `push_export` streams the package to any
# presigned URL the caller supplies. So an export handle hands over the
# weights, which is the thing the restricted set exists to protect, and
# freezing it as "open" asserted a decision nobody had taken.
#
# They stay here, matching what the code does today, until that is ruled on.
# Gating them is a behaviour change and belongs with the gate's own row, not
# with a documentation change.
UNGATED_PENDING_A_RULING = frozenset({"get_export_status", "push_export"})

OWNER_OR_ADMIN = frozenset(
    {
        "start_training",  # only when resume_session_id is given: derives a model
        "stop_training",  # acts on somebody else's run
        "export_model",  # hands out the weights
        "contribute",  # publishes them into the shared pool
    }
)


def _public_methods():
    """Every public async method of the app class, with its params and body."""
    out = {}
    for node in ast.walk(TREE):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        params = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
        out[node.name] = (node, params)
    return out


def _calls_the_gate(node) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name == GATE:
                return True
    return False


METHODS = _public_methods()
RUN_KEYED = {
    name: (node, params)
    for name, (node, params) in METHODS.items()
    if params & RUN_KEY_PARAMS
}


def test_the_PARSE_FOUND_SOMETHING_to_begin_with():
    """THE CONTROL FOR THE FILE.

    Every assertion below is of the form "this set equals that set". A parse
    that silently found nothing makes all of them pass against empty sets --
    the same way a scan that never executed prints the same zero as a clean
    one. So assert the raw counts first.
    """
    assert len(METHODS) > 10, f"only {len(METHODS)} public methods parsed; parse broke"
    assert len(RUN_KEYED) >= 8, (
        f"only {len(RUN_KEYED)} run-keyed methods found: {sorted(RUN_KEYED)}. "
        "Either the signatures changed or RUN_KEY_PARAMS is wrong."
    )
    assert GATE in SOURCE, f"{GATE} is not in entry.py at all; the gate was renamed"


def test_EVERY_run_keyed_method_is_on_exactly_one_side():
    """The reason this file exists: no third, undecided category.

    A method that takes a run identifier either is open to any holder of that
    identifier or requires the owner. Adding one without saying which fails
    here, which is the only moment anybody is reliably forced to decide.
    """
    declared = OPEN_TO_ANY_HOLDER | OWNER_OR_ADMIN | UNGATED_PENDING_A_RULING
    undeclared = set(RUN_KEYED) - declared
    assert undeclared == set(), (
        f"these methods take a run identifier and are on neither side: "
        f"{sorted(undeclared)}. Decide, then add each to OPEN_TO_ANY_HOLDER or "
        "OWNER_OR_ADMIN above."
    )
    for a, b in (
        (OPEN_TO_ANY_HOLDER, OWNER_OR_ADMIN),
        (OPEN_TO_ANY_HOLDER, UNGATED_PENDING_A_RULING),
        (OWNER_OR_ADMIN, UNGATED_PENDING_A_RULING),
    ):
        assert not (a & b), "a method cannot be in two of the three sets"


def test_the_declared_sets_do_not_name_methods_that_no_longer_EXIST():
    """The other direction, so a rename cannot leave a set asserting nothing."""
    missing = (OPEN_TO_ANY_HOLDER | OWNER_OR_ADMIN | UNGATED_PENDING_A_RULING) - set(
        METHODS
    )
    assert (
        missing == set()
    ), f"these are declared but are not public methods of the app: {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(OWNER_OR_ADMIN))
def test_an_OWNER_OR_ADMIN_method_actually_carries_the_gate(name):
    node, _params = METHODS[name]
    assert _calls_the_gate(node), (
        f"{name} is declared owner-or-admin but never calls {GATE}. Either the "
        "gate was dropped or this method belongs in OPEN_TO_ANY_HOLDER."
    )


@pytest.mark.parametrize("name", sorted(OPEN_TO_ANY_HOLDER))
def test_an_OPEN_method_does_NOT_carry_the_gate(name):
    """The inverse, and the half that catches an accidental tightening.

    Gating inference is a user-visible break of the shared-notebook flow, and it
    would otherwise be an easy thing to add while fixing something else.
    """
    node, _params = METHODS[name]
    assert not _calls_the_gate(node), (
        f"{name} is open to any holder of the run identifier, but it now calls "
        f"{GATE}. Several people are meant to be able to try one run's model."
    )


def test_INFERENCE_is_open_which_is_the_decision_that_was_actually_taken():
    """Named separately because it is the part a reader will doubt.

    `infer` and `compute_embedding` run the trained model. They are open anyway,
    because what comes back is an output and not the checkpoint, and because
    several people trying one run's model is the shared-notebook flow working as
    intended rather than a leak.
    """
    assert {"infer", "compute_embedding"} <= OPEN_TO_ANY_HOLDER
    for name in ("infer", "compute_embedding"):
        node, _params = METHODS[name]
        assert not _calls_the_gate(node), f"{name} must not require the owner"


def test_the_WEIGHT_HANDING_methods_are_the_gated_ones():
    """The complement of the above, stated as the rule rather than per method."""
    assert {"export_model", "contribute"} <= OWNER_OR_ADMIN
    node, _params = METHODS["start_training"]
    assert _calls_the_gate(node), (
        "resuming derives a new model from somebody else's checkpoint, so "
        "start_training must gate when resume_session_id is given"
    )


def test_the_README_tells_a_reader_that_sharing_the_id_shares_the_run():
    """DONE WHEN 2. The decision is only recorded if a reader can find it.

    Checked as text because that is what the clause asks for -- a reader
    learning, from the app's own documentation, that an identifier grants this
    and cannot be taken back.
    """
    readme = pathlib.Path(__file__).with_name("README.md").read_text().lower()
    for phrase, why in (
        ("cannot be withdrawn", "that the grant is irrevocable"),
        ("run inference", "that inference is included, not just reading"),
    ):
        assert phrase in readme, f"the README does not say {why} ({phrase!r})"


def test_the_TUTORIAL_says_it_too_where_the_id_is_first_obtained():
    """DONE WHEN 2, the other half.

    The README is where somebody looks for a rule; the notebook is where
    somebody first holds an identifier and is about to paste it to a colleague.
    The note belongs at that moment, not only in the reference.
    """
    import json

    nb = json.loads(
        pathlib.Path(__file__).with_name("tutorial_model_finetune.ipynb").read_text()
    )
    text = "\n".join(
        "".join(c.get("source", []))
        for c in nb["cells"]
        if c.get("cell_type") == "markdown"
    ).lower()
    assert (
        "cannot be withdrawn" in text
    ), "the tutorial does not tell the reader the grant is irrevocable"
    assert "run inference with its model" in text, (
        "the tutorial does not say inference is included, which is the part a "
        "reader would not guess"
    )


@pytest.mark.parametrize("name", sorted(UNGATED_PENDING_A_RULING))
def test_the_UNRULED_methods_are_in_fact_ungated_today(name):
    """Records the state, not an endorsement of it.

    If somebody gates these, this test fails and the fix is to move the method
    into OWNER_OR_ADMIN and update the documentation -- not to delete the
    assertion. That is the whole difference between this set and the open one.
    """
    node, _params = METHODS[name]
    assert not _calls_the_gate(node), (
        f"{name} is now gated. Move it to OWNER_OR_ADMIN and correct the "
        "README, which currently tells readers it is not."
    )


def test_an_EXPORT_RECORD_really_does_hand_over_the_weights():
    """Why the export handle is not a progress token, pinned to the source.

    The README says an `export_id` is equivalent to the weights. That claim has
    to be anchored to the code, or it becomes another sentence that drifts: if
    `download_url` ever stops being part of what `get_export_status` returns,
    the warning should be revisited rather than left standing.
    """
    assert "download_url" in SOURCE, "the export record no longer mentions download_url"
    node, _params = METHODS["get_export_status"]
    body = ast.get_source_segment(SOURCE, node) or ""
    # Review caught the previous form of this: it asserted `"_" in body`, which
    # every method body in Python satisfies. Assert the actual filter instead.
    assert 'startswith("_")' in body or "startswith('_')" in body, (
        "get_export_status no longer returns the record filtered only by a "
        "leading-underscore convention; re-check what it exposes"
    )
    assert "download_url" in (
        ast.get_docstring(node) or ""
    ), "get_export_status's own docstring no longer documents download_url"


def test_the_RESUME_gate_is_applied_to_the_RESUMED_run_not_the_new_one():
    """Nit from review, and a real hole: the gate's ARGUMENT was never checked.

    `_calls_the_gate` passes when the gate is called anywhere in the body with
    any argument, so `start_training` gating the freshly minted `session_id`,
    or gating unconditionally, would have stayed green while the docstring
    claimed it gates "when resume_session_id is given".

    So this checks BOTH halves of that sentence: the argument is
    `resume_session_id`, and the call sits inside a branch conditioned on it.
    An earlier version checked only the argument while the docstring claimed
    both -- the same gap, one level up.
    """
    node, _params = METHODS["start_training"]
    args_seen = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if (getattr(fn, "attr", None) or getattr(fn, "id", None)) == GATE:
                args_seen.append(
                    [
                        a.id if isinstance(a, ast.Name) else type(a).__name__
                        for a in sub.args
                    ]
                )
    assert args_seen, f"{GATE} is not called in start_training at all"
    assert any("resume_session_id" in a for a in args_seen), (
        f"start_training calls {GATE} with {args_seen}, never with "
        "resume_session_id. The gate must be applied to the run being resumed."
    )

    # ...and it must be CONDITIONAL on a resume being asked for. Gating every
    # call would refuse an ordinary new run for anyone but the owner of some
    # other run, which no ruling asks for.
    guarded = False
    for sub in ast.walk(node):
        if not isinstance(sub, ast.If):
            continue
        test_src = ast.get_source_segment(SOURCE, sub.test) or ""
        if "resume_session_id" not in test_src:
            continue
        for inner in ast.walk(sub):
            if isinstance(inner, ast.Call):
                fn = inner.func
                if (getattr(fn, "attr", None) or getattr(fn, "id", None)) == GATE:
                    guarded = True
    assert guarded, (
        f"{GATE} is called in start_training but not under a branch testing "
        "resume_session_id, so it would gate an ordinary new run too."
    )


def test_the_README_warns_that_an_EXPORT_HANDLE_is_the_weights():
    readme = pathlib.Path(__file__).with_name("README.md").read_text().lower()
    assert "download_url" in readme, "the README does not name download_url"
    for phrase in ("export handle", "not yet decided"):
        assert phrase in readme, f"the README does not say {phrase!r}"


def test_the_TUTORIAL_warns_about_the_export_handle_too():
    import json

    nb = json.loads(
        pathlib.Path(__file__).with_name("tutorial_model_finetune.ipynb").read_text()
    )
    text = "\n".join(
        "".join(c.get("source", []))
        for c in nb["cells"]
        if c.get("cell_type") == "markdown"
    ).lower()
    assert (
        "not a progress token" in text
    ), "the tutorial hands the reader an export_id without saying what it grants"


def test_the_README_TABLE_ROW_itself_names_the_weights():
    """Review's surviving mutant: the original wording could come back.

    Restoring "progress of an export already started" in the `get_export_status`
    row left all 20 tests green, because only the separate paragraph below was
    pinned. A reader scanning the table is exactly who the first version of
    this documentation misled, so the row has to carry the warning too.
    """
    readme = pathlib.Path(__file__).with_name("README.md").read_text()
    rows = [
        ln
        for ln in readme.splitlines()
        if ln.startswith("|") and "get_export_status" in ln
    ]
    assert rows, "no README table row for get_export_status"
    for row in rows:
        low = row.lower()
        assert "weights" in low, f"the table row does not name the weights: {row}"
        assert (
            "progress of an export already started" not in low
        ), "the table row has reverted to describing the handle as progress"
