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

# THREE gates, and the number is not the point -- the point is that this file
# must not carry a FIXED LIST of them. Each admits a different set of people:
# `require_caller_may_mutate` is owner-or-admin and governs acting on somebody's
# run; `require_caller_may_resume_run` also admits a MANAGER of the run's
# dataset and governs deriving a model from their checkpoint; and
# `require_caller_may_use_export` binds an export handle to the identity that
# created it.
#
# **A fixed list is how this file produced a VACUOUS PASS, twice.** First it had
# one `GATE` constant, so "is it gated" was the question when the question is
# "by WHICH rule". Then it had two, and a third shipped -- and because the
# detector only knew the two, it answered "not gated" for two methods that had
# just been gated, which is the precise failure this file exists to prevent. The
# detector below therefore DISCOVERS gates by prefix instead of being told them,
# and `test_the_FILE_ACCOUNTS_FOR_EVERY_GATE_THE_APP_DEFINES` fails if the app
# grows a fourth that the classification does not name.
MUTATE_GATE = "require_caller_may_mutate"
RESUME_GATE = "require_caller_may_resume_run"
EXPORT_GATE = "require_caller_may_use_export"

# Any callable whose name starts with this is a caller gate. Matching by PREFIX
# rather than by membership is what makes a new gate visible to the open-set
# check on the day it lands, instead of invisible until somebody notices.
GATE_PREFIX = "require_caller_"

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

# Bound to the identity that MINTED the handle, which is a third rule and not a
# third opinion. `get_export_status` returns the whole export record, and a
# READY record carries `download_url` -- a six-hour presigned link to the
# packaged weights -- plus `resume_checkpoint_file`; `push_export` streams the
# package to any presigned URL the caller supplies. So an export handle hands
# over the weights.
#
# These two spent a while in a set called UNGATED_UNTIL_RESTRICTED, recording
# that the restriction was ruled on but unshipped. It has since shipped, and
# that set is gone: a "not yet" category outlives the thing it describes and
# then quietly asserts something false.
CREATOR_OR_ADMIN = frozenset({"get_export_status", "push_export"})

OWNER_OR_ADMIN = frozenset(
    {
        "stop_training",  # acts on somebody else's run
        "export_model",  # hands out the weights
        "contribute",  # publishes them into the shared pool
    }
)

# Gated, but by the WIDER rule. `start_training` with `resume_session_id`
# derives a model from somebody else's checkpoint, and the shipped decision is
# that a run's collaborators are its dataset's MANAGERS -- so a co-manager may
# resume, while export, contribute and stop stay owner-or-admin. This set used
# to be folded into OWNER_OR_ADMIN, which asserted the narrower rule and went
# red the moment the wider one shipped. The failure was correct: the
# classification had become untrue, and the repair is to name the second rule
# rather than to loosen the first.
OWNER_MANAGER_OR_ADMIN = frozenset({"start_training"})


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


def _gates_called(node) -> frozenset:
    """EVERY caller gate *node* invokes, DISCOVERED by prefix.

    Discovered rather than enumerated, and that is the whole repair. A version
    of this that took the gate names as arguments could only see the gates it
    was handed, so when a third one shipped it reported two freshly-gated
    methods as ungated. Matching `require_caller_*` means a gate nobody has
    told this file about is still visible to the open-set check.
    """
    found = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if isinstance(name, str) and name.startswith(GATE_PREFIX):
                found.add(name)
    return frozenset(found)


def _calls_the_gate(node, gate: str = MUTATE_GATE) -> bool:
    """Does *node* call this specific gate? Built on the discovery above, so
    the two cannot disagree about what counts as a gate."""
    return gate in _gates_called(node)


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
    for gate in (MUTATE_GATE, RESUME_GATE, EXPORT_GATE):
        assert gate in SOURCE, f"{gate} is not in entry.py at all; it was renamed"


def test_the_FILE_ACCOUNTS_FOR_EVERY_GATE_THE_APP_DEFINES():
    """THE CONTROL FOR THE CLASSIFICATION ITSELF, and the repair for a measured
    vacuous pass.

    This file twice reported freshly-gated methods as ungated, because it held a
    FIXED LIST of gate names and the app had grown one it did not know about.
    The detector now discovers gates by prefix, which fixes the open-set check;
    this fixes the other half, the named sets. It reads the gates the app
    DEFINES out of `training.py` and fails if any is not accounted for here.
    So adding a fourth gate breaks this test on the day it lands, and the
    breakage names the gate.
    """
    import re as _re

    training = (pathlib.Path(__file__).with_name("training.py")).read_text()
    defined = set(_re.findall(r"^def (require_caller_\w+)", training, _re.M))
    assert defined, "no caller gates found in training.py; the pattern is wrong"
    accounted = {MUTATE_GATE, RESUME_GATE, EXPORT_GATE}
    assert defined == accounted, (
        f"the app defines {sorted(defined)} but this file accounts for "
        f"{sorted(accounted)}. Unaccounted: {sorted(defined - accounted)}; "
        f"stale: {sorted(accounted - defined)}. Every gate needs a named set, "
        "or a method it guards can be classified as open."
    )


def test_EVERY_run_keyed_method_is_on_exactly_one_side():
    """The reason this file exists: no UNDECIDED category.

    There are four named ones, because the code gives four answers: open to any
    holder of the identifier, owner-or-admin, owner-or-dataset-manager-or-admin,
    and bound to the export's creator. The COUNT is not the rule -- the rule is
    that a method taking a run identifier sits in exactly one of them.

    An earlier version of this docstring said "no third, undecided category"
    while the file carried a set named UNGATED_UNTIL_RESTRICTED. A "not yet"
    category outlives the thing it describes and then asserts something false,
    which is what happened: the restriction shipped and the set did not move.
    """
    declared = (
        OPEN_TO_ANY_HOLDER | OWNER_OR_ADMIN | OWNER_MANAGER_OR_ADMIN | CREATOR_OR_ADMIN
    )
    undeclared = set(RUN_KEYED) - declared
    assert undeclared == set(), (
        f"these methods take a run identifier and are on neither side: "
        f"{sorted(undeclared)}. Decide, then add each to OPEN_TO_ANY_HOLDER, "
        "OWNER_OR_ADMIN, OWNER_MANAGER_OR_ADMIN or CREATOR_OR_ADMIN above."
    )
    sets = {
        "OPEN_TO_ANY_HOLDER": OPEN_TO_ANY_HOLDER,
        "OWNER_OR_ADMIN": OWNER_OR_ADMIN,
        "OWNER_MANAGER_OR_ADMIN": OWNER_MANAGER_OR_ADMIN,
        "CREATOR_OR_ADMIN": CREATOR_OR_ADMIN,
    }
    for a_name, a in sets.items():
        for b_name, b in sets.items():
            if a_name < b_name:
                assert not (a & b), (
                    f"a method cannot be in both {a_name} and {b_name}: "
                    f"{sorted(a & b)}"
                )


def test_the_declared_sets_do_not_name_methods_that_no_longer_EXIST():
    """The other direction, so a rename cannot leave a set asserting nothing."""
    missing = (
        OPEN_TO_ANY_HOLDER | OWNER_OR_ADMIN | OWNER_MANAGER_OR_ADMIN | CREATOR_OR_ADMIN
    ) - set(METHODS)
    assert (
        missing == set()
    ), f"these are declared but are not public methods of the app: {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(OWNER_OR_ADMIN))
def test_an_OWNER_OR_ADMIN_method_actually_carries_the_MUTATE_gate(name):
    node, _params = METHODS[name]
    assert _calls_the_gate(node, MUTATE_GATE), (
        f"{name} is declared owner-or-admin but never calls {MUTATE_GATE}. "
        "Either the gate was dropped, or this method belongs in "
        "OPEN_TO_ANY_HOLDER, or it is now governed by the wider resume rule and "
        "belongs in OWNER_MANAGER_OR_ADMIN."
    )


@pytest.mark.parametrize("name", sorted(OWNER_MANAGER_OR_ADMIN))
def test_an_OWNER_MANAGER_OR_ADMIN_method_carries_the_RESUME_gate(name):
    """The wider rule, asserted as its own thing.

    **This is the delete-control for the resume gate.** Removing the call --
    the always-allow mutant -- makes this red, and so does swapping it for the
    narrower `require_caller_may_mutate`, because that would silently take the
    dataset's managers back out.
    """
    node, _params = METHODS[name]
    assert _calls_the_gate(node, RESUME_GATE), (
        f"{name} is declared owner-manager-or-admin but never calls "
        f"{RESUME_GATE}. Deriving a model from somebody else's checkpoint must "
        "go through the rule that admits the dataset's managers."
    )


@pytest.mark.parametrize("name", sorted(OPEN_TO_ANY_HOLDER))
def test_an_OPEN_method_does_NOT_carry_the_gate(name):
    """The inverse, and the half that catches an accidental tightening.

    Gating inference is a user-visible break of the shared-notebook flow, and it
    would otherwise be an easy thing to add while fixing something else.
    """
    node, _params = METHODS[name]
    gates = _gates_called(node)
    assert gates == frozenset(), (
        f"{name} is open to any holder of the run identifier, but it now calls "
        f"{sorted(gates)}. Several people are meant to be able to try one run's "
        "model. EVERY require_caller_* call counts, not a fixed list of them: a "
        "tightening is a tightening whichever rule it uses, including one this "
        "file has never heard of."
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
        gates = _gates_called(node)
        assert (
            gates == frozenset()
        ), f"{name} must not require the owner (it calls {sorted(gates)})"


def test_the_WEIGHT_HANDING_methods_are_the_gated_ones():
    """The complement of the above, stated as the rule rather than per method."""
    assert {"export_model", "contribute"} <= OWNER_OR_ADMIN
    # start_training is gated too, by the WIDER rule: a dataset's managers may
    # resume a colleague's checkpoint, while nobody but the owner or an admin
    # may export or contribute it. Asserting the mutate gate here is what went
    # red when that decision shipped.
    assert "start_training" in OWNER_MANAGER_OR_ADMIN
    node, _params = METHODS["start_training"]
    assert _calls_the_gate(node, RESUME_GATE), (
        "resuming derives a new model from somebody else's checkpoint, so "
        "start_training must gate when resume_session_id is given"
    )
    assert not _calls_the_gate(node, MUTATE_GATE), (
        "start_training must NOT use the owner-or-admin gate: that would "
        "exclude the dataset managers the resume rule admits"
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


@pytest.mark.parametrize("name", sorted(CREATOR_OR_ADMIN))
def test_a_CREATOR_OR_ADMIN_method_carries_the_EXPORT_gate(name):
    """An export handle is bound to the identity that minted it.

    This replaces a test that asserted these two were UNGATED, which passed for
    a while after they had been gated: the detector it used knew two gate names
    and the export gate was a third, so "not gated" was answered True about
    gated code. The predecessor was written to fail on the day the restriction
    shipped, and it did not. Hence this assertion names the gate, and
    `test_the_FILE_ACCOUNTS_FOR_EVERY_GATE_THE_APP_DEFINES` makes the next new
    gate impossible to miss the same way.
    """
    node, _params = METHODS[name]
    assert _calls_the_gate(node, EXPORT_GATE), (
        f"{name} hands over an export record, which carries a link to the "
        f"weights, so it must call {EXPORT_GATE}"
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
            if (getattr(fn, "attr", None) or getattr(fn, "id", None)) == RESUME_GATE:
                args_seen.append(
                    [
                        a.id if isinstance(a, ast.Name) else type(a).__name__
                        for a in sub.args
                    ]
                )
    assert args_seen, f"{RESUME_GATE} is not called in start_training at all"
    assert any("resume_session_id" in a for a in args_seen), (
        f"start_training calls {RESUME_GATE} with {args_seen}, never with "
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
                if (
                    getattr(fn, "attr", None) or getattr(fn, "id", None)
                ) == RESUME_GATE:
                    guarded = True
    assert guarded, (
        f"{RESUME_GATE} is called in start_training but not under a branch testing "
        "resume_session_id, so it would gate an ordinary new run too."
    )


def test_the_README_warns_that_an_EXPORT_HANDLE_is_the_weights():
    readme = pathlib.Path(__file__).with_name("README.md").read_text().lower()
    assert "download_url" in readme, "the README does not name download_url"
    for phrase in ("export handle", "bound to the identity that created it"):
        assert phrase in readme, f"the README does not say {phrase!r}"

    # THE RETIRED WORDINGS, each of which was true once and is now false. A
    # pinning test is only as good as its pin: this file previously required
    # "will be restricted", so when the restriction SHIPPED the README kept
    # promising a future change that had already happened, and the test held it
    # there. Both the undecided phrasing and the not-yet phrasing are now
    # forbidden, so the documentation cannot drift back to describing the
    # restriction as pending.
    for retired in ("not yet decided", "will be restricted", "until that ships"):
        assert retired not in readme, (
            f"the README still says {retired!r}. The restriction has SHIPPED: "
            "get_export_status and push_export are bound to the export's "
            "creator. A test that pins a pending-change wording freezes a "
            "statement that is no longer true, which is the failure this file "
            "exists to prevent."
        )
    assert "anyone holding the handle" not in readme, (
        "the README table still says an export handle is open to anyone "
        "holding it, which the shipped gate contradicts"
    )


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
    assert "bound to the identity that created it" in text, (
        "the tutorial does not tell the reader that an export handle is bound "
        "to its creator, so a reader still believes passing one works"
    )
    assert "to any caller holding the id" not in text, (
        "the tutorial still says the export record goes to any caller holding "
        "the id, which the shipped gate contradicts"
    )


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
