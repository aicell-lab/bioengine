"""A dataset's managers may list and resume each other's runs — and nothing more.

A run's collaborators are its DATASET's managers. A colleague who co-manages the
dataset a run was trained on can see that run in their listing and continue
training from its checkpoint. They cannot stop it, export it, or contribute it to
a pool: those dispose of somebody else's work, and sharing a dataset is not
consent to them.

Two traps this file exists to pin, both of which the obvious implementation gets
wrong and neither of which announces itself:

* **The manager VALUE and the wildcard KEY are the same string, ``"*"``.** The
  annotation broker writes owner/managers as ``{user: "*"}``, and a PUBLIC
  dataset additionally as ``{"*": "r+"}``. So ``perms.get("*")`` reads the
  public grant, and an implementation that tests "is there a ``*`` in here"
  hands co-manager rights to every caller alive on every public dataset.
* **Annotators hold ``"r+"`` and are NOT managers.** "Write permission" sounds
  like it ought to include them; the ruling says managers, and the page draws
  the same line (``role === 'owner' || role === 'manager'``). An annotator can
  add labels to a dataset — that is not authority over models trained from it.

Each is asserted with a mutant-style control: a test that FAILS if the rule is
widened in exactly the plausible wrong direction, rather than only testing the
happy path.
"""

import json
import re
import time

import pytest

import training

OWNER = "github|49943582"
COMANAGER = "github|222333"
STRANGER = "github|444555"
ADMIN_ID = "github|777000"

DATASET = "some-workspace/shared-dataset"
OTHER_DATASET = "some-workspace/unrelated-dataset"


def ctx(uid, *, anonymous=True, grants=None, email=None):
    user = {"id": uid, "is_anonymous": anonymous}
    if grants is not None:
        user["scope"] = {"workspaces": grants}
    if email is not None:
        user["email"] = email
    return {"user": user}


ADMIN = ctx(ADMIN_ID, anonymous=False, grants={"bioimage-io": "a"})


# The ACL the broker writes for a dataset whose owner is OWNER and whose
# managers include COMANAGER. Taken from `build_acl_permissions`, not invented:
# owner and managers -> "*", annotators -> "r+".
SHARED_ACL = {OWNER: "*", COMANAGER: "*", STRANGER: "r+"}

# The same dataset, made PUBLIC. The wildcard key appears, mapped to the
# annotator-level grant. Nothing about it makes anyone a manager.
PUBLIC_ACL = {OWNER: "*", "*": "r+"}


@pytest.fixture()
def runs(tmp_path, monkeypatch):
    """Plant runs with an owner and, optionally, a dataset."""
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(training, "sessions_root", lambda: root)
    monkeypatch.setattr(training, "session_dir", lambda sid: root / sid)
    monkeypatch.setattr(training, "SESSION_ID_PATTERN", re.compile(r"\A[\w.-]+\Z"))

    def add(sid, owner, *, dataset=None):
        d = root / sid
        d.mkdir()
        status = {
            "status": "COMPLETED",
            "updated_at": time.time(),
            "owner": owner,
            "owner_anonymous": False,
        }
        if dataset is not None:
            status[training.DATASET_FIELD] = dataset
        (d / "status.json").write_text(json.dumps(status))
        return sid

    return add


# ── the fixture's own control ────────────────────────────────────────────────


def test_the_fixture_really_plants_runs_a_caller_can_see(runs):
    """THE CONTROL FOR THE FILE. Every "sees nothing" below is meaningless if a
    caller who SHOULD see a run does not."""
    runs("mine", OWNER, dataset=DATASET)
    assert sorted(training.list_sessions(ctx(OWNER, anonymous=False))) == ["mine"], (
        "the owner must see their own run, or every assertion in this file is "
        "measuring a broken fixture rather than the filter"
    )


# ── caller_manages_dataset: the two traps ────────────────────────────────────


def test_a_manager_is_recognised():
    assert training.caller_manages_dataset(SHARED_ACL, ctx(COMANAGER)) is True


def test_the_owner_of_the_dataset_is_recognised():
    assert training.caller_manages_dataset(SHARED_ACL, ctx(OWNER)) is True


def test_an_ANNOTATOR_is_NOT_a_manager():
    """`r+` lets somebody add annotations to a dataset. It is not authority over
    models trained from it, and the ruling says MANAGERS."""
    assert training.caller_manages_dataset(SHARED_ACL, ctx(STRANGER)) is False


def test_a_caller_absent_from_the_ACL_is_not_a_manager():
    assert training.caller_manages_dataset(SHARED_ACL, ctx("github|999")) is False


def test_a_PUBLIC_dataset_does_NOT_make_everyone_a_manager():
    """THE WILDCARD TRAP, stated as the test that catches it.

    A public dataset's ACL contains `"*": "r+"`. An implementation reading the
    wildcard key -- or testing `"*" in perms.values()` loosely -- makes every
    caller alive a co-manager of every public dataset, and so a resumer of every
    run trained on one. This is the assertion that turns red for it.
    """
    assert training.caller_manages_dataset(PUBLIC_ACL, ctx(STRANGER)) is False
    assert training.caller_manages_dataset(PUBLIC_ACL, ctx("github|anyone")) is False
    # ... while the dataset's actual owner is still a manager, so the test above
    # is not passing merely because the whole ACL was ignored.
    assert training.caller_manages_dataset(PUBLIC_ACL, ctx(OWNER)) is True


def test_a_caller_whose_id_is_literally_a_star_does_not_read_the_public_grant():
    """No such id should exist. If one did, matching it against the ACL would
    read the public grant as that caller's own entry."""
    assert training.caller_manages_dataset(PUBLIC_ACL, ctx("*")) is False


def test_an_ACL_keyed_by_EMAIL_is_matched():
    """The broker prefers id over email when writing the ACL but can write
    either, so checking only one silently misses half the datasets."""
    acl = {"someone@lab.edu": "*"}
    assert (
        training.caller_manages_dataset(acl, ctx("github|1", email="someone@lab.edu"))
        is True
    )


def test_a_malformed_ACL_or_context_is_not_a_manager():
    for perms in (None, [], "rw", {}):
        assert training.caller_manages_dataset(perms, ctx(COMANAGER)) is False
    for bad in (None, {}, {"user": None}, {"user": {}}):
        assert training.caller_manages_dataset(SHARED_ACL, bad) is False


# ── seeing a colleague's run ─────────────────────────────────────────────────


def test_a_CO_MANAGER_sees_a_colleagues_run(runs):
    runs("theirs", OWNER, dataset=DATASET)
    seen = training.list_sessions(
        ctx(COMANAGER, anonymous=False), writable_datasets=frozenset({DATASET})
    )
    assert sorted(seen) == ["theirs"]


def test_a_STRANGER_does_not_see_it(runs):
    """The same run, the same call, a caller who manages nothing."""
    runs("theirs", OWNER, dataset=DATASET)
    assert training.list_sessions(ctx(STRANGER, anonymous=False)) == {}


def test_managing_a_DIFFERENT_dataset_does_not_grant_access(runs):
    """The dataset is matched exactly; holding one does not open another."""
    runs("theirs", OWNER, dataset=DATASET)
    seen = training.list_sessions(
        ctx(COMANAGER, anonymous=False), writable_datasets=frozenset({OTHER_DATASET})
    )
    assert seen == {}


def test_a_run_with_NO_dataset_has_no_co_managers(runs):
    """Every run that exists today predates the field. A co-manager of any
    dataset must not inherit them."""
    runs("private", OWNER)
    seen = training.list_sessions(
        ctx(COMANAGER, anonymous=False), writable_datasets=frozenset({DATASET})
    )
    assert seen == {}


def test_the_owner_still_sees_their_own_run_without_any_dataset(runs):
    runs("private", OWNER)
    assert sorted(training.list_sessions(ctx(OWNER, anonymous=False))) == ["private"]


def test_an_ADMIN_still_does_NOT_see_other_peoples_owned_runs(runs):
    """PINS THE BOUNDARY THIS CHANGE MUST NOT MOVE.

    Admin rights grant the unattributable rows, not everybody's history. That
    was decided when the listing became per-caller and it is unchanged here.

    It is worth stating why this is a test rather than a comment: the obvious
    way to implement co-manager access is one `may_access` function ending in
    `return caller_is_workspace_admin(context)`, which reads perfectly and
    silently hands an administrator every run on the worker. That is exactly
    what the first draft of this change did, and this assertion is what caught
    it.
    """
    runs("theirs", OWNER, dataset=DATASET)
    assert training.list_sessions(ADMIN) == {}


# ── resuming a colleague's run ───────────────────────────────────────────────


def test_a_CO_MANAGER_may_resume(runs):
    sid = runs("theirs", OWNER, dataset=DATASET)
    assert (
        training.caller_may_resume_run(
            sid, ctx(COMANAGER, anonymous=False), writable_datasets=frozenset({DATASET})
        )
        is True
    )


def test_a_STRANGER_may_NOT_resume(runs):
    sid = runs("theirs", OWNER, dataset=DATASET)
    assert training.caller_may_resume_run(sid, ctx(STRANGER, anonymous=False)) is False


def test_the_resume_refusal_raises_NotTheOwner(runs):
    sid = runs("theirs", OWNER, dataset=DATASET)
    with pytest.raises(training.NotTheOwner):
        training.require_caller_may_resume_run(sid, ctx(STRANGER, anonymous=False))


def test_the_resume_refusal_says_nothing_about_the_dataset_or_the_owner(runs):
    """A refusal that names the dataset tells a stranger which datasets exist
    and who manages them."""
    sid = runs("theirs", OWNER, dataset=DATASET)
    with pytest.raises(training.NotTheOwner) as exc:
        training.require_caller_may_resume_run(sid, ctx(STRANGER, anonymous=False))
    message = str(exc.value)
    assert DATASET not in message
    assert OWNER not in message
    assert sid not in message


# ── clause 2: a co-manager may NOT dispose of the run ────────────────────────


def test_a_CO_MANAGER_may_NOT_mutate_a_colleagues_run(runs):
    """CLAUSE 2. Stopping, exporting and contributing stay owner-or-admin.

    `caller_may_mutate` takes no `writable_datasets` at all, which is the
    structural half of this guarantee: there is no argument by which a dataset
    could widen it, so widening it would take a visible change to the signature
    rather than a value flowing in from somewhere.
    """
    sid = runs("theirs", OWNER, dataset=DATASET)
    assert training.caller_may_mutate(sid, ctx(COMANAGER, anonymous=False)) is False


def test_the_mutate_gate_accepts_no_dataset_argument(runs):
    """The structural assertion behind the test above."""
    import inspect

    params = inspect.signature(training.caller_may_mutate).parameters
    assert "writable_datasets" not in params, (
        "caller_may_mutate must not take a dataset argument: a co-manager may "
        "resume a colleague's run but must never stop, export or contribute it"
    )


def test_the_owner_may_still_mutate_their_own_run(runs):
    """THE CONTROL for the two above: both would read the same if the mutate
    gate refused everybody."""
    sid = runs("mine", OWNER, dataset=DATASET)
    assert training.caller_may_mutate(sid, ctx(OWNER, anonymous=False)) is True


# ── the dataset scan that feeds the async resolver ───────────────────────────


def test_dataset_ids_on_disk_returns_each_dataset_once(runs):
    runs("a", OWNER, dataset=DATASET)
    runs("b", COMANAGER, dataset=DATASET)
    runs("c", OWNER, dataset=OTHER_DATASET)
    runs("d", OWNER)
    assert sorted(training.dataset_ids_on_disk()) == sorted({DATASET, OTHER_DATASET})


def test_dataset_ids_on_disk_is_empty_when_no_run_names_one(runs):
    """Today's production store, where nothing recorded a dataset. The resolver
    short-circuits on this and makes no artifact calls at all."""
    runs("a", OWNER)
    runs("b", COMANAGER)
    assert training.dataset_ids_on_disk() == []


# ── the listing must not re-read what it has already read ────────────────────


def test_the_listing_reads_each_run_record_ONCE(runs, monkeypatch):
    """A COST assertion, pinned because the regression is silent.

    The gate used to resolve the session id itself, so every listed run was
    opened three times -- once by `get_status` in the loop, once by the shared
    owner/co-manager helper, once more by the administrator fallback. Nothing
    failed; the listing simply did three times the I/O, and on the production
    store that is 783 file reads where there were 261.

    The floor is TWO reads per run, and it is not mine: `get_status` reads the
    record, and the pre-existing `session_backend` reads it again. Measured on
    the branch BEFORE this change, which gives the same 2N -- so the number
    asserted here is the unchanged baseline, not a target I chose. A listing
    that costs more than this has acquired a re-read.
    """
    PER_RUN = 2
    for i in range(5):
        runs(f"run-{i}", OWNER, dataset=DATASET)

    calls = []
    real = training.read_status
    monkeypatch.setattr(
        training,
        "read_status",
        lambda sid, *a, **k: (calls.append(sid), real(sid, *a, **k))[1],
    )

    training.list_sessions(
        ctx(OWNER, anonymous=False), writable_datasets=frozenset({DATASET})
    )

    assert len(calls) == 5 * PER_RUN, (
        f"expected {5 * PER_RUN} record reads for 5 runs (the pre-change "
        f"baseline of {PER_RUN} each), got {len(calls)}. The gate is re-reading "
        f"a record the listing already parsed."
    )


def test_the_listing_still_reads_once_when_the_caller_is_an_admin(runs, monkeypatch):
    """The administrator path is the one that used to read an EXTRA time, in the
    fallback after the shared legs declined. A count taken only on the owner
    path would not have noticed, because the owner leg returns before reaching
    it."""
    PER_RUN = 2
    for i in range(4):
        runs(f"old-{i}", None)

    calls = []
    real = training.read_status
    monkeypatch.setattr(
        training,
        "read_status",
        lambda sid, *a, **k: (calls.append(sid), real(sid, *a, **k))[1],
    )

    seen = training.list_sessions(ADMIN)
    assert (
        len(seen) == 4
    ), f"the admin should see all four ownerless runs, saw {len(seen)}"
    assert len(calls) == 4 * PER_RUN, (
        f"expected {4 * PER_RUN} record reads for 4 runs on the admin path, "
        f"got {len(calls)}"
    )
