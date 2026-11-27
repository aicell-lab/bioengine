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


# ── the listing must not re-read what it has already read ────────────────────


def test_the_listing_reads_each_run_record_ONCE(runs, monkeypatch):
    """A COST assertion, pinned because the regression is silent.

    The gate used to resolve the session id itself, so every listed run was
    opened three times -- once by `get_status` in the loop, once by the shared
    owner/co-manager helper, once more by the administrator fallback. Nothing
    failed; the listing simply did three times the I/O, and on the production
    store that is 783 file reads where there were 261.

    The floor is now ONE read per run. It used to be two, because
    `session_backend` re-read the record `get_status` had just parsed; it now
    takes the record it is given. **This measures `training.list_sessions`,
    which is an INNER helper** -- see
    `test_the_PUBLISHED_listing_reads_each_record_ONCE` for the path a caller
    actually takes. Review found that distinction the hard way: this test
    passed at 2N while the published path was doing 3N, because the extra pass
    was in the entry point this test never calls.
    """
    PER_RUN = 1
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
    PER_RUN = 1
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


@pytest.mark.asyncio
async def test_the_PUBLISHED_listing_reads_each_record_ONCE(runs, monkeypatch):
    """The cost of the path a CALLER takes, not of an inner helper.

    The two cost tests above measure `training.list_sessions`. Review showed
    that is not where the cost was: `list_training_sessions` resolved the
    dataset ids with a whole extra pass over the store before calling it, so
    the published path did THREE reads per run while those tests sat green at
    two. Measured: 15 reads for 5 runs, which is 783 for the 261 runs on the
    production store -- the number a commit message had claimed was removed.

    So this walks the published method, with the artifact manager faked out,
    and asserts the total.
    """

    class _AM:
        async def read(self, artifact_id):
            return {"config": {"permissions": {}}}

    class _Hypha:
        async def get_service(self, name):
            return _AM()

    for i in range(5):
        runs(f"run-{i}", OWNER, dataset=DATASET)

    calls = []
    real = training.read_status
    monkeypatch.setattr(
        training,
        "read_status",
        lambda sid, *a, **k: (calls.append(sid), real(sid, *a, **k))[1],
    )

    import entry

    cls = None
    for obj in vars(entry).values():
        if isinstance(obj, type) and obj.__name__ == "EntryApp":
            cls = obj
            break
        inner = getattr(obj, "__wrapped_class__", None) or getattr(
            obj, "func_or_class", None
        )
        if isinstance(inner, type) and inner.__name__ == "EntryApp":
            cls = inner
            break
    assert cls is not None, "EntryApp not found"
    inst = cls.__new__(cls)
    inst._hypha = _Hypha()

    await cls.list_training_sessions(inst, context=ctx(OWNER, anonymous=False))

    assert len(calls) == 5, (
        f"the PUBLISHED listing read {len(calls)} records for 5 runs "
        f"({len(calls) / 5:.1f} each). One walk of the store is enough: the "
        "records and the dataset ids come back from the same scan."
    )


@pytest.mark.asyncio
async def test_the_PUBLISHED_listing_LOGS_a_stray_directory(runs, monkeypatch, capsys):
    """The regression the single-scan change introduced.

    `list_sessions` logs the directory names it skipped, and those names are
    caller-controlled, so the log is their only trace. When the published
    entry point began handing it pre-read records it stopped passing the
    skipped names with them, so the scan found a stray directory and nobody
    ever heard about it: a direct `list_sessions()` call still logged, which
    is why the existing tests stayed green.
    """

    class _AM:
        async def read(self, artifact_id):
            return {"config": {"permissions": {}}}

    class _Hypha:
        async def get_service(self, name):
            return _AM()

    runs("run-ok", OWNER, dataset=DATASET)
    # A name the id pattern REJECTS. The fixture relaxes the pattern to
    # `[\w.-]+`, so a hyphenated name is a perfectly valid id there and would
    # be listed rather than skipped -- which is how the first version of this
    # test failed for the wrong reason.
    stray = training.sessions_root() / "not a session"
    stray.mkdir()
    (stray / "status.json").write_text("{}")

    import entry

    cls = None
    for obj in vars(entry).values():
        if isinstance(obj, type) and obj.__name__ == "EntryApp":
            cls = obj
            break
        inner = getattr(obj, "__wrapped_class__", None) or getattr(
            obj, "func_or_class", None
        )
        if isinstance(inner, type) and inner.__name__ == "EntryApp":
            cls = inner
            break
    assert cls is not None, "EntryApp not found"
    inst = cls.__new__(cls)
    inst._hypha = _Hypha()

    capsys.readouterr()
    await cls.list_training_sessions(inst, context=ctx(OWNER, anonymous=False))
    printed = capsys.readouterr().out

    assert "not a session" in printed, (
        "the published listing skipped a stray directory without logging it; "
        f"the name is caller-controlled and this log is its only trace. Got: {printed!r}"
    )


# ── the ruling: the widening is for OWNED runs only ───────────────────────


def test_a_CO_MANAGER_DOES_NOT_SEE_AN_OWNERLESS_RUN_ON_THEIR_DATASET():
    """The lead's ruling on this row, and the code broke it before this test.

    Measured on the branch before the fix: `_owner_or_comanager` returned True
    for a run with NO recorded owner whose `dataset_artifact_id` the caller
    manages — so an ownerless run was visible to a co-manager, and resumable by
    one, because `caller_may_resume_run` shares this leg.

    Two statements agreed with each other before the code did. The ruling: the
    widening applies to OWNED runs only — a co-manager sees the runs of other
    MANAGERS on that dataset — and ownerless runs stay admin-only. And the
    shipped README already said resuming an ownerless run requires `rw`/admin
    on `bioimage-io`, "because with no recorded owner there is nobody for the
    ownership check to match you against".

    The rule is not "a manager of the dataset may act on anything filed under
    it". It is "a manager may act on a COLLEAGUE's run", and an ownerless run
    has no colleague.
    """
    ctx = {"user": {"id": "co-manager", "email": "cm@example.org"}}
    managed = frozenset({"bioimage-io/ds"})

    ownerless = {"dataset_artifact_id": "bioimage-io/ds"}
    assert training._owner_or_comanager(ownerless, ctx, managed) is False

    # ...and an empty-string owner is the same case, not a different one.
    for empty in ("", None):
        rec = {"owner": empty, "dataset_artifact_id": "bioimage-io/ds"}
        assert training._owner_or_comanager(rec, ctx, managed) is False, empty


def test_THE_WIDENING_STILL_WORKS_for_a_COLLEAGUES_run():
    """The control. A leg that refused everything would satisfy the test above
    and remove the feature this row exists to add."""
    ctx = {"user": {"id": "co-manager", "email": "cm@example.org"}}
    managed = frozenset({"bioimage-io/ds"})

    colleague = {"owner": "someone-else", "dataset_artifact_id": "bioimage-io/ds"}
    assert training._owner_or_comanager(colleague, ctx, managed) is True

    # the caller's OWN run, with no dataset recorded at all, still passes on the
    # owner leg — so requiring an owner for the dataset leg did not narrow that.
    own = {"owner": "co-manager"}
    assert training._owner_or_comanager(own, ctx, frozenset()) is True


def test_A_NON_MEMBER_sees_neither_kind():
    """The other control: the dataset leg is what admits a co-manager, so a
    caller who manages nothing must be refused for both kinds of run."""
    ctx = {"user": {"id": "stranger", "email": "s@example.org"}}
    nothing = frozenset()

    owned = {"owner": "someone-else", "dataset_artifact_id": "bioimage-io/ds"}
    ownerless = {"dataset_artifact_id": "bioimage-io/ds"}
    assert training._owner_or_comanager(owned, ctx, nothing) is False
    assert training._owner_or_comanager(ownerless, ctx, nothing) is False
