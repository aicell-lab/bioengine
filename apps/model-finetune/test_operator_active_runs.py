"""An operator can see what a worker is WORKING ON, without seeing whose it is.

The per-caller run listing answers "which runs are mine". Nobody could ask
"what is this worker currently running", and the two are not the same question:
a run started by a caller who was not signed in is owned by an identity Hypha
minted for that connection, so once the connection closes its owner cannot look
and -- because the run IS owned -- no administrator may list it either. It keeps
occupying the worker with nobody able to see or cancel it.

`list_active_training_runs` reports by LIFECYCLE instead. The two properties it
must have, and both are easy to lose:

* it is administrator-only;
* it redacts owners, so it answers the operational question without becoming a
  way to read whose work is whose.

And the thing it must NOT do, which has its own test here as well as in the
co-manager suite: it must not be implemented by widening the administrator leg
of the ordinary listing. That would hand every owned run to every administrator
and undo a privacy decision made deliberately.
"""

import json
import re
import time

import pytest

import training

OWNER = "github|49943582"
OTHER = "github|000111"
ADMIN_ID = "github|777000"


def ctx(uid, *, anonymous=True, grants=None):
    user = {"id": uid, "is_anonymous": anonymous}
    if grants is not None:
        user["scope"] = {"workspaces": grants}
    return {"user": user}


ADMIN = ctx(ADMIN_ID, anonymous=False, grants={"bioimage-io": "a"})
NOT_ADMIN = ctx(OTHER, anonymous=False, grants={"bioimage-io": "r"})


@pytest.fixture()
def runs(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(training, "sessions_root", lambda: root)
    monkeypatch.setattr(training, "session_dir", lambda sid: root / sid)
    monkeypatch.setattr(training, "SESSION_ID_PATTERN", re.compile(r"\A[\w.-]+\Z"))

    def add(sid, owner, status):
        d = root / sid
        d.mkdir()
        (d / "status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "updated_at": time.time(),
                    "owner": owner,
                    "owner_anonymous": owner is not None,
                }
            )
        )
        return sid

    return add


# ── what it reports ──────────────────────────────────────────────────────────


def test_it_reports_a_run_that_has_not_finished(runs):
    runs("busy", OWNER, "TRAINING")
    assert sorted(training.non_terminal_runs()) == ["busy"]


def test_it_reports_PREPARING_and_QUEUED_too(runs):
    """A run wedged before training starts occupies the worker just as much."""
    runs("prep", OWNER, "PREPARING")
    runs("queued", OWNER, "QUEUED")
    assert sorted(training.non_terminal_runs()) == ["prep", "queued"]


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "STOPPED"])
def test_it_does_NOT_report_a_finished_run(runs, status):
    """THE CONTROL that the filter is a filter. Without this, a function that
    returned everything would pass every assertion above."""
    runs("done", OWNER, status)
    assert training.non_terminal_runs() == {}


def test_it_reports_runs_regardless_of_WHOSE_they_are(runs):
    """The whole point: the orphan this exists to find belongs to an identity
    that no longer exists, so ownership cannot be part of the question."""
    runs("mine", OWNER, "TRAINING")
    runs("theirs", OTHER, "TRAINING")
    runs("nobodys", None, "TRAINING")
    assert sorted(training.non_terminal_runs()) == ["mine", "nobodys", "theirs"]


def test_it_REDACTS_the_owner(runs):
    """It answers what is running, not whose it is."""
    runs("busy", OWNER, "TRAINING")
    rec = training.non_terminal_runs()["busy"]
    assert "owner" not in rec and "owner_anonymous" not in rec, rec
    # ... while still carrying what an operator needs to act.
    assert rec.get("status") == "TRAINING"
    assert rec.get("session_id") == "busy"


def test_a_junk_directory_does_not_break_it(runs, tmp_path):
    """Same tolerance as the listing: validate what a caller hands us, tolerate
    what we find. One stray directory must not blind an operator."""
    runs("busy", OWNER, "TRAINING")
    bad = tmp_path / "sessions" / "not a session"
    bad.mkdir()
    (bad / "status.json").write_text("{}")
    assert sorted(training.non_terminal_runs()) == ["busy"]


# ── who may call it ──────────────────────────────────────────────────────────


def _entry():
    import entry

    for obj in vars(entry).values():
        if isinstance(obj, type) and obj.__name__ == "EntryApp":
            return obj
    for obj in vars(entry).values():
        cls = getattr(obj, "__wrapped_class__", None) or getattr(
            obj, "func_or_class", None
        )
        if isinstance(cls, type) and cls.__name__ == "EntryApp":
            return cls
    raise AssertionError("EntryApp not found")


@pytest.mark.asyncio
async def test_an_ADMIN_gets_the_active_runs(runs):
    runs("busy", OTHER, "TRAINING")
    cls = _entry()
    inst = cls.__new__(cls)
    out = await getattr(cls, "list_active_training_runs")(inst, context=ADMIN)
    assert sorted(out) == ["busy"]


@pytest.mark.asyncio
async def test_a_NON_admin_gets_NOTHING(runs):
    """`r` on the workspace is not operator authority -- the same narrowing the
    ownerless rule makes, and for the same reason."""
    runs("busy", OTHER, "TRAINING")
    cls = _entry()
    inst = cls.__new__(cls)
    assert (
        await getattr(cls, "list_active_training_runs")(inst, context=NOT_ADMIN) == {}
    )


@pytest.mark.asyncio
async def test_an_UNREADABLE_context_gets_NOTHING(runs):
    runs("busy", OTHER, "TRAINING")
    cls = _entry()
    inst = cls.__new__(cls)
    for bad in (None, {}, {"user": None}, {"user": {}}):
        assert await getattr(cls, "list_active_training_runs")(inst, context=bad) == {}


# ── the boundary this fix must not cross ─────────────────────────────────────


def test_the_ORDINARY_listing_is_still_narrow_for_an_administrator(runs):
    """CLAUSE 3. The tempting fix is to widen the administrator leg of
    `list_sessions` so operators can see everything. That hands every owned run
    to every administrator and undoes a deliberate privacy decision.

    Pinned HERE as well as in the co-manager suite, because this is the row
    whose fix would be the one to break it, and a guard in a file nobody edits
    while making that mistake is a guard that fires too late.
    """
    runs("theirs", OTHER, "TRAINING")
    assert training.list_sessions(ADMIN) == {}, (
        "the administrator listing must NOT return other people's owned runs; "
        "the operator view is a separate surface for exactly that reason"
    )
    # ... and the operator surface still answers, so the assertion above is a
    # statement about the LISTING and not about the run being absent.
    assert sorted(training.non_terminal_runs()) == ["theirs"]


@pytest.fixture()
def operator_root(tmp_path, monkeypatch):
    """A bare sessions root the projection tests write records into directly."""
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(training, "sessions_root", lambda: root)
    monkeypatch.setattr(training, "session_dir", lambda sid: root / sid)
    monkeypatch.setattr(training, "SESSION_ID_PATTERN", re.compile(r"\A[\w.-]+\Z"))
    return root


# ── what the operator view may and may not carry ─────────────────────────────


def _write_run(root, sid, **fields):
    d = root / sid
    d.mkdir(parents=True, exist_ok=True)
    rec = {"status": "TRAINING", "updated_at": time.time()}
    rec.update(fields)
    (d / "status.json").write_text(json.dumps(rec))
    return d


def test_the_operator_view_carries_NO_DATASET_ID_and_NO_LABEL(operator_root):
    """The leak the owner-key check could not see.

    Stripping `owner` and `owner_anonymous` left `dataset_artifact_id` in the
    record. A dataset its manager owns lives in that manager's own workspace,
    so the id names them -- which is the thing this view promised not to
    publish. The free-text label is a second channel of the same kind.
    """
    _write_run(
        operator_root,
        "2026-11-25-100000-aaaaaaaa",
        status="TRAINING",
        owner="ws-user-someone/alice",
        dataset_artifact_id="ws-user-anonymouz-abc/my-cells",
        label="alice's third attempt at the mitochondria set",
    )
    out = training.non_terminal_runs()
    rec = out["2026-11-25-100000-aaaaaaaa"]
    for leaky in ("dataset_artifact_id", "label", "owner", "owner_anonymous"):
        assert leaky not in rec, f"the operator view still carries {leaky!r}: {rec}"


def test_the_operator_view_is_an_ALLOW_LIST_not_a_strip_list(operator_root):
    """A field nobody thought about must not reach an operator.

    This is the difference that matters: with a strip-list, a field added to
    the record later arrives in the operator view by default and nothing fails.
    """
    _write_run(
        operator_root,
        "2026-11-25-100000-bbbbbbbb",
        status="TRAINING",
        some_future_field="a value nobody has considered yet",
        contact_email="alice@example.org",
    )
    rec = training.non_terminal_runs()["2026-11-25-100000-bbbbbbbb"]
    unexpected = set(rec) - set(training.OPERATOR_FIELDS) - {"stale", "stale_for_s"}
    assert (
        unexpected == set()
    ), f"fields nobody allowed reached the operator view: {sorted(unexpected)}"


def test_it_still_carries_what_an_OPERATOR_ACTUALLY_NEEDS(operator_root):
    """THE CONTROL for the two tests above.

    A projection that returned `{}` would satisfy both of them and be useless.
    """
    _write_run(
        operator_root,
        "2026-11-25-100000-cccccccc",
        status="TRAINING",
        start_time=time.time() - 60,
        model_type="cpsam",
    )
    rec = training.non_terminal_runs()["2026-11-25-100000-cccccccc"]
    for needed in ("status", "start_time", "model_type"):
        assert needed in rec, f"an operator cannot act without {needed!r}: {rec}"


# ── a non-terminal status is not a running process ───────────────────────────


def test_a_long_QUEUED_run_is_marked_STALE(operator_root):
    """The phantom case, and the one the stale sweep cannot reach.

    `get_status` sweeps a quiet run to STOPPED only from PREPARING or TRAINING.
    QUEUED is exempt on purpose, so a run that was QUEUED when its replica
    restarted stays QUEUED for ever -- exactly the outlived-its-process case
    this view exists for. Reported unmarked, it shows a busy worker that is
    idle.
    """
    _write_run(
        operator_root,
        "2026-11-25-100000-dddddddd",
        status="QUEUED",
        updated_at=time.time() - 7 * 24 * 3600,
    )
    rec = training.non_terminal_runs()["2026-11-25-100000-dddddddd"]
    assert rec["status"] == "QUEUED", "the sweep is not expected to move QUEUED"
    assert rec["stale"] is True, f"a 7-day-old QUEUED run is not marked stale: {rec}"
    assert rec["stale_for_s"] > 6 * 24 * 3600


def test_a_FRESH_run_is_NOT_marked_stale(operator_root):
    """THE CONTROL: a mark that is always set says nothing."""
    _write_run(
        operator_root,
        "2026-11-25-100000-eeeeeeee",
        status="TRAINING",
        updated_at=time.time(),
    )
    rec = training.non_terminal_runs()["2026-11-25-100000-eeeeeeee"]
    assert rec["stale"] is False, f"a run updated just now is marked stale: {rec}"


def test_an_UNREADABLE_record_is_reported_but_marked(operator_root):
    """A corrupt record arrives as status UNKNOWN; it must not read as busy."""
    d = operator_root / "2026-11-25-100000-ffffffff"
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text("{ this is not json")
    out = training.non_terminal_runs()
    rec = out.get("2026-11-25-100000-ffffffff")
    assert rec is not None, "an unreadable record vanished from the operator view"
    assert rec["stale"] is True, f"an unreadable record is not marked stale: {rec}"
