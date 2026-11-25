"""The run history is shared, and sharing it does not widen who may see it.

Run history was one worker's filesystem, so two instances of the app answered
differently. Measured on one worker as one administrator: the production
instance listed 261 runs, a second instance listed 19, and the two had NOTHING
in common -- a user's previous runs depended on which instance their tab was
pinned to, and nothing reported that it had changed.

The fix is one app-owned record every instance reads. That makes the per-caller
filter the only thing standing between callers, so these tests spend most of
their weight on the ways a SHARED record could become a route to a run the
local rule would have refused:

* a record from another instance is filtered by the same rule as a local run;
* a session this instance holds is never re-read from the shared copy, not
  even when the local rule just refused it to this caller;
* a record that says nothing about its owner is shown to NOBODY, rather than
  being read as ownerless and handed to an administrator;
* the owner travels INSIDE the record -- it has to, or a run from another
  instance cannot be attributed at all -- and still never reaches a caller.

The last one is why the record keeps a field the listing strips: the fence is
the artifact's own permissions (private, standalone, app-credential only --
anonymous read, anonymous list_files and anonymous HTTP all measured as
refused) plus `redact_private` on the way out.
"""

from __future__ import annotations

import asyncio
import json
import re
import time

import pytest

import training

ME = "github|49943582"
OTHER = "github|000111"
ADMIN_ID = "github|777000"


def ctx(uid, *, anonymous=True, grants=None):
    user = {"id": uid, "is_anonymous": anonymous}
    if grants is not None:
        user["scope"] = {"workspaces": grants}
    return {"user": user}


ADMIN = ctx(ADMIN_ID, anonymous=False, grants={"bioimage-io": "a"})
MINE = ctx(ME, anonymous=False)
THEIRS = ctx(OTHER, anonymous=False)


@pytest.fixture()
def local(tmp_path, monkeypatch):
    """A sessions root this instance holds, plus a planting helper.

    Readable ids are used, so the shape rule is relaxed exactly as the
    per-caller suite relaxes it -- and for the same reason: these tests are
    about WHO MAY SEE WHICH RUN, and the real pattern is pinned elsewhere. The
    relaxed pattern is applied to the shared-record leg too, otherwise the
    records would all be dropped as malformed and every assertion below would
    pass while measuring nothing.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(training, "sessions_root", lambda: root)
    monkeypatch.setattr(training, "session_dir", lambda sid: root / sid)
    monkeypatch.setattr(training, "SESSION_ID_PATTERN", re.compile(r"\A[\w.-]+\Z"))

    def add(sid, owner, **fields):
        d = root / sid
        d.mkdir(parents=True, exist_ok=True)
        status = {"status": "COMPLETED", "updated_at": 0, **fields}
        if owner is not None:
            status["owner"] = owner
        (d / "status.json").write_text(json.dumps(status))
        return sid

    return add


def record(sid, owner, **fields):
    """A record in the shape `run_record` produces."""
    status = {"status": "COMPLETED", "updated_at": 0, **fields}
    if owner is not None:
        status["owner"] = owner
    return {
        "record_version": training.RUN_RECORD_VERSION,
        "session_id": sid,
        "config": {"n_epochs": 1},
        "status": status,
    }


# ── a run from another instance is listed, and filtered the same way ────────


def test_a_run_from_ANOTHER_instance_is_listed_to_its_owner(local):
    """The clause this row exists for: two instances, one history."""
    local("mine-here", ME)
    shared = {"mine-elsewhere": record("mine-elsewhere", ME)}
    out = training.list_sessions(MINE, records=shared)
    assert set(out) == {"mine-here", "mine-elsewhere"}, (
        "a run this caller started on another instance is missing, which is "
        "the split history this record is meant to end"
    )


def test_and_NOT_to_a_different_caller(local):
    """Paired with the test above, which proves the run is really there.

    Without the pairing, a listing that returned nothing at all would pass
    this and say nothing.
    """
    shared = {"mine-elsewhere": record("mine-elsewhere", ME)}
    assert training.list_sessions(MINE, records=shared) != {}
    assert training.list_sessions(THEIRS, records=shared) == {}


def test_an_OWNERLESS_shared_record_goes_only_to_an_administrator(local):
    shared = {"ancient": record("ancient", None)}
    assert training.list_sessions(THEIRS, records=shared) == {}
    assert set(training.list_sessions(ADMIN, records=shared)) == {"ancient"}


def test_a_record_with_a_NULL_owner_is_ownerless_not_a_match(local):
    """`caller_identity` returns None for an unidentified caller and the
    PREPARING write spreads it, so the key is present and null. A comparison
    that matched None to None would hand every unattributed run to every
    unidentified caller."""
    shared = {"null-owner": record("null-owner", None)}
    shared["null-owner"]["status"]["owner"] = None
    anon = ctx("anonymouz-dot-basement-35411643")
    assert training.list_sessions(anon, records=shared) == {}
    assert set(training.list_sessions(ADMIN, records=shared)) == {"null-owner"}


# ── the shared copy must not become a second route to a local run ───────────


def test_a_LOCAL_run_this_caller_cannot_see_is_NOT_resurfaced_by_the_record(local):
    """THE bypass this design could have introduced.

    The local rule refuses somebody else's run. If the shared leg then read
    the same session from the record -- which carries the owner, so it would
    be filtered too -- it would still be a second evaluation of the same
    question, and a record whose owner went missing in a migration would be
    shown to an administrator while the local copy was correctly attributed.
    So a session this instance HOLDS is skipped by id, before any visibility
    question is asked.
    """
    local("theirs", OTHER)
    # The shared copy of the SAME session, with its owner lost.
    shared = {"theirs": record("theirs", None)}
    assert training.list_sessions(ADMIN, records=shared) == {}, (
        "an administrator was shown a run through the shared record that the "
        "local rule attributes to somebody else"
    )
    assert set(training.list_sessions(THEIRS, records=shared)) == {"theirs"}


def test_the_LOCAL_status_wins_over_a_stale_record(local):
    """The record is only as fresh as the last sync; local disk is re-read on
    every call. A run that has moved on must not be reported with its old
    state."""
    # `updated_at` has to be NOW: `get_status` sweeps a TRAINING run with no
    # recent update to STOPPED, so a zero timestamp would have this test
    # comparing the record against the stale sweep rather than against the
    # local status -- it would pass for the wrong reason, or fail for one.
    local("mine", ME, status="TRAINING", updated_at=time.time())
    shared = {"mine": record("mine", ME, status="COMPLETED")}
    out = training.list_sessions(MINE, records=shared)
    assert out["mine"]["status"] == "TRAINING", out["mine"]


# ── a shared run is reported exactly as a local one ─────────────────────────
#
# THE RED-MAKER the first review round named, and it was right. The first
# version returned the stored document raw while the local path went through
# `get_status`, so the same run read both ways came back:
#
#   LOCAL  STOPPED  [checkpoint_available, config, elapsed_s, message,
#                    session_id, start_time, status, updated_at]
#   REMOTE TRAINING [start_time, status, updated_at]
#
# A run whose instance had died reported TRAINING forever on every other
# instance, because the stale sweep lives in the shaping and the shared path
# skipped it; and four fields the page reads were simply absent.


def _record_from_local(sid):
    """The record this run would be shared as, built the way the app builds
    it -- from the run's own files, not hand-written."""
    return training.run_record(sid)


def test_a_run_LISTED_BOTH_WAYS_comes_back_IDENTICAL(local):
    local("mine", ME, status="COMPLETED", updated_at=time.time(), start_time=1.0)
    rec = _record_from_local("mine")

    from_local = training.list_sessions(MINE)["mine"]
    # Same run, now reachable only through the shared copy.
    (training.sessions_root() / "mine" / "status.json").unlink()
    from_shared = training.list_sessions(MINE, records={"mine": rec})["mine"]

    assert from_shared == from_local, (
        "a shared run is reported differently from the same run read locally; "
        f"local-only keys {sorted(set(from_local) - set(from_shared))}, "
        f"shared-only keys {sorted(set(from_shared) - set(from_local))}"
    )


def test_a_STALE_shared_run_is_SWEPT_like_a_local_one(local):
    """The sharper half: the sweep is what stops a dead instance's run from
    reporting TRAINING forever everywhere else."""
    local("mine", ME, status="TRAINING", updated_at=time.time() - 86400, start_time=1.0)
    rec = _record_from_local("mine")
    assert rec["status"]["status"] == "TRAINING", "the stored document is unswept"

    (training.sessions_root() / "mine" / "status.json").unlink()
    out = training.list_sessions(MINE, records={"mine": rec})["mine"]
    assert out["status"] == "STOPPED", out
    assert "stale window" in out.get("message", ""), out


def test_a_shared_run_carries_the_FIELDS_THE_PAGE_READS(local):
    local("mine", ME, status="COMPLETED", updated_at=time.time(), start_time=1.0)
    rec = _record_from_local("mine")
    (training.sessions_root() / "mine" / "status.json").unlink()
    out = training.list_sessions(MINE, records={"mine": rec})["mine"]
    for field in ("session_id", "config", "elapsed_s", "checkpoint_available"):
        assert field in out, f"{field} is missing from a shared run"
    assert out["session_id"] == "mine"


def test_a_shared_run_is_NOT_claimed_to_be_SERVABLE_here(local):
    """`checkpoint_available` is a fact about a PLACE. The weights are on the
    instance that trained the run, so claiming the record's value would have
    the page offer a model no replica here holds."""
    local("mine", ME, status="COMPLETED", updated_at=time.time())
    rec = _record_from_local("mine")
    rec["status"]["checkpoint_available"] = True
    (training.sessions_root() / "mine" / "status.json").unlink()
    out = training.list_sessions(MINE, records={"mine": rec})["mine"]
    assert out["checkpoint_available"] is False, out


# ── a record that says nothing fails CLOSED ─────────────────────────────────


@pytest.mark.parametrize(
    "bad",
    [
        {"record_version": 1, "session_id": "x"},  # no status at all
        {"status": "COMPLETED"},  # status is a string, not the document
        {"status": None},
        None,
        "not even a mapping",
        [],
    ],
)
def test_a_record_with_no_usable_status_is_shown_to_NOBODY(local, bad):
    """A missing status has no owner in it, and the ownerless rule would hand
    it to an administrator. That is the wrong direction for a record this
    instance cannot interpret, so it is dropped and counted."""
    shared = {"junk": bad}
    assert training.list_sessions(ADMIN, records=shared) == {}
    assert training.list_sessions(MINE, records=shared) == {}


def test_a_record_key_that_is_not_a_session_id_is_dropped(local, monkeypatch):
    """The key becomes a session id in the answer, so an id the server could
    not have issued must not travel through the shared record into a
    listing."""
    monkeypatch.setattr(
        training, "SESSION_ID_PATTERN", re.compile(r"\A\d{4}-\d{2}-\d{2}-\d{6}-\w+\Z")
    )
    shared = {"../escape": record("../escape", ADMIN_ID)}
    assert training.list_sessions(ADMIN, records=shared) == {}


# ── the owner is carried and never published ────────────────────────────────


def test_the_OWNER_is_in_the_record_and_NOT_in_the_answer(local):
    shared = {"mine-elsewhere": record("mine-elsewhere", ME)}
    assert shared["mine-elsewhere"]["status"]["owner"] == ME
    out = training.list_sessions(MINE, records=shared)
    for field in training.PRIVATE_STATUS_FIELDS:
        assert field not in out["mine-elsewhere"], f"{field} reached the caller"
    assert ME not in json.dumps(out), "the owner identity reached the caller"


def test_redacting_the_answer_does_not_MUTATE_the_record(local):
    """The records mapping belongs to the caller of `list_sessions`, which
    re-uses it across two listings in the same request path. A strip that
    edited it in place would make the second listing see a record with no
    owner -- and the ownerless rule would then hand it to an administrator."""
    shared = {"mine-elsewhere": record("mine-elsewhere", ME)}
    training.list_sessions(MINE, records=shared)
    assert shared["mine-elsewhere"]["status"]["owner"] == ME
    assert training.list_sessions(ADMIN, records=shared) == {}


# ── no records at all is a degraded answer, not an error ────────────────────


def test_no_records_still_lists_the_local_runs(local):
    local("mine", ME)
    assert set(training.list_sessions(MINE)) == {"mine"}
    assert set(training.list_sessions(MINE, records=None)) == {"mine"}
    assert set(training.list_sessions(MINE, records={})) == {"mine"}
