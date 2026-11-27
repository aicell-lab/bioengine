"""At the code level: a run on ONE replica's disk is listed by ANOTHER.

WHAT IT COVERS, AND WHY NOTHING ELSE DOES. The clauses say two replicas must
list the SAME runs, and that a run started on either appears in both. Every
existing test builds ONE instance against ONE sessions root -- including
`test_ONE_REPLICAS_bookkeeping_does_not_reach_ANOTHERS`, which does make a
second instance but shares the first's root and is about class-dict pollution,
not visibility. So the one thing this file exists to cover has no test that would
notice it breaking.

THE TWO REPLICAS HAVE SEPARATE DISKS AND ONE SHARED ARTIFACT. That is the real
topology: `sessions_root` is a module-level function, so it is switched per
replica around each call rather than patched once -- a single patch would give
both instances the same disk and the test would pass without the shared record
doing any work at all.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import training
from test_run_record_sync import FakeAM, FakeHTTP, _entry_class

APP = "model-finetune"


class Replica:
    """One app instance with a sessions root of its own."""

    def __init__(self, root, am, store_id=APP):
        self.root = root
        self.store_id = store_id
        cls = _entry_class()
        inst = cls.__new__(cls)
        inst._runs_pushed = {}
        inst._runs_cache = None
        inst._runs_meta = {}
        inst._runs_cache_at = 0.0
        inst.RUNS_RETRY_DELAY_S = 0
        inst.RUNS_CACHE_TTL_S = 0

        async def _am():
            return am

        inst._runs_artifact_manager = _am
        inst._http = FakeHTTP(am)
        self.inst = inst
        self.cls = cls

    def plant(self, sid, **fields):
        d = self.root / sid
        d.mkdir(parents=True, exist_ok=True)
        status = {"status": "COMPLETED", "updated_at": 0, "owner": "github|1", **fields}
        (d / "status.json").write_text(json.dumps(status))
        (d / "training_params.json").write_text(json.dumps({"n_epochs": 1}))
        return sid


@pytest.fixture()
def two_replicas(tmp_path, monkeypatch):
    """A and B: separate disks, ONE artifact between them."""
    am = FakeAM()
    a_root = tmp_path / "replica-a"
    b_root = tmp_path / "replica-b"
    a_root.mkdir()
    b_root.mkdir()

    current = {"root": a_root, "store": APP}
    # Switched per replica rather than patched once. Patching once would hand
    # both instances the same disk, and then B "seeing" A's run would prove
    # nothing about the shared record.
    monkeypatch.setattr(training, "sessions_root", lambda: current["root"])
    monkeypatch.setattr(training, "session_dir", lambda sid: current["root"] / sid)
    monkeypatch.setattr(training, "record_store_id", lambda: current["store"])

    A = Replica(a_root, am)
    B = Replica(b_root, am)

    def as_replica(r):
        current["root"] = r.root
        current["store"] = r.store_id
        return r

    return A, B, am, as_replica, current


def _list(cls, inst, context=None):
    return asyncio.run(cls.list_training_sessions(inst, context=context))


ADMIN = {
    "user": {
        "id": "github|1",
        "is_anonymous": False,
        "scope": {"workspaces": {"bioimage-io": "a"}},
    }
}


def test_a_run_on_ONE_replicas_disk_is_LISTED_BY_THE_OTHER(two_replicas):
    """Clause 3's substance: B has an empty disk and must still see A's run."""
    A, B, am, as_replica, _cur = two_replicas

    as_replica(A)
    sid = A.plant("2026-11-26-120000-aaaaaaaa")
    assert asyncio.run(A.inst._persist_run_record(sid)) is True

    as_replica(B)
    assert not list(
        B.root.iterdir()
    ), "B's disk must be empty for this to mean anything"
    seen = _list(B.cls, B.inst, ADMIN)

    assert sid in seen, (
        "B cannot see a run A recorded, so the two replicas do NOT list the "
        f"same runs: {sorted(seen)}"
    )


def test_BOTH_replicas_list_the_SAME_run_ids(two_replicas):
    """Clause 1's wording: the same runs, not merely overlapping ones."""
    A, B, am, as_replica, _cur = two_replicas

    as_replica(A)
    a_run = A.plant("2026-11-26-120000-aaaaaaaa")
    assert asyncio.run(A.inst._persist_run_record(a_run)) is True

    as_replica(B)
    b_run = B.plant("2026-11-26-130000-bbbbbbbb")
    assert asyncio.run(B.inst._persist_run_record(b_run)) is True

    as_replica(A)
    A.inst._runs_cache = None
    from_a = set(_list(A.cls, A.inst, ADMIN))
    as_replica(B)
    B.inst._runs_cache = None
    from_b = set(_list(B.cls, B.inst, ADMIN))

    assert from_a == from_b, (
        f"the replicas disagree: only A sees {sorted(from_a - from_b)}, "
        f"only B sees {sorted(from_b - from_a)}"
    )
    assert {a_run, b_run} <= from_a, sorted(from_a)


def test_a_replica_that_CANNOT_NAME_ITS_DEPLOYMENT_sees_only_its_own_disk(two_replicas):
    """The control that stops the first test passing for the wrong reason.

    If B saw A's run through some leak other than the shared record -- a shared
    module-level cache, a root that did not really switch -- then a replica
    with sharing OFF would see it too. Sharing off, B must see nothing.
    """
    A, B, am, as_replica, _cur = two_replicas

    as_replica(A)
    sid = A.plant("2026-11-26-120000-aaaaaaaa")
    assert asyncio.run(A.inst._persist_run_record(sid)) is True

    B.store_id = None  # record_store_id() -> None: sharing is OFF
    as_replica(B)
    B.inst._runs_cache = None
    seen = _list(B.cls, B.inst, ADMIN)

    assert sid not in seen, (
        "a replica with sharing OFF still saw another replica's run, so the "
        "first test's visibility does not come from the shared record"
    )


def test_ANOTHER_DEPLOYMENTS_run_is_NOT_listed(two_replicas):
    """One artifact is shared by every deployment; histories must not merge."""
    A, B, am, as_replica, _cur = two_replicas

    as_replica(A)
    sid = A.plant("2026-11-26-120000-aaaaaaaa")
    assert asyncio.run(A.inst._persist_run_record(sid)) is True

    B.store_id = "model-finetune-dev"  # a different deployment, same artifact
    as_replica(B)
    B.inst._runs_cache = None
    seen = _list(B.cls, B.inst, ADMIN)

    assert sid not in seen, (
        "a throwaway deployment listed production's run; the per-deployment "
        "directory is not holding"
    )
