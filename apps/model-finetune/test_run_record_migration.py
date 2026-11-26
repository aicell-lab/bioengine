"""The backfill of on-disk runs into the shared record, and what it refuses.

Clause 2 wants every existing on-disk run in the record, with the count before
and after equal. Two things make that easy to satisfy wrongly, and both are
pinned here.

THE STORE IS ENUMERATED BY THE FILESYSTEM. The per-caller listing returns what
the CALLER may see -- measured at 269 of 291 on the production store, because a
run owned by somebody else, or by an anonymous identity that no longer exists,
is invisible to it by design. A backfill driven by that listing skips exactly
the runs most worth preserving, and `before == after` still passes, because
both sides count the same subset.

EQUALITY IS BY ID SET, NOT BY COUNT. Two sets of the same size are not the same
set.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import training

ADMIN = {"user": {"id": "gh|admin", "is_anonymous": False,
                  "scope": {"workspaces": {"bioimage-io": "a"}}}}
PLAIN = {"user": {"id": "gh|plain", "is_anonymous": False}}


def _entry_cls():
    import entry

    return getattr(entry.EntryApp, "func_or_class", entry.EntryApp)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(training, "sessions_root", lambda: root)
    monkeypatch.setattr(training, "session_dir", lambda sid: root / sid)
    monkeypatch.setattr(training, "record_store_id", lambda: "an-app")

    def add(sid, owner=None, **fields):
        d = root / sid
        d.mkdir(parents=True, exist_ok=True)
        st = {"status": "COMPLETED", "updated_at": 0, **fields}
        if owner is not None:
            st["owner"] = owner
        (d / "status.json").write_text(json.dumps(st))
        return sid

    return add


def _app(persisted=None, fail=()):
    cls = _entry_cls()
    inst = object.__new__(cls)

    async def _persist(session_id, status=None, *, log_failure=False):
        if session_id in fail:
            return False
        if persisted is not None:
            persisted.append(session_id)
        return True

    inst._persist_run_record = _persist
    return cls, inst


def _run(cls, inst, **kw):
    return asyncio.run(cls.migrate_run_records(inst, context=ADMIN, **kw))


def test_the_PLAN_covers_every_run_ON_DISK_including_other_peoples(store):
    """The runs a caller cannot see are the point of the backfill."""
    store("2026-01-01-000001-aaaaaaaa", owner="me")
    store("2026-01-01-000002-bbbbbbbb", owner="someone-else")
    store("2026-01-01-000003-cccccccc", owner=None)  # ownerless: invisible to all
    cls, inst = _app()
    plan = _run(cls, inst)
    assert plan["dry_run"] is True
    assert plan["planned"] == 3, plan
    assert plan["ids"] == [
        "2026-01-01-000001-aaaaaaaa",
        "2026-01-01-000002-bbbbbbbb",
        "2026-01-01-000003-cccccccc",
    ], plan["ids"]


def test_a_DRY_RUN_WRITES_NOTHING(store):
    store("2026-01-01-000001-aaaaaaaa", owner="me")
    persisted = []
    cls, inst = _app(persisted)
    plan = _run(cls, inst)
    assert persisted == [], "the dry run wrote records"
    assert plan["applied"] == 0, plan


def test_APPLY_without_the_CONFIRM_COUNT_is_refused(store):
    """`apply=true` alone must not be enough: the count proves the caller read
    the plan rather than passing a flag."""
    store("2026-01-01-000001-aaaaaaaa", owner="me")
    persisted = []
    cls, inst = _app(persisted)
    with pytest.raises(ValueError) as exc:
        _run(cls, inst, apply=True)
    assert "confirm" in str(exc.value)
    assert persisted == [], "a refused apply still wrote"


def test_APPLY_with_the_RIGHT_count_writes_every_id(store):
    for i, owner in enumerate(("me", "other", None), start=1):
        store(f"2026-01-01-00000{i}-aaaaaaa{i}", owner=owner)
    persisted = []
    cls, inst = _app(persisted)
    plan = _run(cls, inst, apply=True, confirm="3")
    assert plan["applied"] == 3, plan
    assert plan["complete"] is True, plan
    assert sorted(persisted) == plan["ids"], (persisted, plan["ids"])


def test_a_write_that_FAILS_makes_the_run_INCOMPLETE(store):
    """`complete` is the claim the clause rests on, so one failure must sink it
    rather than being averaged away by a count."""
    for i in (1, 2):
        store(f"2026-01-01-00000{i}-aaaaaaa{i}", owner="me")
    cls, inst = _app(fail={"2026-01-01-000002-aaaaaaa2"})
    plan = _run(cls, inst, apply=True, confirm="2")
    assert plan["complete"] is False, plan
    assert plan["failed"] == ["2026-01-01-000002-aaaaaaa2"], plan


def test_a_NON_ADMIN_is_refused(store):
    store("2026-01-01-000001-aaaaaaaa", owner="me")
    cls, inst = _app()
    with pytest.raises(training.NotTheOwner):
        asyncio.run(cls.migrate_run_records(inst, context=PLAIN))


def test_a_replica_that_CANNOT_NAME_ITS_DEPLOYMENT_refuses(store, monkeypatch):
    """Writing into a fallback bucket shared by every such replica is the thing
    `record_store_id` returning None exists to prevent."""
    monkeypatch.setattr(training, "record_store_id", lambda: None)
    store("2026-01-01-000001-aaaaaaaa", owner="me")
    cls, inst = _app()
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(cls.migrate_run_records(inst, context=ADMIN))
    assert "which deployment" in str(exc.value)


def test_an_UNREADABLE_run_blocks_APPLY_but_not_the_DRY_RUN(store, monkeypatch):
    """A partial history must not be written and then called complete. The dry
    run still reports, so an operator can see what is wrong."""
    store("2026-01-01-000001-aaaaaaaa", owner="me")
    real = training.get_status

    def _boom(name, **kw):
        if name.endswith("aaaaaaaa"):
            raise OSError("unreadable")
        return real(name, **kw)

    monkeypatch.setattr(training, "get_status", _boom)
    cls, inst = _app()
    plan = _run(cls, inst)
    assert plan["planned"] == 0 and len(plan["unreadable"]) == 1, plan
    with pytest.raises(RuntimeError) as exc:
        _run(cls, inst, apply=True, confirm="0")
    assert "could not be read" in str(exc.value)
