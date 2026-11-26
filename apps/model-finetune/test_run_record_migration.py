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

ADMIN = {
    "user": {
        "id": "gh|admin",
        "is_anonymous": False,
        "scope": {"workspaces": {"bioimage-io": "a"}},
    }
}
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


class FakeAM:
    """Enough artifact manager to be READ BACK, which is the point.

    `complete` is no longer this replica's account of its own writes -- the
    migration lists and downloads the records again and hashes them -- so a
    fake that only recorded calls could not exercise the check at all.
    """

    def __init__(self, fail_list=False):
        self.files = {}  # full path -> text
        self.fail_list = fail_list

    def store(self, path, text):
        self.files[path] = text

    async def edit(self, artifact_id=None, stage=None, **kw):
        pass

    async def put_file(self, artifact_id=None, file_path=None, **kw):
        return f"https://upload.example/{file_path}"

    async def commit(self, artifact_id=None, **kw):
        pass

    async def list_files(self, artifact_id=None, dir_path=None, **kw):
        if self.fail_list:
            raise RuntimeError("the artifact could not be listed")
        prefix = f"{dir_path}/"
        return [
            {
                "type": "file",
                "name": p[len(prefix) :],
                "size": len(t),
                "last_modified": 1.0,
            }
            for p, t in sorted(self.files.items())
            if p.startswith(prefix)
        ]

    async def get_file(self, artifact_id=None, file_path=None, **kw):
        return f"https://download.example/{file_path}"


class FakeHTTP:
    class _Resp:
        def __init__(self, status, text=""):
            self.status_code = status
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    def __init__(self, am):
        self.am = am

    @staticmethod
    def _path(url):
        return url.split("://", 1)[1].split("/", 1)[1]

    async def put(self, url, content=None, **kw):
        self.am.store(self._path(url), content.decode("utf-8"))
        return self._Resp(200)

    async def get(self, url, **kw):
        path = self._path(url)
        if path not in self.am.files:
            return self._Resp(404)
        return self._Resp(200, self.am.files[path])


def _wire(inst, am=None):
    """Give an instance a fake artifact manager and the HTTP client over it."""
    am = am or FakeAM()
    inst._am = am
    inst._http = FakeHTTP(am)
    inst._runs_pushed = {}
    inst.RUNS_RETRY_DELAY_S = 0

    async def _get_am():
        return am

    inst._runs_artifact_manager = _get_am
    return am


def _app(persisted=None, fail=(), tamper=(), am=None):
    """An instance whose `_persist_run_record` is STUBBED.

    The stub still deposits the record it was handed into the fake artifact, so
    the read-back has something real to read; `tamper` makes it deposit
    something else instead, which is how a record that lands with the WRONG
    CONTENT is exercised. The real write path is covered separately, by
    `test_the_PLANNED_HASH_is_a_hash_of_the_BYTES_ACTUALLY_UPLOADED`.
    """
    cls = _entry_cls()
    inst = object.__new__(cls)
    am = _wire(inst, am)

    async def _persist(session_id, status=None, *, log_failure=False, record=None):
        if session_id in fail:
            return False
        if persisted is not None:
            persisted.append(session_id)
        body = (
            json.dumps({"tampered": session_id})
            if session_id in tamper
            else json.dumps(record)
        )
        am.store(f"{training.run_record_dir('an-app')}/{session_id}.json", body)
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


def test_an_UNREADABLE_run_blocks_APPLY_but_not_the_DRY_RUN(store):
    """A partial history must not be written and then called complete. The dry
    run still reports, so an operator can see what is wrong.

    The run is made unreadable ON DISK rather than by monkeypatching a reader.
    The earlier version patched `training.get_status`, which the plan stopped
    calling when it moved to building the record the writer builds -- so the
    test went green by no longer reaching its own condition. A corrupt
    `status.json` is unreadable to whichever reader the plan happens to use.
    """
    sid = store("2026-01-01-000001-aaaaaaaa", owner="me")
    (training.sessions_root() / sid / "status.json").write_text("{not json at all")
    cls, inst = _app()
    plan = _run(cls, inst)
    assert plan["planned"] == 0 and len(plan["unreadable"]) == 1, plan
    with pytest.raises(RuntimeError) as exc:
        _run(cls, inst, apply=True, confirm="0")
    assert "could not be read" in str(exc.value)


# ── the plan's hash must describe the bytes that were really uploaded ────────


def test_the_PLANNED_HASH_is_a_hash_of_the_BYTES_ACTUALLY_UPLOADED(store):
    """The one test that would have caught the plan hashing a document nobody wrote.

    It uses the REAL `_persist_run_record`, captures what the fake artifact
    manager received, and hashes THAT. An earlier version hashed
    `run_record(name, get_status(...))` -- the shaped status, with a
    clock-derived `elapsed_s` and the stale sweep applied -- while the write
    produced the raw document, so the two disagreed on every run while
    `complete` still came back True.
    """
    import hashlib

    store("2026-01-01-000001-aaaaaaaa", owner="me")
    store("2026-01-01-000002-bbbbbbbb", owner=None)
    # A run left mid-flight with an old timestamp: this is the one the stale
    # sweep rewrites in the shaped status, so the shaped and raw documents
    # differ in the STATUS ITSELF and not merely in extra keys.
    store(
        "2026-01-01-000003-cccccccc",
        owner="me",
        status="TRAINING",
        updated_at=1.0,
        start_time=0.0,
    )

    cls = _entry_cls()
    inst = object.__new__(cls)
    am = _wire(inst)

    plan = _run(cls, inst, apply=True, confirm="3")

    assert plan["applied"] == 3, plan
    uploaded = {p.rsplit("/", 1)[-1][: -len(".json")]: t for p, t in am.files.items()}
    assert set(uploaded) == set(plan["hashes"]), (
        sorted(uploaded),
        sorted(plan["hashes"]),
    )
    for sid, text in uploaded.items():
        got = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert got == plan["hashes"][sid], (
            f"{sid}: the artifact received bytes hashing to {got}, but the plan "
            f"promised {plan['hashes'][sid]}"
        )
    assert plan["complete"] is True, plan
    assert plan["mismatched"] == [], plan


def test_a_record_that_lands_with_the_WRONG_CONTENT_is_NOT_complete(store):
    """`complete` must rest on what the ARTIFACT holds, not on the write returning True.

    Every write succeeds here; one of them deposits a different document. The
    id sets match and nothing failed, so a check built on either would pass.
    """
    for i, sid in enumerate(
        ("2026-01-01-000001-aaaaaaaa", "2026-01-01-000002-bbbbbbbb"), start=1
    ):
        store(sid, owner="me")
    cls, inst = _app(tamper={"2026-01-01-000002-bbbbbbbb"})
    plan = _run(cls, inst, apply=True, confirm="2")

    assert plan["applied"] == 2, plan
    assert plan["failed"] == [], plan
    assert plan["mismatched"] == ["2026-01-01-000002-bbbbbbbb"], plan
    assert plan["complete"] is False, plan


def test_a_READ_BACK_THAT_FAILS_is_not_a_complete_migration(store):
    """Unverifiable is not the same as verified, and must not read as success."""
    store("2026-01-01-000001-aaaaaaaa", owner="me")
    cls, inst = _app(am=FakeAM(fail_list=True))
    plan = _run(cls, inst, apply=True, confirm="1")

    assert plan["applied"] == 1, plan
    assert plan["failed"] == [], plan
    assert plan["verify_error"] is not None, plan
    assert plan["complete"] is False, plan


# ── a corrupt run must not be preserved as an empty one ─────────────────────


def test_a_CORRUPT_run_is_NOT_backfilled_as_an_EMPTY_record(store):
    """The data-loss path, asserted on the artifact rather than on an exception.

    `read_status` answers `{}` for a document that will not parse. Copying that
    into the permanent record would replace the run's history with nothing and
    report success -- and because nothing raised, the `unreadable` refusal
    could not fire either. Here the apply must refuse AND leave the artifact
    untouched.
    """
    good = store("2026-01-01-000001-aaaaaaaa", owner="me")
    bad = store("2026-01-01-000002-bbbbbbbb", owner="me")
    (training.sessions_root() / bad / "status.json").write_text("{not json at all")

    cls, inst = _app()
    plan = _run(cls, inst)
    assert plan["ids"] == [good], plan["ids"]
    assert [u["id"] for u in plan["unreadable"]] == [bad], plan["unreadable"]

    with pytest.raises(RuntimeError) as exc:
        _run(cls, inst, apply=True, confirm="1")
    assert "could not be read" in str(exc.value)
    assert inst._am.files == {}, "a refused apply still wrote to the artifact: " + repr(
        inst._am.files
    )


def test_the_STRICT_reader_and_the_ORDINARY_one_DISAGREE_on_a_corrupt_document(store):
    """The whole fix in one assertion, so neither reader can quietly become the other.

    If `read_status_strict` ever stopped raising, the migration would go back
    to writing empty records and every test above would still pass -- they
    would simply stop reaching their condition, which is exactly how the
    original version of this guard was decorative.
    """
    sid = store("2026-01-01-000001-aaaaaaaa", owner="me")
    (training.sessions_root() / sid / "status.json").write_text("{not json at all")

    assert training.read_status(sid) == {}, "the ordinary reader is meant to fail soft"
    with pytest.raises(ValueError):
        training.read_status_strict(sid)


def test_the_STRICT_reader_AGREES_with_the_ordinary_one_on_a_GOOD_document(store):
    """The paired control: strictness must not change what a readable run says."""
    sid = store("2026-01-01-000001-aaaaaaaa", owner="me", status="COMPLETED")
    assert training.read_status_strict(sid) == training.read_status(sid)


# ── which document gets written, not merely that two of ours agree ──────────


def test_a_BACKFILLED_record_is_IDENTICAL_to_what_the_ORDINARY_SYNC_writes(store):
    """Self-consistency is not enough: both halves can pick the SAME wrong document.

    Hashing the uploaded bytes against the plan catches the plan and the write
    DISAGREEING. It cannot catch them agreeing on the wrong thing -- feed the
    shaped status to both and the hashes still match, which is a mutation that
    passed. What actually has to be true is that a backfilled record is the
    record the ordinary per-poll sync would have written for the same run, so
    a migrated history and a synced one cannot be told apart.

    The stale run is the discriminator: the shaped status has the stale sweep
    applied and reports STOPPED, while the document on disk says TRAINING.
    """
    store("2026-01-01-000001-aaaaaaaa", owner="me")
    stale = store(
        "2026-01-01-000003-cccccccc",
        owner="me",
        status="TRAINING",
        updated_at=1.0,
        start_time=0.0,
    )

    cls = _entry_cls()
    inst = object.__new__(cls)
    am = _wire(inst)
    plan = _run(cls, inst, apply=True, confirm="2")
    assert plan["complete"] is True, plan

    for path, text in am.files.items():
        sid = path.rsplit("/", 1)[-1][: -len(".json")]
        assert text == json.dumps(training.run_record(sid)), (
            f"{sid}: the backfilled record is not the document the ordinary "
            f"sync writes for the same run"
        )

    # And name the discriminator explicitly, so this cannot pass because every
    # run in the fixture happened to be finished.
    written_stale = json.loads(
        am.files[f"{training.run_record_dir('an-app')}/{stale}.json"]
    )
    assert (written_stale.get("status") or {}).get("status") == "TRAINING", (
        "the stale sweep reached the permanent record: it holds a status the "
        "run's own document on disk does not have"
    )


def test_the_writer_USES_the_record_it_was_GIVEN(store):
    """`record=` is a contract, and a writer that rebuilds silently breaks it.

    The migration hashes the document before writing it. If the writer rebuilds
    instead, it writes a document nobody hashed -- harmless while the run sits
    still, and wrong the moment it does not.
    """
    sid = store("2026-01-01-000001-aaaaaaaa", owner="me")
    cls = _entry_cls()
    inst = object.__new__(cls)
    am = _wire(inst)

    sentinel = {"session_id": sid, "handed_in": True}
    ok = asyncio.run(cls._persist_run_record(inst, sid, record=sentinel))

    assert ok is True
    path = f"{training.run_record_dir('an-app')}/{sid}.json"
    assert (
        json.loads(am.files[path]) == sentinel
    ), "the writer rebuilt the record instead of writing the one it was handed"
