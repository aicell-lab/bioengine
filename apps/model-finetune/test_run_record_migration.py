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

    def __init__(self, fail_list=False, page_size=None, ignores_offset=False):
        self.files = {}  # full path -> text
        self.fail_list = fail_list
        #: Serve at most this many entries per call, as the real store does.
        #: `None` means "no cap of my own" -- the caller's limit decides.
        self.page_size = page_size
        #: Serve page one forever, as a store that dropped `offset` would.
        self.ignores_offset = ignores_offset
        #: Hard ceiling on how many times this fake may be LISTED. See the
        #: note on `list_files`: it turns a non-terminating caller into an
        #: error rather than a hang.
        self.max_asks = 50
        self.list_asks = []  # (dir_path, limit, offset) as ASKED, not as served
        #: How many times the artifact was STAGED and COMMITTED. Counted
        #: because the cost this fake cannot otherwise show is round trips:
        #: the backfill used to stage and commit once PER RECORD.
        self.edits = 0
        self.commits = 0
        #: Make the first N stage/commit calls RAISE. The artifact-level stage
        #: flag is cleared by any other instance's commit, so both of these
        #: fail in production; the backfill must report rather than raise.
        self.fail_edits = 0
        self.fail_commits = 0
        #: Uploads in flight, and the high-water mark, so a test can assert
        #: the concurrency BOUND rather than hoping.
        self.in_flight = 0
        self.max_in_flight = 0

    def store(self, path, text):
        self.files[path] = text

    async def edit(self, artifact_id=None, stage=None, **kw):
        self.edits += 1
        if self.fail_edits > 0:
            self.fail_edits -= 1
            raise RuntimeError("artifact is not in staging mode")

    async def put_file(self, artifact_id=None, file_path=None, **kw):
        return f"https://upload.example/{file_path}"

    async def commit(self, artifact_id=None, **kw):
        self.commits += 1
        if self.fail_commits > 0:
            self.fail_commits -= 1
            raise RuntimeError("artifact is not in staging mode")

    async def list_files(
        self, artifact_id=None, dir_path=None, limit=1000, offset=0, **kw
    ):
        """A PAGING listing, because the real one pages.

        This fake used to return the whole directory on every call, which
        models a server that does not exist: the real `list_files` defaults to
        a 1000-entry page, caps at 10000, and takes an `offset`. Against the
        old fake, a read-back that stopped at page one was indistinguishable
        from one that followed the pagination, so the truncation this row is
        about could not have been caught here.

        It records what each call ASKED for, not only what was served, because
        the entire defect is that those two differ.

        **BOUNDED, so a missing termination guard FAILS instead of HANGING.**
        Removing the production loop's no-progress guard used not to turn this
        suite red -- it hung, and had to be killed. A hang is not a verdict:
        pytest-timeout is not installed, so CI would hang the same way rather
        than report, and the very test that says "it must stop, not spin" was
        the one outcome it could not report. The ceiling turns that mutant into
        a loud error.
        """
        if self.fail_list:
            raise RuntimeError("the artifact could not be listed")
        self.list_asks.append((dir_path, limit, offset))
        if len(self.list_asks) > self.max_asks:
            raise RuntimeError(
                f"list_files asked {len(self.list_asks)} times, past the "
                f"{self.max_asks}-call ceiling: the caller is not terminating. "
                "This ceiling exists so that a missing no-progress guard fails "
                "LOUDLY instead of hanging a suite that cannot time out."
            )
        prefix = f"{dir_path}/"
        rows = [
            {
                "type": "file",
                "name": p[len(prefix) :],
                "size": len(t),
                "last_modified": 1.0,
            }
            for p, t in sorted(self.files.items())
            if p.startswith(prefix)
        ]
        served = min(limit, self.page_size) if self.page_size else limit
        start = 0 if self.ignores_offset else offset
        return rows[start : start + served]

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
        self.am.in_flight += 1
        self.am.max_in_flight = max(self.am.max_in_flight, self.am.in_flight)
        try:
            # Yield so overlapping uploads are actually interleaved by the loop;
            # without this every PUT completes before the next starts and the
            # high-water mark would read 1 however concurrent the caller is.
            await asyncio.sleep(0)
            self.am.store(self._path(url), content.decode("utf-8"))
            return self._Resp(200)
        finally:
            self.am.in_flight -= 1

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
    """An instance whose fake UPLOAD records what the artifact received.

    These levers used to hang off a stubbed `_persist_run_record`. The apply
    path no longer calls that helper at all -- it stages once, uploads every
    record, and commits once, because calling the helper per record meant one
    artifact COMMIT per record and about 1565 serial round trips for a
    313-record store. A stub there now intercepts nothing: `persisted` stayed
    empty and `fail`/`tamper` reached no code, while `applied` and `complete`
    still read correct, which is exactly the shape of a test that has stopped
    testing.

    So they moved onto the HTTP PUT the backfill actually performs. That
    exercises the real `put_file` -> PUT -> `commit` sequence, and `persisted`
    now means "the artifact received this record" rather than "this private
    method was called" -- a stronger statement, and one that survives the next
    refactor of the write path.
    """
    cls = _entry_cls()
    inst = object.__new__(cls)
    am = _wire(inst, am)
    real_put = inst._http.put

    def _id_from(url):
        return url.rsplit("/", 1)[-1][: -len(".json")]

    async def _put(url, content=None, **kw):
        sid = _id_from(url)
        if sid in fail:
            # The real failure arrives as a raising PUT, which the upload path
            # catches per record; returning a 500 would exercise
            # `raise_for_status` instead. Both land in `failed`.
            raise RuntimeError(f"upload refused for {sid}")
        if persisted is not None:
            persisted.append(sid)
        if sid in tamper:
            content = json.dumps({"tampered": sid}).encode("utf-8")
        return await real_put(url, content=content, **kw)

    inst._http.put = _put
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


# ── the round-trip cost, which is what the ceiling was about ───────────────


def test_a_BACKFILL_STAGES_ONCE_AND_COMMITS_ONCE_however_many_records(store):
    """THE property the fix buys. The timeout was a round-trip count, not a write.

    The apply used to call `_persist_run_record` per record, and that helper wraps
    `edit(stage=True)` ... `commit()` around ONE file -- so a 313-record backfill
    performed 313 artifact commits and about 1565 round trips, taken serially. The
    stage/commit count has to be independent of how many records there are.
    """
    for i in range(1, 13):
        store(f"2026-01-01-0000{i:02d}-aaaaaaa{i % 10}")
    am = FakeAM()
    cls, inst = _app(am=am)

    plan = _run(cls, inst, apply=True, confirm="12")

    assert plan["applied"] == 12, plan
    assert plan["complete"] is True, plan
    assert am.edits == 1, f"staged {am.edits} times for 12 records"
    assert am.commits == 1, f"committed {am.commits} times for 12 records"


def test_the_upload_concurrency_is_BOUNDED_by_the_class_attribute(store):
    """Bounded, not unbounded -- 313 simultaneous connections is a different outage.

    Asserted on the high-water mark of uploads in flight, and on it being ABOVE one,
    so this cannot pass because the uploads happened to serialise.
    """
    for i in range(1, 13):
        store(f"2026-01-01-0000{i:02d}-bbbbbbb{i % 10}")
    am = FakeAM()
    cls, inst = _app(am=am)
    inst.RUNS_MIGRATE_CONCURRENCY = 3

    _run(cls, inst, apply=True, confirm="12")

    assert am.max_in_flight > 1, "the uploads never overlapped, so this asserts nothing"
    assert am.max_in_flight <= 3, f"concurrency reached {am.max_in_flight}, bound was 3"


def test_a_concurrency_of_ONE_still_completes(store):
    """THE CONTROL on the bound: it is a ceiling, not a requirement to overlap.

    A fix that only worked when requests overlapped would pass the test above and
    break on a store that forced serialisation.
    """
    for i in range(1, 5):
        store(f"2026-01-01-0000{i:02d}-ccccccc{i}")
    am = FakeAM()
    cls, inst = _app(am=am)
    inst.RUNS_MIGRATE_CONCURRENCY = 1

    plan = _run(cls, inst, apply=True, confirm="4")

    assert plan["complete"] is True, plan
    assert am.max_in_flight == 1, am.max_in_flight
    assert am.commits == 1, am.commits


# ── neither the stage nor the commit may raise out of the apply ────────────


def test_a_commit_that_loses_the_race_ONCE_is_retried_and_still_completes(store):
    """The stage flag is artifact-level and any other instance's commit clears it.

    `_persist_run_record` retried for exactly this reason. The bulk path holds the
    flag open for the WHOLE upload phase instead of one file, and this replica's own
    sync loop is a concurrent stage/commit writer, so the window is WIDER here, not
    narrower. One lost race must not cost the run.
    """
    for i in range(1, 5):
        store(f"2026-01-01-0000{i:02d}-ddddddd{i}")
    am = FakeAM()
    am.fail_commits = 1
    cls, inst = _app(am=am)

    plan = _run(cls, inst, apply=True, confirm="4")

    assert plan["complete"] is True, plan
    assert plan["commit_error"] is None, plan["commit_error"]
    assert am.commits == 2, f"the commit was not retried (commits={am.commits})"


def test_a_commit_that_KEEPS_failing_RETURNS_A_REPORT_rather_than_raising(store):
    """THE regression this answers. An exception here is the row's own harm.

    From the caller's side a raised error is indistinguishable from "the write did
    not happen" -- which is exactly what this operation exists to let them tell
    apart. The per-record path retried and still returned a report; the first
    version of the bulk path let the error escape with no report at all.
    """
    for i in range(1, 5):
        store(f"2026-01-01-0000{i:02d}-eeeeeee{i}")
    am = FakeAM()
    am.fail_commits = 99
    cls, inst = _app(am=am)

    plan = _run(cls, inst, apply=True, confirm="4")  # must NOT raise

    assert plan["complete"] is False, plan
    assert plan["commit_error"], "the commit failure was not recorded"
    assert "staging" in plan["commit_error"], plan["commit_error"]
    # the read-back still ran, so the caller gets a verdict and not a silence
    assert "verified" in plan and "mismatched" in plan, plan


def test_a_STAGE_that_keeps_failing_also_returns_a_report(store):
    """Same property one step earlier: nothing was uploaded, and that is reported."""
    for i in range(1, 5):
        store(f"2026-01-01-0000{i:02d}-fffffff{i}")
    am = FakeAM()
    am.fail_edits = 99
    cls, inst = _app(am=am)

    plan = _run(cls, inst, apply=True, confirm="4")  # must NOT raise

    assert plan["complete"] is False, plan
    assert plan["stage_error"], "the stage failure was not recorded"
    assert plan["applied"] == 0, plan
    assert plan["failed"] == plan["ids"], plan


def test_the_backfill_RECORDS_WHAT_IT_PUSHED_so_the_sync_loop_does_not_redo_it(store):
    """Otherwise the 313-commit cost is moved into the background loop, not removed.

    The per-record path recorded every push in `_runs_pushed`. Without it the next
    `_sync_run_records` sees every migrated record as unpushed and re-pushes all of
    them, one commit each -- the exact cost this change exists to remove.
    """
    ids = []
    for i in range(1, 5):
        sid = f"2026-01-01-0000{i:02d}-aaabbbb{i}"
        store(sid)
        ids.append(sid)
    am = FakeAM()
    cls, inst = _app(am=am)

    plan = _run(cls, inst, apply=True, confirm="4")

    assert plan["complete"] is True, plan
    assert sorted(inst._runs_pushed) == sorted(ids), sorted(inst._runs_pushed)


def test_nothing_is_recorded_as_pushed_when_the_commit_FAILED(store):
    """THE CONTROL on the test above: the bookkeeping must not lie.

    Recording a push whose commit never landed would make the sync loop skip the
    very records that are missing from the artifact.
    """
    for i in range(1, 5):
        store(f"2026-01-01-0000{i:02d}-cccdddd{i}")
    am = FakeAM()
    am.fail_commits = 99
    cls, inst = _app(am=am)

    plan = _run(cls, inst, apply=True, confirm="4")

    assert plan["complete"] is False, plan
    assert inst._runs_pushed == {}, inst._runs_pushed
