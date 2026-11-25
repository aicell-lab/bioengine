"""Sharing the records: what is written, where, how often, and what fails well.

The write runs with the APP's own credential against ONE artifact the app
names itself, so these tests are mostly about the things that must NOT be
possible or must NOT escalate: no caller can choose the artifact, a failed
share cannot fail a run, and a record that cannot be read cannot empty a run
history.

The fake artifact manager records every call rather than asserting inside
itself, so a test says what it wants in its own body. It also mimics the two
behaviours measured on the live artifact before this was built: a commit does
NOT create a new version (the version list stayed at one entry across two
commits), and a second write to the same path replaces the first.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

import training


def _entry_class():
    import entry

    for obj in vars(entry).values():
        if isinstance(obj, type) and obj.__name__ == "EntryApp":
            return obj
        inner = getattr(obj, "__wrapped_class__", None) or getattr(
            obj, "func_or_class", None
        )
        if isinstance(inner, type) and inner.__name__ == "EntryApp":
            return inner
    raise AssertionError("EntryApp not found")


class FakeAM:
    """Enough artifact manager to see what the app did."""

    def __init__(self, *, fail_commit_times=0, fail_list=False):
        self.calls = []
        self.files = {}
        self.versions = ["v0"]
        self._fail_commit_times = fail_commit_times
        self._fail_list = fail_list
        self._pending = None

    async def edit(self, artifact_id=None, stage=None, **kw):
        self.calls.append(("edit", artifact_id, stage))

    async def put_file(self, artifact_id=None, file_path=None, **kw):
        self.calls.append(("put_file", artifact_id, file_path))
        self._pending = file_path
        return f"https://upload.example/{file_path}"

    async def commit(self, artifact_id=None, **kw):
        self.calls.append(("commit", artifact_id))
        if self._fail_commit_times > 0:
            self._fail_commit_times -= 1
            raise RuntimeError("nothing staged — another writer committed first")
        # Measured on the live artifact: a commit does not add a version.
        assert len(self.versions) == 1

    async def list_files(self, artifact_id=None, dir_path=None, **kw):
        self.calls.append(("list_files", artifact_id, dir_path))
        if self._fail_list:
            raise RuntimeError("the artifact could not be listed")
        return [{"name": n, "type": "file"} for n in sorted(self.files)]

    async def get_file(self, artifact_id=None, file_path=None, **kw):
        self.calls.append(("get_file", artifact_id, file_path))
        return f"https://download.example/{file_path}"


class FakeHTTP:
    """An httpx-shaped client over the FakeAM's file store."""

    class _Resp:
        def __init__(self, status=200, text=""):
            self.status_code = status
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    def __init__(self, am, *, put_status=200, corrupt=()):
        self.am = am
        self.put_status = put_status
        self.corrupt = set(corrupt)
        self.puts = []

    async def put(self, url, content=None, **kw):
        name = url.rsplit("/", 1)[-1]
        self.puts.append((name, content))
        if self.put_status < 400:
            # The second write to a path replaces the first, as measured.
            self.am.files[name] = content.decode("utf-8")
        return self._Resp(self.put_status)

    async def get(self, url, **kw):
        name = url.rsplit("/", 1)[-1]
        if name in self.corrupt:
            return self._Resp(200, "{not json at all")
        return self._Resp(200, self.am.files.get(name, ""))


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """An instance with a fake artifact manager and a sessions root of its own."""
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(training, "sessions_root", lambda: root)
    monkeypatch.setattr(training, "session_dir", lambda sid: root / sid)

    cls = _entry_class()
    inst = cls.__new__(cls)
    am = FakeAM()
    http = FakeHTTP(am)
    inst._runs_pushed = {}
    # No real wait in the retry path.
    inst.RUNS_RETRY_DELAY_S = 0

    async def _am():
        return am

    inst._runs_artifact_manager = _am
    inst._http = http
    inst._am = am
    inst._root = root

    def plant(sid, **fields):
        d = root / sid
        d.mkdir(parents=True, exist_ok=True)
        status = {"status": "COMPLETED", "updated_at": 0, "owner": "github|1", **fields}
        (d / "status.json").write_text(json.dumps(status))
        (d / "training_params.json").write_text(json.dumps({"n_epochs": 1}))
        return sid

    inst._plant = plant
    return inst


S1 = "2026-11-25-120000-aaaaaaaa"
S2 = "2026-11-25-130000-bbbbbbbb"


# ── the write ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_record_goes_to_the_APPS_OWN_artifact_under_runs(app):
    app._plant(S1)
    assert await app._persist_run_record(S1) is True
    artifacts = {c[1] for c in app._am.calls if c[0] in ("edit", "put_file", "commit")}
    assert artifacts == {training.RUNS_ARTIFACT_ID}, artifacts
    paths = [c[2] for c in app._am.calls if c[0] == "put_file"]
    assert paths == [f"runs/{S1}.json"], paths
    # staged, uploaded, committed -- in that order, since a put_file URL is
    # only valid against a staged artifact.
    order = [c[0] for c in app._am.calls]
    assert order.index("edit") < order.index("put_file") < order.index("commit")


@pytest.mark.asyncio
async def test_what_is_written_is_the_RECORD_including_the_owner(app):
    """The owner has to be IN the shared copy or a run from another instance
    cannot be attributed to anybody -- see test_shared_run_history.py. The
    fence is the artifact's permissions, not the absence of the field."""
    app._plant(S1, owner="github|49943582")
    await app._persist_run_record(S1)
    written = json.loads(app._am.files[f"{S1}.json"])
    assert written["session_id"] == S1
    assert written["record_version"] == training.RUN_RECORD_VERSION
    assert written["status"]["owner"] == "github|49943582"
    assert written["config"] == {"n_epochs": 1}


@pytest.mark.asyncio
async def test_a_bad_session_id_NEVER_REACHES_the_artifact(app):
    """An id the server could not have issued is refused before any call: the
    id becomes a path inside the artifact."""
    for bad in ["../escape", "runs/../../x", "", "not a session"]:
        assert await app._persist_run_record(bad) is False
    assert app._am.calls == [], app._am.calls


@pytest.mark.asyncio
async def test_a_LOST_COMMIT_RACE_is_retried_once_and_then_given_up_on(app):
    """Two instances can hold `stage=True` on one artifact, and the first
    commit clears the single artifact-level flag -- so a commit can find
    nothing staged through no fault of this writer. The bytes are already
    uploaded by then, so one retry and then the next poll."""
    app._plant(S1)
    app._am._fail_commit_times = 1
    assert await app._persist_run_record(S1) is True
    assert [c[0] for c in app._am.calls].count("commit") == 2

    app._am._fail_commit_times = 5
    app._runs_pushed.clear()
    assert await app._persist_run_record(S1) is False, (
        "a permanently failing commit must be reported as not landed, not "
        "swallowed into a True that makes the sync skip it forever"
    )


@pytest.mark.asyncio
async def test_a_FAILED_SHARE_does_not_raise_into_the_caller(app):
    """Every caller is doing something else that must not fail with this."""
    app._plant(S1)
    app._http.put_status = 503
    assert await app._persist_run_record(S1) is False


# ── the poll ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_sync_pushes_each_run_once_and_then_skips_it(app):
    app._plant(S1)
    app._plant(S2)
    first = await app._sync_run_records()
    assert first == {"pushed": 2, "unchanged": 0, "failed": 0}, first
    second = await app._sync_run_records()
    assert second == {"pushed": 0, "unchanged": 2, "failed": 0}, second


@pytest.mark.asyncio
async def test_a_CHANGED_record_is_pushed_again_even_without_a_status_transition(app):
    """Content-addressed, not transition-addressed. A per-epoch metric arriving
    while the run stays TRAINING is most of what a run history shows, and a
    sync keyed on the status string would never share it."""
    app._plant(S1, status="TRAINING")
    await app._sync_run_records()
    app._plant(S1, status="TRAINING", epochs=[{"epoch": 0, "train_loss": 0.4}])
    counts = await app._sync_run_records()
    assert counts["pushed"] == 1, counts
    written = json.loads(app._am.files[f"{S1}.json"])
    assert written["status"]["epochs"] == [{"epoch": 0, "train_loss": 0.4}]


@pytest.mark.asyncio
async def test_a_directory_that_is_not_a_session_is_not_pushed(app):
    (app._root / "lost+found").mkdir()
    (app._root / "lost+found" / "status.json").write_text("{}")
    app._plant(S1)
    counts = await app._sync_run_records()
    assert counts == {"pushed": 1, "unchanged": 0, "failed": 0}, counts


# ── the read ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_records_come_back_keyed_by_session_id(app):
    app._plant(S1)
    app._plant(S2)
    await app._sync_run_records()
    got = await app._read_run_records()
    assert set(got) == {S1, S2}, got
    assert got[S1]["session_id"] == S1


@pytest.mark.asyncio
async def test_an_UNLISTABLE_artifact_returns_None_rather_than_an_empty_history(app):
    """None and {} are different answers. {} says the shared history is empty;
    None says it could not be read, and only the second is a reason to log and
    fall back to this instance's own disk. Collapsing them would make an
    outage look like a project with no runs."""
    app._am._fail_list = True
    assert await app._read_run_records() is None


@pytest.mark.asyncio
async def test_ONE_unreadable_record_does_not_empty_the_history(app):
    app._plant(S1)
    app._plant(S2)
    await app._sync_run_records()
    app._http.corrupt = {f"{S1}.json"}
    got = await app._read_run_records()
    assert set(got) == {S2}, got


@pytest.mark.asyncio
async def test_the_listing_asks_for_the_shared_records_and_hides_the_owner(app):
    """End to end over the two halves: share, then list as the owner."""
    app._plant(S1, owner="github|49943582")
    await app._sync_run_records()
    # The listing reads the shared copy; this instance also holds S1 locally,
    # so drop the local directory to make the shared leg the only source.
    (app._root / S1 / "status.json").unlink()
    cls = _entry_class()
    fn = getattr(cls, "list_training_sessions")
    ctx = {"user": {"id": "github|49943582", "is_anonymous": False}}
    out = await fn(app, context=ctx)
    assert set(out) == {S1}, out
    for field in training.PRIVATE_STATUS_FIELDS:
        assert field not in out[S1]
    assert "github|49943582" not in json.dumps(out)
