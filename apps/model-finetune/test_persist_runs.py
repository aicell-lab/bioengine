"""Run history belongs to the dataset, not to whichever worker answered.

``persist_run`` writes one run's record into a dataset artifact and ``list_runs``
reads them back. The artifact manager is faked here: what is under test is the
write shape, the idempotency and the reporting of an unreadable record, none of
which needs Hypha.

The staged write is the part worth pinning. A ``put_file`` URL is only valid
against a staged artifact, so the order has to be edit(stage=True) -> put_file ->
PUT -> commit; a put that lands without the commit leaves the file in a staging
area that no reader sees, which looks exactly like a successful write from the
caller's side.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import entry
import training


class _Resp:
    def __init__(self, text="", status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Http:
    """Records every PUT, and serves GETs from what was PUT."""

    def __init__(self):
        self.objects = {}
        self.puts = []
        self.fail_get_for = set()

    async def put(self, url, content=None):
        self.puts.append(url)
        self.objects[url] = content
        return _Resp()

    async def get(self, url):
        if url in self.fail_get_for:
            return _Resp("", status=500)
        return _Resp((self.objects.get(url) or b"").decode("utf-8"))


class _Am:
    """A fake artifact manager that enforces the staging rule."""

    def __init__(self):
        self.files = {}
        self.calls = []
        self.staged = False
        self.commits = 0

    async def edit(self, artifact_id=None, stage=None):
        self.calls.append(("edit", artifact_id, stage))
        self.staged = bool(stage)

    async def put_file(self, artifact_id, file_path=None):
        self.calls.append(("put_file", artifact_id, file_path))
        if not self.staged:
            raise RuntimeError("put_file against an unstaged artifact")
        url = f"https://put.invalid/{artifact_id}/{file_path}"
        self.files[file_path] = url
        return url

    async def commit(self, artifact_id):
        self.calls.append(("commit", artifact_id))
        self.commits += 1
        self.staged = False

    async def list_files(self, artifact_id, dir_path=None):
        self.calls.append(("list_files", artifact_id, dir_path))
        prefix = f"{dir_path}/"
        return [{"name": p[len(prefix) :]} for p in self.files if p.startswith(prefix)]

    async def get_file(self, artifact_id, file_path=None):
        self.calls.append(("get_file", artifact_id, file_path))
        if file_path not in self.files:
            raise FileNotFoundError(file_path)
        return self.files[file_path]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "sessions_root", lambda: tmp_path / "s")
    inst = object.__new__(entry.EntryApp.func_or_class)
    inst._http = _Http()
    am = _Am()

    async def _am(_caller=None):
        return am

    inst._runs_artifact_manager = _am
    inst._am_fake = am

    # _http_retry is used by list_runs; route it at the fake.
    async def _retry(method, url, **kw):
        return await inst._http.get(url)

    inst._http_retry = _retry
    return inst


def _session(session_id, status, params=None):
    d = training.session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps(status))
    if params is not None:
        (d / "training_params.json").write_text(json.dumps(params))


def _call(coro):
    return asyncio.run(coro)


def test_a_run_is_written_staged_then_committed(app):
    """The order is the contract: a put_file URL is only valid while staged, and
    a put without the commit is invisible to every reader while reading as
    success to the caller."""
    _session("s1", {"status": "COMPLETED"}, {"n_epochs": 3})
    out = _call(app.persist_run("s1", "ws/ds", context=None))

    verbs = [c[0] for c in app._am_fake.calls]
    assert verbs == ["edit", "put_file", "commit"], verbs
    assert app._am_fake.calls[0][2] is True, "the artifact was not staged first"
    assert out["file_path"] == "runs/s1.json"
    assert out["status"] == "COMPLETED"
    assert app._am_fake.commits == 1


def test_the_written_bytes_are_the_run_record(app):
    _session("s1", {"status": "FAILED", "message": "boom"}, {"n_epochs": 2})
    _call(app.persist_run("s1", "ws/ds", context=None))
    url = app._am_fake.files["runs/s1.json"]
    stored = json.loads(app._http.objects[url].decode("utf-8"))
    assert stored["session_id"] == "s1"
    assert stored["status"]["message"] == "boom"
    assert stored["config"]["n_epochs"] == 2
    assert stored["record_version"] == training.RUN_RECORD_VERSION


def test_a_presigned_reference_is_redacted_on_the_way_in(app):
    """The record is redacted by ``run_record``; this pins that the redaction is
    actually on the path that writes to a dataset, which is where a signature
    would otherwise be archived for anyone with read access."""
    _session(
        "s1",
        {"status": "COMPLETED"},
        {"train_images": ["https://s3.example/i.tif?X-Amz-Signature=abc123"]},
    )
    _call(app.persist_run("s1", "ws/ds", context=None))
    url = app._am_fake.files["runs/s1.json"]
    assert b"abc123" not in app._http.objects[url]


def test_re_persisting_replaces_rather_than_appends(app):
    """Named for the session, so the record is idempotent. A run is persisted
    when it finishes and may be re-persisted by a caller catching up."""
    _session("s1", {"status": "TRAINING"}, {"n_epochs": 1})
    _call(app.persist_run("s1", "ws/ds", context=None))
    _session("s1", {"status": "COMPLETED"}, {"n_epochs": 1})
    _call(app.persist_run("s1", "ws/ds", context=None))

    listed = _call(app.list_runs("ws/ds", context=None))
    assert len(listed["runs"]) == 1, "re-persisting created a second record"
    assert listed["runs"][0]["status"]["status"] == "COMPLETED"


def test_list_runs_reads_the_dataset_not_this_workers_sessions(app):
    """Two runs written, then the worker's own session directory emptied. The
    history must still list both — that is the whole point of persisting."""
    for sid, st in (("s1", "COMPLETED"), ("s2", "FAILED")):
        _session(sid, {"status": st}, {"n_epochs": 1})
        _call(app.persist_run(sid, "ws/ds", context=None))
    for sid in ("s1", "s2"):
        (training.session_dir(sid) / "status.json").unlink()

    listed = _call(app.list_runs("ws/ds", context=None))
    got = {r["session_id"]: r["status"]["status"] for r in listed["runs"]}
    assert got == {"s1": "COMPLETED", "s2": "FAILED"}
    assert listed["unreadable"] == []


def test_an_unreadable_record_is_reported_not_skipped(app):
    """A runs page that silently drops a broken record shows a shorter history
    and looks correct. The gap has to be visible."""
    for sid in ("s1", "s2"):
        _session(sid, {"status": "COMPLETED"}, {"n_epochs": 1})
        _call(app.persist_run(sid, "ws/ds", context=None))
    app._http.fail_get_for.add(app._am_fake.files["runs/s2.json"])

    listed = _call(app.list_runs("ws/ds", context=None))
    assert [r["session_id"] for r in listed["runs"]] == ["s1"]
    assert len(listed["unreadable"]) == 1
    assert listed["unreadable"][0]["file"] == "s2.json"


def test_no_runs_directory_is_an_empty_history_not_an_error(app):
    """A dataset nobody has trained on yet."""

    class _Empty(_Am):
        async def list_files(self, artifact_id, dir_path=None):
            raise FileNotFoundError(dir_path)

    empty = _Empty()

    async def _am(_caller=None):
        return empty

    app._runs_artifact_manager = _am
    listed = _call(app.list_runs("ws/ds", context=None))
    assert listed == {"runs": [], "unreadable": []}
