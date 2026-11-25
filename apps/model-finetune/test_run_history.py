"""A fine-tuning run must outlive the session directory it ran in.

Why: the runs the Colab page lists lived only in ``~/.bioengine`` on whichever
node the entry replica happened to be. That survives a replica restart while the
same host mount is there, and nothing else — a worker rebuild, a move to another
cluster or a scale-to-zero onto a fresh node take every run with them, and a user
is shown no history at all. So each run is written to the app's workspace storage
as one small record, and the listing merges what is stored with what is live.

The arms below are built around the three things that make such a feature wrong
in practice rather than in theory:

* a record that quietly publishes a credential (``start_training`` resolves
  ``init_checkpoint`` into a presigned URL, and the record is readable by
  anything with the workspace);
* a stored snapshot shown as if it were live — a run frozen mid-flight reading
  as ``TRAINING`` for ever, or a checkpoint claimed for weights that went with
  the disk;
* a history feature that can fail the training run it is recording.

No Hypha, no S3, no Ray: the instance is built with ``object.__new__`` carrying
only the attributes these methods touch, and storage is a dict.
"""

from __future__ import annotations

import json

import pytest

import entry
import training


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeStore:
    """The S3 service's three calls, as measured against the live one.

    The measured trap this reproduces: ``get_file`` mints a URL for an object
    that is NOT there, so absence surfaces as a 404 on the GET and nowhere
    earlier.
    """

    def __init__(self):
        self.objects = {}
        self.stamps = {}
        self.gets = []
        self.put_failures = 0

    async def put_file(self, file_path, ttl=None):
        if self.put_failures:
            self.put_failures -= 1
            raise RuntimeError("storage is down")
        return f"https://store.invalid/put/{file_path}?sig=abc"

    async def get_file(self, file_path, use_proxy=False):
        return f"https://store.invalid/get/{file_path}?sig=abc"

    async def list_files(self, path):
        prefix = f"{path}/"
        return [
            {
                "type": "file",
                "name": name[len(prefix) :],
                "size": len(body),
                "last_modified": self.stamps.get(name, 1.0),
            }
            for name, body in sorted(self.objects.items())
            if name.startswith(prefix)
        ]


def _app(store, tmp_path, monkeypatch):
    """An EntryApp carrying only what the history methods read."""
    monkeypatch.setattr(training, "sessions_root", lambda: tmp_path)
    inst = object.__new__(entry.EntryApp.func_or_class)
    inst._s3 = store
    inst._run_record_cache = {}
    inst._record_tasks = set()

    async def _http_retry(method, url, **kwargs):
        name = url.split("/", 4)[-1].split("?")[0]
        if method == "PUT":
            store.objects[name] = kwargs["content"]
            store.stamps.setdefault(name, 1.0)
            return _FakeResponse(200)
        store.gets.append(name)
        if name not in store.objects:
            return _FakeResponse(404)
        return _FakeResponse(200, json.loads(store.objects[name]))

    inst._http_retry = _http_retry
    return inst


def _finished_session(session_id="run-1", **extra):
    """A session on disk, as the real workers leave one behind."""
    training.write_status(
        session_id,
        status="COMPLETED",
        model_type="cpsam",
        backend="cellpose",
        start_time=1.0,
        end_time=2.0,
        n_epochs=5,
        n_epochs_completed=5,
        n_epochs_completed_basis="measured",
        epochs=[{"epoch": 0, "train_loss": 2.0, "val_loss": 1.9, "lr": 1e-4}],
        train_losses=[2.0, 1.5, 1.2, 1.1, 1.0],
        **extra,
    )


# --- what a record carries --------------------------------------------------


async def test_the_record_carries_the_config_the_metrics_and_the_outcome(
    tmp_path, monkeypatch
):
    """Clause by clause, these three are what the history is FOR."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    _finished_session("run-1")
    training.write_training_params(
        "run-1", {"n_epochs": 5, "learning_rate": 1e-5, "batch_size": 2}
    )

    assert await app._persist_run("run-1") is True
    record = json.loads(store.objects["model-finetune/runs/run-1.json"])

    assert record["session_id"] == "run-1"
    assert record["record_version"] == 1
    assert record["config"]["n_epochs"] == 5  # config
    assert record["config"]["learning_rate"] == 1e-5
    assert record["status"]["epochs"][0]["train_loss"] == 2.0  # per-epoch metrics
    assert record["status"]["train_losses"] == [2.0, 1.5, 1.2, 1.1, 1.0]
    assert record["status"]["status"] == "COMPLETED"  # outcome
    assert record["status"]["n_epochs_completed_basis"] == "measured"


async def test_a_presigned_url_in_the_record_is_redacted(tmp_path, monkeypatch):
    """``start_training`` resolves ``init_checkpoint`` into a presigned URL and
    writes it into the status. The record outlives the URL's own validity and is
    readable by anything holding the workspace, so the signature must not travel
    with it — while the path, which says what the run started FROM, must."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    _finished_session(
        "run-1",
        pool_provenance={
            "weights_url": "https://hypha.invalid/w.pt?X-Amz-Signature=deadbeef"
        },
    )
    await app._persist_run("run-1")
    body = store.objects["model-finetune/runs/run-1.json"].decode()

    assert "deadbeef" not in body, "a presigned signature was published"
    assert "https://hypha.invalid/w.pt?<redacted>" in body


async def test_the_status_keys_that_are_dropped_are_actually_dropped(
    tmp_path, monkeypatch
):
    """CONTROL for the arm above: redaction only helps for values that stay. The
    resolved checkpoint URL and the host path are withheld outright, so this
    asserts both the drop AND that the record is otherwise non-empty — a record
    that dropped everything would pass a naive 'the secret is absent' check."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    _finished_session(
        "run-1",
        init_checkpoint="https://hypha.invalid/ckpt.pt?sig=secret",
        checkpoint_path="/home/someone/.bioengine/x/best.pt",
    )
    await app._persist_run("run-1")
    record = json.loads(store.objects["model-finetune/runs/run-1.json"])

    assert "init_checkpoint" not in record["status"]
    assert "checkpoint_path" not in record["status"]
    assert record["status"]["status"] == "COMPLETED", "the record kept nothing"


async def test_the_config_is_an_allow_list_not_a_deny_list(tmp_path, monkeypatch):
    """A later version adding a secret to ``params`` must not publish it by
    default. The failure mode of a deny-list is silent and permanent, so the
    record names what it keeps."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    _finished_session("run-1")
    training.write_training_params(
        "run-1",
        {
            "n_epochs": 5,
            "checkpoint_path": "/home/someone/secret/best.pt",
            "some_future_token": "SHOULD-NOT-TRAVEL",
        },
    )
    await app._persist_run("run-1")
    body = store.objects["model-finetune/runs/run-1.json"].decode()

    assert "SHOULD-NOT-TRAVEL" not in body
    assert "/home/someone/secret" not in body
    assert json.loads(body)["config"] == {"n_epochs": 5}


async def test_a_run_with_no_params_file_yet_is_still_recorded(tmp_path, monkeypatch):
    """The record written when a run STARTS is the one that makes an interrupted
    run visible at all, and ``training_params.json`` does not exist yet then —
    reading it raises. Config passed in, so the starting record is not the one
    that never lands."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    training.write_status("run-1", status="PREPARING", model_type="cpsam")

    assert await app._persist_run("run-1", config={"n_epochs": 3}) is True
    record = json.loads(store.objects["model-finetune/runs/run-1.json"])
    assert record["config"] == {"n_epochs": 3}


async def test_a_storage_failure_cannot_fail_the_run(tmp_path, monkeypatch):
    """History is worth less than the run that produced it."""
    store = _FakeStore()
    store.put_failures = 1
    app = _app(store, tmp_path, monkeypatch)
    _finished_session("run-1")

    assert await app._persist_run("run-1") is False
    assert store.objects == {}


# --- what a listing shows ---------------------------------------------------


async def test_a_stored_run_is_listed_after_its_session_directory_is_gone(
    tmp_path, monkeypatch
):
    """THE arm this feature exists for."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    _finished_session("run-1")
    await app._persist_run("run-1")

    # The restart: the status file goes with the node, the record does not.
    (tmp_path / "run-1" / "status.json").unlink()
    assert training.list_sessions() == {}, "the premise did not hold"

    listed = await app.list_training_sessions()
    assert list(listed) == ["run-1"]
    assert listed["run-1"]["status"] == "COMPLETED"
    assert listed["run-1"]["persisted"] is True
    assert listed["run-1"]["config"] is not None


async def test_a_record_never_claims_a_checkpoint_it_cannot_serve(
    tmp_path, monkeypatch
):
    """The weights went with the session directory. A row offering Serve or
    Export for them would fail at the click, so the record says false whatever
    it said when it was written."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    _finished_session("run-1")
    await app._persist_run("run-1")
    stored = json.loads(store.objects["model-finetune/runs/run-1.json"])
    # The record itself was written while the session was intact...
    stored["status"]["checkpoint_available"] = True
    store.objects["model-finetune/runs/run-1.json"] = json.dumps(stored).encode()
    (tmp_path / "run-1" / "status.json").unlink()

    listed = await app.list_training_sessions()
    assert listed["run-1"]["checkpoint_available"] is False


async def test_a_live_session_wins_over_its_own_record(tmp_path, monkeypatch):
    """Both exist for every run that is still going: the record was written at
    the start and the session is still being updated. The fresher one must win,
    or a running job would be displayed with its starting snapshot."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    training.write_status("run-1", status="PREPARING", model_type="cpsam")
    await app._persist_run("run-1", config={"n_epochs": 5})
    training.write_status("run-1", status="TRAINING", n_epochs_completed=2)

    listed = await app.list_training_sessions()
    assert listed["run-1"]["status"] == "TRAINING"
    assert listed["run-1"]["n_epochs_completed"] == 2
    assert "persisted" not in listed["run-1"]


async def test_a_run_recorded_mid_flight_is_not_shown_as_still_training(
    tmp_path, monkeypatch
):
    """A stored snapshot cannot advance, and nothing is going to finish it. Left
    alone it would read as TRAINING for ever — the same mistake
    ``training.get_status``'s stale window exists to avoid for live sessions."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    training.write_status("run-1", status="TRAINING", model_type="cpsam")
    await app._persist_run("run-1", config={"n_epochs": 5})
    (tmp_path / "run-1" / "status.json").unlink()

    listed = await app.list_training_sessions()
    assert listed["run-1"]["status"] == "STOPPED"
    assert "no outcome" in listed["run-1"]["message"]


async def test_the_listing_can_be_restricted_to_live_sessions(tmp_path, monkeypatch):
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    _finished_session("run-1")
    await app._persist_run("run-1")
    (tmp_path / "run-1" / "status.json").unlink()

    assert await app.list_training_sessions(include_persisted=False) == {}
    assert list(await app.list_training_sessions()) == ["run-1"]


async def test_a_missing_object_is_not_parsed_as_a_record(tmp_path, monkeypatch):
    """MEASURED TRAP: the live ``get_file`` mints a URL for an object that is not
    there, so absence shows up as a 404 on the GET and nowhere earlier. A listing
    that trusted the mint would hand a page an error body as a run."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    store.objects["model-finetune/runs/ghost.json"] = b'{"session_id": "ghost"}'
    store.stamps["model-finetune/runs/ghost.json"] = 1.0
    # PREMISE: with the object readable, this listing is non-empty. Without this
    # the assertion below would hold for an empty listing of any cause.
    assert list(await app._persisted_runs()) == ["ghost"]

    # Same listing, but every read 404s — the object was listed and is gone. The
    # object stays in the fake store's listing on purpose: a listing that failed
    # would also return {} and would prove nothing about the read path.
    app._run_record_cache.clear()
    reads = []

    async def _http_retry(method, url, **kwargs):
        reads.append(url)
        return _FakeResponse(404)

    app._http_retry = _http_retry
    assert await app._persisted_runs() == {}
    assert reads, "nothing was read, so the 404 path was never exercised"


async def test_an_absent_prefix_is_an_empty_history_not_an_error(tmp_path, monkeypatch):
    store = _FakeStore()

    async def _boom(path):
        raise RuntimeError("NoSuchKey")

    store.list_files = _boom
    app = _app(store, tmp_path, monkeypatch)
    assert await app._persisted_runs() == {}
    assert await app.list_training_sessions() == {}


# --- polling cost -----------------------------------------------------------


async def test_an_unchanged_record_is_not_fetched_twice(tmp_path, monkeypatch):
    """A runs page polls. A record is immutable once its run is terminal, so
    re-reading every object on every poll spends one GET per run per poll for
    bytes that cannot have changed."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    _finished_session("run-1")
    await app._persist_run("run-1")

    await app._persisted_runs()
    first = len(store.gets)
    assert first == 1
    await app._persisted_runs()
    assert len(store.gets) == first, "the cached record was fetched again"


async def test_a_rewritten_record_is_fetched_again(tmp_path, monkeypatch):
    """CONTROL for the cache: one keyed on the name alone would serve the
    starting snapshot of every run for ever, which is worse than no cache."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    training.write_status("run-1", status="TRAINING", model_type="cpsam")
    await app._persist_run("run-1", config={"n_epochs": 5})
    await app._persisted_runs()

    training.write_status("run-1", status="COMPLETED", end_time=9.0)
    await app._persist_run("run-1")
    store.stamps["model-finetune/runs/run-1.json"] = 2.0  # object rewritten

    runs = await app._persisted_runs()
    assert runs["run-1"]["status"]["status"] == "COMPLETED"
    assert len(store.gets) == 2


async def test_the_listing_reads_the_newest_records_first(tmp_path, monkeypatch):
    """The cap bounds a poll, so it has to drop the OLDEST runs, not an
    arbitrary set: a page showing the five oldest runs of fifty is worse than
    one showing the five newest."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    for i in range(5):
        _finished_session(f"run-{i}")
        await app._persist_run(f"run-{i}")
        store.stamps[f"model-finetune/runs/run-{i}.json"] = float(i)

    runs = await app._persisted_runs(limit=2)
    assert sorted(runs) == ["run-3", "run-4"]


# --- the status fallback ----------------------------------------------------


async def test_polling_a_stored_run_does_not_report_no_such_session(
    tmp_path, monkeypatch
):
    """A page lists the history and then polls a row. Without the fallback the
    poll answers UNKNOWN about a run the same worker had just listed."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    _finished_session("run-1")
    await app._persist_run("run-1")
    (tmp_path / "run-1" / "status.json").unlink()

    status = await app.get_training_status("run-1")
    assert status["status"] == "COMPLETED"
    assert status["persisted"] is True


async def test_a_live_session_is_never_answered_from_a_record(tmp_path, monkeypatch):
    """CONTROL for the fallback: it must trigger on UNKNOWN only. A record is
    always staler than the session it was written from."""
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    training.write_status("run-1", status="PREPARING", model_type="cpsam")
    await app._persist_run("run-1", config={"n_epochs": 5})
    training.write_status("run-1", status="TRAINING")

    status = await app.get_training_status("run-1")
    assert status["status"] == "TRAINING"
    assert "persisted" not in status
    assert store.gets == [], "a live session cost a storage read"


async def test_an_unknown_session_with_no_record_stays_unknown(tmp_path, monkeypatch):
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)
    status = await app.get_training_status("never-existed")
    assert status["status"] == "UNKNOWN"


async def test_a_run_writes_its_record_from_run_training_itself(tmp_path, monkeypatch):
    """THE CALL-SITE ARM. ``_run_training``'s ``finally`` starts the write
    through a guard that cannot raise, because an exception from a ``finally``
    would replace the run's own outcome with a failure to write history. That
    tolerance is only safe if something asserts the happy path really writes —
    otherwise deleting the call, or the plumbing it needs, leaves every test
    green and every run unrecorded.

    Driven to the FAILED branch on purpose: an outcome is history too, and a run
    that died is exactly the one a user comes back to look for.
    """
    store = _FakeStore()
    app = _app(store, tmp_path, monkeypatch)

    class _Replica:
        _training_tasks: dict = {}
        _s3 = store
        _run_record_cache: dict = {}
        _record_tasks: set = set()
        _http_retry = staticmethod(app._http_retry)
        _persist_run = app._persist_run
        _persist_run_in_background = app._persist_run_in_background
        _run_record = app._run_record

        async def _preparing_heartbeat(self, *_a, **_k):
            return None

        async def _prepare_training_data(self, *_a, **_k):
            raise RuntimeError("no data")

    await entry.EntryApp.func_or_class._run_training(
        _Replica(), "run-1", "cpsam", [], [], [], [], {}
    )

    record = json.loads(store.objects["model-finetune/runs/run-1.json"])
    assert record["status"]["status"] == "FAILED"
    assert "no data" in record["status"]["message"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://h.invalid/a.pt?sig=x", "https://h.invalid/a.pt?<redacted>"),
        ("https://h.invalid/a.pt", "https://h.invalid/a.pt"),
        ("/local/path?not=a-url", "/local/path?not=a-url"),
        (5, 5),
        (None, None),
    ],
)
def test_redaction_touches_urls_and_leaves_everything_else(value, expected):
    assert entry._redact_urls(value) == expected
