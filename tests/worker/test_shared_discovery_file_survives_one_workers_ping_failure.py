"""One worker's failed ping must not take the data server away from the others.

Every worker on a workspace directory reads the same
``datasets/bioengine_current_server`` file, and that file is the only reference
to the data server. It is written by the data server at startup and removed by
it at shutdown -- it is never rewritten while the server runs. So when
``_ping_data_server`` deleted it on a local ping failure, one worker performed a
global delete of a record only a restart of a different process could restore,
and every other worker on the directory lost a data server that was still
answering.

Measured on europa 2026-11-13 against a six-worker fleet, bioengine 0.16.1: the
data server ``Up 43 minutes (healthy)`` and answering ``/ping`` with HTTP 200
from inside a worker container, the discovery directory empty, and
``get_worker_info().datasets == {}`` for the worker on the shared directory
while a worker on its own directory reported four. That pair is the
discriminating control the incident turned on -- it separates "the data server
is down" from "this worker deleted its own way back to it".

The tests below are built the way the incident was: dataset counts are asserted
on BOTH workers, before and after, because an assertion only about the failing
worker cannot tell the two causes apart either.
"""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from bioengine.worker import worker as worker_module
from bioengine.worker.worker import BioEngineWorker

DATASETS = {"alpha": {}, "beta": {}, "gamma": {}, "delta": {}}
SERVER_URL = "http://10.0.0.7:39527"


class StubDatasets:
    """Stands in for BioEngineDatasets. ``reachable`` is flipped per worker."""

    reachable = True

    def __init__(self, data_server_url, hypha_token=None, logger=None):
        self.data_server_url = data_server_url

    async def ping_data_server(self):
        if not type(self).reachable:
            raise RuntimeError("connection refused")

    async def list_datasets(self):
        if not type(self).reachable:
            raise RuntimeError("connection refused")
        return dict(DATASETS)


def make_worker(workspace_dir: Path, datasets_cls) -> BioEngineWorker:
    """A worker carrying only the state the data-server methods read."""
    worker = BioEngineWorker.__new__(BioEngineWorker)
    worker.logger = logging.getLogger("test")  # __del__ reaches for it
    worker.start_time = 1_700_000_000.0  # and for this
    worker.workspace_dir = workspace_dir
    worker.data_server = None
    worker.available_datasets = {}
    worker._data_server_unreachable = False
    worker.apps_manager = SimpleNamespace(
        app_builder=SimpleNamespace(
            data_server_url=None,
            update_data_server_url=lambda url: None,
        )
    )
    return worker


async def monitoring_pass(worker: BioEngineWorker) -> None:
    """The three data-server steps the monitoring loop runs, in its order."""
    await worker._ping_data_server()
    await worker._discover_data_server()
    await worker._refresh_datasets()


@pytest.fixture
def shared_workspace(tmp_path, monkeypatch):
    """One discovery directory, as a deployment that shares it would have."""
    discovery = tmp_path / "shared" / "datasets" / "bioengine_current_server"
    discovery.parent.mkdir(parents=True)
    discovery.write_text(SERVER_URL)

    # Each worker gets its own stub class so one can fail while the other does
    # not -- the two-worker split the incident needed to be visible at all.
    class DatasetsA(StubDatasets):
        reachable = True

    class DatasetsB(StubDatasets):
        reachable = True

    monkeypatch.setattr(worker_module, "BioEngineDatasets", DatasetsA)
    worker_a = make_worker(tmp_path / "shared", DatasetsA)
    worker_b = make_worker(tmp_path / "shared", DatasetsB)
    return discovery, worker_a, worker_b, DatasetsA, DatasetsB


async def both_discover(discovery, worker_a, worker_b, cls_a, cls_b, monkeypatch):
    """Bring both workers up on the shared file, one at a time."""
    monkeypatch.setattr(worker_module, "BioEngineDatasets", cls_a)
    await monitoring_pass(worker_a)
    monkeypatch.setattr(worker_module, "BioEngineDatasets", cls_b)
    await monitoring_pass(worker_b)


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------


async def test_one_workers_ping_failure_leaves_the_other_worker_serving(
    shared_workspace, monkeypatch
):
    discovery, worker_a, worker_b, cls_a, cls_b = shared_workspace
    await both_discover(discovery, worker_a, worker_b, cls_a, cls_b, monkeypatch)

    # BEFORE -- both workers, not only the one that will fail.
    assert discovery.exists()
    assert len(worker_a.available_datasets) == 4
    assert len(worker_b.available_datasets) == 4

    # Worker A alone loses its connection. The server keeps answering B.
    cls_a.reachable = False
    monkeypatch.setattr(worker_module, "BioEngineDatasets", cls_a)
    await monitoring_pass(worker_a)

    # AFTER -- the shared record survives, so B is untouched.
    assert discovery.exists(), (
        "worker A deleted the shared discovery file on its own ping failure; "
        "only the data server may remove it, and nothing would put it back"
    )
    assert discovery.read_text().strip() == SERVER_URL

    monkeypatch.setattr(worker_module, "BioEngineDatasets", cls_b)
    await monitoring_pass(worker_b)
    assert len(worker_b.available_datasets) == 4, (
        "worker B lost a data server that was still healthy"
    )

    # Worker A's own view degrades, which is correct for A and is the
    # presence assertion that keeps the two above from passing vacuously: if
    # the failure never took effect, this would still read 4.
    assert worker_a.available_datasets == {}
    assert worker_a.data_server is None


async def test_the_worker_reconnects_by_itself_once_the_server_answers_again(
    shared_workspace, monkeypatch
):
    """Keeping the file is what makes recovery possible without a restart.

    Deleting it removed the only reference to the data server, and the server
    rewrites it only at startup -- so the old behaviour could not recover even
    after the server came back. This is the other half of the same change.
    """
    discovery, worker_a, worker_b, cls_a, cls_b = shared_workspace
    await both_discover(discovery, worker_a, worker_b, cls_a, cls_b, monkeypatch)

    cls_a.reachable = False
    monkeypatch.setattr(worker_module, "BioEngineDatasets", cls_a)
    await monitoring_pass(worker_a)
    assert worker_a.available_datasets == {}

    cls_a.reachable = True
    await monitoring_pass(worker_a)
    assert len(worker_a.available_datasets) == 4
    assert worker_a.data_server is not None


async def test_a_standing_failure_is_logged_once_not_every_pass(
    shared_workspace, monkeypatch, caplog
):
    """The retry runs every monitoring interval (10 s by default), so logging
    each failure would turn one unreachable server into a log flood. The error
    belongs on the transition, and the transition must be re-armed by a
    recovery or a second outage goes unreported."""
    discovery, worker_a, worker_b, cls_a, cls_b = shared_workspace
    await both_discover(discovery, worker_a, worker_b, cls_a, cls_b, monkeypatch)

    cls_a.reachable = False
    monkeypatch.setattr(worker_module, "BioEngineDatasets", cls_a)
    with caplog.at_level(logging.ERROR, logger="test"):
        for _ in range(5):
            await monitoring_pass(worker_a)
    first_outage = [r for r in caplog.records if "pinging dataset server" in r.message]
    assert len(first_outage) == 1, (
        f"five failing passes produced {len(first_outage)} error lines"
    )

    # Re-armed by recovery: a second outage must report again rather than be
    # swallowed by the flag that silenced the first.
    cls_a.reachable = True
    await monitoring_pass(worker_a)
    cls_a.reachable = False
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="test"):
        await monitoring_pass(worker_a)
    assert [r for r in caplog.records if "pinging dataset server" in r.message], (
        "a second outage after a recovery was never reported"
    )


async def test_the_data_server_remains_the_only_writer_of_the_file():
    """The worker is a reader. If it ever writes or unlinks that path again the
    shared-state defect is back in a new spelling, so this asserts on the
    source rather than on behaviour -- a behavioural test cannot see a delete
    that is guarded by a condition the test did not happen to hit."""
    source = Path(worker_module.__file__).read_text()
    lines = source.splitlines()

    # Positive control first: an emptied or moved file satisfies every absence
    # assertion below while proving nothing.
    assert "bioengine_current_server" in source, "the discovery path is gone from worker.py"
    assert "_discover_data_server" in source
    # And a second one, opposite-signed: worker.py DOES write the two files it
    # owns, so "no writes found" would mean the scan is broken, not that the
    # worker is well behaved.
    assert [n for n, line in enumerate(lines, 1) if ".write_text(" in line], (
        "the write scan matched nothing at all, so its silence means nothing"
    )

    deletes = [f"{n}: {line.strip()}" for n, line in enumerate(lines, 1) if ".unlink()" in line]
    assert not deletes, f"worker.py deletes a file it does not own: {deletes}"

    discovery_refs = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(lines, 1)
        if "bioengine_current_server" in line
    ]
    assert len(discovery_refs) == 1, (
        f"the discovery path is referenced {len(discovery_refs)} times; it is read "
        f"in _discover_data_server and nowhere else: {discovery_refs}"
    )
