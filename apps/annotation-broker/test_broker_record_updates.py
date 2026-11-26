"""Tests for the locked read-modify-write on a dataset's metadata record.

The bug these cover: a handler read the record, awaited something slow (an
artifact-manager call, a batch of HTTP header fetches), then wrote the record
back. The write carried a snapshot taken *before* the await, so every change
made during it was silently discarded. ``get_dataset_index`` building its
dimensions cache would overwrite a label created while it ran, and
``create_label`` was the usual victim because it has no await of its own and
so looks correct in isolation.

Two of the cases below are controls rather than tests of the fix:

* ``test_the_old_unlocked_shape_really_does_lose_a_change`` runs the *old*
  read-then-write shape and asserts a change IS lost. Without it, the
  concurrency test above it could pass for the boring reason that the two
  threads never actually overlapped, and nothing would say so.
* ``test_a_competing_whole_record_write_cannot_strand_the_returned_record``
  is the one that would have caught the original report: the caller must never
  be handed a record describing a write that is not on disk.
"""

from __future__ import annotations

import threading
import time

import pytest

import broker_core as core
from test_broker_ngff import bk  # the broker module, with its runtime stubs

ARTIFACT = "ws/ds"


def _seed(root, artifact_id=ARTIFACT):
    """A committed record to mutate."""
    core.write_metadata(
        core.new_metadata(artifact_id, owner={"id": "u1"}), root=root
    )
    return artifact_id


def _run_both(first, second, timeout=20):
    """Run two callables in threads released from a barrier together."""
    start = threading.Barrier(2, timeout=10)
    errors = []

    def wrap(fn):
        def inner():
            try:
                start.wait()
                fn()
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        return inner

    threads = [threading.Thread(target=wrap(first)), threading.Thread(target=wrap(second))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    assert not any(t.is_alive() for t in threads), "a writer thread did not finish"
    return errors


# ---------------------------------------------------------------------------
# Clause 1 — two concurrent changes to one record, both survive
# ---------------------------------------------------------------------------


class TestConcurrentRecordChanges:
    def test_two_concurrent_changes_to_one_record_both_survive(self, tmp_path):
        """A label created while another writer is mid-update must not be lost,
        and neither must the other writer's field."""
        aid = _seed(tmp_path)

        def add_label():
            def mutate(record):
                record.setdefault("labels", []).append(
                    {"name": "nuclei", "description": ""}
                )
                # Held INSIDE the critical section, so the other writer is
                # genuinely contending rather than politely arriving later.
                time.sleep(0.2)

            core.update_metadata(aid, mutate, root=tmp_path)

        def make_public():
            def mutate(record):
                record["public"] = True
                time.sleep(0.2)

            core.update_metadata(aid, mutate, root=tmp_path)

        errors = _run_both(add_label, make_public)
        assert not errors, errors

        final = core.read_metadata(aid, root=tmp_path)
        assert [l["name"] for l in final.get("labels") or []] == ["nuclei"], final
        assert final.get("public") is True, final

    def test_the_old_unlocked_shape_really_does_lose_a_change(self, tmp_path):
        """CONTROL for the test above.

        Runs the shape the handlers used to have — read the record, pause, write
        the whole snapshot back — and asserts that exactly one of the two
        changes survives. If this ever passes with BOTH changes present, the
        test above is proving nothing about locking and this file needs
        rewriting.
        """
        aid = _seed(tmp_path)

        def unlocked_update(mutate):
            record = core.read_metadata(aid, root=tmp_path)  # read, unlocked
            mutate(record)
            time.sleep(0.2)  # the await the handler used to do here
            core.write_metadata(record, root=tmp_path)  # whole snapshot

        def add_label():
            unlocked_update(
                lambda r: r.setdefault("labels", []).append(
                    {"name": "nuclei", "description": ""}
                )
            )

        def make_public():
            unlocked_update(lambda r: r.__setitem__("public", True))

        errors = _run_both(add_label, make_public)
        assert not errors, errors

        final = core.read_metadata(aid, root=tmp_path)
        survived = [
            bool(final.get("labels")),
            bool(final.get("public")),
        ]
        assert survived.count(True) == 1, (
            "the unlocked shape kept both changes, so the concurrency test "
            f"above is not actually exercising contention: {final}"
        )


# ---------------------------------------------------------------------------
# update_metadata's own contract
# ---------------------------------------------------------------------------


class TestUpdateMetadata:
    def test_it_returns_exactly_what_is_on_disk(self, tmp_path):
        aid = _seed(tmp_path)
        returned = core.update_metadata(
            aid, lambda r: r.__setitem__("public", True), root=tmp_path
        )
        assert returned == core.read_metadata(aid, root=tmp_path)

    def test_it_raises_rather_than_creating_a_missing_record(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            core.update_metadata(
                "ws/never-registered", lambda r: None, root=tmp_path
            )
        assert core.read_metadata("ws/never-registered", root=tmp_path) is None

    def test_it_refuses_an_async_mutate(self, tmp_path):
        """An awaiting mutate would re-open the window the lock closes."""
        aid = _seed(tmp_path)

        async def mutate(record):  # pragma: no cover - must not be called
            record["public"] = True

        with pytest.raises(TypeError, match="synchronous"):
            core.update_metadata(aid, mutate, root=tmp_path)
        assert core.read_metadata(aid, root=tmp_path).get("public") in (None, False)

    def test_a_raising_mutate_leaves_the_record_untouched(self, tmp_path):
        aid = _seed(tmp_path)
        before = core.read_metadata(aid, root=tmp_path)

        def mutate(record):
            record["public"] = True
            raise ValueError("bad label name")

        with pytest.raises(ValueError):
            core.update_metadata(aid, mutate, root=tmp_path)
        assert core.read_metadata(aid, root=tmp_path) == before

    def test_the_lock_is_reentrant(self, tmp_path):
        """update_metadata holds the lock while calling a mutate that may
        itself write; a non-reentrant lock would deadlock here."""
        aid = _seed(tmp_path)

        def mutate(record):
            record["public"] = True
            core.write_metadata(record, root=tmp_path)

        core.update_metadata(aid, mutate, root=tmp_path)
        assert core.read_metadata(aid, root=tmp_path).get("public") is True


class TestCreateMetadataIfAbsent:
    def test_it_does_not_overwrite_an_existing_record(self, tmp_path):
        aid = _seed(tmp_path)
        core.update_metadata(
            aid,
            lambda r: r.setdefault("labels", []).append({"name": "keep"}),
            root=tmp_path,
        )
        record, created = core.create_metadata_if_absent(
            aid, lambda: core.new_metadata(aid, owner={"id": "u2"}), root=tmp_path
        )
        assert created is False
        assert [l["name"] for l in record.get("labels") or []] == ["keep"]
        assert record["owner"]["id"] == "u1"

    def test_it_creates_when_absent(self, tmp_path):
        record, created = core.create_metadata_if_absent(
            "ws/fresh",
            lambda: core.new_metadata("ws/fresh", owner={"id": "u9"}),
            root=tmp_path,
        )
        assert created is True
        assert core.read_metadata("ws/fresh", root=tmp_path) == record

    def test_only_one_of_two_concurrent_creations_wins(self, tmp_path):
        """Both callers must end up with the same record, not one silently
        overwriting the other's."""
        aid = "ws/raced"
        seen = []

        def register(owner_id):
            def go():
                def factory():
                    time.sleep(0.2)
                    return core.new_metadata(aid, owner={"id": owner_id})

                seen.append(
                    core.create_metadata_if_absent(aid, factory, root=tmp_path)
                )

            return go

        errors = _run_both(register("u1"), register("u2"))
        assert not errors, errors
        assert [c for _r, c in seen].count(True) == 1, seen
        on_disk = core.read_metadata(aid, root=tmp_path)
        assert all(r["owner"] == on_disk["owner"] for r, _c in seen), seen


# ---------------------------------------------------------------------------
# Clause 2 — create_label persists or raises, and never lies about it
# ---------------------------------------------------------------------------


@pytest.fixture
def broker(tmp_path, monkeypatch):
    """A broker whose artifact I/O is a no-op, so the test exercises the record
    write rather than Hypha. The record is read from disk on every
    ``_require_role``, as the real one does."""
    monkeypatch.setattr(bk, "STATE_ROOT", tmp_path)
    _seed(tmp_path)
    b = bk.AnnotationBroker()

    def _require_role(_context, artifact_id, _minimum):
        record = core.read_metadata(artifact_id, root=tmp_path)
        if record is None:
            raise PermissionError(f"no record for {artifact_id}")
        return record, {"id": "u1", "email": "u1@example.org"}, "owner"

    async def _ensure_staged(_artifact_id, fn):
        return await fn()

    async def _write_json_file(*_args, **_kwargs):
        return None

    b._require_role = _require_role
    b._canonical_id = lambda a: a
    b._ensure_staged = _ensure_staged
    b._write_json_file = _write_json_file
    return b


class TestCreateLabelPersistsOrRaises:
    async def test_the_returned_record_is_the_one_on_disk(self, broker, tmp_path):
        returned = await broker.create_label(
            artifact_id=ARTIFACT, name="nuclei", description="cell nuclei"
        )
        on_disk = core.read_metadata(ARTIFACT, root=tmp_path)
        assert [l["name"] for l in returned.get("labels") or []] == ["nuclei"]
        assert returned == on_disk

    async def test_a_bad_label_name_raises_and_writes_nothing(
        self, broker, tmp_path
    ):
        before = core.read_metadata(ARTIFACT, root=tmp_path)
        with pytest.raises(ValueError):
            await broker.create_label(artifact_id=ARTIFACT, name="Not Valid!")
        assert core.read_metadata(ARTIFACT, root=tmp_path) == before

    async def test_a_failing_write_raises_rather_than_returning_a_record(
        self, broker, tmp_path, monkeypatch
    ):
        """If the record cannot be stored, the caller gets an exception — never
        a record describing a label that is not there."""

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(core.os, "replace", boom)
        with pytest.raises(OSError):
            await broker.create_label(artifact_id=ARTIFACT, name="nuclei")
        monkeypatch.undo()
        assert not (core.read_metadata(ARTIFACT, root=tmp_path).get("labels") or [])

    async def test_a_competing_whole_record_write_cannot_strand_the_returned_record(
        self, broker, tmp_path
    ):
        """The original report, as a test.

        A writer that holds a pre-read snapshot and writes it whole runs
        concurrently with create_label. Whatever create_label returns must
        agree with disk: it may not claim a label that was overwritten.
        """
        results = {}

        def create():
            import asyncio

            results["returned"] = asyncio.run(
                broker.create_label(artifact_id=ARTIFACT, name="nuclei")
            )

        def competing_whole_record_write():
            # Exactly what get_dataset_index used to do: a record read before
            # the slow work, written back whole afterwards.
            def mutate(record):
                record["image_dims"] = {"img1": [512, 512]}
                time.sleep(0.2)

            core.update_metadata(ARTIFACT, mutate, root=tmp_path)

        errors = _run_both(create, competing_whole_record_write)
        assert not errors, errors

        on_disk = core.read_metadata(ARTIFACT, root=tmp_path)
        returned = results["returned"]
        # The label create_label reported is really there ...
        assert [l["name"] for l in returned.get("labels") or []] == ["nuclei"]
        assert [l["name"] for l in on_disk.get("labels") or []] == ["nuclei"], on_disk
        # ... and the competing writer's field survived too.
        assert on_disk.get("image_dims") == {"img1": [512, 512]}, on_disk
