"""Two concurrent record changes on one dataset must BOTH survive.

Plain pytest, no Ray, no live Hypha. ``root`` points at ``tmp_path`` so nothing touches the real
``~/annotation_broker`` state directory.

WHAT THIS IS ABOUT. ``write_metadata`` writes the WHOLE record (``.tmp`` then ``os.replace``). That
is atomic per file, so a reader never sees torn JSON — but it used to have no compare-and-set, so a
write blindly replaced whatever was on disk. Two callers that each ``read_metadata`` -> mutate ->
``write_metadata`` lost one update entirely, and the loser was handed back the record it *believed*
it wrote.

THE INTERLEAVE IS THE ONE MEASURED IN PRODUCTION, not an invented one. In the reported failure ``create_label``
returned a record carrying the new label and the image dimensions of the FIRST upload batch only,
while the store afterwards held all four images and no label. That ordering is:

    create_label reads  ->  upload reads  ->  create_label writes  ->  upload writes

so the upload's write, built on a pre-label read, is what clobbered the label. The loser is not the
last writer by wall clock; it is whoever wrote from the older copy. The fix therefore has to cover
every read-modify-write path, not just ``create_label``.

THE TWO PROPERTIES ARE SEPARATE, and the tests keep them separate:
  * ``write_metadata`` alone gives NO SILENT LOSS — a stale write raises ``RecordConflict``.
  * ``update_metadata`` gives BOTH CHANGES SURVIVE — it re-reads and re-applies on conflict.
A fix with only the first would still drop the user's label, just loudly, so clause 1 is tested
through ``update_metadata`` and the raising behaviour is tested on its own.
"""

from __future__ import annotations

import json
import threading

import pytest

import broker_core as bc

AID = "bioimage-io/annotation-concurrent-1019"
OWNER = {"id": "owner-1", "email": "owner@example.com"}


def _seed(root):
    """A stored record with no labels and one image, as after a first upload batch."""
    meta = bc.new_metadata(AID, OWNER)
    meta["image_dims"] = {"picked-scan": [384, 384]}
    return bc.write_metadata(meta, root=root, expect_revision=None)


def _stored(root):
    got = bc.read_metadata(AID, root=root)
    assert got is not None, "the seeded record vanished"
    return got


def _label_names(meta):
    return sorted(l.get("name") for l in (meta.get("labels") or []))


def _add_label(name):
    def mutate(record):
        bc.add_label(record, name, "")
        return record

    return mutate


def _add_image(stem):
    def mutate(record):
        record.setdefault("image_dims", {})[stem] = [384, 384]
        return record

    return mutate


# ---------------------------------------------------------------------------
# clause 1 — both concurrent changes survive
# ---------------------------------------------------------------------------


def test_interleaved_label_and_upload_writes_both_survive(tmp_path):
    """The exact production ordering, sequenced by hand so it cannot flake.

    Both writers go through ``update_metadata``, so the one that finds the record moved re-applies
    its change to the winner's version instead of discarding it.
    """
    _seed(tmp_path)
    bc.update_metadata(AID, _add_label("cells"), root=tmp_path)
    bc.update_metadata(AID, _add_image("picked-scan-2"), root=tmp_path)

    stored = _stored(tmp_path)
    assert _label_names(stored) == ["cells"], f"label lost: {stored.get('labels')!r}"
    assert sorted(stored["image_dims"]) == ["picked-scan", "picked-scan-2"], (
        f"second image lost: {stored.get('image_dims')!r}"
    )


def test_stale_copies_racing_through_update_metadata_both_survive(tmp_path):
    """Both callers read BEFORE either writes — the actual race, not two tidy sequential updates.

    This is the arm that would fail if ``update_metadata`` merely wrote once and reported a
    conflict: the second writer holds a pre-label copy, so it must re-read and re-apply.
    """
    _seed(tmp_path)
    first = bc.read_metadata(AID, root=tmp_path)
    second = bc.read_metadata(AID, root=tmp_path)
    assert first["revision"] == second["revision"], "both sides must start from one revision"

    bc.add_label(first, "cells", "")
    bc.write_metadata(first, root=tmp_path)  # lands; revision moves on

    # `second` is now stale. Writing it raw must NOT silently discard the label (asserted below),
    # and going through update_metadata must land the image AND keep the label.
    with pytest.raises(bc.RecordConflict):
        second["image_dims"]["picked-scan-2"] = [384, 384]
        bc.write_metadata(second, root=tmp_path)

    bc.update_metadata(AID, _add_image("picked-scan-2"), root=tmp_path)
    stored = _stored(tmp_path)
    assert _label_names(stored) == ["cells"]
    assert sorted(stored["image_dims"]) == ["picked-scan", "picked-scan-2"]


def test_threaded_label_and_upload_writes_both_survive(tmp_path):
    """The same claim under real threads, both held at a barrier until each has read."""
    _seed(tmp_path)
    ready = threading.Barrier(2, timeout=30)
    errors: list[BaseException] = []

    def run(mutate):
        def body():
            try:
                bc.read_metadata(AID, root=tmp_path)  # read before the barrier: a real race
                ready.wait()
                bc.update_metadata(AID, mutate, root=tmp_path)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        return body

    threads = [
        threading.Thread(target=run(_add_label("cells"))),
        threading.Thread(target=run(_add_image("picked-scan-2"))),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a writer thread did not finish"
    assert not errors, f"a writer raised: {errors!r}"

    stored = _stored(tmp_path)
    assert _label_names(stored) == ["cells"], f"label lost under concurrency: {stored.get('labels')!r}"
    assert sorted(stored["image_dims"]) == ["picked-scan", "picked-scan-2"], (
        f"image lost under concurrency: {stored.get('image_dims')!r}"
    )


def test_many_concurrent_label_writes_all_survive(tmp_path):
    """Eight writers, eight distinct labels, all of them stored.

    A single retry would be enough for two writers; this fails if the retry budget or the locking
    only copes with one collision.
    """
    _seed(tmp_path)
    names = [f"label-{i}" for i in range(8)]
    ready = threading.Barrier(len(names), timeout=30)
    errors: list[BaseException] = []

    def body(name):
        def run():
            try:
                ready.wait()
                bc.update_metadata(AID, _add_label(name), root=tmp_path)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        return run

    threads = [threading.Thread(target=body(n)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    assert not errors, f"a writer raised: {errors!r}"
    assert _label_names(_stored(tmp_path)) == sorted(names)


# ---------------------------------------------------------------------------
# no silent loss, on its own
# ---------------------------------------------------------------------------


def test_a_stale_whole_record_write_raises_instead_of_discarding(tmp_path):
    """The property that covers the call sites this change does not individually rewrite."""
    _seed(tmp_path)
    stale = bc.read_metadata(AID, root=tmp_path)
    bc.update_metadata(AID, _add_label("cells"), root=tmp_path)  # record moves on

    stale["public"] = True
    with pytest.raises(bc.RecordConflict) as exc:
        bc.write_metadata(stale, root=tmp_path)
    assert exc.value.artifact_id == AID
    assert exc.value.expected != exc.value.found, "the error must name both revisions"
    # and the concurrent change is still there, not replaced by the stale copy
    assert _label_names(_stored(tmp_path)) == ["cells"]
    assert _stored(tmp_path)["public"] is False


def test_revision_advances_on_every_write(tmp_path):
    first = _seed(tmp_path)
    second = bc.update_metadata(AID, _add_label("cells"), root=tmp_path)
    third = bc.update_metadata(AID, _add_image("picked-scan-2"), root=tmp_path)
    assert first["revision"] < second["revision"] < third["revision"]


def test_update_metadata_returns_what_is_stored_not_the_caller_copy(tmp_path):
    """clause 2's shape: a caller is never handed a record describing a write that did not land."""
    _seed(tmp_path)
    returned = bc.update_metadata(AID, _add_label("cells"), root=tmp_path)
    assert returned == _stored(tmp_path), "the returned record differs from the stored one"


def test_update_metadata_on_a_missing_record_raises(tmp_path):
    with pytest.raises(KeyError):
        bc.update_metadata("bioimage-io/annotation-does-not-exist", _add_label("x"), root=tmp_path)


# ---------------------------------------------------------------------------
# upgrade safety and the property the fix must not regress
# ---------------------------------------------------------------------------


def test_a_record_written_before_this_change_is_accepted_once_and_stamped(tmp_path):
    """State already on disk carries no `revision`; the upgrade must not brick it."""
    legacy = bc.new_metadata(AID, OWNER)
    legacy.pop("revision", None)
    p = bc.metadata_path(AID, root=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(legacy, indent=2))
    assert "revision" not in json.loads(p.read_text())

    out = bc.update_metadata(AID, _add_label("cells"), root=tmp_path)
    assert isinstance(out["revision"], int)
    assert _label_names(_stored(tmp_path)) == ["cells"]


def test_write_metadata_is_still_atomic_per_file(tmp_path):
    """Guards what write_metadata already got right, so the fix cannot regress it."""
    _seed(tmp_path)
    bc.update_metadata(AID, _add_label("cells"), root=tmp_path)
    p = bc.metadata_path(AID, root=tmp_path)
    assert p.exists(), "the record file is missing"
    leftovers = sorted(q.name for q in p.parent.glob("*.tmp"))
    assert leftovers == [], f"a temp file was left behind: {leftovers}"
    assert _label_names(_stored(tmp_path)) == ["cells"]
