"""The staging file a record write goes through, and the widest writer in the broker.

`test_concurrent_record_writes.py` covers the lost update itself: two changes race,
both survive, a stale whole-record write raises. These are the two things that sat
*underneath* that fix and which serialising does not reach.

1. The scratch file. `write_metadata` used to stage through
   `p.with_suffix(".json.tmp")`, a name derived from the artifact id alone, so two
   writers staging at the same instant shared one file and `os.replace` could publish
   a document assembled from both halves. That loses the WHOLE record, not one field,
   and it is strictly worse than the lost update the revision check prevents.

   The lock is not an answer to it. `fcntl.flock` is advisory and answers ENOLCK on an
   NFSv3 mount without `local_lock`, which is the kind of mount the broker's state root
   lives on. So these tests do not assume the lock holds; the per-writer name has to
   remove the shared-file failure on its own.

2. `import_ngff_image`, which reads the record at entry and writes it back after
   awaiting staging, an upload and a commit. The revision check already stops it
   erasing a concurrent change, but it would refuse the write *after* the image had
   been imported -- stranding a successful import behind a failed credential write.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

import broker_core as core


def _record(tmp_path, artifact_id="ws/ds"):
    core.write_metadata(
        core.new_metadata(artifact_id, {"id": "u1"}),
        root=tmp_path,
        expect_revision=None,
    )
    return artifact_id


# ── the scratch file is per writer ─────────────────────────────────────────


def _staged_names(tmp_path, artifact_id, n=2, monkeypatch=None):
    """The source path each write stages through, captured at `os.replace`."""
    seen = []
    real = core.os.replace

    def _spy(src, dst):
        seen.append(str(src))
        return real(src, dst)

    monkeypatch.setattr(core.os, "replace", _spy)
    for i in range(n):
        rec = core.read_metadata(artifact_id, root=tmp_path)
        rec["labels"] = [{"name": f"l{i}", "description": ""}]
        core.write_metadata(rec, root=tmp_path)
    return seen


def test_two_writes_do_NOT_share_one_staging_file(tmp_path, monkeypatch):
    """THE case. A shared scratch name is how a torn record gets published.

    Asserted on the name rather than by racing two processes, because a race that
    happens to interleave safely would report a pass for the wrong reason. If the
    names differ, the shared-file failure cannot occur at all -- there is nothing to
    interleave.
    """
    aid = _record(tmp_path)
    names = _staged_names(tmp_path, aid, n=2, monkeypatch=monkeypatch)

    assert len(names) == 2, names
    assert names[0] != names[1], f"both writes staged through the same file: {names[0]}"


def test_the_staging_name_is_not_derived_from_the_artifact_id_alone(
    tmp_path, monkeypatch
):
    """THE CONTROL on the test above, against a fix that only looks right.

    Two different *records* would also produce two different names, so "the names
    differ" alone is satisfiable without a per-writer component. This pins that two
    writes to the SAME record differ, and that neither is the old shared name.
    """
    aid = _record(tmp_path)
    names = _staged_names(tmp_path, aid, n=2, monkeypatch=monkeypatch)

    old_shared = str(core.metadata_path(aid, tmp_path).with_suffix(".json.tmp"))
    for n in names:
        assert n != old_shared, f"still staging through the shared name {old_shared}"
    assert len(set(names)) == 2


def test_a_failed_replace_leaves_no_scratch_file_behind(tmp_path, monkeypatch):
    """Otherwise the state root silently accumulates files nothing will ever read."""
    aid = _record(tmp_path)

    def _boom(src, dst):
        raise OSError("replace refused")

    monkeypatch.setattr(core.os, "replace", _boom)
    rec = core.read_metadata(aid, root=tmp_path)
    with pytest.raises(OSError):
        core.write_metadata(rec, root=tmp_path)

    leftovers = sorted(p.name for p in core.datasets_dir(tmp_path).glob("*.tmp"))
    assert leftovers == [], leftovers


def test_the_new_staging_name_is_still_invisible_to_list_dataset_ids(
    tmp_path, monkeypatch
):
    """A renamed scratch file must not start showing up as a dataset.

    `list_dataset_ids` globs `*.json` and then filters `.tmp`; the new name has to stay
    excluded by that, or every in-flight write would appear as a phantom dataset.
    """
    aid = _record(tmp_path)
    assert core.list_dataset_ids(tmp_path) == [aid]

    # A scratch file of the shape the writer now produces, left in place deliberately.
    p = core.metadata_path(aid, tmp_path)
    stray = p.with_name(f"{p.name}.4242.deadbeef.tmp")
    stray.write_text("{}")

    assert core.list_dataset_ids(tmp_path) == [
        aid
    ], "a scratch file leaked into the listing"


def test_the_record_survives_a_write_with_the_LOCK_NEUTERED(tmp_path, monkeypatch):
    """The property the per-writer name buys, with the lock removed on purpose.

    `fcntl.flock` answers ENOLCK on an NFSv3 mount without `local_lock`, so the state
    root may well be a place where it does nothing. With it neutered the revision
    compare-and-set is no longer atomic -- that is a known limitation, recorded in
    `write_metadata`'s docstring -- but the stored document must still be a WHOLE,
    parseable record from exactly one writer, never a mixture of two.
    """
    monkeypatch.setattr(core.fcntl, "flock", lambda *a, **k: None)
    aid = _record(tmp_path)

    for i in range(5):
        rec = core.read_metadata(aid, root=tmp_path)
        rec["labels"] = [{"name": f"l{i}", "description": ""}]
        core.write_metadata(rec, root=tmp_path)

    stored = json.loads(core.metadata_path(aid, tmp_path).read_text())
    assert stored["artifact_id"] == aid
    assert [l["name"] for l in stored["labels"]] == ["l4"]


# ── the widest writer goes through update_metadata ─────────────────────────


def _method(name):
    src = (pathlib.Path(__file__).parent / "broker.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(
        f"{name} is not defined in broker.py -- it was renamed or removed, so this "
        f"guard no longer covers it and must be updated deliberately"
    )


def _calls(node, attr):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == attr
    ]


def test_import_ngff_image_does_NOT_write_the_whole_record_it_read():
    """Structural, and said so: this reads the source rather than importing Ray.

    `import_ngff_image` reads the record at entry and finishes long after -- staging,
    an upload and a commit all await in between, which is the widest window in the
    broker. A whole-record write there is refused by the revision check, correctly,
    but only after the import has already happened, so a successful import is stranded
    behind a failed credential write. It must re-read and re-apply instead.
    """
    node = _method("import_ngff_image")

    assert _calls(node, "write_metadata") == [], (
        "import_ngff_image still writes the whole record it read at entry; "
        "a conflict there fails after the image has been imported"
    )
    assert _calls(
        node, "update_metadata"
    ), "import_ngff_image no longer records the credential through update_metadata"


def test_the_guard_above_can_see_a_raw_write_elsewhere():
    """THE CONTROL. The assertion is an absence, so prove the probe finds presences.

    A guard that scanned the wrong node, or an `ast` walk that silently matched
    nothing, would also report "no write_metadata calls". This names a method that
    still writes the whole record on purpose -- a short, non-spanning one -- and
    requires the same probe to see it.
    """
    still_raw = [
        m
        for m in ("register_dataset", "set_dataset_sharing", "delete_label")
        if _calls_safely(m)
    ]
    assert still_raw, (
        "the probe found no raw write_metadata call anywhere, so its absence in "
        "import_ngff_image proves nothing about the fix"
    )


def _calls_safely(name):
    try:
        return bool(_calls(_method(name), "write_metadata"))
    except AssertionError:
        return False


# ── update_metadata's own contract ─────────────────────────────────────────


def test_a_mutate_that_edits_IN_PLACE_and_returns_None_is_not_discarded(tmp_path):
    """`update_metadata`'s docstring invites an in-place mutate. It has to work.

    It did not: the deep copy handed to `mutate` was not retained, so a mutate that
    edited it and returned `None` -- the natural way to write one -- fell back to the
    unmodified record, stored that, and reported success. The only symptom was a key
    quietly missing afterwards, which is the same class of silent loss this whole
    change exists to remove.

    Found by using the documented contract: `import_ngff_image`'s credential write
    vanished exactly this way.
    """
    aid = _record(tmp_path)

    def _in_place(record):
        record["ngff_credentials"] = {"https://example/": "secret"}
        # returns None deliberately

    out = core.update_metadata(aid, _in_place, root=tmp_path)

    assert out.get("ngff_credentials") == {"https://example/": "secret"}, out
    stored = core.read_metadata(aid, root=tmp_path)
    assert stored.get("ngff_credentials") == {"https://example/": "secret"}, stored


def test_a_mutate_that_RETURNS_a_replacement_still_works(tmp_path):
    """THE CONTROL on the test above: the other documented shape must not regress.

    A fix that only honoured the returned value, or only the in-place edit, would pass
    one of these two and fail the other.
    """
    aid = _record(tmp_path)

    def _returning(record):
        record["labels"] = [{"name": "cells", "description": ""}]
        return record

    out = core.update_metadata(aid, _returning, root=tmp_path)

    assert [l["name"] for l in out["labels"]] == ["cells"]
    stored = core.read_metadata(aid, root=tmp_path)
    assert [l["name"] for l in stored["labels"]] == ["cells"]


def test_the_mutate_cannot_edit_the_callers_own_record_object(tmp_path):
    """Why it is a deep COPY: `mutate` must not reach the stored dict the reader holds.

    Keeping the copy (the fix above) must not become "pass `current` straight in",
    which would make a failed attempt leave half-applied edits on the object the retry
    loop re-reads.
    """
    aid = _record(tmp_path)
    seen = []

    def _grab(record):
        seen.append(record)
        record["labels"] = [{"name": "x", "description": ""}]

    core.update_metadata(aid, _grab, root=tmp_path)

    fresh = core.read_metadata(aid, root=tmp_path)
    assert (
        seen and seen[0] is not fresh
    ), "mutate was handed a live reference, not a copy"
