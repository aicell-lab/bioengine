"""The staging file a record write goes through, and the widest writer in the broker.

`test_concurrent_record_writes.py` covers the lost update itself: two changes race,
both survive, a stale whole-record write raises. These are the two things that sat
*underneath* that fix and which serialising does not reach.

1. The scratch file. `write_metadata` used to stage through
   `p.with_suffix(".json.tmp")`, a name derived from the artifact id alone, so two
   writers staging at the same instant shared one file and `os.replace` could publish
   a document assembled from both halves. That loses the WHOLE record, not one field,
   and it is strictly worse than the lost update the revision check prevents.

   The lock is not an answer to it. `fcntl.flock` is advisory and there are filesystems
   on which it does not hold; which one this state root is on is not measured here, and
   no claim is made about it. These tests simply do not assume the lock holds -- the
   per-writer name has to remove the shared-file failure on its own.

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


# A `test_the_record_survives_a_write_with_the_LOCK_NEUTERED` arm stood here. It wrote
# five times IN SEQUENCE with `flock` stubbed out, and a review showed it stayed green
# with the shared `.json.tmp` restored -- so it tested nothing its name promised, which
# is worse than having no arm at all. Sequential writes cannot interleave, and making
# them genuinely overlap in-process is not something this file can do deterministically.
# Removed rather than left as decoration; the per-writer name is pinned by the two
# name-based arms above.


# ── a record written BEFORE the revision stamp existed ─────────────────────


def test_a_LEGACY_record_cannot_be_overwritten_by_a_SECOND_stale_copy(tmp_path):
    """THE case that mattered most, and it was live on every deployed dataset.

    `expected is None` meant two different things: "the caller wants an unconditional
    write" and "this record predates the revision stamp". Conflating them skipped the
    comparison for every record already on disk -- and every record on a deployed
    worker predates this change. Measured before the fix: a legacy record, a concurrent
    label write that stamped revision 1, then a stale whole-record write -> ACCEPTED,
    and the label erased. So the reported bug survived the fix on every existing
    dataset until that record happened to get its first post-upgrade write.

    The first stale write is still accepted and stamps the record -- that is the
    documented migration step. It is the SECOND one that must be refused.
    """
    aid = "ws/legacy"
    p = core.metadata_path(aid, tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"artifact_id": aid, "owner": {"id": "u1"}, "labels": []}, indent=2)
    )
    assert "revision" not in json.loads(p.read_text())

    stale = core.read_metadata(aid, root=tmp_path)
    assert stale.get("revision") is None

    core.update_metadata(
        aid,
        lambda r: r.__setitem__("labels", [{"name": "cells", "description": ""}]),
        root=tmp_path,
    )
    assert core.read_metadata(aid, root=tmp_path).get("revision") == 1

    with pytest.raises(core.RecordConflict):
        core.write_metadata(stale, root=tmp_path)

    final = core.read_metadata(aid, root=tmp_path)
    assert [l["name"] for l in final["labels"]] == ["cells"], final


def test_a_legacy_record_is_still_accepted_ONCE_and_stamped(tmp_path):
    """THE CONTROL on the test above: the migration step must not have been broken.

    A fix that refused every write whose in-hand revision was missing would satisfy the
    test above and make every pre-existing dataset permanently unwritable.
    """
    aid = "ws/legacy2"
    p = core.metadata_path(aid, tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"artifact_id": aid, "owner": {"id": "u1"}, "labels": []}))

    rec = core.read_metadata(aid, root=tmp_path)
    rec["labels"] = [{"name": "first", "description": ""}]
    out = core.write_metadata(rec, root=tmp_path)

    assert out["revision"] == 1
    assert [l["name"] for l in core.read_metadata(aid, root=tmp_path)["labels"]] == [
        "first"
    ]


def test_creating_a_record_that_does_not_exist_is_not_refused(tmp_path):
    """THE OTHER CONTROL: a fresh record carries revision 1 and nothing is stored.

    Comparing that against a missing file would refuse every creation, so the check
    deliberately skips when there is no stored record to have moved on. Without this
    arm, tightening the comparison looks free.
    """
    out = core.write_metadata(
        core.new_metadata("ws/brand-new", {"id": "u1"}), root=tmp_path
    )

    assert out["artifact_id"] == "ws/brand-new"
    assert core.read_metadata("ws/brand-new", root=tmp_path) is not None


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


@pytest.mark.parametrize(
    "method", ["create_label", "get_dataset_index", "delete_label", "import_ngff_image"]
)
def test_every_AWAIT_SPANNING_writer_goes_through_update_metadata(method):
    """Each method that reads the record, awaits, and then writes it back.

    A review found that removing the fix from `create_label` or `get_dataset_index`
    left all 639 tests green -- the two sites that actually produce the reported bug
    were unguarded, because every arm called `core.update_metadata` directly instead
    of driving the method. These pin the call sites, so putting the old
    read/modify/write-the-whole-record shape back is RED.

    `delete_label` is here for a sharper reason than the others: its awaits DELETE the
    label's files, so a refused write leaves the label in the record with its folder
    gone. An earlier version of this file listed it as "short, non-spanning". That was
    wrong -- it holds six awaits between its read and its write.
    """
    node = _method(method)

    assert _calls(node, "write_metadata") == [], (
        f"{method} still writes the whole record it read before awaiting; "
        f"a conflict there either erases a concurrent change or half-commits"
    )
    assert _calls(
        node, "update_metadata"
    ), f"{method} does not go through update_metadata"


def test_the_guard_above_can_see_a_raw_write_elsewhere():
    """THE CONTROL. The assertion is an absence, so prove the probe finds presences.

    A guard that scanned the wrong node, or an `ast` walk that silently matched
    nothing, would also report "no write_metadata calls".

    The two methods named here hold **zero** awaits between their read and their write,
    measured by walking their bodies -- so they are genuinely non-spanning and are
    deliberately left writing the whole record. That is what makes them a fair control
    rather than a list of names that happens to be convenient.
    """
    still_raw = [
        m for m in ("request_access", "dismiss_access_request") if _calls_safely(m)
    ]
    assert still_raw, (
        "the probe found no raw write_metadata call anywhere, so its absence in the "
        "spanning writers proves nothing about the fix"
    )


@pytest.mark.parametrize("method", ["request_access", "dismiss_access_request"])
def test_the_control_methods_really_are_non_spanning(method):
    """And the control's premise, asserted rather than asserted-about.

    If one of these grew an await between its read and its write, it would become a
    spanning writer and the control would be quietly defending a defect.
    """
    node = _method(method)
    awaits = [a for a in ast.walk(node) if isinstance(a, ast.Await)]

    assert awaits == [], (
        f"{method} now awaits {len(awaits)} time(s), so it spans a window and must go "
        f"through update_metadata like the others -- it can no longer serve as the "
        f"non-spanning control"
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
