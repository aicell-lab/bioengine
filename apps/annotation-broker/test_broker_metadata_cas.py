"""A dataset record must not lose one of two concurrent changes.

A user uploaded images, created a label, got a record back that contained it, and the label was
gone on the next read. No error anywhere. Annotating then failed with
``ValueError: Unknown label 'cells'`` — naming the label they had just made.

The cause is not the label code. ``get_dataset_index`` reads the whole record at entry, awaits
``_image_manifest`` and one HTTP range GET per image, and then writes that record back WHOLE to
cache image dimensions. Anything written during those awaits is erased. ``create_label`` is the
victim, not the racer: its own read-modify-write has no ``await`` in it.

``write_metadata`` was already atomic at the file level — tmp file plus ``os.replace`` — and that
is probably why this survived so long. Atomicity of the write says nothing about staleness of the
payload; the overwrite was clean, complete, and wrong.

The arms below pin the behaviour, and arm 2 replays the same interleaving through the OLD
unguarded write to show the loss is real and that these arms can see it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import broker_core as bc

ART = "ws/ds-1"


def _fresh(root: Path) -> dict:
    meta = bc.new_metadata(ART, {"id": "u1", "email": "u1@example.org"})
    return bc.write_metadata(meta, root=root)


# ───────────────────────────── clause 1: two concurrent changes both survive ─────────────────


def test_two_interleaved_changes_BOTH_survive(tmp_path: Path) -> None:
    """The exact interleaving from the report, at record level.

    A = create_label (fast, no awaits).  B = the dims cache write, which read the record BEFORE
    A ran and writes after it. Under compare-and-set B must notice and retry, so the label
    survives AND the dims land.
    """
    _fresh(tmp_path)

    # B reads first -- this is the stale copy that used to do the damage.
    b_read = bc.read_metadata(ART, tmp_path)
    b_token = b_read["updated_at"]

    # A (create_label) reads, adds the label, and writes. It wins cleanly.
    bc.update_metadata(ART, lambda m: bc.add_label(m, "cells", ""), root=tmp_path)

    # B now tries to write its stale record back. CAS must refuse it.
    with pytest.raises(bc.MetadataConflict):
        stale = dict(b_read)
        stale["image_dims"] = {"scan-1": [384, 384]}
        bc.write_metadata(stale, root=tmp_path, expect_updated_at=b_token)

    # B retried properly, as the fixed call site does.
    def _merge(m: dict) -> None:
        merged = dict(m.get("image_dims") or {})
        merged.update({"scan-1": [384, 384]})
        m["image_dims"] = merged

    final = bc.update_metadata(ART, _merge, root=tmp_path)

    on_disk = bc.read_metadata(ART, tmp_path)
    assert [l["name"] for l in on_disk["labels"]] == ["cells"], "the label was lost"
    assert on_disk["image_dims"] == {"scan-1": [384, 384]}, "the dims were lost"
    assert final == on_disk, "update_metadata returned something other than the stored record"


def test_the_SAME_interleaving_loses_the_label_through_the_OLD_unguarded_write(
    tmp_path: Path,
) -> None:
    """Arm 2 exists to prove arm 1 is not vacuous.

    Identical ordering, but B writes with no ``expect_updated_at`` -- which is exactly what the
    code did before. The label must disappear. If this ever stops losing it, arm 1 has stopped
    testing anything and both need re-reading.
    """
    _fresh(tmp_path)
    b_read = bc.read_metadata(ART, tmp_path)

    bc.update_metadata(ART, lambda m: bc.add_label(m, "cells", ""), root=tmp_path)
    assert [l["name"] for l in bc.read_metadata(ART, tmp_path)["labels"]] == ["cells"]

    stale = dict(b_read)
    stale["image_dims"] = {"scan-1": [384, 384]}
    bc.write_metadata(stale, root=tmp_path)  # the old, unguarded path

    on_disk = bc.read_metadata(ART, tmp_path)
    assert on_disk["labels"] == [], (
        "the unguarded write did NOT lose the label, so these arms cannot see the bug "
        "they exist for"
    )


def test_a_create_requires_that_no_record_exists(tmp_path: Path) -> None:
    """``expect_updated_at=None`` means 'there must be nothing there yet'."""
    meta = bc.new_metadata(ART, {"id": "u1", "email": "u1@example.org"})
    bc.write_metadata(meta, root=tmp_path, expect_updated_at=None)  # ok, nothing there
    with pytest.raises(bc.MetadataConflict):
        bc.write_metadata(meta, root=tmp_path, expect_updated_at=None)  # now there is


def test_an_unguarded_write_still_works_so_untouched_callers_are_unaffected(
    tmp_path: Path,
) -> None:
    """The guard is opt-in. Nine other call sites still pass no token, and this change must not
    have altered what they do -- 288 existing arms depend on that."""
    _fresh(tmp_path)
    out = bc.write_metadata(
        {**bc.read_metadata(ART, tmp_path), "public": True}, root=tmp_path
    )
    assert out["public"] is True
    assert bc.read_metadata(ART, tmp_path)["public"] is True


# ───────────── clause 2: the caller is never handed a record describing a lost write ─────────


def test_update_metadata_returns_the_STORED_record_not_the_callers_dict(
    tmp_path: Path,
) -> None:
    """This is the half of the bug the user actually saw.

    ``create_label`` returned the dict it had just mutated, so the reply contained a label that
    was never stored -- which is why the failure surfaced much later, as a save against a label
    the user had watched themselves create.
    """
    _fresh(tmp_path)
    returned = bc.update_metadata(
        ART, lambda m: bc.add_label(m, "cells", "desc"), root=tmp_path
    )
    stored = bc.read_metadata(ART, tmp_path)
    assert returned == stored
    # And it is a distinct object, so a later mutation of the reply cannot rewrite history.
    returned["labels"].append({"name": "ghost", "description": ""})
    assert [l["name"] for l in bc.read_metadata(ART, tmp_path)["labels"]] == ["cells"]


def test_a_persistent_conflict_RAISES_rather_than_reporting_success(
    tmp_path: Path,
) -> None:
    """"Persists or raises" -- never "reports a success it cannot support".

    The mutate callback writes the record itself on every attempt, so every compare-and-set
    fails. The call must give up loudly.
    """
    _fresh(tmp_path)

    def _always_conflict(m: dict) -> None:
        # Land an unrelated write between this attempt's read and its write.
        bc.write_metadata(
            {**bc.read_metadata(ART, tmp_path), "public": True}, root=tmp_path
        )
        bc.add_label(m, "cells", "")

    with pytest.raises(bc.MetadataConflict):
        bc.update_metadata(ART, _always_conflict, root=tmp_path, attempts=3)


def test_a_missing_record_raises_rather_than_creating_one(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        bc.update_metadata("ws/nope", lambda m: None, root=tmp_path)


# ───────────────────────── the WIRING, because the mutants here are deletions ────────────────
#
# The arms above exercise broker_core. They would all still pass if broker.py went back to
# writing its stale `meta` -- which is the entire defect. These read the call sites out of the
# AST instead.

BROKER = Path(__file__).with_name("broker.py")


def _func(tree: ast.AST, name: str) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in broker.py; this guard is testing nothing")


def _calls(node: ast.AST) -> list:
    return [
        n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]


@pytest.mark.parametrize("fname", ["create_label", "get_dataset_index"])
def test_the_racing_call_sites_go_through_update_metadata(fname: str) -> None:
    """Both implicated writers must re-read under the guard.

    Guarding only `create_label` would not fix this: the write that destroys data is the one in
    `get_dataset_index`.
    """
    fn = _func(ast.parse(BROKER.read_text()), fname)
    called = _calls(fn)
    assert "update_metadata" in called, (
        f"{fname} does not call update_metadata, so it still writes a record it read "
        f"before its awaits"
    )
    assert "write_metadata" not in called, (
        f"{fname} still calls write_metadata directly, which overwrites the whole record"
    )
