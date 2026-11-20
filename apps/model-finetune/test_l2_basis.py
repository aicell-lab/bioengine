"""every ``l2_from_base`` says what it was measured AGAINST.

Two halves, both from the 2026-11-14 ruling:

1. a stock-initialised deposit carries a NON-NULL ``l2_from_base``, measured
   against the stock base it actually started from;
2. every ``l2_from_base`` carries an explicit basis label, and a record written
   BEFORE this change reads as ambiguous rather than as zero.

The third sentence is the one a shortcut gets wrong. Every pre-the basis label record holds
a distance and no label, and the tempting default — "unlabelled means the
session-init base, because that was the only one that produced a number" — is the
inference 0.27.6 makes unsafe. ``l2_basis`` must refuse it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pool  # noqa: E402


def test_the_four_bases_are_distinct() -> None:
    """Control. If any two collided, every per-case assertion below could pass
    while the resolver returned a single constant for everything."""
    four = [
        pool.L2_BASIS_SESSION_INIT,
        pool.L2_BASIS_STOCK,
        pool.L2_BASIS_ABSENT,
        pool.L2_BASIS_UNDETERMINED,
    ]
    assert len(set(four)) == 4, f"expected four distinct bases, got {four}"


# ── the resolution table ─────────────────────────────────────────────────────

def test_a_labelled_session_init_record_resolves_to_that_base() -> None:
    rec = {"l2_from_base": 1.5, "l2_from_base_basis": pool.L2_BASIS_SESSION_INIT}
    assert pool.l2_basis(rec) == pool.L2_BASIS_SESSION_INIT


def test_a_labelled_stock_record_resolves_to_that_base() -> None:
    rec = {"l2_from_base": 1.5, "l2_from_base_basis": pool.L2_BASIS_STOCK}
    assert pool.l2_basis(rec) == pool.L2_BASIS_STOCK


def test_no_distance_is_its_own_answer_not_undetermined() -> None:
    """"the pool holds no distance" and "the pool holds a distance whose origin is
    unknown" are different facts, and folding them together would lose the one
    the basis label exists to record."""
    assert pool.l2_basis({"l2_from_base": None}) == pool.L2_BASIS_ABSENT
    assert pool.l2_basis({}) == pool.L2_BASIS_ABSENT
    assert pool.L2_BASIS_ABSENT != pool.L2_BASIS_UNDETERMINED


def test_a_pre_change_record_is_UNDETERMINED_not_a_base_and_not_zero() -> None:
    """THE case this row exists for. A distance with no label must not be
    attributed to either base, and must not read as zero movement."""
    pre = {"l2_from_base": 2.75}          # written before the basis label: value, no label
    got = pool.l2_basis(pre)
    assert got == pool.L2_BASIS_UNDETERMINED
    assert got != pool.L2_BASIS_SESSION_INIT, (
        "an unlabelled distance was attributed to the session-init base; that is "
        "the guess 0.27.6 makes unsafe"
    )
    assert got != pool.L2_BASIS_STOCK
    assert got != pool.L2_BASIS_ABSENT, (
        "a record that HOLDS a distance was reported as holding none — that is the "
        "null-vs-zero confusion inverted"
    )
    # And the value itself is untouched: the label is about it, not a replacement.
    assert pre["l2_from_base"] == 2.75


def test_an_unrecognised_label_is_undetermined_not_passed_through() -> None:
    """A label written by a newer app must not be echoed back as if this build
    understood it."""
    rec = {"l2_from_base": 1.0, "l2_from_base_basis": "some_future_base_0122"}
    assert pool.l2_basis(rec) == pool.L2_BASIS_UNDETERMINED


@pytest.mark.parametrize("record", [None, "not a record", 7, []])
def test_a_non_record_is_undetermined(record: Any) -> None:
    assert pool.l2_basis(record) == pool.L2_BASIS_UNDETERMINED


# ── the writer stamps it unconditionally ─────────────────────────────────────

def _meta(**over: Any) -> Dict[str, Any]:
    kw: Dict[str, Any] = dict(
        contribution_id="c-0122",
        model_type="cpsam",
        app_version="0.30.21",
        cellpose_version="4.0.0",
        architecture_signature="sig-cpsam",
        base_checkpoint_id=None,
        base_checkpoint_sha256=None,
        samples_seen=40,
        training_params={},
        l2_from_base=1.25,
        dataset_fingerprint=None,
        solo_val_metric=None,
        weights_sha256="a" * 64,
        uploaded_at=1.0,
    )
    kw.update(over)
    return pool.build_contribution_metadata(**kw)


def test_the_metadata_writer_stamps_the_label_even_when_null() -> None:
    """Null included, for the reason app_version is: a label absent when unknown
    cannot be told apart from one written before labels existed."""
    meta = _meta()                      # caller passes no basis
    assert "l2_from_base_basis" in meta, (
        "the key is missing entirely, so a reader cannot tell this writer from one "
        "that predates the label"
    )
    assert meta["l2_from_base_basis"] is None
    assert pool.l2_basis(meta) == pool.L2_BASIS_UNDETERMINED


def test_the_metadata_writer_carries_a_supplied_label_through() -> None:
    meta = _meta(l2_from_base_basis=pool.L2_BASIS_STOCK)
    assert meta["l2_from_base_basis"] == pool.L2_BASIS_STOCK
    assert pool.l2_basis(meta) == pool.L2_BASIS_STOCK


def test_the_stamp_does_not_disturb_the_digest_basis() -> None:
    """Paired assertion: the stored-basis stamp's label must still be stamped beside the basis label's, or
    this change traded one attributable field for another."""
    meta = _meta(l2_from_base_basis=pool.L2_BASIS_STOCK)
    assert meta["weights_sha256_basis"] == pool.CONTRIBUTION_DIGEST_BASIS


# ── half 1: both runtime branches set the label ──────────────────────────────

def test_both_runtime_branches_label_the_distance_they_measured() -> None:
    """Read off the source, because exercising it needs torch and a real
    checkpoint. The claim is structural: the stock branch must compute a distance
    AND name its base, or the basis label's null simply moves to the label.

    Controls: a constant that appears in neither branch must be absent, and the
    file must actually have been read.
    """
    src = (HERE / "runtime_cellpose.py").read_text(encoding="utf-8")
    assert len(src) > 1000, "runtime_cellpose.py read back implausibly short"

    assert "pool.L2_BASIS_SESSION_INIT" in src, "the init-checkpoint branch names no base"
    assert "pool.L2_BASIS_STOCK" in src, "the stock branch names no base"
    # The stock branch must MEASURE, not just label.
    assert re.search(
        r'out\["l2_from_base"\]\s*=\s*pool\.state_dict_l2_from_base\(sd,\s*stock\)', src
    ), "the stock branch does not compute a distance against the stock base"
    # Negative control: a constant no branch uses.
    assert "pool.L2_BASIS_UNDETERMINED" not in src, (
        "a runtime branch stamped UNDETERMINED as if it were a base; undetermined "
        "is a READER's answer to a missing label, never something a writer stamps"
    )
    assert "ZZZNOSUCH_L2_BASIS_0122" not in src
