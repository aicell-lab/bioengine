"""No contributor-facing surface may present the disjointness wording 0.27.15 withdrew.

Two metric basis strings asserted, as fact, that a split was disjoint from every
contributor's training data. App 0.27.15 (commit ``e762a17b5af5``) retracted both,
because split A and split B arrive as caller-supplied arrays and this app cannot
check either claim — the most it can say is that the coordinator declared it. The
retraction changed the constants. It did not change the records already written, and
those records are still read back and shown to contributors (#0418).

The fix here is a READ-PATH correction plus a WRITE-SIDE refusal, and the two halves
are tested apart because they fail apart: a reader that stopped matching serves the
retracted claim again in silence, and a writer with no guard lets a caller round-trip
an old string back into a new record.

Why the controls below are not ceremony. This is a string search, and a string search
**fails toward clean** — a fix that broke the projection outright would make every
retracted string vanish from the output and read exactly like a fix that worked. So
every search for the retracted wording is paired with:

* a **presence** assertion on strings the retraction never touched
  (``BASELINE_BASIS``, ``EMPTY_SELECTION_BASIS``), which must still be there, byte for
  byte. Two of the four distinct stored strings being unaffected is what makes this a
  discrimination rather than a search that matched everything; and
* a **negative control** in the same pass: the identical search for wording that was
  never used anywhere must return zero. Without it, a search that stopped matching
  reports clean.

The retracted strings are duplicated as literals in this file on purpose. They are
historical data, not prose: if someone tidies the constant in ``pool.py`` it stops
matching the stored records it exists to recognise, and nothing else would notice.

stdlib + pytest only. No torch, no numpy, no live Hypha.
"""

from __future__ import annotations

import copy
import json
import sys
import types
from typing import Any, Dict, Optional

import pytest

import pool


# ── the retracted wording, re-stated independently of pool.py ────────────────
#
# A second copy, byte for byte, taken from the records as stored. If this and the
# constant in pool.py ever disagree, one of them has drifted and the correction has
# stopped firing on real data.

STORED_SELECTION = (
    "the score the greedy gate admitted contributions against, on split A, a public "
    "benchmark disjoint from every contributor's training data"
)
STORED_WITNESS = (
    "scored on the disjoint witness split B, held back from the selection split and "
    "from every contributor, the only number the improvement curve plots"
)

# The negative control's subject. Shaped like a basis string so it would be found by
# any search loose enough to be matching on shape rather than on content, and never
# written by any release.
NEVER_WRITTEN = (
    "scored on the imaginary split Q, withheld from nobody in particular, a basis "
    "string no release of this app has ever produced"
)

POOL_ID = "bioimage-io/cellpose-sam-community"


def test_the_literals_in_this_file_match_the_constants_in_pool():
    """The pin that makes every other test in this file about real data.

    If the constant in ``pool.py`` is reworded — tidied, re-wrapped, a typo
    "corrected" — the map keyed on it stops matching the stored records and the
    projection quietly resumes publishing the retracted claim. Nothing else fails
    when that happens, which is why it is asserted here against an independent copy
    rather than against itself.
    """
    assert pool.RETRACTED_SELECTION_BASIS == STORED_SELECTION
    assert pool.RETRACTED_WITNESS_BASIS == STORED_WITNESS


def test_the_retracted_wording_is_not_the_current_wording():
    """Discrimination control for the test above. If a retracted string happened to
    equal its replacement, every "corrected" assertion below would pass on a
    correction that changed nothing."""
    assert pool.RETRACTED_SELECTION_BASIS != pool.SELECTION_BASIS
    assert pool.RETRACTED_WITNESS_BASIS != pool.WITNESS_BASIS


def test_no_retracted_string_maps_onto_another_retracted_string():
    """A map whose values were themselves retracted would correct a record onto
    wording that is also withdrawn, and the search for retracted text would still
    come back non-zero — or, worse, come back clean because the second string was
    not in the map."""
    for replacement in pool._RETRACTED_BASIS.values():
        assert replacement not in pool._RETRACTED_BASIS


# ── corrected_metric: the unit ───────────────────────────────────────────────

def _metric(basis: str, role: str = "selection") -> Dict[str, Any]:
    """A stored metric record, hand-built.

    Deliberately NOT built through ``round_metric`` — that path now refuses the
    retracted wording, which is the point of the writer half. These records
    represent what is already on disk, written before the guard existed.
    """
    return {
        "role": role,
        "name": pool.METRIC_NAME,
        "aggregate": 0.8123,
        "aggregate_basis": basis,
        "aggregate_scope": None,
        "higher_is_better": True,
        "n_sites_scored": None,
        "n_datasets_scored": None,
        "per_site": None,
        "per_site_basis": None,
        "aggregate_withheld": None,
    }


class TestCorrectedMetric:
    def test_the_retracted_selection_wording_is_restated(self):
        out = pool.corrected_metric(_metric(STORED_SELECTION))
        assert STORED_SELECTION not in out["aggregate_basis"]
        assert out["aggregate_basis"].startswith(pool.SELECTION_BASIS)

    def test_the_retracted_witness_wording_is_restated(self):
        """The half this row did not name, and the worse half of the two: it is
        attached to the only number the improvement curve plots."""
        out = pool.corrected_metric(_metric(STORED_WITNESS, role="witness"))
        assert STORED_WITNESS not in out["aggregate_basis"]
        assert out["aggregate_basis"].startswith(pool.WITNESS_BASIS)

    def test_the_correction_is_visible_rather_than_seamless(self):
        """A silent substitution leaves a contributor who fetched the stored record
        themselves with an unexplained difference — the same "is this corrupt?" dead
        end #0417 describes. The note says the stored record is unchanged and why."""
        out = pool.corrected_metric(_metric(STORED_SELECTION))
        assert pool.BASIS_CORRECTION_NOTE in out["aggregate_basis"]
        assert "0.27.15" in out["aggregate_basis"]

    def test_nothing_but_the_basis_changes(self):
        """A correction that also moved a SCORE would be a data defect dressed as a
        wording fix."""
        before = _metric(STORED_SELECTION)
        after = pool.corrected_metric(before)
        assert set(after) == set(before)
        for key in after:
            if key != "aggregate_basis":
                assert after[key] == before[key], key

    def test_the_stored_record_is_not_mutated(self):
        """The index is the historical record of what the campaign published. This
        is the read path; it must leave the argument exactly as it found it."""
        before = _metric(STORED_SELECTION)
        snapshot = copy.deepcopy(before)
        pool.corrected_metric(before)
        assert before == snapshot

    def test_an_unaffected_record_is_returned_as_the_SAME_object(self):
        """Identity, not equality, so "was this corrected?" is answerable without
        re-comparing strings — and so a correction that rebuilt every record
        unconditionally would be caught here rather than passing as a no-op."""
        for basis in (pool.SELECTION_BASIS, pool.WITNESS_BASIS,
                      pool.BASELINE_BASIS, pool.EMPTY_SELECTION_BASIS, NEVER_WRITTEN):
            m = _metric(basis)
            assert pool.corrected_metric(m) is m

    def test_absent_and_malformed_metrics_pass_through(self):
        """``witness_metric`` is legitimately None on a merge that admitted nothing.
        A render path that raised on it would turn a normal state into an outage."""
        assert pool.corrected_metric(None) is None
        assert pool.corrected_metric({}) == {}
        odd = {"aggregate_basis": 17}
        assert pool.corrected_metric(odd) is odd
        assert pool.corrected_metric("not a dict") == "not a dict"


# ── round_metric: a record written AFTER the fix cannot reintroduce it ───────

class TestWriterRefusesRetractedWording:
    @pytest.mark.parametrize("basis,role", [
        (STORED_SELECTION, "selection"),
        (STORED_WITNESS, "witness"),
    ])
    def test_a_new_record_cannot_carry_retracted_wording(self, basis, role):
        """The read-path correction fixes the records that exist. This stops a new
        one being written with the withdrawn claim — which is not hypothetical: the
        obvious way to reproduce an old metric is to copy its basis string."""
        with pytest.raises(ValueError) as exc:
            pool.round_metric(role=role, aggregate=0.5, aggregate_basis=basis)
        assert "0.27.15" in str(exc.value)

    def test_the_refusal_names_the_constant_to_use_instead(self):
        """An error that only says no sends the caller back to guess."""
        with pytest.raises(ValueError) as exc:
            pool.round_metric(role="selection", aggregate=0.5,
                              aggregate_basis=STORED_SELECTION)
        assert "SELECTION_BASIS" in str(exc.value)
        with pytest.raises(ValueError) as exc:
            pool.round_metric(role="witness", aggregate=0.5,
                              aggregate_basis=STORED_WITNESS)
        assert "WITNESS_BASIS" in str(exc.value)

    def test_every_basis_the_app_actually_writes_is_still_accepted(self):
        """The control that keeps the guard from being a blanket refusal. These four
        are every ``aggregate_basis`` value ``entry.py`` passes; a guard that
        rejected one of them would stop the merge path dead."""
        for basis in (pool.SELECTION_BASIS, pool.WITNESS_BASIS,
                      pool.BASELINE_BASIS, pool.EMPTY_SELECTION_BASIS):
            m = pool.round_metric(role="witness", aggregate=0.5, aggregate_basis=basis)
            assert m["aggregate_basis"] == basis


# ── the projection: every surface a contributor reads ────────────────────────

def _legacy_index() -> Dict[str, Any]:
    """A stored ``pool_index.json`` as a pre-0.27.15 release wrote it.

    Every one of the five places a metric record reaches the wire carries retracted
    wording, because a fix that covered four of five would otherwise pass. The two
    unaffected strings are present too — they are the thing a broken projection
    would also destroy.
    """
    return {
        "model_type": "cpsam",
        "architecture_signature": "sig-1",
        "current_community": {
            "version": "v2",
            "path": "community/v2/weights.pt",
            "sha256": "c" * 64,
            "selection_metric": _metric(STORED_SELECTION),
            "witness_metric": _metric(STORED_WITNESS, role="witness"),
        },
        "baseline": {
            "model": "stock-cpsam",
            # Unaffected: BASELINE_BASIS was never retracted. Asserted present below.
            "witness_metric": _metric(pool.BASELINE_BASIS, role="witness"),
            # Carries retracted wording and is CURRENTLY NOT PROJECTED at all —
            # pool_wire_state publishes only baseline.witness_metric as
            # baseline_metric. Four records in the live pools sit here, which is why
            # the measured stored count (28 in the indexes) exceeds the measured
            # presented count (24). It is in this fixture deliberately: the
            # whole-tree test below is what stops a later change that starts
            # projecting it from reintroducing the withdrawn claim in silence.
            "selection_metric": _metric(STORED_SELECTION),
        },
        "merges": [
            {
                "created_at": 1_750_100_000.0,
                "members": [{"contribution_id": "alpha-0", "marginal_score": 0.1,
                             "gate_score": 0.1, "included": True}],
                "selection_metric": _metric(STORED_SELECTION),
                "witness_metric": _metric(STORED_WITNESS, role="witness"),
                "global_sha256": "d" * 64,
                # The field that puts a merge in soups[] rather than empty_merges[].
                "published_version": "v1",
            },
            {
                "created_at": 1_750_200_000.0,
                "members": [{"contribution_id": "beta-0", "marginal_score": 0.0,
                             "gate_score": 0.0, "included": False}],
                # Unaffected: an empty merge's bar is EMPTY_SELECTION_BASIS.
                "selection_metric": _metric(pool.EMPTY_SELECTION_BASIS),
            },
        ],
        "contributions": [],
        "decided_contribution_ids": [],
    }


def _all_strings(node: Any) -> str:
    """Every string anywhere in the projected structure, concatenated.

    A whole-tree search rather than a field-by-field one: the clause is that NO
    surface presents the wording, and a field-by-field check can only ever cover the
    fields someone remembered.
    """
    return json.dumps(node, default=str)


class TestWireProjection:
    @pytest.fixture()
    def wire(self):
        return pool.pool_wire_state(_legacy_index(), pool_artifact_id=POOL_ID)

    def test_no_retracted_wording_reaches_the_wire(self, wire):
        blob = _all_strings(wire)
        assert STORED_SELECTION not in blob
        assert STORED_WITNESS not in blob

    def test_negative_control_the_search_still_discriminates(self, wire):
        """Run in the same pass as the test above and on the same haystack. A search
        that stopped matching — a typo in the needle, a projection that returned {} —
        reports clean for the real strings AND for this one. The real search is only
        evidence while this one can still tell the difference."""
        blob = _all_strings(wire)
        assert NEVER_WRITTEN not in blob            # never written: must be absent
        assert pool.SELECTION_BASIS in blob         # ...and the haystack is not empty
        assert pool.WITNESS_BASIS in blob

    def test_unaffected_wording_survives_byte_for_byte(self, wire):
        """The other half of the control, and the one that catches the failure mode
        that reads most like success: a fix that broke the projection makes every
        retracted string vanish. These two were never retracted and must still be
        there, unchanged — a correction that touched them is over-reaching."""
        blob = _all_strings(wire)
        assert pool.BASELINE_BASIS in blob
        assert pool.EMPTY_SELECTION_BASIS in blob
        assert wire["baseline_metric"]["aggregate_basis"] == pool.BASELINE_BASIS
        assert wire["empty_merges"][0]["selection_metric"]["aggregate_basis"] == \
            pool.EMPTY_SELECTION_BASIS

    @pytest.mark.parametrize("path", [
        ("soups", 0, "selection_metric"),
        ("soups", 0, "witness_metric"),
        ("current_community", "selection_metric"),
        ("current_community", "witness_metric"),
    ])
    def test_each_corrected_site_carries_the_note(self, wire, path):
        """Named site by site as well as by whole-tree search, so a failure says
        WHICH surface regressed instead of only that one did."""
        node = wire
        for step in path:
            node = node[step]
        assert pool.BASIS_CORRECTION_NOTE in node["aggregate_basis"]

    def test_the_baselines_selection_metric_is_dropped_not_corrected(self):
        """Says out loud WHY the whole-tree test is green for this one field.

        ``pool_wire_state`` publishes only ``baseline.witness_metric``, as
        ``baseline_metric``; the baseline's SELECTION metric is not projected
        anywhere. Four live records carry retracted wording there and are therefore
        not contributor-facing through this path — the gap between the measured
        stored count and the measured presented count.

        Pinned so the two tests fail differently. If a later change starts
        projecting it, this one goes red naming the field, rather than only the
        whole-tree search going red on a string with no address.
        """
        wire = pool.pool_wire_state(_legacy_index(), pool_artifact_id=POOL_ID)
        assert wire["baseline_metric"]["aggregate_basis"] == pool.BASELINE_BASIS
        assert "baseline_selection_metric" not in wire
        assert STORED_SELECTION not in _all_strings(wire)

    def test_the_stored_index_is_not_mutated_by_rendering(self):
        """``pool_wire_state`` is a projection. If it edited the index in place, the
        next writer would commit the corrected text back into the artifact — turning
        a render-time correction into the artifact rewrite this fix deliberately did
        not do."""
        index = _legacy_index()
        snapshot = copy.deepcopy(index)
        pool.pool_wire_state(index, pool_artifact_id=POOL_ID)
        assert index == snapshot

    def test_a_pool_written_after_the_fix_is_untouched_by_the_correction(self):
        """End-to-end for the writer half: build the index through ``round_metric``,
        project it, and assert nothing was corrected — identity on every metric. If
        this ever fails, a current constant has been added to the retracted map."""
        index = _legacy_index()
        index["current_community"]["selection_metric"] = pool.round_metric(
            role="selection", aggregate=0.8, aggregate_basis=pool.SELECTION_BASIS)
        index["current_community"]["witness_metric"] = pool.round_metric(
            role="witness", aggregate=0.8, aggregate_basis=pool.WITNESS_BASIS)
        wire = pool.pool_wire_state(index, pool_artifact_id=POOL_ID)
        cc = wire["current_community"]
        assert cc["selection_metric"]["aggregate_basis"] == pool.SELECTION_BASIS
        assert cc["witness_metric"]["aggregate_basis"] == pool.WITNESS_BASIS
        assert pool.BASIS_CORRECTION_NOTE not in _all_strings(cc)


# ── the second surface: a published community manifest ───────────────────────
#
# A manifest at community/<version>/manifest.json is a SEPARATE stored file that does
# NOT pass through pool_wire_state. Correcting only the index would leave a
# contributor who asks for a named community version reading the withdrawn claim.

class _Resp:
    status_code = 200

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


class _AM:
    async def get_file(self, artifact_id: str, file_path: str) -> str:
        return "https://example.invalid/" + file_path


class _Entry:
    """Just enough of ``EntryApp`` for ``_read_community_manifest`` to reach its
    return. The manifest payload is the stub; the function under test is real."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    async def _http_retry(self, method: str, url: str, **kw: Any) -> _Resp:
        return _Resp(self._payload)


def _read_manifest_of(instance):
    """The real coroutine, bound to a stub.

    ``entry.EntryApp`` is a Ray Serve ``Deployment`` once ``@bioengine.app`` has run,
    so its methods are not reachable through it; ``func_or_class`` is the wrapped
    class. Imported lazily so a missing optional dependency surfaces as a test
    failure rather than a collection error.
    """
    import entry

    return entry.EntryApp.func_or_class._read_community_manifest.__get__(instance)


@pytest.mark.asyncio
async def test_a_published_manifest_is_corrected_on_read_back():
    stub = _Entry({
        "community_version": "v2",
        "weights_sha256": "e" * 64,
        "selection_metric": _metric(STORED_SELECTION),
        "witness_metric": _metric(STORED_WITNESS, role="witness"),
    })
    out = await _read_manifest_of(stub)(_AM(), POOL_ID, "v2")

    blob = _all_strings(out)
    assert STORED_SELECTION not in blob
    assert STORED_WITNESS not in blob
    assert NEVER_WRITTEN not in blob                      # negative control, same pass
    assert pool.SELECTION_BASIS in blob                   # and the haystack is real
    assert pool.WITNESS_BASIS in blob
    assert out["version"] == "v2"
    assert out["sha256"] == "e" * 64


@pytest.mark.asyncio
async def test_a_manifest_with_current_wording_is_passed_through_unchanged():
    """Discrimination control for the test above: if the read-back path rewrote every
    manifest unconditionally, the test above would pass on a function that corrupts
    current records too."""
    current = _metric(pool.SELECTION_BASIS)
    stub = _Entry({
        "community_version": "v3",
        "weights_sha256": "f" * 64,
        "selection_metric": current,
        "witness_metric": None,
    })
    out = await _read_manifest_of(stub)(_AM(), POOL_ID, "v3")

    assert out["selection_metric"] is current
    assert out["witness_metric"] is None
    assert pool.BASIS_CORRECTION_NOTE not in _all_strings(out)
