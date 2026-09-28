"""Contract tests for ``pool.py``'s campaign wire projection.

Plain pytest, no torch, no numpy, no live Hypha — ``pool``'s heavy imports are all
function-local, so the schema surface imports on stdlib alone. These pin the
0.13.0-draft wire contract the campaign page reads (``pool.pool_wire_state``): the
role-tagged RoundMetric shape, the soups/empty-merges partition, and — the guard
that matters — that a SELECTION value can never surface on the witness curve. The
directional test fails on a misplaced VALUE, not merely a mislabelled tag.

The field-sets below were derived from the page's verified 0.13.0-draft fixture
corpus (``cellpose-sam-community.json``). The corpus is not vendored (it is the
page's artifact and rots on draft bumps); ``test_matches_fixture_corpus`` cross-
checks these constants against it when a path is supplied via
``MODEL_FINETUNE_CORPUS_DIR``, so a draft bump that moves a field is caught the next
time the test is run against the fresh corpus.
"""

from __future__ import annotations

import os
import pathlib

import pytest

import pool


# Field-sets pinned from the 0.13.0-draft fixture corpus.
ROUNDMETRIC_KEYS = {
    "role", "name", "aggregate", "aggregate_basis", "aggregate_scope",
    "higher_is_better", "n_sites_scored", "n_datasets_scored", "per_site",
    "per_site_basis", "aggregate_withheld",
}
SOUP_KEYS = {
    "soup_id", "index", "merged_at", "assessed", "contributions",
    "selection_metric", "witness_metric", "global_sha256", "weights",
    "transport", "community_model",
}
EMPTY_MERGE_KEYS = {"merged_at", "assessed", "selection_metric", "transport"}
MERGE_TRIGGER_KEYS = {
    "kind", "contributions_per_merge", "next_merge_at", "decided_by", "agent",
}
COMMUNITY_MODEL_KEYS = {"artifact_id", "version", "url", "n_contributions_cumulative"}
TRANSPORT_KEYS = {"bytes_in", "bytes_out", "n_transfers", "sources_complete"}

POOL_ID = "bioimage-io/cellpose-sam-community"

# Distinct split-A (selection) and split-B (witness) scores per merge, so a swap of
# the two moves a real number. B (witness) is the plotted curve; A (selection) is
# the internal gate and must never appear on it.
BASELINE_B = 0.6412
SOUP0_A, SOUP0_B = 0.8000, 0.7235
EMPTY_BAR_A = 0.6985
SOUP1_A, SOUP1_B = 0.8300, 0.8478


def _member(cid, score, included):
    return {"contribution_id": cid, "gate_score": score, "included": included}


def _synthetic_index():
    """A stored pool_index.json as ``aggregate()`` would write it: a baseline, a
    published soup, an empty merge (nothing beat the bar), then another soup. Uses
    ``pool.round_metric`` exactly as the entry does, so the test exercises the real
    role-stamping path, not a hand-built record."""
    baseline = {
        "model": "stock-cpsam",
        "selection_metric": pool.round_metric(
            role="selection", aggregate=0.62, aggregate_basis=pool.SELECTION_BASIS),
        "witness_metric": pool.round_metric(
            role="witness", aggregate=BASELINE_B, aggregate_basis=pool.BASELINE_BASIS),
        "created_at": 1_750_000_000.0,
    }
    merges = [
        {
            "published_version": "v1", "baseline_version": "stock-cpsam",
            "members": [_member("alpha-0", SOUP0_A, True), _member("beta-0", 0.70, False)],
            "selection_metric": pool.round_metric(
                role="selection", aggregate=SOUP0_A, aggregate_basis=pool.SELECTION_BASIS),
            "witness_metric": pool.round_metric(
                role="witness", aggregate=SOUP0_B, aggregate_basis=pool.WITNESS_BASIS),
            "global_sha256": "a" * 64,
            "transport": {"bytes_in": 2_600_000_000, "bytes_out": 1_300_000_000,
                          "n_transfers": 3, "sources_complete": True},
            "deferred": [], "created_at": 1_750_100_000.0,
        },
        {
            "published_version": None, "baseline_version": "v1",
            "members": [_member("gamma-0", 0.66, False)],
            "selection_metric": pool.round_metric(
                role="selection", aggregate=EMPTY_BAR_A, aggregate_basis=pool.EMPTY_SELECTION_BASIS),
            "witness_metric": None,
            "global_sha256": None,
            "transport": {"bytes_in": 1_300_000_000, "bytes_out": 0,
                          "n_transfers": 1, "sources_complete": True},
            "deferred": [], "created_at": 1_750_200_000.0,
        },
        {
            "published_version": "v2", "baseline_version": "v1",
            "members": [_member("delta-0", SOUP1_A, True)],
            "selection_metric": pool.round_metric(
                role="selection", aggregate=SOUP1_A, aggregate_basis=pool.SELECTION_BASIS),
            "witness_metric": pool.round_metric(
                role="witness", aggregate=SOUP1_B, aggregate_basis=pool.WITNESS_BASIS),
            "global_sha256": "b" * 64,
            "transport": {"bytes_in": 1_300_000_000, "bytes_out": 1_300_000_000,
                          "n_transfers": 2, "sources_complete": True},
            "deferred": [], "created_at": 1_750_300_000.0,
        },
    ]
    return {
        "schema_version": pool.SCHEMA_VERSION,
        "model_type": "cpsam",
        "architecture_signature": "sig123",
        "current_community": {"version": "v2", "path": "community/v2/weights.pt",
                              "sha256": "b" * 64,
                              "selection_metric": merges[2]["selection_metric"],
                              "witness_metric": merges[2]["witness_metric"]},
        "contributions": [
            {"contribution_id": "alpha-0", "uploaded_at": 1_750_050_000.0,
             "n_bytes": 1_300_000_000, "n_train_images": 400, "samples_seen": 4000},
            {"contribution_id": "beta-0", "uploaded_at": 1_750_060_000.0,
             "n_bytes": 1_300_000_000, "n_train_images": 200, "samples_seen": 2000},
            {"contribution_id": "gamma-0", "uploaded_at": 1_750_150_000.0,
             "n_bytes": 1_300_000_000, "n_train_images": 100, "samples_seen": 1000},
            {"contribution_id": "delta-0", "uploaded_at": 1_750_250_000.0,
             "n_bytes": 1_300_000_000, "n_train_images": 300, "samples_seen": 3000},
            {"contribution_id": "epsilon-0", "uploaded_at": 1_750_350_000.0,
             "n_bytes": 1_300_000_000, "n_train_images": 500, "samples_seen": 5000},
        ],
        "decided_contribution_ids": ["alpha-0", "beta-0", "gamma-0", "delta-0"],
        "baseline": baseline,
        "merges": merges,
    }


@pytest.fixture()
def wire():
    return pool.pool_wire_state(_synthetic_index(), pool_artifact_id=POOL_ID)


class TestRoundMetric:
    def test_exact_field_set(self):
        m = pool.round_metric(role="witness", aggregate=0.5, aggregate_basis="x")
        assert set(m) == ROUNDMETRIC_KEYS

    def test_role_is_rejected_when_invalid(self):
        with pytest.raises(ValueError):
            pool.round_metric(role="curve", aggregate=0.5, aggregate_basis="x")

    def test_holdout_scope_refuses_per_site_map(self):
        with pytest.raises(ValueError):
            pool.round_metric(role="witness", aggregate=0.5, aggregate_basis="x",
                              per_site={"site-1": 0.5})

    def test_selection_and_witness_share_the_metric_name(self):
        s = pool.round_metric(role="selection", aggregate=0.5, aggregate_basis="a")
        w = pool.round_metric(role="witness", aggregate=0.5, aggregate_basis="b")
        assert s["name"] == w["name"] == pool.METRIC_NAME


class TestMergeTrigger:
    def test_five_fields_all_null(self, wire):
        mt = wire["merge_trigger"]
        assert set(mt) == MERGE_TRIGGER_KEYS
        assert all(v is None for v in mt.values())


class TestWireShape:
    def test_partition_counts(self, wire):
        assert len(wire["soups"]) == 2
        assert len(wire["empty_merges"]) == 1

    def test_soup_key_set(self, wire):
        for soup in wire["soups"]:
            assert set(soup) == SOUP_KEYS
            assert set(soup["community_model"]) == COMMUNITY_MODEL_KEYS
            assert set(soup["transport"]) == TRANSPORT_KEYS
            assert set(soup["selection_metric"]) == ROUNDMETRIC_KEYS
            assert set(soup["witness_metric"]) == ROUNDMETRIC_KEYS

    def test_empty_merge_key_set_has_no_witness(self, wire):
        em = wire["empty_merges"][0]
        assert set(em) == EMPTY_MERGE_KEYS
        assert "witness_metric" not in em
        assert em["selection_metric"]["role"] == "selection"

    def test_soup_ordinals_and_cumulative(self, wire):
        s0, s1 = wire["soups"]
        assert (s0["soup_id"], s0["index"]) == ("soup-0", 0)
        assert (s1["soup_id"], s1["index"]) == ("soup-1", 1)
        assert s0["community_model"]["n_contributions_cumulative"] == 1
        assert s1["community_model"]["n_contributions_cumulative"] == 2

    def test_community_model_url(self, wire):
        s0 = wire["soups"][0]
        assert s0["community_model"]["artifact_id"] == POOL_ID
        assert s0["community_model"]["url"] == "#/models/cellpose-sam-community?version=v1"

    def test_merged_at_is_iso_utc_millis(self, wire):
        merged_at = wire["soups"][0]["merged_at"]
        assert merged_at.endswith("Z") and "T" in merged_at
        assert merged_at.count(".") == 1 and merged_at.split(".")[1] == "000Z"

    def test_weights_never_leave_the_site(self, wire):
        assert all(s["weights"] is None for s in wire["soups"])


class TestAssessedIsIdsOnly:
    """A per-candidate gate_score in the wire payload is a contract violation:
    ``assessed`` is IDs only, and the audit-only gate_score must not leak."""

    def test_assessed_and_contributions_are_id_strings(self, wire):
        for soup in wire["soups"]:
            assert all(isinstance(x, str) for x in soup["assessed"])
            assert all(isinstance(x, str) for x in soup["contributions"])
        assert all(isinstance(x, str) for x in wire["empty_merges"][0]["assessed"])

    def test_taken_is_a_subset_of_assessed(self, wire):
        s0 = wire["soups"][0]
        assert s0["assessed"] == ["alpha-0", "beta-0"]
        assert s0["contributions"] == ["alpha-0"]

    def test_no_gate_score_anywhere_in_the_wire(self, wire):
        import json

        assert "gate_score" not in json.dumps(wire)


class TestContributionDisposition:
    def test_disposition_tracks_merge_outcome(self, wire):
        by_id = {c["contribution_id"]: c for c in wire["contributions"]}
        assert by_id["alpha-0"]["disposition"] == "included"
        assert by_id["alpha-0"]["merged_into"] == "soup-0"
        assert by_id["beta-0"]["disposition"] == "excluded"
        assert by_id["beta-0"]["merged_into"] is None
        assert by_id["gamma-0"]["disposition"] == "excluded"
        assert by_id["epsilon-0"]["disposition"] == "pending"


def _witness_curve(wire):
    """The values the page would plot on the improvement curve: the aggregate of
    every soup metric tagged ``role='witness'`` — the page's own selection rule."""
    return [s["witness_metric"]["aggregate"]
            for s in wire["soups"] if s["witness_metric"]["role"] == "witness"]


def _selection_values(wire):
    return [s["selection_metric"]["aggregate"] for s in wire["soups"]]


class TestDirectionalGuard:
    """The load-bearing test: publish must not surface a SELECTION value under
    role='witness'. These assertions fail on a misplaced VALUE, not just a swapped
    tag — the witness curve is checked against the known split-B numbers."""

    def test_baseline_metric_is_witness_role(self, wire):
        assert wire["baseline_metric"]["role"] == "witness"
        assert wire["baseline_metric"]["aggregate"] == BASELINE_B

    def test_witness_curve_is_split_b_not_split_a(self, wire):
        assert _witness_curve(wire) == [SOUP0_B, SOUP1_B]
        # split-A (selection) values are distinct and must not appear on the curve
        assert _selection_values(wire) == [SOUP0_A, SOUP1_A]
        assert set(_witness_curve(wire)).isdisjoint(_selection_values(wire))

    def test_roles_match_their_field_names(self, wire):
        for soup in wire["soups"]:
            assert soup["selection_metric"]["role"] == "selection"
            assert soup["witness_metric"]["role"] == "witness"

    def test_a_misplaced_selection_value_is_caught(self):
        """Simulate the exact defect: publish builds the witness metric from the
        split-A value (role still 'witness', but the wrong number). The curve check
        must reject it — proving the guard catches a misplaced value, not only a bad
        tag."""
        idx = _synthetic_index()
        merge0 = idx["merges"][0]
        merge0["witness_metric"] = pool.round_metric(
            role="witness", aggregate=SOUP0_A, aggregate_basis=pool.WITNESS_BASIS)
        corrupted = pool.pool_wire_state(idx, pool_artifact_id=POOL_ID)
        # Tag still reads witness, so a tag-only check would pass...
        assert corrupted["soups"][0]["witness_metric"]["role"] == "witness"
        # ...but the curve now carries a split-A value, which the guard rejects.
        assert _witness_curve(corrupted) != [SOUP0_B, SOUP1_B]
        with pytest.raises(AssertionError):
            assert _witness_curve(corrupted) == [SOUP0_B, SOUP1_B]


@pytest.mark.skipif(
    not os.environ.get("MODEL_FINETUNE_CORPUS_DIR"),
    reason="set MODEL_FINETUNE_CORPUS_DIR to the verified 0.13.0-draft fixture corpus to cross-check field-sets",
)
def test_matches_fixture_corpus():
    """Cross-check the pinned field-sets against the page's verified fixture, so a
    draft bump that moves a field is caught the next time this runs against the
    fresh corpus."""
    import json

    d = pathlib.Path(os.environ["MODEL_FINETUNE_CORPUS_DIR"])
    doc = json.loads((d / "cellpose-sam-community.json").read_text())
    prog = doc["progress"]
    assert doc["schema_version"] == pool.SCHEMA_VERSION
    assert set(prog["merge_trigger"]) == MERGE_TRIGGER_KEYS
    assert set(prog["baseline_metric"]) == ROUNDMETRIC_KEYS
    soup = prog["soups"][1]
    assert set(soup) == SOUP_KEYS
    assert set(soup["community_model"]) == COMMUNITY_MODEL_KEYS
    assert set(soup["transport"]) == TRANSPORT_KEYS
    assert set(soup["selection_metric"]) == ROUNDMETRIC_KEYS
    assert set(soup["witness_metric"]) == ROUNDMETRIC_KEYS
    assert set(prog["empty_merges"][0]) == EMPTY_MERGE_KEYS
