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

import hashlib
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
    return {"contribution_id": cid, "marginal_score": score, "gate_score": score, "included": included}


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
    """A per-candidate score in the wire payload is a contract violation:
    ``assessed`` is IDs only, and the audit-only per-member scores must not leak."""

    def test_assessed_and_contributions_are_id_strings(self, wire):
        for soup in wire["soups"]:
            assert all(isinstance(x, str) for x in soup["assessed"])
            assert all(isinstance(x, str) for x in soup["contributions"])
        assert all(isinstance(x, str) for x in wire["empty_merges"][0]["assessed"])

    def test_taken_is_a_subset_of_assessed(self, wire):
        s0 = wire["soups"][0]
        assert s0["assessed"] == ["alpha-0", "beta-0"]
        assert s0["contributions"] == ["alpha-0"]

    def test_no_per_member_score_anywhere_in_the_wire(self, wire):
        import json

        assert "gate_score" not in json.dumps(wire)
        assert "marginal_score" not in json.dumps(wire)


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


# ── soup member cache eviction (pool.evict_lru_to_cap) ───────────────────────
# The cache this bounds is the one whose unbounded growth OOM-killed the soup
# replica. It is filled by candidates WEIGHED (up to max_batch, default 32), not
# by members admitted, and it outlives any one merge — so a cap, not a per-merge
# clear, is what bounds it.


def _cached_member(directory, name, size, mtime):
    path = directory / name
    path.write_bytes(b"\0" * size)
    os.utime(path, (mtime, mtime))
    return path


def test_evicts_oldest_first_until_under_cap(tmp_path):
    a = _cached_member(tmp_path, "member-a.pt", 100, mtime=1_000)
    b = _cached_member(tmp_path, "member-b.pt", 100, mtime=2_000)
    c = _cached_member(tmp_path, "member-c.pt", 100, mtime=3_000)

    assert pool.evict_lru_to_cap(tmp_path, "member-*.pt", cap_bytes=250) == 1
    assert not a.exists()
    assert b.exists() and c.exists()


def test_incoming_bytes_make_room_before_the_write(tmp_path):
    """``_fetch_member`` evicts BEFORE writing, so the file about to land has to be
    counted or the cap is exceeded by exactly one member every time."""
    _cached_member(tmp_path, "member-a.pt", 100, mtime=1_000)
    _cached_member(tmp_path, "member-b.pt", 100, mtime=2_000)

    assert pool.evict_lru_to_cap(tmp_path, "member-*.pt", cap_bytes=250) == 0
    assert pool.evict_lru_to_cap(
        tmp_path, "member-*.pt", cap_bytes=250, incoming_bytes=100) == 1


def test_under_cap_evicts_nothing(tmp_path):
    kept = _cached_member(tmp_path, "member-a.pt", 100, mtime=1_000)
    assert pool.evict_lru_to_cap(tmp_path, "member-*.pt", cap_bytes=10_000) == 0
    assert kept.exists()


def test_only_the_pattern_is_evicted(tmp_path):
    """The directory also holds transient soup-*/head-* scratch that the merge
    unlinks itself; eviction must not reach into it."""
    _cached_member(tmp_path, "member-a.pt", 100, mtime=1_000)
    scratch = _cached_member(tmp_path, "soup-deadbeef.pt", 5_000, mtime=500)

    pool.evict_lru_to_cap(tmp_path, "member-*.pt", cap_bytes=50)
    assert scratch.exists()


def test_cap_is_actually_enforced_not_just_approached(tmp_path):
    """Discrimination control for the three above: they assert a delete COUNT, which
    a policy that evicts one file and stops would also satisfy. This asserts the
    post-condition the cap exists for — total on disk is at or under it — against a
    directory that needs several evictions, and fails for a policy that under-evicts."""
    for i in range(10):
        _cached_member(tmp_path, f"member-{i}.pt", 100, mtime=1_000 + i)

    pool.evict_lru_to_cap(tmp_path, "member-*.pt", cap_bytes=250)
    remaining = sorted(p.name for p in tmp_path.glob("member-*.pt"))
    assert sum(p.stat().st_size for p in tmp_path.glob("member-*.pt")) <= 250
    # and it kept the NEWEST, not an arbitrary two
    assert remaining == ["member-8.pt", "member-9.pt"]


# ── a deposit record must always name the app version that wrote it (#0116) ──
#
# 0.27.6 moved every ``l2_from_base`` value without touching ``schema_version``,
# so the record's shape cannot date its own contents. These pin the field that
# can. The null case is the load-bearing one: an ABSENT key is what a pre-0116
# writer produces, so writing the key with None is the only thing that tells
# "this writer did not know its version" apart from "this writer did not stamp".

def _metadata(app_version):
    return pool.build_contribution_metadata(
        contribution_id="c-1",
        model_type="cpsam",
        app_version=app_version,
        cellpose_version="4.0.1",
        architecture_signature="sig",
        base_checkpoint_id="stock-cpsam",
        base_checkpoint_sha256=None,
        samples_seen=10,
        training_params={},
        l2_from_base=None,
        dataset_fingerprint=None,
        solo_val_metric=None,
        weights_sha256="deadbeef",
        uploaded_at=0.0,
    )


def test_app_version_is_recorded():
    assert _metadata("0.27.14")["app_version"] == "0.27.14"


def test_app_version_key_is_present_even_when_unknown():
    """The key, not the value, is what a consumer partitions on."""
    assert "app_version" in _metadata(None)
    assert _metadata(None)["app_version"] is None


def test_app_version_does_not_collide_with_the_other_version_fields():
    """Discrimination control: three distinct version-ish fields live in this
    record and a fix that overwrote the wrong one would still pass the two
    tests above. cellpose_version names the library, schema_version the record
    shape, app_version the code that computed the values."""
    m = _metadata("0.27.14")
    assert m["cellpose_version"] == "4.0.1"
    assert m["schema_version"] == pool.SCHEMA_VERSION
    assert m["app_version"] != m["cellpose_version"] != m["schema_version"]


# ---------------------------------------------------------------------------
# #0139: a contribution with no verified digest must not enter a soup.
# ---------------------------------------------------------------------------

def _batch_of_three():
    """A pool batch as `aggregate` sees it: one pre-digest record, one that banked
    the key as null, one good. All three are otherwise complete and mergeable, so
    anything that excludes more or fewer than two is reacting to the digest."""
    return [
        {"contribution_id": "legacy-0", "weights_path": "c/legacy-0/weights.pt",
         "n_bytes": 1_300_000_000, "samples_seen": 4000},
        {"contribution_id": "null-0", "weights_path": "c/null-0/weights.pt",
         "weights_sha256": None, "n_bytes": 1_300_000_000, "samples_seen": 2000},
        {"contribution_id": "good-0", "weights_path": "c/good-0/weights.pt",
         "weights_sha256": "c" * 64, "n_bytes": 1_300_000_000, "samples_seen": 1000},
    ]


def test_an_absent_and_a_null_digest_are_both_unverifiable_and_a_good_one_is_not():
    assert pool.unverifiable_contributions(_batch_of_three()) == ["legacy-0", "null-0"]


def test_the_verifiable_contribution_survives_the_filter():
    """The other half of the clause: excluding the two is only correct if the third
    still merges. A guard that returned every id would pass the test above."""
    batch = _batch_of_three()
    excluded = set(pool.unverifiable_contributions(batch))
    kept = [c["contribution_id"] for c in batch if c["contribution_id"] not in excluded]
    assert kept == ["good-0"]


def test_a_digest_is_the_only_field_this_keys_on():
    """Discrimination control. `soup_admission` deliberately fails OPEN on a missing
    `n_bytes`, so a guard that keyed on 'any absent metadata' would silently reverse
    that decision for the memory gate as well. A row with a digest and nothing else
    must pass."""
    batch = [{"contribution_id": "sparse-0", "weights_path": "c/sparse-0/weights.pt",
              "weights_sha256": "d" * 64}]
    assert pool.unverifiable_contributions(batch) == []


def test_an_empty_batch_excludes_nothing():
    """Discrimination control: the filter is reached with an empty batch on a pool
    whose contributions are all already decided, and must not invent an exclusion."""
    assert pool.unverifiable_contributions([]) == []


def test_matching_bytes_are_accepted():
    content = b"the tensors as attested"
    assert pool.member_digest_refusal(
        "c/good-0/weights.pt", hashlib.sha256(content).hexdigest(), content) is None


def test_bytes_that_do_not_match_the_recorded_digest_are_refused():
    content = b"not the tensors that were attested"
    expected = hashlib.sha256(b"something else entirely").hexdigest()
    refusal = pool.member_digest_refusal("c/good-0/weights.pt", expected, content)
    assert refusal is not None
    # An operator has to be able to act on this without re-deriving it: which member,
    # what was promised, what arrived.
    assert "c/good-0/weights.pt" in refusal
    assert expected in refusal
    assert hashlib.sha256(content).hexdigest() in refusal


def test_no_recorded_digest_means_no_check_at_the_replica():
    """Discrimination control, and it pins a deliberate split of responsibility: the
    replica sees one member and cannot know a pool-wide rule, so the absent-digest
    refusal belongs to admission (tested above) and NOT here. A `member_digest_refusal`
    that refused on None would make the two guards fight — admission would exclude the
    contribution and this would also raise for any caller that skipped admission."""
    assert pool.member_digest_refusal("c/legacy-0/weights.pt", None, b"anything") is None
    assert pool.member_digest_refusal("c/legacy-0/weights.pt", "", b"anything") is None
