"""Contract tests for ``pool.py``'s campaign wire projection.

Plain pytest, no torch, no numpy, no live Hypha — ``pool``'s heavy imports are all
function-local, so the schema surface imports on stdlib alone. These pin the
0.15.0-draft wire contract the campaign page reads (``pool.pool_wire_state``): the
role-tagged RoundMetric shape, the soups/empty-merges partition, and — the guard
that matters — that a SELECTION value can never surface on the witness curve. The
directional test fails on a misplaced VALUE, not merely a mislabelled tag.

The field-sets below were first derived from the page's verified 0.13.0-draft fixture
corpus (``cellpose-sam-community.json``) and re-verified unchanged against the
regenerated 0.15.0-draft corpus (bioimage.io 6a7fac8, 2026-11-13). The corpus is not
vendored (it is the page's artifact and rots on draft bumps);
``test_matches_fixture_corpus`` cross-checks these constants against it when a path is
supplied via ``MODEL_FINETUNE_CORPUS_DIR``, so a draft bump that moves a field is
caught the next time the test is run against the fresh corpus.

⚠ The corpus check covers the merge/transport field-sets and NOT the contribution
stub, so it could not have caught the ``bytes_out`` → ``n_bytes`` rename that defines
rung 0.15 — which is why ``CONTRIBUTION_KEYS`` is asserted here directly. It is also
not merely an omission waiting to be filled: measured against the 0.15 corpus, the
page's contribution record and this backend's stub do not carry the same fields (the
page adds ``base_version``, ``contributor_id``, ``dataset_name`` and ``declared``;
this backend publishes ``samples_seen``, which the page's does not). Asserting set
equality here would be red on a question about what the backend ought to publish, not
about the wire contract this file pins.
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

import pool


# Field-sets pinned from the fixture corpus; unchanged between 0.13.0-draft and
# 0.15.0-draft, re-verified against the latter.
#
# WIRE_ROUNDMETRIC_KEYS is the page's own record at rung 0.15.0-draft — the eleven
# fields the corpus carries. ROUNDMETRIC_KEYS is what THIS backend writes, which is
# those eleven plus the per-image breakdown. The two are separate constants because
# they answer different questions, and collapsing them is what makes an additive
# backend field look like a schema break: the corpus genuinely does not carry
# per_image, and asserting equality against it would go RED for a record that is
# correct. The relation that holds, and that the corpus test asserts, is that the
# backend writes every key the page's record has and the only extras are these.
PER_IMAGE_KEYS = {"per_image", "per_image_basis"}
WIRE_ROUNDMETRIC_KEYS = {
    "role", "name", "aggregate", "aggregate_basis", "aggregate_scope",
    "higher_is_better", "n_sites_scored", "n_datasets_scored", "per_site",
    "per_site_basis", "aggregate_withheld",
}
ROUNDMETRIC_KEYS = WIRE_ROUNDMETRIC_KEYS | PER_IMAGE_KEYS
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
CONTRIBUTION_KEYS = {
    "contribution_id", "received_at", "weights_sha256", "n_bytes",
    "n_train_images", "samples_seen", "disposition", "merged_into",
}

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
            {"contribution_id": "alpha-0", "weights_sha256": "a1" * 32, "uploaded_at": 1_750_050_000.0,
             "n_bytes": 1_300_000_000, "n_train_images": 400, "samples_seen": 4000},
            {"contribution_id": "beta-0", "weights_sha256": "b2" * 32, "uploaded_at": 1_750_060_000.0,
             "n_bytes": 1_300_000_000, "n_train_images": 200, "samples_seen": 2000},
            {"contribution_id": "gamma-0", "weights_sha256": "c3" * 32, "uploaded_at": 1_750_150_000.0,
             "n_bytes": 1_300_000_000, "n_train_images": 100, "samples_seen": 1000},
            {"contribution_id": "delta-0", "weights_sha256": "d4" * 32, "uploaded_at": 1_750_250_000.0,
             "n_bytes": 1_300_000_000, "n_train_images": 300, "samples_seen": 3000},
            {"contribution_id": "epsilon-0", "weights_sha256": "e5" * 32, "uploaded_at": 1_750_350_000.0,
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


class TestContributionBytesAreGreppable:
    """soup_admission fails OPEN when a pool never recorded ``n_bytes``, so the
    operator's question is literally "does the view show n_bytes?". Publishing it
    as ``bytes_out`` answered that question with the wrong word AND collided with
    transport.bytes_out, which carries unrelated bytes one level up."""

    def test_contribution_publishes_the_stored_name(self, wire):
        for c in wire["contributions"]:
            assert c["n_bytes"] == 1_300_000_000

    def test_contribution_key_set(self, wire):
        for c in wire["contributions"]:
            assert set(c) == CONTRIBUTION_KEYS

    def test_bytes_out_means_transport_only(self, wire):
        # Paired with the presence assertions above on purpose: "bytes_out is
        # absent" alone is also satisfied by the stub disappearing. And transport
        # must still HAVE it, so renaming the wrong level fails here.
        for c in wire["contributions"]:
            assert "bytes_out" not in c
        for soup in wire["soups"]:
            assert "bytes_out" in soup["transport"]


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
    reason=(
        "set MODEL_FINETUNE_CORPUS_DIR to a 0.15.0-draft fixture corpus to "
        "cross-check field-sets. The corpus lives in the OTHER repo, which is "
        "why this is opt-in rather than vendored: bioimage.io, branch "
        "feat/federation-campaigns-page, src/services/__fixtures__/corpus. It "
        "must be a corpus at THIS backend's rung — take it from the commit "
        "that defined the rung (6a7fac8, 2026-11-13), and re-read the rung "
        "from PAGE_RUNGS rather than assuming the branch tip still sits on "
        "it. The rung itself is checked "
        "unconditionally by test_schema_version_is_the_rung_this_backend_"
        "targets, so this skip no longer hides a wrong SCHEMA_VERSION (#0419)"
    ),
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
    # Guard against being handed a corpus from a different rung: the assertion
    # above would then fail for a reason that looks like a backend bug.
    assert _major_minor(doc["schema_version"]) == BACKEND_TARGETS_RUNG, (
        f"the corpus at MODEL_FINETUNE_CORPUS_DIR declares "
        f"{doc['schema_version']!r}, but this backend targets rung "
        f"{BACKEND_TARGETS_RUNG!r} — wrong corpus, not a backend defect"
    )
    assert set(prog["merge_trigger"]) == MERGE_TRIGGER_KEYS
    soup = prog["soups"][1]
    assert set(soup) == SOUP_KEYS
    assert set(soup["community_model"]) == COMMUNITY_MODEL_KEYS
    assert set(soup["transport"]) == TRANSPORT_KEYS
    assert set(prog["empty_merges"][0]) == EMPTY_MERGE_KEYS
    # The metric records are checked by RELATION, not equality: the corpus is the
    # page's record and does not carry the per-image breakdown this backend adds.
    # Equality here would report an additive backend field as a corpus mismatch —
    # and would equally hide a field this backend FAILED to write, since both
    # directions collapse into one failing assertion.
    for where, metric in (("baseline_metric", prog["baseline_metric"]),
                          ("selection_metric", soup["selection_metric"]),
                          ("witness_metric", soup["witness_metric"])):
        keys = set(metric)
        assert keys - ROUNDMETRIC_KEYS == set(), (
            f"the corpus's {where} carries {sorted(keys - ROUNDMETRIC_KEYS)}, which "
            f"this backend never writes — the page's record has moved")
        assert ROUNDMETRIC_KEYS - keys <= PER_IMAGE_KEYS, (
            f"this backend writes {sorted(ROUNDMETRIC_KEYS - keys - PER_IMAGE_KEYS)} "
            f"on {where} and the corpus does not — a backend-only field that is not "
            f"the per-image breakdown is a schema break, not an addition")


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
    # ...and that the constant is itself the rung this backend targets. Without
    # this line the assertion above is satisfied by both sides being wrong
    # together, which is the whole of #0419. This line carries no weight of its
    # own unless BACKEND_TARGETS_RUNG is independent of pool.SCHEMA_VERSION —
    # see test_the_targeted_rung_is_written_down_and_not_derived (#0442), which
    # is what keeps it so. It is the quieter of the two sites a derivation
    # silences, and the one nobody had reported.
    assert _major_minor(m["schema_version"]) == BACKEND_TARGETS_RUNG
    assert m["app_version"] != m["cellpose_version"] != m["schema_version"]


# ---------------------------------------------------------------------------
# SCHEMA_VERSION must be a rung, and the right one
# ---------------------------------------------------------------------------
#
# The page compares schema versions on MAJOR.MINOR ONLY:
#
#     const major = (v: string) => v.split('.').slice(0, 2).join('.');
#     if (major(received) !== major(CAMPAIGN_SCHEMA_VERSION)) {
#       throw new CampaignSchemaMismatchError(received);
#     }
#
# (bioimage.io, src/services/campaignService.ts, assertSchema. The local name
# says "major"; the slice takes TWO components.) So the patch number and the
# "-draft" suffix are invisible to the page, and a backend that declares a
# major.minor some earlier shape already used does not raise a mismatch — it
# reports AGREEMENT, falsely. The failure is silent, which is why it needs a
# test rather than care.
#
# Every rung the page has committed, measured 2026-11-13 on bioimage.io branch
# feat/federation-campaigns-page. The date and sha are the page commit that
# DEFINED the rung — the FIRST one to carry that major.minor, not the last.
#
# Re-measure against the file that DEFINES the constant, which is NOT the file
# quoted above: campaignService.ts only imports it (2 commits ever), while
# src/types/campaign.ts is where it is declared (26 commits).
#   for c in $(git log --format=%h <branch> -- src/types/campaign.ts); do
#     git show "${c}:src/types/campaign.ts" \
#       | grep -m1 -E '^ *export +const +CAMPAIGN_SCHEMA_VERSION'; done
# Read it oldest-first and keep the first sha per major.minor. Control: every
# one of those commits must yield a DECLARATION, and 26 of 26 do.
#
# Both halves of that pipeline are written the way they are because the obvious
# way fails toward a FALSE "the pattern has drifted", and a control that merely
# tests for non-empty output passes in both cases. Measured 2026-11-13:
#   - unanchored `grep -m1 CAMPAIGN_SCHEMA_VERSION` matches line 35 — a prose
#     comment, "Bump CAMPAIGN_SCHEMA_VERSION on any breaking change…" — on all
#     26 commits, never the declaration. It yields 26 identical non-version
#     lines while the non-empty control reads 26/26 success.
#   - unbraced `$c:src/types/campaign.ts` makes zsh read `:s` as a parameter
#     modifier (substitute), which needs a pattern it does not find, so the
#     word never expands: zsh 5.8 says `bad substitution`, rc=1, and `git show`
#     is never handed a ref at all. Every iteration yields nothing, so the
#     sweep reports 26/26 drifted. bash expands it fine, which is why the
#     recipe can look correct in one shell and fail in the other.
# Anchor the pattern and brace the ref, and assert the output parses as a
# version string rather than merely being non-empty.
# (0.8 and 0.9 were never used.)
PAGE_RUNGS = {
    "0.1": ("0.1.0-draft", "2026-09-06", "86f381e"),
    "0.2": ("0.2.0-draft", "2026-09-06", "1f37721"),
    "0.3": ("0.3.0-draft", "2026-09-07", "8485ce6"),
    "0.4": ("0.4.0-draft", "2026-09-07", "10307e8"),
    "0.5": ("0.5.0-draft", "2026-09-07", "fa7be54"),
    "0.6": ("0.6.0-draft", "2026-09-07", "83692b4"),
    "0.7": ("0.7.0-draft", "2026-09-07", "ffa8229"),
    "0.10": ("0.10.0-draft", "2026-09-12", "3ea7449"),
    "0.11": ("0.11.0-draft", "2026-09-12", "31adbc1"),
    "0.12": ("0.12.0-draft", "2026-09-12", "4bf7b65"),
    "0.13": ("0.13.0-draft", "2026-09-12", "a39d6a7"),
    "0.14": ("0.14.0-draft", "2026-09-24", "9d5ef26"),
    "0.15": ("0.15.0-draft", "2026-11-13", "6a7fac8"),
}

# The rung whose SHAPE this backend implements. This is NOT derived from
# pool.SCHEMA_VERSION — that is the point. Deriving it would recreate the
# self-referential assertion #0419 is about, where the fixture agrees with
# whatever the constant happens to hold.
#
# Bumping the wire contract is therefore a TWO-SIDED edit, deliberately: move
# pool.SCHEMA_VERSION and move this, in the same commit, after checking the
# page has committed the rung you are moving to. A one-sided edit goes red, and
# the message tells you which side you forgot.
BACKEND_TARGETS_RUNG = "0.15"


def _major_minor(value: str) -> str:
    """Exactly what the page does: split on '.', keep the first two parts."""
    return ".".join(value.split(".")[:2])


def _rung_violation(schema_version: str) -> str | None:
    """Why ``schema_version`` is wrong for this backend, or None if it is right.

    Split out from the test so the CHECK can be exercised against known-bad
    values without mutating the constant — a test that only ever sees the real
    value cannot show it is able to reject anything.
    """
    got = _major_minor(schema_version)
    if got == BACKEND_TARGETS_RUNG:
        return None
    if got in PAGE_RUNGS:
        full, date, sha = PAGE_RUNGS[got]
        return (
            f"SCHEMA_VERSION is {schema_version!r}, whose major.minor is {got!r} — "
            f"but {got!r} is a rung the campaign page already committed on {date} "
            f"({sha}, {full}) to mean a DIFFERENT record shape. This backend "
            f"implements rung {BACKEND_TARGETS_RUNG!r}. The page compares "
            f"major.minor only, so it will report agreement with the {got!r} shape "
            f"instead of raising a mismatch, and mis-render rather than refuse. "
            f"Either restore {BACKEND_TARGETS_RUNG!r}, or — if this backend really "
            f"did move to {got!r} — update BACKEND_TARGETS_RUNG in the same commit."
        )
    return (
        f"SCHEMA_VERSION is {schema_version!r}, whose major.minor is {got!r}, but "
        f"this backend targets rung {BACKEND_TARGETS_RUNG!r} and {got!r} is not a "
        f"rung the page has ever committed. If this is a deliberate two-sided "
        f"bump, add {got!r} to PAGE_RUNGS with the page commit that defines it "
        f"and set BACKEND_TARGETS_RUNG to it, in the same commit."
    )


def test_schema_version_is_the_rung_this_backend_targets():
    """The assertion #0419 found missing. Compares the constant against an
    independently-recorded rung, not against itself."""
    assert _rung_violation(pool.SCHEMA_VERSION) is None, _rung_violation(
        pool.SCHEMA_VERSION
    )


def test_the_targeted_rung_is_one_the_page_actually_defined():
    assert BACKEND_TARGETS_RUNG in PAGE_RUNGS, (
        f"BACKEND_TARGETS_RUNG is {BACKEND_TARGETS_RUNG!r}, which the page has "
        f"never committed. Known rungs: {sorted(PAGE_RUNGS)}"
    )


def _a_rung_the_page_has_never_defined() -> str:
    """The lowest major.minor the page has not committed — DERIVED, never named.

    A literal here would be true only until the page committed that rung, and
    would then go red for a reason that has nothing to do with the backend —
    which is the "next person to bump deletes the test" failure this whole file
    exists to avoid. An earlier draft of this test did hardcode one, and the
    mutation harness's legitimate-bump arm is what caught it.
    """
    for minor in range(1, 1000):
        candidate = f"0.{minor}"
        if candidate not in PAGE_RUNGS and candidate != BACKEND_TARGETS_RUNG:
            return candidate
    raise AssertionError("no undefined rung below 0.1000 — widen the search")


def test_the_check_rejects_a_superseded_page_rung():
    """#0114's actual defect, pinned so it cannot recur silently. 0.14.0-draft
    is a real rung the page committed for the dropped-synchronous-arm shape."""
    # The pin's premise, stated so a future bump to 0.14 fails with a reason
    # rather than cryptically. A rung already committed stays committed, so
    # this is the only way the historical fact below can stop applying.
    assert BACKEND_TARGETS_RUNG != "0.14", (
        "this test pins #0114's defect, whose premise is that the backend does "
        "NOT target 0.14; if it now deliberately does, rewrite the pin against "
        "whichever rung was superseded — do not delete it"
    )
    why = _rung_violation("0.14.0-draft")
    assert why is not None, "0.14.0-draft was accepted — the check cannot go red"
    assert "0.14" in why and "2026-09-24" in why and "9d5ef26" in why, (
        f"the rejection does not NAME the collision, so a red run would not say "
        f"what is wrong: {why!r}"
    )


def test_the_check_rejects_a_rung_the_page_never_defined():
    """A backend that invents a rung is out of lock-step even though it collides
    with nothing — the page raises a mismatch on it."""
    rung = _a_rung_the_page_has_never_defined()
    why = _rung_violation(f"{rung}.0-draft")
    assert why is not None, f"{rung}.0-draft was accepted — the check cannot go red"
    assert "not a rung the page has ever committed" in why, (
        f"the rejection does not say WHY {rung} is wrong: {why!r}"
    )
    # ...and it must say so for a different reason than a collision does, or the
    # two failure modes are indistinguishable in a red run.
    assert "2026-" not in why


def test_the_check_ignores_the_patch_and_the_draft_suffix():
    """Discrimination control: the check must match the page's comparison, not
    be stricter than it. Pinning the exact literal would make every patch bump
    red, and the next person to hit that deletes the test."""
    rung = PAGE_RUNGS[BACKEND_TARGETS_RUNG][0]
    assert _rung_violation(rung) is None
    assert _rung_violation(f"{BACKEND_TARGETS_RUNG}.7") is None
    assert _rung_violation(f"{BACKEND_TARGETS_RUNG}.0-rc1") is None


def test_the_check_is_not_self_referential(monkeypatch):
    """The regression guard for #0419 itself. If someone later rewrites the
    rung check to read pool.SCHEMA_VERSION on both sides, every test above
    still passes and this one fails."""
    monkeypatch.setattr(pool, "SCHEMA_VERSION", "0.14.0-draft")
    assert _rung_violation(pool.SCHEMA_VERSION) is not None, (
        "with SCHEMA_VERSION monkeypatched to a known-wrong rung the check still "
        "passed, so it is comparing the constant against itself"
    )


# ---------------------------------------------------------------------------
# ...and the target must be WRITTEN DOWN, not computed (#0442)
# ---------------------------------------------------------------------------
#
# The test above monkeypatches pool.SCHEMA_VERSION AFTER import, so it can only
# see a self-reference that happens at call time. Derive BACKEND_TARGETS_RUNG
# from pool.SCHEMA_VERSION at module level instead — a one-line change that
# reads like a tidy-up — and the derived value FREEZES at import, before any
# monkeypatch runs. The comparison is the constant against itself again, one
# level up from where #0419 removed it, and the test above can only redden when
# the frozen value already IS the rung it patches to.
#
# Measured 2026-11-13, sweeping the derivation over all thirteen rungs PAGE_RUNGS
# records (harness .dev/verify/0442/sweep-rung-derivation.sh): exactly ONE goes
# red (0.14, and only because test_the_check_rejects_a_superseded_page_rung
# hardcodes that rung as its premise). Twelve stay fully GREEN, of which eleven
# are wrong values a revert could land on. That 1-of-13 is an artefact of one
# hardcoded literal, not a property of the check.
#
# So the assertion below is deliberately about the target's INDEPENDENCE and not
# about its value. "The target equals this specific table" is the shape that just
# failed — it pins a literal, goes red on the next legitimate rung bump, and gets
# deleted by whoever hits it. "The target is not derived from the constant it
# guards" survives a bump, because a bump rewrites the literal and leaves it a
# literal. ast.literal_eval asks exactly that question: a literal is written
# down, and anything it refuses is computed.
#
# One assertion covers every consumer, since they all read the same constant:
# _rung_violation's `got == BACKEND_TARGETS_RUNG`, the line in
# test_app_version_does_not_collide_with_the_other_version_fields, and the two
# page-table tests.

_THIS_FILE = pathlib.Path(__file__).resolve()


def _module_level_value_node(name: str) -> ast.expr:
    """The right-hand side of this file's own module-level ``name = ...``.

    Reading the source rather than the imported object is the point: by the time
    a test runs, a derived value and a written-down one are the same string.
    """
    tree = ast.parse(_THIS_FILE.read_text())
    found = [
        stmt.value
        for stmt in tree.body
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None
        for target in (
            stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        )
        if isinstance(target, ast.Name) and target.id == name
    ]
    # Exactly one, because grading the first of two would silently grade the
    # wrong line, and zero means the assignment moved inside an `if` or a
    # function where this check cannot see it.
    assert len(found) == 1, (
        f"expected exactly one module-level assignment to {name} in "
        f"{_THIS_FILE.name}, found {len(found)} — this guard reads the source, "
        f"so {name} has to be assigned once, at module level, unconditionally"
    )
    return found[0]


def _written_down(node: ast.expr) -> tuple[bool, object]:
    """``(True, value)`` if ``node`` writes its value down, ``(False, None)`` if
    it computes it.

    Split out from the tests so the predicate can be exercised against a
    known-derived expression — see
    test_this_guard_can_actually_refuse_a_derivation.
    """
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return False, None


def _not_written_down(name: str, live_value: object) -> str | None:
    """Why ``name`` is not an independent literal in this file, or None if it is."""
    node = _module_level_value_node(name)
    is_literal, literal = _written_down(node)
    if not is_literal:
        return (
            f"{name} is COMPUTED, not written down: {_THIS_FILE.name} assigns it "
            f"`{ast.unparse(node)}`. A computed target cannot discriminate — if it "
            f"is derived from pool.SCHEMA_VERSION, directly or through a helper, "
            f"the rung check compares the constant against itself and goes quiet "
            f"for every value except the one or two that some other test happens "
            f"to hardcode (measured: 1 of the 13 rungs the page has committed). "
            f"Write the rung down as a literal and bump it in the same commit as "
            f"pool.SCHEMA_VERSION — the two-sided edit is the mechanism, not an "
            f"inconvenience."
        )
    if literal != live_value:
        return (
            f"{name} is a literal in the source but the imported value differs "
            f"(source {literal!r}, imported {live_value!r}), so something rebinds "
            f"it after assignment and this guard is reading a line that no longer "
            f"decides anything"
        )
    return None


def test_the_targeted_rung_is_written_down_and_not_derived():
    """#0442. The new check is "the target is not derived from the constant it
    guards" — NOT "the target equals this specific table". Only the first
    survives a legitimate rung bump; the second is the shape that just failed."""
    why = _not_written_down("BACKEND_TARGETS_RUNG", BACKEND_TARGETS_RUNG)
    assert why is None, why


def test_the_rung_table_is_written_down_and_not_derived():
    """The same hole one level further out. If PAGE_RUNGS were computed from
    pool.SCHEMA_VERSION then test_the_targeted_rung_is_one_the_page_actually
    _defined becomes self-referential too, and _a_rung_the_page_has_never_defined
    starts handing out rungs the page has in fact committed."""
    why = _not_written_down("PAGE_RUNGS", PAGE_RUNGS)
    assert why is None, why


def test_this_guard_can_actually_refuse_a_derivation():
    """Discrimination control, and the reason the two tests above are evidence.

    They assert an absence, so on their own they are equally consistent with
    "nothing is derived" and "the instrument cannot see a derivation". This runs
    the same predicate over the exact one-line derivation #0442 is about and
    requires it to come back refused, naming the expression it refused.
    """
    derivation = 'BACKEND_TARGETS_RUNG = ".".join(pool.SCHEMA_VERSION.split(".")[:2])'
    is_literal, _ = _written_down(ast.parse(derivation).body[0].value)
    assert not is_literal, (
        f"the predicate accepted {derivation!r} as written down, so it cannot "
        f"refuse the derivation #0442 is about and the two tests above are green "
        f"for no reason"
    )
    # ...and it must still ACCEPT a plain literal, or it refuses everything and
    # is equally useless — an always-red guard gets deleted as fast as a
    # never-red one.
    is_literal, value = _written_down(
        ast.parse('BACKEND_TARGETS_RUNG = "0.15"').body[0].value
    )
    assert is_literal and value == "0.15", (
        "the predicate refused a plain string literal, so it would refuse every "
        "legitimate rung too"
    )


# ---------------------------------------------------------------------------
# #0141: the wire view must publish the digest the index already holds.
# ---------------------------------------------------------------------------

def test_every_contribution_publishes_its_stored_weights_digest(wire):
    """The fixture gives all five contributions DISTINCT digests, so this cannot
    pass against a hardcoded value or a constant leaked from a sibling row — which
    is the discrimination that matters here, the projection having previously
    dropped the field rather than mangled it."""
    stored = {c["contribution_id"]: c["weights_sha256"]
              for c in _synthetic_index()["contributions"]}
    published = {c["contribution_id"]: c.get("weights_sha256")
                 for c in wire["contributions"]}
    assert published == stored
    assert len(set(published.values())) == len(published)


def test_a_contribution_without_a_stored_digest_publishes_null_rather_than_raising():
    """Control on the choice of `.get()` over a subscript here. The merge path
    subscripts, because souping unchecked bytes is the harm it guards; this is a
    page projection, where one absent field must not take the whole view down."""
    idx = _synthetic_index()
    del idx["contributions"][0]["weights_sha256"]
    wire = pool.pool_wire_state(idx, pool_artifact_id=POOL_ID)
    assert wire["contributions"][0]["weights_sha256"] is None
    assert wire["contributions"][1]["weights_sha256"] == "b2" * 32
