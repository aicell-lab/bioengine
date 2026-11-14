"""The per-image breakdown behind the gate's scalar — #0121.

The greedy gate admits a contribution when the soup's mean instance F1 on split A
does not drop. That decision is recorded as one number, and one number cannot be
audited against any admission rule: a merge that gained 0.02 by lifting every image
and a merge that gained 0.02 by trading a collapse on two images against the rest
leave identical records. The vector behind the mean already existed inside
``mean_instance_f1`` and was discarded; this banks it.

Two halves, and the second is not optional. Serialising the vector onto the metric
record is the visible half. The cache half is forced by ``score_key``, which hashes
the split id and the sorted member paths and NOTHING about what the caller asked
for: a journal holding only scalars serves a hit for a pass whose vector it does not
have, so the breakdown would be missing on exactly the merges that were interrupted
— and that absence is indistinguishable from never having measured it.

Scope, stated plainly: these drive the real ``pool`` helpers and read ``entry.py`` at
source level for the wiring. They do NOT execute ``aggregate()`` (it needs
``bioengine``, ``hypha_rpc``, ``pydantic``, ``yaml`` and ``httpx``) and they do not
run cellpose. The end-to-end proof is a live merge and is not claimed here.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import pool

np = pytest.importorskip("numpy")

GATE = {"artifact_id": "bioimage-io/splits", "sha256": "a" * 64}
WITNESS = {"artifact_id": "bioimage-io/splits", "sha256": "b" * 64}
DIAM = 30.0


def _map(*instances) -> "np.ndarray":
    """A 1-D label map with one run per instance. ``_map((1, 4), (2, 3))`` is label 1
    over four pixels then label 2 over three."""
    out = []
    for label, n in instances:
        out.extend([label] * n)
    return np.asarray(out, dtype=np.int32).reshape(1, -1)


PERFECT = _map((1, 4), (2, 4))
HALF = _map((1, 4), (0, 4))      # finds one of the two instances
MISS = _map((0, 8))              # finds nothing

# A second geometry with room for a false positive, so an image can score exactly
# 0.5: one instance matched, one predicted onto background, one ground truth missed.
GT3 = _map((1, 4), (2, 4), (0, 4))
PERFECT3 = GT3
SWAPPED3 = _map((1, 4), (0, 4), (2, 4))   # 1 tp, 1 fp, 1 fn -> f1 0.5
MISS3 = _map((0, 12))


# ── the vector exists, and the scalar is its mean ────────────────────────────

def test_the_vector_has_one_score_per_image():
    v = pool.per_image_instance_f1([PERFECT, HALF, MISS], [PERFECT, PERFECT, PERFECT])
    assert len(v) == 3


def test_the_scalar_is_exactly_the_mean_of_the_published_vector():
    """The property the record's guard rests on. ``mean_instance_f1`` is DEFINED as
    the mean of ``per_image_instance_f1``, so this is equality and not closeness —
    an implementation that re-derived the scalar separately would have to be compared
    with a tolerance, and would then be free to drift inside it."""
    preds, gts = [PERFECT, HALF, MISS], [PERFECT, PERFECT, PERFECT]
    v = pool.per_image_instance_f1(preds, gts)
    assert pool.mean_instance_f1(preds, gts) == pool.mean_of_per_image(v)


def test_the_vector_discriminates_splits_the_scalar_cannot():
    """Why the row exists, as a measurement rather than an argument: two splits with
    the SAME mean, one uniform and one carrying a collapse. Without this the whole
    change could be satisfied by publishing a constant vector of the right length."""
    uniform = pool.per_image_instance_f1([SWAPPED3, SWAPPED3], [GT3, GT3])
    mixed = pool.per_image_instance_f1([PERFECT3, MISS3], [GT3, GT3])
    assert uniform == [0.5, 0.5] and mixed == [1.0, 0.0]
    assert pool.mean_of_per_image(uniform) == pool.mean_of_per_image(mixed)
    assert uniform != mixed


def test_an_empty_split_raises_rather_than_inventing_a_score():
    with pytest.raises(ValueError):
        pool.per_image_instance_f1([], [])
    with pytest.raises(ValueError):
        pool.mean_of_per_image([])


def test_mismatched_lengths_raise_instead_of_scoring_a_prefix():
    """``zip`` stops at the shorter argument. Before the vector was published that
    produced a mean over a silently truncated prefix; now the vector's LENGTH is read
    as the size of the split, so the truncation would misstate what was scored."""
    with pytest.raises(ValueError, match="prefix"):
        pool.per_image_instance_f1([PERFECT], [PERFECT, PERFECT])


def test_the_truncation_this_guards_was_really_reachable():
    """Control for the test above: if zip over these arguments did NOT truncate, the
    guard would be pinning a case that cannot occur and would pass for nothing."""
    assert len(list(zip([PERFECT], [PERFECT, PERFECT]))) == 1


# ── the record: the pairing, and the aggregate it must be the mean of ────────

EXPECTED_KEYS = {
    "role", "name", "aggregate", "aggregate_basis", "aggregate_scope",
    "higher_is_better", "n_sites_scored", "n_datasets_scored", "per_site",
    "per_site_basis", "aggregate_withheld", "per_image", "per_image_basis",
}


def test_the_record_carries_the_two_new_fields_and_nothing_else_moved():
    m = pool.round_metric(role="witness", aggregate=0.5, aggregate_basis="x",
                          per_image=[0.4, 0.6], per_image_basis=pool.PER_IMAGE_BASIS)
    assert set(m) == EXPECTED_KEYS
    assert m["per_image"] == [0.4, 0.6]
    assert m["per_image_basis"] == pool.PER_IMAGE_BASIS


def test_a_record_without_a_breakdown_still_carries_the_fields_as_null():
    """An absent key and a null value read identically through ``.get``, so the
    fields are written unconditionally: a consumer can tell "this app does not
    measure breakdowns" from "this record has none" only if the key is always there."""
    m = pool.round_metric(role="witness", aggregate=0.5, aggregate_basis="x")
    assert set(m) == EXPECTED_KEYS
    assert m["per_image"] is None and m["per_image_basis"] is None


@pytest.mark.parametrize("kwargs", [
    {"per_image": [0.5]},
    {"per_image_basis": pool.PER_IMAGE_BASIS},
])
def test_half_a_pair_is_refused(kwargs):
    """A vector with no basis does not say what its numbers are or how they are
    ordered; a basis with no vector describes nothing."""
    with pytest.raises(ValueError, match="together or not at all"):
        pool.round_metric(role="witness", aggregate=0.5, aggregate_basis="x", **kwargs)


def test_a_breakdown_from_a_different_pass_is_refused():
    """The guard that makes the breakdown worth publishing. A record whose vector
    belongs to another scoring pass would read as an audit trail while auditing
    nothing — worse than publishing no vector at all."""
    with pytest.raises(ValueError, match="different scoring pass"):
        pool.round_metric(role="selection", aggregate=0.90, aggregate_basis="x",
                          per_image=[0.1, 0.2], per_image_basis=pool.PER_IMAGE_BASIS)


def test_the_rounding_the_call_sites_do_is_inside_the_tolerance():
    """Positive control for the refusal above: entry.py publishes ``round(x, 4)``
    beside a vector that is not rounded to 4, so a guard demanding exact equality
    would refuse every honest record. 1/3 is chosen because it is not representable
    and rounds the hard way."""
    vector = [1 / 3, 1 / 3, 1 / 3]
    exact = pool.mean_of_per_image(vector)
    m = pool.round_metric(role="selection", aggregate=round(exact, 4),
                          aggregate_basis="x", per_image=vector,
                          per_image_basis=pool.PER_IMAGE_BASIS)
    assert m["aggregate"] != exact, "the fixture must actually exercise the slack"
    assert abs(m["aggregate"] - pool.mean_of_per_image(m["per_image"])) \
        <= pool.PER_IMAGE_AGGREGATE_TOLERANCE


def test_a_disagreement_just_outside_the_tolerance_is_refused():
    """Discrimination control for the tolerance: it must be a bound, not a licence.
    A record off by 10x the slack is a different pass and must still be refused."""
    with pytest.raises(ValueError, match="different scoring pass"):
        pool.round_metric(role="selection", aggregate=0.5 + 10 * pool.PER_IMAGE_AGGREGATE_TOLERANCE,
                          aggregate_basis="x", per_image=[0.5, 0.5],
                          per_image_basis=pool.PER_IMAGE_BASIS)


@pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")]])
def test_an_unusable_vector_is_refused(vector):
    with pytest.raises(ValueError):
        pool.round_metric(role="witness", aggregate=0.5, aggregate_basis="x",
                          per_image=vector, per_image_basis=pool.PER_IMAGE_BASIS)


def test_a_vector_with_no_aggregate_is_refused():
    """Nothing to check the vector against, so the guard above would be unreachable."""
    with pytest.raises(ValueError, match="mean of"):
        pool.round_metric(role="witness", aggregate=None, aggregate_basis="x",
                          per_image=[0.5], per_image_basis=pool.PER_IMAGE_BASIS)


def test_per_image_is_allowed_where_per_site_is_refused():
    """The reason this is a NEW field rather than ``per_site``. Every record this app
    writes is campaign_holdout-scoped, and a holdout metric publishing a per-site map
    is a self-contradiction the page refuses — it says the number came from a
    campaign-owned holdout and then breaks it down by participant. A per-IMAGE vector
    makes no claim about participants, so the same scope admits it."""
    with pytest.raises(ValueError, match="per_site"):
        pool.round_metric(role="witness", aggregate=0.5, aggregate_basis="x",
                          per_site={"site-1": 0.5})
    ok = pool.round_metric(role="witness", aggregate=0.5, aggregate_basis="x",
                           per_image=[0.4, 0.6], per_image_basis=pool.PER_IMAGE_BASIS)
    assert ok["aggregate_scope"] == "campaign_holdout"
    assert ok["per_site"] is None and ok["per_image"] == [0.4, 0.6]


def test_the_basis_says_the_aggregate_is_the_mean_of_these_numbers():
    """The basis is published provenance, so it may only state what the app checks —
    and it states exactly the invariant the record enforces."""
    assert "mean of exactly these numbers" in pool.PER_IMAGE_BASIS


# ── the cache half: the journal banks both, or serves neither ────────────────

def test_the_journal_banks_the_vector_beside_the_scalar():
    j = pool.new_merge_journal(GATE, WITNESS, DIAM)
    pool.record_score(j, "k", 0.5, per_image=[0.4, 0.6])
    assert j["scores"]["k"] == 0.5
    assert j["per_image_scores"]["k"] == [0.4, 0.6]


def test_the_journal_refuses_a_scalar_and_a_vector_from_different_passes():
    j = pool.new_merge_journal(GATE, WITNESS, DIAM)
    with pytest.raises(ValueError, match="not the same scoring pass"):
        pool.record_score(j, "k", 0.9, per_image=[0.1, 0.2])


def test_the_vector_is_required_not_defaulted():
    """``per_image`` is keyword-only and required. A caller that could omit it would
    bank a scalar whose breakdown is gone for good — the journal is precisely where
    re-measuring is not an option."""
    j = pool.new_merge_journal(GATE, WITNESS, DIAM)
    with pytest.raises(TypeError):
        pool.record_score(j, "k", 0.5)


def test_eviction_drops_both_halves_together():
    """A cap that evicted only the scalars would leave orphan vectors growing without
    bound; one that evicted only the vectors would leave half-entries that
    ``usable_merge_journal`` then prunes, silently costing re-scores."""
    j = pool.new_merge_journal(GATE, WITNESS, DIAM)
    for i in range(10):
        pool.record_score(j, f"k{i}", i / 10, per_image=[i / 10], cap=4)
    assert list(j["scores"]) == ["k6", "k7", "k8", "k9"]
    assert list(j["per_image_scores"]) == ["k6", "k7", "k8", "k9"]


def _scalar_only_journal():
    """A journal as an app that banked only scalars left it: the four stamps are
    current (the breakdown is additive, so none of them moves) and the vectors are
    absent."""
    j = pool.new_merge_journal(GATE, WITNESS, DIAM)
    pool.record_score(j, "paired", 0.5, per_image=[0.4, 0.6])
    pool.record_score(j, "orphan", 0.7, per_image=[0.7])
    del j["per_image_scores"]["orphan"]
    return j


def test_a_scalar_only_entry_is_pruned_rather_than_served_without_its_vector():
    kept = pool.usable_merge_journal(_scalar_only_journal(), GATE, WITNESS, DIAM)
    assert set(kept["scores"]) == {"paired"}
    assert set(kept["per_image_scores"]) == {"paired"}


def test_the_stamps_really_do_let_that_journal_through():
    """Control for the test above. If a scalar-only journal were already discarded
    wholesale by the stamp check, the pruning would be pinning an unreachable case —
    and the test would pass for a function that returned an empty journal always."""
    j = _scalar_only_journal()
    assert all(j[f] == pool.new_merge_journal(GATE, WITNESS, DIAM)[f]
               for f in ("schema_version", "gate_split", "witness_split", "diam_mean"))
    assert "orphan" in j["scores"], "the fixture must present the half-entry at all"
    assert len(pool.usable_merge_journal(j, GATE, WITNESS, DIAM)["scores"]) == 1


@pytest.mark.parametrize("vectors", [None, "not-a-dict", {}])
def test_a_journal_with_no_vectors_at_all_degrades_to_empty_not_to_bare_scalars(vectors):
    j = pool.new_merge_journal(GATE, WITNESS, DIAM)
    pool.record_score(j, "k", 0.5, per_image=[0.5])
    if vectors is None:
        del j["per_image_scores"]
    else:
        j["per_image_scores"] = vectors
    assert pool.usable_merge_journal(j, GATE, WITNESS, DIAM)["scores"] == {}


# ── the two halves together: a resumed merge still publishes a breakdown ─────

def _replay(members, journal, killed_at=None):
    """aggregate()'s scoring sequence against a journal, carrying both halves —
    the same replay ``test_merge_resume`` uses, extended to the vector. Returns
    (keys paid for, the head's pass on the witness split)."""
    gate, wit = pool.split_identity(GATE), pool.split_identity(WITNESS)
    paid = []

    def score(paths, split):
        key = pool.score_key(split, paths)
        if key in journal["scores"]:
            return {"f1": journal["scores"][key],
                    "per_image": list(journal["per_image_scores"][key])}
        if killed_at is not None and len(paid) == killed_at:
            raise RuntimeError("replica killed")
        vector = [round(0.4 + len(paths) / 100, 6), round(0.6 + len(paths) / 100, 6)]
        value = pool.mean_of_per_image(vector)
        paid.append(key)
        pool.record_score(journal, key, value, per_image=vector)
        return {"f1": value, "per_image": vector}

    score([], gate)
    score([], wit)
    soup = []
    score(soup, gate)
    for m in members:
        score(soup + [m], gate)
    for m in members:
        score(soup + [m], gate)
        soup = soup + [m]
    return paid, score(soup, wit)


def test_a_resumed_merge_publishes_the_same_breakdown_a_cold_one_does():
    """The failure the cache half exists to prevent: the vector missing on exactly
    the merges that were interrupted, where its absence cannot be told apart from
    never having been measured."""
    members = [f"m{i}.pt" for i in range(8)]

    journal = pool.new_merge_journal(GATE, WITNESS, DIAM)
    with pytest.raises(RuntimeError):
        _replay(members, journal, killed_at=5)
    resumed = pool.usable_merge_journal(journal, GATE, WITNESS, DIAM)
    paid_on_resume, head = _replay(members, resumed)

    paid_cold, head_cold = _replay(members, pool.new_merge_journal(GATE, WITNESS, DIAM))

    assert len(paid_on_resume) == len(paid_cold) - 5, "it must still resume where it died"
    assert head["per_image"] == head_cold["per_image"] != []
    record = pool.round_metric(role="witness", aggregate=round(head["f1"], 4),
                               aggregate_basis=pool.WITNESS_BASIS,
                               per_image=head["per_image"],
                               per_image_basis=pool.PER_IMAGE_BASIS)
    assert record["per_image"] == head_cold["per_image"]


def test_a_scalar_only_journal_would_have_published_no_breakdown():
    """Discrimination control: without the cache half the resumed head's vector is
    EMPTY, and ``round_metric`` then refuses the pair rather than publishing a
    half-record — so the failure is loud, but the breakdown is still gone."""
    members = [f"m{i}.pt" for i in range(8)]
    journal = pool.new_merge_journal(GATE, WITNESS, DIAM)
    with pytest.raises(RuntimeError):
        _replay(members, journal, killed_at=5)

    stripped = pool.usable_merge_journal(journal, GATE, WITNESS, DIAM)
    stripped["per_image_scores"] = {k: [] for k in stripped["scores"]}

    def score_without_vectors(paths, split):
        key = pool.score_key(split, paths)
        return {"f1": stripped["scores"][key], "per_image": stripped["per_image_scores"][key]}

    hit = score_without_vectors([], pool.split_identity(GATE))
    assert hit["per_image"] == []
    with pytest.raises(ValueError):
        pool.round_metric(role="selection", aggregate=round(hit["f1"], 4),
                          aggregate_basis=pool.SELECTION_BASIS,
                          per_image=hit["per_image"],
                          per_image_basis=pool.PER_IMAGE_BASIS)


# ── it reaches the page: the wire projects the record whole ──────────────────

def test_the_breakdown_survives_the_wire_projection():
    """``pool_wire_state`` passes metric records through without a key whitelist, so
    an added field really does reach the page rather than being projected away. If
    that ever changes, the field is dead weight and this says so."""
    vector = [0.4, 0.6]
    metric = pool.round_metric(role="witness", aggregate=0.5,
                               aggregate_basis=pool.WITNESS_BASIS,
                               per_image=vector, per_image_basis=pool.PER_IMAGE_BASIS)
    index = {
        "model_type": "cpsam",
        "architecture_signature": "sig",
        "contributions": [],
        "decided_contribution_ids": [],
        "merges": [],
        "baseline": {"model": "stock-cpsam", "witness_metric": metric,
                     "selection_metric": metric, "created_at": 1.0},
    }
    wire = pool.pool_wire_state(index, pool_artifact_id="bioimage-io/cellpose-finetuning-pool")
    assert wire["baseline_metric"]["per_image"] == vector
    assert wire["baseline_metric"]["per_image_basis"] == pool.PER_IMAGE_BASIS


def test_the_basis_correction_path_keeps_the_breakdown():
    """A record carrying retracted basis wording is rewritten on read. That path
    copies the dict, so it must carry the vector across — a correction that dropped
    it would delete breakdowns from exactly the oldest records."""
    metric = dict(pool.round_metric(role="selection", aggregate=0.5,
                                    aggregate_basis="x", per_image=[0.4, 0.6],
                                    per_image_basis=pool.PER_IMAGE_BASIS))
    metric["aggregate_basis"] = pool.RETRACTED_SELECTION_BASIS
    out = pool.corrected_metric(metric)
    assert out is not metric, "the fixture must actually reach the correcting branch"
    assert out["per_image"] == [0.4, 0.6]


# ── the wiring, pinned at source ─────────────────────────────────────────────

ENTRY = (pathlib.Path(__file__).parent / "entry.py").read_text()
RUNTIME = (pathlib.Path(__file__).parent / "runtime_cellpose.py").read_text()


def test_the_runtime_returns_the_vector_it_reduced():
    """``_score`` reads ``per_image`` off the runtime's reply and the two ship as one
    bundle, so a runtime that returned only the scalar would break every merge at its
    first scoring pass — loud, but only on a GPU replica, which is the slowest place
    there is to find it. ``mean_of_per_image`` rather than a second F1 pass is the
    load-bearing half: it makes the published scalar the reduction of the published
    vector by construction, not by two calls that happen to agree."""
    body = RUNTIME.split("    async def score_soup(")[1].split("\n    async def ")[0]
    assert "pool.per_image_instance_f1(" in body, "the runtime must produce the vector"
    assert '"per_image": per_image' in body, "and return it"
    assert "pool.mean_of_per_image(per_image)" in body, (
        "the scalar must be the reduction of that vector, not a separate measurement")
    assert "pool.mean_instance_f1(" not in body, (
        "a second scoring pass would make the two halves independent again")


def test_every_published_metric_goes_through_one_builder():
    """``_metric`` is where the vector and its basis are attached as a pair. A second
    call site building a record directly could publish the scalar alone, and no unit
    test of ``pool`` could see it — the record would simply be missing a field."""
    calls = re.findall(r"pool\.round_metric\(", ENTRY)
    assert len(calls) == 1, (
        f"expected exactly one pool.round_metric call (inside _metric, which attaches "
        f"the breakdown); found {len(calls)}")
    builder = ENTRY.split("def _metric(")[1].split("\n            cc = ")[0]
    assert "pool.PER_IMAGE_BASIS" in builder
    assert "per_image=" in builder


def test_the_scoring_pass_carries_both_halves_to_the_journal():
    """``_score`` is the only path to a score and the only path to the journal, so the
    vector must be banked there or not at all."""
    inside = ENTRY.split("async def _score(")[1].split("\n            def _as_pass(")[0]
    assert "per_image=per_image" in inside, "the vector must reach record_score"
    assert "journal[\"per_image_scores\"][key]" in inside, "a journal hit must serve it too"
