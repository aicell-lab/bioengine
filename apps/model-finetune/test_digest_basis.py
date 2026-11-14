"""#0417 — a stored head digest must say WHICH digest it is, or say it cannot.

``global_sha256`` (and its ``current_community.sha256`` and manifest
``weights_sha256`` copies — one value, three places) digested the file
``torch.save`` wrote before commit d3040c4f8165 (app 0.27.12) and digests the
tensors after it. Same name, same type, same 64-hex shape. A contributor who
re-merges an identical member set gets a tensor digest, compares it against a
stored file digest, and the only conclusion open to them is that the community
model is corrupt. It is not.

The property under test is NOT "every value is attributed". It is that a value is
attributed only on evidence, and that the no-evidence case says so out loud. The
present behaviour — returning nothing at all — is the silent default this removes,
and a fix that defaulted the other way (assume tensors, it is the current code)
would read exactly like success on every record written after the cutoff.

So three things are pinned here, and the second is the one with teeth:

  1. a pre-cutoff record resolves to the FILE meaning (the code for the other did
     not exist yet, so this direction is a proof, not a guess);
  2. a post-cutoff record with no stamp resolves to UNDETERMINED — deployment lags
     a commit, so the late side is evidence of nothing. ``test_timestamps_alone_
     never_yield_the_tensor_meaning`` sweeps the whole timeline and asserts no
     input reaches the tensor answer on a date alone;
  3. ``test_the_pinned_soup_contract_is_untouched`` is the over-reach control: the
     page's 0.13.0-draft soup field set, copied here independently of
     test_pool.py's copy, must be IDENTICAL after this change. A fix that widened
     it would be a silent backend/page contract break — exactly what
     SCHEMA_VERSION exists to make loud.

Plain pytest, no torch (``state_dict_digest`` only touches ``.shape``, ``.dtype``
and ``.detach().cpu().contiguous().numpy()``, so a numpy-backed stub reproduces it
byte for byte and the golden vector below pins the framing on a machine with no
GPU stack at all).
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

import pool


# ── fixed points ─────────────────────────────────────────────────────────────
# The cutoff, as a THIRD independent copy. pool.py has one and the measurement
# harness has another; a constant imported from the module under test cannot
# detect that module silently moving it.
# `git show -s --format=%ct d3040c4f8165`, not a hand-converted date.
CUTOFF = 1793004206.0                       # 2026-10-26T08:43:26Z, commit d3040c4f8165
OLDEST_REAL_MERGE = 1790589154.005547       # 2026-09-28T09:52:34Z, cpsam-soup-demo-pool v1
AFTER_CUTOFF = 1793500000.0                 # 2026-11-01T02:26:40Z

SHA = "a" * 64
OTHER_SHA = "b" * 64

# A basis value no release of this app has ever written. Shaped like the real ones
# so a check loose enough to accept anything string-ish would accept it too.
NEVER_WRITTEN_BASIS = "sha256_of_the_imaginary_thing"

POOL_ID = "bioimage-io/cpsam-soup-demo-pool"


def _merge(
    *,
    version: Optional[str] = "v1",
    sha: Optional[str] = SHA,
    created_at: Any = OLDEST_REAL_MERGE,
    basis: Any = "<omit>",
) -> Dict[str, Any]:
    """A merge record in the stored shape. ``basis='<omit>'`` writes no basis key at
    all, which is what every record written before this change looks like — distinct
    from a key present and null."""
    rec: Dict[str, Any] = {
        "published_version": version,
        "baseline_version": "stock-cpsam",
        "members": [{"contribution_id": "c-1", "marginal_score": 0.5,
                     "gate_score": 0.6, "included": True}],
        "selection_metric": {"role": "selection", "aggregate": 0.6},
        "witness_metric": {"role": "witness", "aggregate": 0.55},
        "global_sha256": sha,
        "transport": {"bytes_in": 1, "bytes_out": 2, "n_transfers": 1,
                      "sources_complete": True},
        "deferred": [],
        "created_at": created_at,
    }
    if basis != "<omit>":
        rec["global_sha256_basis"] = basis
    return rec


def _index(merges: List[Dict[str, Any]], *, current: Any = "<derive>") -> Dict[str, Any]:
    """A pool index around ``merges``. ``current='<derive>'`` points at the last
    published merge the way ``set_current_community`` would have, WITHOUT a basis
    stamp — i.e. as every pool on the live service looks today."""
    published = [m for m in merges if m.get("published_version")]
    if current == "<derive>":
        current = None
        if published:
            last = published[-1]
            current = {
                "version": last["published_version"],
                "path": f"community/{last['published_version']}/weights.pt",
                "sha256": last["global_sha256"],
                "selection_metric": last["selection_metric"],
                "witness_metric": last["witness_metric"],
            }
    return {
        "schema_version": pool.SCHEMA_VERSION,
        "model_type": "cpsam",
        "architecture_signature": "sig",
        "current_community": current,
        "contributions": [{"contribution_id": "c-1", "weights_path": "p",
                           "metadata_path": "m", "weights_sha256": "c" * 64,
                           "architecture_signature": "sig", "samples_seen": 10,
                           "uploaded_at": OLDEST_REAL_MERGE}],
        "decided_contribution_ids": ["c-1"],
        "baseline": {"witness_metric": {"role": "witness", "aggregate": 0.5}},
        "merges": merges,
    }


# ── the resolver ─────────────────────────────────────────────────────────────

class TestResolver:
    def test_a_pre_cutoff_record_is_the_file_digest(self):
        """The only direction a date can prove: state_dict_digest did not exist."""
        assert pool.digest_basis(_merge(created_at=CUTOFF - 1)) == pool.DIGEST_BASIS_FILE

    def test_every_real_stored_merge_resolves_to_the_file_digest(self):
        """The eight live values, by their measured timestamps."""
        for ts in REAL_MERGE_TIMES:
            assert pool.digest_basis(_merge(created_at=ts)) == pool.DIGEST_BASIS_FILE

    def test_a_post_cutoff_record_without_a_stamp_is_undetermined(self):
        """The row's fourth clause. A worker can still be running the old build a
        week after the commit, so the late side is evidence of nothing."""
        out = pool.digest_basis(_merge(created_at=AFTER_CUTOFF))
        assert out == pool.DIGEST_BASIS_UNDETERMINED
        assert out != pool.DIGEST_BASIS_TENSORS

    def test_the_cutoff_instant_itself_is_undetermined(self):
        """Strictly-before, so the boundary falls on the honest side."""
        assert pool.digest_basis(_merge(created_at=CUTOFF)) == pool.DIGEST_BASIS_UNDETERMINED

    def test_timestamps_alone_never_yield_the_tensor_meaning(self):
        """The asymmetry, swept rather than sampled: across the whole timeline there
        is NO unstamped input for which a date resolves to the new meaning. Guessing
        it would reinstate the silent default one version later."""
        probes = [0.0, 1.0, OLDEST_REAL_MERGE, CUTOFF - 1e-6, CUTOFF,
                  CUTOFF + 1e-6, AFTER_CUTOFF, 4_000_000_000.0]
        seen = {pool.digest_basis(_merge(created_at=t)) for t in probes}
        assert pool.DIGEST_BASIS_TENSORS not in seen
        # ...and the sweep is a real one: both honest answers DO occur, so an
        # assertion of absence is not passing because nothing was evaluated.
        assert seen == {pool.DIGEST_BASIS_FILE, pool.DIGEST_BASIS_UNDETERMINED}

    @pytest.mark.parametrize("stamp,expected", [
        (pool.DIGEST_BASIS_TENSORS, pool.DIGEST_BASIS_TENSORS),
        (pool.DIGEST_BASIS_FILE, pool.DIGEST_BASIS_FILE),
    ])
    def test_an_explicit_stamp_beats_the_timestamp_in_both_directions(self, stamp, expected):
        """Including the combination a date would get wrong: a file digest written
        after the cutoff (an old build still deployed) and a tensor digest dated
        before it (a clock that is wrong, or a record copied forward)."""
        assert pool.digest_basis(_merge(created_at=AFTER_CUTOFF, basis=stamp)) == expected
        assert pool.digest_basis(_merge(created_at=CUTOFF - 1, basis=stamp)) == expected

    def test_an_unrecognised_stamp_is_undetermined_not_defaulted(self):
        """A basis written by a future app this build has never heard of. Mapping it
        onto one of the two known meanings is the same mistake, one version later."""
        rec = _merge(created_at=CUTOFF - 1, basis=NEVER_WRITTEN_BASIS)
        assert pool.digest_basis(rec) == pool.DIGEST_BASIS_UNDETERMINED

    def test_no_digest_is_its_own_answer(self):
        """A merge that admitted nothing publishes no head. 'There is no value to
        attribute' and 'the value cannot be attributed' are different answers."""
        out = pool.digest_basis(_merge(version=None, sha=None))
        assert out == pool.DIGEST_BASIS_ABSENT
        assert out != pool.DIGEST_BASIS_UNDETERMINED

    def test_a_stamp_cannot_conjure_a_digest_that_is_not_there(self):
        rec = _merge(version=None, sha=None, basis=pool.DIGEST_BASIS_TENSORS)
        assert pool.digest_basis(rec) == pool.DIGEST_BASIS_ABSENT

    @pytest.mark.parametrize("ts", [
        None, "", "2026-09-28T09:52:34Z", "not-a-number", [], {},
        float("nan"), float("inf"), float("-inf"),
    ])
    def test_an_unusable_created_at_is_undetermined(self, ts):
        """-inf is in here on purpose: it is arithmetically 'before the cutoff' and a
        bare ``<`` would confidently return the file meaning for a nonsense value."""
        assert pool.digest_basis(_merge(created_at=ts)) == pool.DIGEST_BASIS_UNDETERMINED

    @pytest.mark.parametrize("record", [None, "", 0, [], "a" * 64])
    def test_a_non_record_is_undetermined(self, record):
        assert pool.digest_basis(record) == pool.DIGEST_BASIS_UNDETERMINED

    def test_the_keys_are_parameterisable_for_the_other_two_surfaces(self):
        """The same value is stored under three names; one resolver reads all three."""
        manifest = {"weights_sha256": SHA, "created_at": CUTOFF - 1}
        assert pool.digest_basis(manifest, digest_key="weights_sha256") == pool.DIGEST_BASIS_FILE
        pointer = {"sha256": SHA, "sha256_basis": pool.DIGEST_BASIS_TENSORS}
        assert pool.digest_basis(pointer, digest_key="sha256") == pool.DIGEST_BASIS_TENSORS


# Measured 2026-11-13 from the six live bioimage-io pools (the snapshot the row's
# figures were re-checked against). Eight merges carry a digest; these are their
# created_at values. Written out rather than derived so this file does not need the
# snapshot to run, and so a change to the snapshot cannot quietly change the test.
REAL_MERGE_TIMES: List[float] = [
    1790589154.005547,   # cpsam-soup-demo-pool       v1  2026-09-28T09:52:34Z
    1790590608.3255925,  # cpsam-soup-demo-pool-v2    v1  2026-09-28T10:16:48Z
    1790591145.4142523,  # cpsam-soup-demo-pool-v2    v2  2026-09-28T10:25:45Z
    1790591632.632655,   # cpsam-soup-demo-pool-v2    v3  2026-09-28T10:33:52Z
    1790596196.454121,   # cpsam-soup-concurrent-v2   v1  2026-09-28T11:49:56Z
    1790615628.9399433,  # cpsam-soup-overnight-gen2  v1  2026-09-28T17:13:48Z
    1790617463.887484,   # cpsam-soup-overnight-gen2  v2  2026-09-28T17:44:23Z
    1790619791.5862625,  # cpsam-soup-overnight-gen2  v3  2026-09-28T18:23:11Z
]


def test_the_measured_live_timestamps_are_all_before_the_cutoff():
    """The row's claim, as a fixture-level assertion: if a pool is later written by
    a newer app and this list is refreshed, a value on the other side of the cutoff
    fails HERE, naming the reason, instead of silently changing what the suite above
    is testing."""
    assert len(REAL_MERGE_TIMES) == 8
    assert max(REAL_MERGE_TIMES) < CUTOFF
    assert CUTOFF - max(REAL_MERGE_TIMES) > 20 * 24 * 3600   # ~28 days of margin


# ── the published form ───────────────────────────────────────────────────────

class TestProvenance:
    @pytest.mark.parametrize("basis", [
        pool.DIGEST_BASIS_TENSORS, pool.DIGEST_BASIS_FILE,
        pool.DIGEST_BASIS_UNDETERMINED, pool.DIGEST_BASIS_ABSENT,
    ])
    def test_every_basis_has_a_note(self, basis):
        assert basis in pool.DIGEST_BASIS_NOTES
        assert len(pool.DIGEST_BASIS_NOTES[basis]) > 80

    def test_the_four_bases_are_distinct(self):
        bases = {pool.DIGEST_BASIS_TENSORS, pool.DIGEST_BASIS_FILE,
                 pool.DIGEST_BASIS_UNDETERMINED, pool.DIGEST_BASIS_ABSENT}
        assert len(bases) == 4

    def test_undetermined_reproducibility_is_none_not_false(self):
        """Tri-state. 'We cannot tell' rendered as 'no' is a claim nobody made."""
        out = pool.digest_provenance(_merge(created_at=AFTER_CUTOFF))
        assert out["reproducible_by_re_merge"] is None
        assert out["reproducible_by_re_merge"] is not False

    def test_the_file_basis_says_a_re_merge_cannot_reproduce_it(self):
        out = pool.digest_provenance(_merge(created_at=CUTOFF - 1))
        assert out["basis"] == pool.DIGEST_BASIS_FILE
        assert out["reproducible_by_re_merge"] is False
        # The clause: the reader is TOLD WHY, not left to infer corruption.
        note = out["note"]
        assert "filename" in note
        assert "not evidence about the weights" in note
        assert pool.TENSOR_DIGEST_SINCE_APP_VERSION in note

    def test_the_tensor_basis_says_a_re_merge_does_reproduce_it(self):
        out = pool.digest_provenance(_merge(basis=pool.DIGEST_BASIS_TENSORS))
        assert out["basis"] == pool.DIGEST_BASIS_TENSORS
        assert out["reproducible_by_re_merge"] is True

    def test_the_undetermined_note_does_not_claim_corruption_either_way(self):
        note = pool.digest_provenance(_merge(created_at=AFTER_CUTOFF))["note"]
        assert "UNEXPLAINED" in note
        assert "not as evidence about the weights" in note


# ── the drift pin on state_dict_digest ───────────────────────────────────────
# CURRENT_DIGEST_BASIS names what state_dict_digest computes. Nothing in the
# language ties them together, so a change to the function that forgets the
# constant would start stamping new records with a basis that is no longer true —
# the same defect as #0417, with a stamp on it. The golden vector below is the tie.


class _StubTensor:
    """The slice of torch.Tensor that state_dict_digest touches, over numpy.

    ``str(t.dtype)`` must read ``torch.float32``, because that exact string is what
    goes into the hash for a real checkpoint — a stub reporting numpy's ``float32``
    would pin a digest no real pool has ever had.
    """

    class _DType:
        def __init__(self, name: str) -> None:
            self._name = name

        def __str__(self) -> str:
            return self._name

    def __init__(self, values, dtype_name: str = "torch.float32", np_dtype="<f4") -> None:
        self._a = np.asarray(values, dtype=np.dtype(np_dtype))
        self.dtype = self._DType(dtype_name)

    @property
    def shape(self):
        return self._a.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self._a


def _golden_state_dict():
    return {
        "encoder.weight": _StubTensor([[1.0, 2.0], [3.0, 4.0]]),
        "decoder.bias": _StubTensor([0.5, -0.5]),
        "diam_labels": _StubTensor([30.0]),
    }


# Produced by pool.state_dict_digest over _golden_state_dict() at the commit that
# introduced this file. If this value moves, the digest's meaning moved: update
# CURRENT_DIGEST_BASIS in the same commit, or explain why the stamp still holds.
GOLDEN = "fd1c314f4a093e4adceb16bdb132dc36182ab55c695c7d79fa62901e91b1c941"


class TestDigestFramingIsPinned:
    def test_the_golden_vector_still_holds(self):
        assert pool.state_dict_digest(_golden_state_dict()) == GOLDEN

    def test_insertion_order_does_not_change_the_digest(self):
        """Sorted keys, so two replicas building the same soup agree."""
        sd = _golden_state_dict()
        reordered = {k: sd[k] for k in reversed(list(sd))}
        assert pool.state_dict_digest(reordered) == pool.state_dict_digest(sd)

    def test_the_dtype_is_inside_the_hash(self):
        """A discrimination control on the framing: if the name|shape|dtype prefix
        stopped being hashed, the golden vector above would still pass for the wrong
        reason, because the raw bytes are unchanged."""
        same_bytes_other_dtype = {
            "encoder.weight": _StubTensor([[1.0, 2.0], [3.0, 4.0]], dtype_name="torch.float16"),
            "decoder.bias": _StubTensor([0.5, -0.5]),
            "diam_labels": _StubTensor([30.0]),
        }
        assert pool.state_dict_digest(same_bytes_other_dtype) != GOLDEN

    def test_the_key_names_are_inside_the_hash(self):
        renamed = {("x." + k if k != "diam_labels" else k): v
                   for k, v in _golden_state_dict().items()}
        assert pool.state_dict_digest(renamed) != GOLDEN

    def test_the_current_stamp_names_the_tensor_meaning(self):
        assert pool.CURRENT_DIGEST_BASIS == pool.DIGEST_BASIS_TENSORS


# ── writers ──────────────────────────────────────────────────────────────────

class TestWritersStampTheBasis:
    def test_a_new_community_manifest_carries_the_basis(self):
        man = pool.build_community_manifest(
            community_version="v1", combine="uniform", selected_contribution_ids=["c-1"],
            gate_split={"sha256": "ab"}, witness_split={"sha256": "cd"},
            selection_metric={"role": "selection"}, witness_metric={"role": "witness"},
            members=[], architecture_signature="sig", weights_sha256=SHA,
            created_at=AFTER_CUTOFF,
        )
        assert man["weights_sha256_basis"] == pool.CURRENT_DIGEST_BASIS
        assert pool.digest_basis(man, digest_key="weights_sha256") == pool.DIGEST_BASIS_TENSORS

    def test_a_stamped_manifest_resolves_with_no_timestamp_at_all(self):
        """The whole point of the stamp: attribution without date-reasoning."""
        man = pool.build_community_manifest(
            community_version="v1", combine="uniform", selected_contribution_ids=[],
            gate_split={}, witness_split={}, selection_metric={}, witness_metric={},
            members=[], architecture_signature="sig", weights_sha256=SHA,
            created_at=AFTER_CUTOFF,
        )
        del man["created_at"]
        assert pool.digest_basis(man, digest_key="weights_sha256") == pool.DIGEST_BASIS_TENSORS

    def test_the_index_pointer_carries_the_basis(self):
        """The pointer has no created_at of its own, so without this stamp it is the
        one surface of the three that cannot be attributed from itself."""
        index = {"current_community": None}
        pool.set_current_community(
            index, community_version="v1", weights_path="community/v1/weights.pt",
            sha256=SHA, selection_metric={}, witness_metric={},
        )
        cur = index["current_community"]
        assert "created_at" not in cur
        assert cur["sha256_basis"] == pool.CURRENT_DIGEST_BASIS
        assert pool.digest_basis(cur, digest_key="sha256") == pool.DIGEST_BASIS_TENSORS


# ── #0509: the contribution surface ──────────────────────────────────────────
#
# The third writer. ``append_contribution`` stamped nothing, so a contribution's
# ``weights_sha256`` was the one published digest with no attribution at all — a
# client whose upload predates the cut-over re-hashes its own file, gets a
# different value from the one the pool publishes, and has no way to tell "the
# pool measured something else" from "my upload is corrupt".
#
# Deliberately NOT asserted through the writer's own return value or its argument
# dict: the row asks for the STORED entry read back out of the index, because a
# stamp applied to the caller's dict and then dropped on the way in would satisfy
# any assertion made on the dict.

CONTRIB_SHA = "d" * 64
CONTRIB_OTHER_SHA = "e" * 64
# Two DISTINCT pre-cutoff instants. Both before the cut-off on purpose: the
# stamped record must resolve to the tensor meaning even though its own date would
# say "file", which is the only way to show the answer comes from the stamp rather
# than from the clock.
UPLOADED_OLD = OLDEST_REAL_MERGE
UPLOADED_NEW = OLDEST_REAL_MERGE + 3600.0


def _contribution(cid: str, digest: str, uploaded_at: Any, basis: Any = "<omit>") -> Dict[str, Any]:
    """A contribution stub in the stored shape. ``basis='<omit>'`` writes no basis
    key at all — what every record written before #0509 looks like, and distinct
    from a key present and null."""
    rec: Dict[str, Any] = {
        "contribution_id": cid,
        "weights_path": f"contributions/{cid}/weights.pt",
        "metadata_path": f"contributions/{cid}/metadata.json",
        "weights_sha256": digest,
        "architecture_signature": "sig",
        "samples_seen": 10,
        "n_train_images": 5,
        "n_bytes": 1218617275,
        "uploaded_at": uploaded_at,
    }
    if basis != "<omit>":
        rec["weights_sha256_basis"] = basis
    return rec


class TestContributionsStampTheBasis:
    def test_a_new_contribution_carries_the_basis_on_the_stored_entry(self):
        """Clause 1, read back out of the index rather than off the writer."""
        index: Dict[str, Any] = {}
        pool.append_contribution(
            index, _contribution("c-new", CONTRIB_SHA, UPLOADED_NEW)
        )
        stored = index["contributions"][-1]
        assert stored["weights_sha256_basis"] == pool.CURRENT_DIGEST_BASIS
        assert pool.contribution_digest_basis(stored) == pool.DIGEST_BASIS_TENSORS

    def test_a_stamped_contribution_resolves_with_no_timestamp_at_all(self):
        """The point of the stamp: attribution without date-reasoning."""
        index: Dict[str, Any] = {}
        pool.append_contribution(
            index, _contribution("c-new", CONTRIB_SHA, UPLOADED_NEW)
        )
        stored = dict(index["contributions"][-1])
        del stored["uploaded_at"]
        assert pool.contribution_digest_basis(stored) == pool.DIGEST_BASIS_TENSORS

    def test_two_contributions_under_different_bases_are_distinguishable(self):
        """THE discrimination this row is for, and the test with teeth.

        One legacy member (no stamp, pre-cutoff date) and one newly appended
        member. Both dates are pre-cutoff and the two differ, so:

          * a resolver hardcoding the tensor meaning reads the legacy member wrong;
          * a resolver hardcoding the file meaning reads the new member wrong;
          * a resolver leaking a sibling's basis gives both the SAME answer;
          * a resolver that ignored the stamp and dated the record would call the
            new member "file", because its own date says so.

        Only reading each record's own stamp first, then its own date, passes.
        """
        index: Dict[str, Any] = {
            "contributions": [
                _contribution("c-legacy", CONTRIB_OTHER_SHA, UPLOADED_OLD)
            ]
        }
        pool.append_contribution(
            index, _contribution("c-new", CONTRIB_SHA, UPLOADED_NEW)
        )
        legacy, new = index["contributions"]

        assert legacy["contribution_id"] == "c-legacy"
        assert new["contribution_id"] == "c-new"
        assert legacy["weights_sha256"] != new["weights_sha256"]
        assert legacy["uploaded_at"] != new["uploaded_at"]
        assert legacy["uploaded_at"] < CUTOFF and new["uploaded_at"] < CUTOFF

        assert pool.contribution_digest_basis(legacy) == pool.DIGEST_BASIS_FILE
        assert pool.contribution_digest_basis(new) == pool.DIGEST_BASIS_TENSORS
        assert pool.contribution_digest_basis(legacy) != pool.contribution_digest_basis(new)

    def test_a_post_cutoff_unstamped_contribution_is_undetermined_not_tensors(self):
        """Clause 2's decision, as the behaviour rather than as prose: an absent
        basis is NOT quietly read as the current one. This is the case where a
        silent default would look most like success."""
        legacy = _contribution("c-late", CONTRIB_SHA, AFTER_CUTOFF)
        assert "weights_sha256_basis" not in legacy
        assert pool.contribution_digest_basis(legacy) == pool.DIGEST_BASIS_UNDETERMINED
        assert pool.contribution_digest_basis(legacy) != pool.DIGEST_BASIS_TENSORS

    def test_an_unrecognised_stamp_is_undetermined_not_passed_through(self):
        """A record written by a NEWER app than this one. Shaped like a real basis,
        so a check loose enough to accept any string would accept it."""
        rec = _contribution("c-future", CONTRIB_SHA, UPLOADED_NEW, basis=NEVER_WRITTEN_BASIS)
        assert pool.contribution_digest_basis(rec) == pool.DIGEST_BASIS_UNDETERMINED

    def test_a_contribution_with_no_digest_says_so_rather_than_undetermined(self):
        """``no_digest`` and ``undetermined`` are different answers — "nothing to
        attribute" vs "cannot attribute" — and collapsing them hides the second."""
        rec = _contribution("c-empty", "", UPLOADED_NEW)
        assert pool.contribution_digest_basis(rec) == pool.DIGEST_BASIS_ABSENT

    def test_the_default_keys_misattribute_a_contribution_two_different_ways(self):
        """Why ``contribution_digest_basis`` exists rather than three keyword
        arguments at each call site. A contribution is the one record shape whose
        digest key AND timestamp key both differ from the defaults, so there are two
        wrong answers available and neither raises.

        The second is the dangerous one: "undetermined" for a record whose date
        proves the file meaning is indistinguishable from the honest refusal."""
        legacy = _contribution("c-legacy", CONTRIB_OTHER_SHA, UPLOADED_OLD)

        assert pool.digest_basis(legacy) == pool.DIGEST_BASIS_ABSENT
        assert (
            pool.digest_basis(legacy, digest_key="weights_sha256")
            == pool.DIGEST_BASIS_UNDETERMINED
        )
        # The named resolver is the only one of the three that is right.
        assert pool.contribution_digest_basis(legacy) == pool.DIGEST_BASIS_FILE

    def test_appending_back_fills_nothing_and_does_not_mutate_the_caller(self):
        """Clause 5. Existing stored records keep having no basis — they resolve by
        date, the one direction that is a proof — and the dict the caller passed in
        is left alone, so nothing it might also be holding is rewritten."""
        legacy = _contribution("c-legacy", CONTRIB_OTHER_SHA, UPLOADED_OLD)
        index: Dict[str, Any] = {"contributions": [legacy]}
        before = dict(legacy)

        fresh = _contribution("c-new", CONTRIB_SHA, UPLOADED_NEW)
        pool.append_contribution(index, fresh)

        assert index["contributions"][0] is legacy, "the stored record was replaced"
        assert legacy == before, "the stored record was rewritten"
        assert "weights_sha256_basis" not in index["contributions"][0]
        assert "weights_sha256_basis" not in fresh, "the caller's dict was mutated"
        # Paired with the above so "nothing was written" cannot pass: the new entry
        # DID get the stamp, which is the whole change.
        assert index["contributions"][1]["weights_sha256_basis"] == pool.CURRENT_DIGEST_BASIS

    def test_the_refused_duplicate_append_stamps_nothing(self):
        """The #0376 dedupe backstop still fires, and the stamp does not reach the
        index on the path that raises."""
        index: Dict[str, Any] = {
            "contributions": [_contribution("c-first", CONTRIB_SHA, UPLOADED_OLD)]
        }
        with pytest.raises(ValueError):
            pool.append_contribution(
                index, _contribution("c-retry", CONTRIB_SHA, UPLOADED_NEW)
            )
        assert len(index["contributions"]) == 1
        assert "weights_sha256_basis" not in index["contributions"][0]


# ── the wire projection ──────────────────────────────────────────────────────

# An INDEPENDENT copy of the page's 0.13.0-draft soup field set. test_pool.py has
# its own; a copy imported from there could not detect that file being edited to
# accept a widened contract, which is the thing this controls for.
PINNED_SOUP_KEYS_0_13_0_DRAFT = {
    "soup_id", "index", "merged_at", "assessed", "contributions",
    "selection_metric", "witness_metric", "global_sha256", "weights",
    "transport", "community_model",
}
PINNED_EMPTY_MERGE_KEYS_0_13_0_DRAFT = {
    "merged_at", "assessed", "selection_metric", "transport",
}


class TestWireProjection:
    def test_the_pinned_soup_contract_is_untouched(self):
        """Over-reach control. The attribution is additive and lives OUTSIDE every
        field set the page pins, so this must be IDENTICAL to the pre-change set —
        not merely 'still contains the old keys'."""
        wire = pool.pool_wire_state(
            _index([_merge(version="v1"), _merge(version=None, sha=None)]),
            pool_artifact_id=POOL_ID,
        )
        assert set(wire["soups"][0]) == PINNED_SOUP_KEYS_0_13_0_DRAFT
        assert set(wire["empty_merges"][0]) == PINNED_EMPTY_MERGE_KEYS_0_13_0_DRAFT
        assert wire["schema_version"] == pool.SCHEMA_VERSION

    def test_a_legacy_pool_attributes_every_published_head_to_the_file_digest(self):
        """A pool as the live ones are: pre-cutoff, no stamp anywhere."""
        wire = pool.pool_wire_state(
            _index([_merge(version="v1", sha=SHA),
                    _merge(version="v2", sha=OTHER_SHA)]),
            pool_artifact_id=POOL_ID,
        )
        prov = wire["digest_provenance"]["soups"]
        assert set(prov) == {"soup-0", "soup-1"}
        assert all(p["basis"] == pool.DIGEST_BASIS_FILE for p in prov.values())
        assert all(p["reproducible_by_re_merge"] is False for p in prov.values())

    def test_an_empty_merge_gets_no_entry(self):
        """It publishes no head, so there is no digest to attribute and inventing an
        entry would make 'nothing was published' look like a failed attribution."""
        wire = pool.pool_wire_state(
            _index([_merge(version=None, sha=None), _merge(version="v1")]),
            pool_artifact_id=POOL_ID,
        )
        assert set(wire["digest_provenance"]["soups"]) == {"soup-0"}
        assert len(wire["empty_merges"]) == 1

    def test_the_pointer_borrows_the_publishing_merges_date(self):
        """``current_community`` carries neither a stamp nor a date on a legacy pool,
        so it is resolved by joining to the merge that published that version."""
        wire = pool.pool_wire_state(
            _index([_merge(version="v1", created_at=OLDEST_REAL_MERGE)]),
            pool_artifact_id=POOL_ID,
        )
        assert wire["digest_provenance"]["current_community"]["basis"] == pool.DIGEST_BASIS_FILE
        assert wire["current_community"]["sha256_provenance"]["basis"] == pool.DIGEST_BASIS_FILE

    def test_the_pointer_is_undetermined_when_no_merge_publishes_its_version(self):
        """An index whose pointer names a version the timeline does not contain — an
        aggregate interrupted between its two commits. The join must refuse rather
        than borrow a date from some other merge."""
        index = _index([_merge(version="v1")])
        index["current_community"]["version"] = "v9"
        wire = pool.pool_wire_state(index, pool_artifact_id=POOL_ID)
        assert wire["digest_provenance"]["current_community"]["basis"] == pool.DIGEST_BASIS_UNDETERMINED

    def test_a_post_cutoff_unstamped_pool_reports_undetermined(self):
        """The negative control at projection level: a pool written in the window
        between the commit and this change, where neither evidence source applies."""
        wire = pool.pool_wire_state(
            _index([_merge(version="v1", created_at=AFTER_CUTOFF)]),
            pool_artifact_id=POOL_ID,
        )
        prov = wire["digest_provenance"]
        assert prov["soups"]["soup-0"]["basis"] == pool.DIGEST_BASIS_UNDETERMINED
        assert prov["current_community"]["basis"] == pool.DIGEST_BASIS_UNDETERMINED

    def test_a_pool_written_by_this_app_resolves_to_tensors_end_to_end(self):
        """Index built the way the entry now builds one: stamp on the merge record
        and on the pointer. No date reasoning is involved in either answer."""
        index = _index([_merge(version="v1", created_at=AFTER_CUTOFF,
                               basis=pool.CURRENT_DIGEST_BASIS)], current=None)
        pool.set_current_community(
            index, community_version="v1", weights_path="community/v1/weights.pt",
            sha256=SHA, selection_metric={}, witness_metric={},
        )
        wire = pool.pool_wire_state(index, pool_artifact_id=POOL_ID)
        prov = wire["digest_provenance"]
        assert prov["soups"]["soup-0"]["basis"] == pool.DIGEST_BASIS_TENSORS
        assert prov["soups"]["soup-0"]["reproducible_by_re_merge"] is True
        assert prov["current_community"]["basis"] == pool.DIGEST_BASIS_TENSORS

    def test_a_pool_with_no_published_head_has_no_pointer_provenance(self):
        wire = pool.pool_wire_state(
            _index([_merge(version=None, sha=None)]), pool_artifact_id=POOL_ID)
        assert wire["current_community"] is None
        assert wire["digest_provenance"]["current_community"] is None
        assert wire["digest_provenance"]["soups"] == {}


# ── the second stored surface: the published community manifest ──────────────
# It does not pass through pool_wire_state, so it is attributed in entry.py or
# nowhere. The real coroutine is driven here; only the HTTP payload is a stub.


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
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    async def _http_retry(self, method: str, url: str, **kw: Any) -> _Resp:
        return _Resp(self._payload)


def _read_manifest_of(instance):
    """The real coroutine bound to a stub. ``entry.EntryApp`` is a Ray Serve
    Deployment once ``@bioengine.app`` has run, so ``func_or_class`` is the class."""
    import entry

    return entry.EntryApp.func_or_class._read_community_manifest.__get__(instance)


@pytest.mark.asyncio
async def test_a_legacy_manifest_is_attributed_on_read_back():
    stub = _Entry({
        "community_version": "v1",
        "weights_sha256": SHA,
        "created_at": OLDEST_REAL_MERGE,
        "selection_metric": {"role": "selection"},
        "witness_metric": {"role": "witness"},
    })
    out = await _read_manifest_of(stub)(_AM(), POOL_ID, "v1")

    assert out["sha256"] == SHA
    assert out["sha256_provenance"]["basis"] == pool.DIGEST_BASIS_FILE
    assert out["sha256_provenance"]["reproducible_by_re_merge"] is False
    assert "filename" in out["sha256_provenance"]["note"]


@pytest.mark.asyncio
async def test_a_stamped_manifest_is_attributed_without_its_date():
    stub = _Entry({
        "community_version": "v4",
        "weights_sha256": SHA,
        "weights_sha256_basis": pool.DIGEST_BASIS_TENSORS,
        "created_at": AFTER_CUTOFF,
        "selection_metric": {},
        "witness_metric": {},
    })
    out = await _read_manifest_of(stub)(_AM(), POOL_ID, "v4")
    assert out["sha256_provenance"]["basis"] == pool.DIGEST_BASIS_TENSORS
    assert out["sha256_provenance"]["reproducible_by_re_merge"] is True


@pytest.mark.asyncio
async def test_a_manifest_written_in_the_unstamped_window_is_undetermined():
    """Neither a stamp nor a usable date — the state the fix must not paper over."""
    stub = _Entry({
        "community_version": "v4",
        "weights_sha256": SHA,
        "created_at": AFTER_CUTOFF,
        "selection_metric": {},
        "witness_metric": {},
    })
    out = await _read_manifest_of(stub)(_AM(), POOL_ID, "v4")
    assert out["sha256_provenance"]["basis"] == pool.DIGEST_BASIS_UNDETERMINED
    assert out["sha256_provenance"]["reproducible_by_re_merge"] is None


# ── the two WRITER surfaces: the merge record, and aggregate()'s own return ───
# Neither passes through pool_wire_state or _read_community_manifest, so each is
# attributed in aggregate() or nowhere — and until this block each was pinned by
# nothing. That was measured, not supposed: deleting `global_sha256_basis` from the
# append_merge call, and deleting `global_sha256_provenance` from the return, each
# left the whole 199-case app suite GREEN.
#
# The gap was structural rather than an oversight. The only other harness that
# drives the real aggregate (test_merge_serialisation.py) returns a pool whose
# model_type is not cpsam on purpose, so the coroutine raises at its model_type
# guard — which is upstream of both lines, and of everything else in the merge.
#
# So the merge is driven to completion here: every collaborator is a stub, every
# pool.* helper is the real one. `head_sha` comes from the runtime, which is what
# lets ONE harness reach both branches — a merge that publishes a head, and a merge
# that admits nothing and publishes none. The second is the ``None`` half of the
# stamp, and it is the half a test that only ever admits something cannot see.

GATE_SPLIT = {"artifact_id": "bioimage-io/val-gate", "sha256": "c" * 64}
WITNESS_SPLIT = {"artifact_id": "bioimage-io/val-witness", "sha256": "d" * 64}
HEAD_SHA = "e" * 64

# Large enough that pool.soup_admission never refuses: this harness is about the
# attribution of a digest, and a refusal would exit before one exists.
_BUDGET = {"declared_bytes": 64 * 1024 ** 3, "rss_bytes": 1024 ** 3}


class _AggResp(_Resp):
    """``_Resp`` plus the attribute the download path reads. Unused while the pool
    has no current head, and present so that a change which starts downloading one
    fails on a payload rather than on an AttributeError."""

    content = b"{}"


class _AggAM(_AM):
    async def put_file(self, artifact_id: str, file_path: str) -> str:
        return "https://example.invalid/put/" + file_path


class _Runtime:
    """The GPU deployment, reduced to the three calls aggregate() makes on it.

    One constant score for every pass: whether a candidate is admitted is then a
    pure function of ``f1`` against the recorded baseline, so a caller picks the
    branch under test by choosing one number.
    """

    def __init__(self, f1: float, sha: Optional[str] = HEAD_SHA) -> None:
        self._f1 = f1
        self._sha = sha

    async def soup_memory_budget(self) -> Dict[str, Any]:
        return dict(_BUDGET)

    async def score_soup(self, specs, imgs, gts, diam_mean) -> Dict[str, Any]:
        return {"f1": self._f1, "replica": "r0", "soup_dl_bytes": 0}

    async def materialize_soup(self, specs, url) -> Dict[str, Any]:
        return {"sha256": self._sha, "n_bytes": 2048,
                "replica": "r0", "soup_dl_bytes": 0}


class _Merging:
    """Just enough of ``EntryApp`` for ``aggregate`` to run a merge end to end."""

    def __init__(self, index: Dict[str, Any], runtime: _Runtime) -> None:
        self._pool_locks: Dict[str, Any] = {}
        self._merges_in_flight = 0
        self.index = index
        self._runtime = runtime
        self.commits = 0

    async def _check_runtime_available(self, model_type: str) -> None:
        return None

    def _runtime_for(self, model_type: str) -> _Runtime:
        return self._runtime

    async def _resolve_val_split(self, images, labels):
        return list(images), list(labels)

    async def _pool_service(self, pool_token: str):
        class _Client:
            async def disconnect(self_inner) -> None:
                return None

        return _AggAM(), _Client()

    async def _read_pool_index(self, am, pool_artifact_id: str) -> Dict[str, Any]:
        return self.index

    async def _http_retry(self, method: str, url: str, **kw: Any) -> _AggResp:
        return _AggResp({})

    def _bump_transport(self, **kw: Any) -> None:
        return None

    @contextlib.asynccontextmanager
    async def _staged(self, am, pool_artifact_id: str):
        yield

    async def _commit_with_retry(self, am, pool_artifact_id: str) -> None:
        self.commits += 1

    async def _read_merge_journal(self, am, pool_artifact_id: str) -> None:
        return None

    async def _write_merge_journal(self, am, pool_artifact_id: str, journal) -> None:
        return None

    async def _next_free_community_version(self, am, pool_artifact_id, index) -> str:
        return "v2"


def _aggregate_of(instance):
    """The real ``aggregate`` coroutine bound to a stub instance."""
    import entry

    return entry.EntryApp.func_or_class.aggregate.__get__(instance)


def _pool_index() -> Dict[str, Any]:
    """A cpsam pool with one undecided contribution and a recorded baseline.

    The baseline is pre-recorded deliberately: without it aggregate() first scores
    stock cpsam on both splits, which is two more runtime calls and not one line
    more coverage of the two surfaces under test.
    """
    return {
        "model_type": "cpsam",
        "architecture_signature": "sig-1",
        "baseline": {
            "model": "stock-cpsam",
            "selection_metric": {"role": "selection", "aggregate": 0.50},
            "witness_metric": {"role": "witness", "aggregate": 0.40},
            "created_at": OLDEST_REAL_MERGE,
        },
        "contributions": [{
            "contribution_id": "c-1",
            "weights_path": "contributions/c-1/weights.pt",
            "n_bytes": 4096,
            "uploaded_at": OLDEST_REAL_MERGE,
            "weights_sha256": OTHER_SHA,
        }],
    }


async def _run_merge(f1: float):
    """Drive the real aggregate() to completion. Returns ``(result, index)``.

    ``f1 >= 0.50`` beats the recorded baseline and publishes a head; below it
    nothing is admitted and no head exists to attribute.
    """
    index = _pool_index()
    stub = _Merging(index, _Runtime(f1))
    result = await _aggregate_of(stub)(
        pool_artifact_id=POOL_ID,
        pool_token="t",
        gate_val_images=[], gate_val_labels=[],
        witness_val_images=[], witness_val_labels=[],
        gate_split=dict(GATE_SPLIT), witness_split=dict(WITNESS_SPLIT),
    )
    return result, index


@pytest.mark.asyncio
async def test_a_merge_record_this_build_writes_stamps_the_current_basis():
    """The forward-looking half of the fix: the primary of the three writers, and
    the thing that makes every FUTURE merge record self-describing so that no later
    reader has to date-reason the way this module has to for pre-0.27.12 records."""
    result, index = await _run_merge(0.90)

    assert result["published_version"] == "v2", "this merge must have published a head"
    record = index["merges"][-1]
    assert record["global_sha256"] == HEAD_SHA
    assert record["global_sha256_basis"] == pool.CURRENT_DIGEST_BASIS
    # Read back cold, the record attributes itself with no reference to its date.
    assert pool.digest_basis(record) == pool.DIGEST_BASIS_TENSORS

    # The discrimination control, and the reason the stamp is load-bearing rather
    # than decorative: this record is written NOW, which is after the cutoff, so
    # its timestamp proves nothing. Strip the stamp and the same record is
    # UNDETERMINED. That gap is exactly what the stamp buys.
    unstamped = {k: v for k, v in record.items() if k != "global_sha256_basis"}
    assert pool.digest_basis(unstamped) == pool.DIGEST_BASIS_UNDETERMINED


@pytest.mark.asyncio
async def test_a_merge_that_admitted_nothing_stamps_a_null_basis_not_an_absent_one():
    """The ``None`` half of the stamp. Written unconditionally, null included: a
    basis field absent because there is no digest reads identically to one absent
    because the app predates the stamp, and that ambiguity is the whole reason the
    field exists."""
    result, index = await _run_merge(0.10)

    assert result["published_version"] is None, "nothing should have been admitted"
    record = index["merges"][-1]
    assert record["global_sha256"] is None
    # Present and null, NOT missing — an absent key and a null value read
    # identically through .get(), so the key's existence is asserted first.
    assert "global_sha256_basis" in record
    assert record["global_sha256_basis"] is None
    assert pool.digest_basis(record) == pool.DIGEST_BASIS_ABSENT


@pytest.mark.asyncio
async def test_aggregate_returns_the_attribution_of_the_digest_it_just_published():
    """The moment the #0417 false alarm actually happens: a caller re-merges an
    identical member set, gets a number, and compares it against a stored one. Both
    numbers are in their hands HERE, so the attribution has to travel with the
    return value and not only on a page they would have to go and read."""
    result, _ = await _run_merge(0.90)

    assert result["global_sha256"] == HEAD_SHA
    prov = result["global_sha256_provenance"]
    assert prov["basis"] == pool.DIGEST_BASIS_TENSORS
    assert prov["reproducible_by_re_merge"] is True
    assert prov["note"], "the sentence that explains a mismatch must travel too"


@pytest.mark.asyncio
async def test_aggregate_declines_to_attribute_a_head_it_did_not_publish():
    """``no_digest``, not ``undetermined``: "there is no value to attribute" and
    "the value cannot be attributed" are different answers, and a merge that
    admitted nothing is the first."""
    result, _ = await _run_merge(0.10)

    assert result["global_sha256"] is None
    prov = result["global_sha256_provenance"]
    assert prov["basis"] == pool.DIGEST_BASIS_ABSENT
    assert prov["reproducible_by_re_merge"] is None
