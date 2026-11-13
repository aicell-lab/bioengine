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
