"""#0139 — a pool member whose bytes cannot be vouched for is quarantined.

Plain pytest, no torch, no numpy, no live Hypha: ``pool``'s heavy imports are all
function-local, so this surface imports on stdlib alone.

The fixture has FOUR rows, and each one is a distinct failure this change has to tell
apart, not four copies of one case:

- ``c-legacy``  — the ``weights_sha256`` key is ABSENT. A record written before the
  digest was required; nothing can ever be done about it.
- ``c-null``    — the key is present and null. The writer knew the field existed and
  had nothing to put in it, which is a different bug from never having written it,
  and the two must land in the same quarantine bucket because operationally they mean
  the identical thing: there is nothing here to check the bytes against.
- ``c-good``    — digest recorded, bytes hash to it. The control that keeps every
  assertion below from being satisfiable by quarantining the whole pool.
- ``c-mismatch``— digest recorded, bytes hash to something ELSE. The bytes in the
  pool are not the bytes that were deposited. This shares no reason code with the
  first two: it is a possible integrity incident and a human has to see it, and a
  single code for both would bury it inside the boring case forever.

The last case is the one the whole row turns on. Before this change the merge
hashed nothing on load, so ``c-mismatch`` and ``c-good`` produced byte-identical
behaviour and an identical published checkpoint.
"""

from __future__ import annotations

import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pool  # noqa: E402

GOOD_BYTES = b"the bytes this contribution attested to"
GOOD_SHA = hashlib.sha256(GOOD_BYTES).hexdigest()
OTHER_SHA = hashlib.sha256(b"some other bytes entirely").hexdigest()


def _rows():
    """The four-row fixture. Built per call so a test mutating a row cannot leak."""
    return [
        # weights_sha256 absent entirely — the legacy writer
        {"contribution_id": "c-legacy", "weights_path": "contrib/c-legacy/w.pt"},
        {"contribution_id": "c-null", "weights_path": "contrib/c-null/w.pt",
         "weights_sha256": None},
        {"contribution_id": "c-good", "weights_path": "contrib/c-good/w.pt",
         "weights_sha256": GOOD_SHA},
        {"contribution_id": "c-mismatch", "weights_path": "contrib/c-mismatch/w.pt",
         "weights_sha256": GOOD_SHA},
    ]


def _actuals():
    """What hashing the bytes on the replica returns for the rows that get hashed.

    ``c-legacy`` and ``c-null`` are absent on purpose: a member with no digest is
    quarantined from the index alone and its bytes are never downloaded, so a map
    that contained them would be describing work this change does not do.
    """
    return {
        "contrib/c-good/w.pt": GOOD_SHA,
        "contrib/c-mismatch/w.pt": OTHER_SHA,
    }


# ── the digest predicate ─────────────────────────────────────────────────────

class TestUsableDigest:
    @pytest.mark.parametrize("value", [None, "", "deadbeef", GOOD_SHA[:63],
                                       GOOD_SHA + "a", 0, 123, [], {}, b"x" * 64,
                                       "z" * 64, GOOD_SHA[:-1] + "g"])
    def test_unusable_values_are_unusable(self, value):
        """Each of these has been seen or is one keystroke from being seen. The
        non-strings matter as much as the strings: ``len()`` on an int raises and
        ``len()`` on a list does not, so a predicate written as a bare length check
        passes some of these and crashes on others."""
        assert pool.is_usable_digest(value) is False

    def test_a_real_digest_is_usable(self):
        assert pool.is_usable_digest(GOOD_SHA) is True

    def test_case_does_not_decide_provenance(self):
        """An upper-case copy of the right digest is the right digest. Quarantining
        it would exclude a member whose provenance is in fact established, on a
        presentation detail of whatever tool the string travelled through."""
        assert pool.is_usable_digest(GOOD_SHA.upper()) is True

    def test_the_predicate_is_not_satisfied_by_length_alone(self):
        """A discrimination control: 64 characters is not the property. Without
        this, ``len(value) == 64`` passes every test above."""
        assert len("z" * 64) == 64
        assert pool.is_usable_digest("z" * 64) is False


# ── the split ────────────────────────────────────────────────────────────────

class TestDigestQuarantine:
    def test_the_good_row_is_admitted_and_it_is_the_only_one(self):
        admitted, quarantined = pool.digest_quarantine(_rows(), _actuals())
        assert [r["contribution_id"] for r in admitted] == ["c-good"]
        assert len(quarantined) == 3

    def test_absent_and_null_share_one_reason_and_the_mismatch_does_not(self):
        """The ruling's first implementability condition, as an assertion. If this
        ever goes green with all three reasons equal, a reader filtering the
        community manifest for integrity incidents finds every legacy record and
        gives up on the filter."""
        _, quarantined = pool.digest_quarantine(_rows(), _actuals())
        by_id = {q["contribution_id"]: q["reason"] for q in quarantined}
        assert by_id["c-legacy"] == pool.QUARANTINE_NO_DIGEST
        assert by_id["c-null"] == pool.QUARANTINE_NO_DIGEST
        assert by_id["c-mismatch"] == pool.QUARANTINE_DIGEST_MISMATCH
        assert pool.QUARANTINE_NO_DIGEST != pool.QUARANTINE_DIGEST_MISMATCH

    def test_a_mismatch_records_both_hashes(self):
        """What the escalation needs: the contribution, and both numbers. A reason
        code with one hash cannot be acted on — it says something is wrong and
        withholds which of the two values is the surprising one."""
        _, quarantined = pool.digest_quarantine(_rows(), _actuals())
        mis = next(q for q in quarantined if q["contribution_id"] == "c-mismatch")
        assert mis["expected_sha256"] == GOOD_SHA
        assert mis["actual_sha256"] == OTHER_SHA
        assert mis["weights_path"] == "contrib/c-mismatch/w.pt"

    def test_a_no_digest_row_keeps_what_the_pool_actually_recorded(self):
        """``None`` is stored as ``None`` rather than normalised to a string: the
        difference between a record that carried nothing and one that carried junk
        is the only evidence there is about which writer produced it."""
        _, quarantined = pool.digest_quarantine(_rows(), _actuals())
        by_id = {q["contribution_id"]: q for q in quarantined}
        assert by_id["c-null"]["expected_sha256"] is None
        assert by_id["c-legacy"]["expected_sha256"] is None

    def test_case_difference_alone_does_not_quarantine(self):
        rows = [{"contribution_id": "c-up", "weights_path": "p",
                 "weights_sha256": GOOD_SHA.upper()}]
        admitted, quarantined = pool.digest_quarantine(rows, {"p": GOOD_SHA})
        assert [r["contribution_id"] for r in admitted] == ["c-up"]
        assert quarantined == []

    def test_a_member_nobody_hashed_raises_rather_than_being_admitted(self):
        """The load-bearing line. The failure this row exists to prevent is souping
        bytes that nothing checked; a version of ``digest_quarantine`` that treated
        an unhashed member as "nothing to compare against, carry on" would restore
        that failure exactly, and every other test in this file would still pass."""
        rows = [{"contribution_id": "c-good", "weights_path": "contrib/c-good/w.pt",
                 "weights_sha256": GOOD_SHA}]
        with pytest.raises(ValueError, match="ever being hashed"):
            pool.digest_quarantine(rows, {})

    def test_a_nonsense_actual_digest_raises(self):
        """A verifier that returns junk must not be allowed to quarantine a member
        as a mismatch: that would publish an integrity incident about the pool on
        the strength of a broken hasher."""
        rows = [{"contribution_id": "c-good", "weights_path": "p",
                 "weights_sha256": GOOD_SHA}]
        with pytest.raises(ValueError, match="not a sha256"):
            pool.digest_quarantine(rows, {"p": "oops"})

    def test_an_empty_pool_quarantines_nothing(self):
        assert pool.digest_quarantine([], {}) == ([], [])


class TestMembersToHash:
    def test_only_the_rows_with_something_to_compare_against_are_hashed(self):
        paths = pool.members_to_hash(_rows())
        assert paths == ["contrib/c-good/w.pt", "contrib/c-mismatch/w.pt"]

    def test_a_mismatch_is_hashed_not_skipped(self):
        """A plausible shortcut is to hash only members the pool has not seen
        before, which would never re-read ``c-mismatch`` after its first merge. The
        mismatch is only findable where the bytes land."""
        assert "contrib/c-mismatch/w.pt" in pool.members_to_hash(_rows())

    def test_duplicate_paths_are_hashed_once(self):
        rows = _rows() + [_rows()[2]]
        assert pool.members_to_hash(rows).count("contrib/c-good/w.pt") == 1

    def test_what_it_returns_is_exactly_what_the_split_demands(self):
        """The two functions are a contract: anything ``members_to_hash`` omits,
        ``digest_quarantine`` must not require — otherwise the orchestration raises
        on a pool it should have merged. Checked by running the real pair rather
        than by reading them."""
        actual = {p: GOOD_SHA for p in pool.members_to_hash(_rows())}
        pool.digest_quarantine(_rows(), actual)  # must not raise


class TestQuarantineRecord:
    def test_an_unknown_reason_is_refused(self):
        with pytest.raises(ValueError, match="not a quarantine reason"):
            pool.quarantine_record(
                contribution_id="c", weights_path="p", reason="no-digest",
                expected_sha256=None, actual_sha256=None)

    def test_counts_name_every_reason_even_at_zero(self):
        """An absent key and a zero read identically through ``.get()``. A caller
        testing whether this merge hit an integrity incident must get 0, not a
        KeyError, on the overwhelmingly common merge where it did not."""
        counts = pool.quarantine_counts([])
        assert counts == {pool.QUARANTINE_NO_DIGEST: 0,
                          pool.QUARANTINE_DIGEST_MISMATCH: 0}

    def test_counts_partition_the_fixture(self):
        _, quarantined = pool.digest_quarantine(_rows(), _actuals())
        counts = pool.quarantine_counts(quarantined)
        assert counts == {pool.QUARANTINE_NO_DIGEST: 2,
                          pool.QUARANTINE_DIGEST_MISMATCH: 1}
        assert sum(counts.values()) == len(quarantined)


# ── the two floors ───────────────────────────────────────────────────────────

class TestHeadIntegrityRefusal:
    def test_a_head_that_survives_intact_does_not_refuse(self):
        assert pool.head_integrity_refusal(
            head_version="v3", head_member_ids=["a", "b"], quarantined_ids=set(),
            floor=2) is None

    def test_losing_a_head_member_refuses_and_names_it(self):
        reason = pool.head_integrity_refusal(
            head_version="v3", head_member_ids=["a", "b"], quarantined_ids={"b"},
            floor=2)
        assert reason is not None
        assert "'b'" in reason or "b" in reason
        assert "v3" in reason

    def test_the_refusal_says_nothing_was_published_or_removed(self):
        """A refusal that does not say what it left behind reads like data loss to
        whoever meets it, and the first thing they will do is try to repair a pool
        that is fine."""
        reason = pool.head_integrity_refusal(
            head_version="v3", head_member_ids=["a"], quarantined_ids={"a"}, floor=1)
        assert "published" in reason and "removed" in reason

    def test_a_first_merge_with_no_head_is_not_a_refusal(self):
        """There is no composition to preserve, so this guard has nothing to say.
        Without the empty case it would refuse every brand-new pool."""
        assert pool.head_integrity_refusal(
            head_version=None, head_member_ids=[], quarantined_ids={"x"},
            floor=1) is None


class TestPublishFloorRefusal:
    def test_at_the_floor_it_publishes(self):
        assert pool.publish_floor_refusal(
            member_ids=["a", "b"], floor=2, quarantined=[]) is None

    def test_below_the_floor_it_refuses_with_the_counts(self):
        _, quarantined = pool.digest_quarantine(_rows(), _actuals())
        reason = pool.publish_floor_refusal(
            member_ids=["a"], floor=3, quarantined=quarantined)
        assert reason is not None
        assert "floor of 3" in reason
        assert pool.QUARANTINE_DIGEST_MISMATCH in reason

    def test_the_refusal_is_not_satisfied_by_quarantining_everything(self):
        """A negative control. ``member_ids=[]`` must refuse, not pass trivially —
        the empty soup is the one case where a thinness guard written as "fewer
        than before" would say nothing at all."""
        assert pool.publish_floor_refusal(
            member_ids=[], floor=1, quarantined=[]) is not None


# ── the published record ─────────────────────────────────────────────────────

class TestManifestCarriesTheQuarantine:
    def _manifest(self, quarantined):
        return pool.build_community_manifest(
            community_version="v2", combine="uniform",
            selected_contribution_ids=["c-good"],
            gate_split={"sha256": "ab"}, witness_split={"sha256": "cd"},
            selection_metric={"role": "selection"},
            witness_metric={"role": "witness"},
            members=[{"contribution_id": "c-good", "marginal_score": 0.5,
                      "gate_score": 0.5, "included": True}],
            quarantined=quarantined, architecture_signature="sig",
            weights_sha256=GOOD_SHA, created_at=1.0,
        )

    def test_the_manifest_records_every_quarantined_member(self):
        _, quarantined = pool.digest_quarantine(_rows(), _actuals())
        man = self._manifest(quarantined)
        assert {q["contribution_id"] for q in man["quarantined"]} == {
            "c-legacy", "c-null", "c-mismatch"}
        assert man["quarantine_counts"] == {pool.QUARANTINE_NO_DIGEST: 2,
                                            pool.QUARANTINE_DIGEST_MISMATCH: 1}

    def test_a_quarantined_member_is_not_filed_as_a_rejected_member(self):
        """``members`` means "evaluated, not included" and carries the scores that
        decision rests on. A quarantined member was never scored; filing it there
        would report an integrity failure as an ordinary rejection and hand the
        reader two null scores to interpret."""
        _, quarantined = pool.digest_quarantine(_rows(), _actuals())
        man = self._manifest(quarantined)
        assert {m["contribution_id"] for m in man["members"]} == {"c-good"}

    def test_a_clean_merge_still_states_that_it_was_clean(self):
        """The field is present and empty rather than absent. Absence would be
        indistinguishable from a writer that predates this change, which is the
        exact ambiguity the row was filed about."""
        man = self._manifest([])
        assert man["quarantined"] == []
        assert man["quarantine_counts"][pool.QUARANTINE_DIGEST_MISMATCH] == 0

    def test_the_manifest_cannot_be_written_without_deciding(self):
        """``quarantined`` has no default. An omitted list and an empty one would
        publish the identical manifest, so a caller that forgot to thread the
        quarantine through would attest "nothing was excluded" for a merge that
        excluded members — silently, and forever, in an immutable artifact."""
        with pytest.raises(TypeError, match="quarantined"):
            pool.build_community_manifest(
                community_version="v2", combine="uniform",
                selected_contribution_ids=[], gate_split={}, witness_split={},
                selection_metric={}, witness_metric={}, members=[],
                architecture_signature="sig", weights_sha256=GOOD_SHA,
                created_at=1.0,
            )
