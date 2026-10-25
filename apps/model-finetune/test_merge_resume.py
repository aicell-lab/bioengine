"""Tests for aggregate()'s resume journal — the clause-2 half of #0068.

A merge is ``2 * len(batch) + 3`` scoring passes and each pass is minutes of
exclusive GPU. The run that prompted this was killed at pass 11 of 35 and returned
nothing, losing 62 minutes. The journal records every score as it is earned so a
kill costs the pass in flight rather than the merge.

Scope, stated plainly: these drive the REAL ``pool`` helpers through the REAL call
pattern ``aggregate()`` makes, and ``test_no_scoring_call_bypasses_the_journal``
pins the wiring in ``entry.py`` at source level. What they do NOT do is execute
``aggregate()`` itself — it needs ``bioengine``, ``hypha_rpc``, ``pydantic``,
``yaml`` and ``httpx``, and a stub stack that deep would be testing the stubs. The
end-to-end proof is a deNBI run and is not claimed here.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import pool

GATE = {"artifact_id": "bioimage-io/splits", "sha256": "a" * 64}
WITNESS = {"artifact_id": "bioimage-io/splits", "sha256": "b" * 64}
DIAM = 30.0


# ── the key: a soup is a SET of members ──────────────────────────────────────

def test_score_key_is_order_independent():
    """A soup is a uniform mean, so [a, b] and [b, a] are the same checkpoint and
    must not be paid for twice. This is also what lets the greedy admission pass
    reuse the ordering pass's score for whichever candidate it takes first."""
    gate = pool.split_identity(GATE)
    assert pool.score_key(gate, ["a.pt", "b.pt"]) == pool.score_key(gate, ["b.pt", "a.pt"])


def test_score_key_separates_different_member_sets():
    """Discrimination control for the test above: a key that ignored its members
    entirely would also pass order-independence, and would serve one score for
    every soup."""
    gate = pool.split_identity(GATE)
    keys = {
        pool.score_key(gate, []),
        pool.score_key(gate, ["a.pt"]),
        pool.score_key(gate, ["b.pt"]),
        pool.score_key(gate, ["a.pt", "b.pt"]),
    }
    assert len(keys) == 4


def test_score_key_separates_the_two_splits():
    """Selection reads split A and the reported curve is split B. A key that did
    not separate them would serve a gate score as a witness score — the exact
    circularity the two-split design exists to prevent."""
    assert pool.score_key(pool.split_identity(GATE), ["a.pt"]) != \
        pool.score_key(pool.split_identity(WITNESS), ["a.pt"])


# ── invalidation: a score is only true for the inputs it was measured with ───

@pytest.mark.parametrize("field,value", [
    ("schema_version", "0.0.1-other"),
    ("gate_split", "bioimage-io/splits@" + "c" * 64),
    ("witness_split", "bioimage-io/splits@" + "d" * 64),
    ("diam_mean", 17.0),
])
def test_each_stamp_invalidates_the_journal(field, value):
    saved = pool.new_merge_journal(GATE, WITNESS, DIAM)
    pool.record_score(saved, "k", 0.5)
    saved[field] = value

    fresh = pool.usable_merge_journal(saved, GATE, WITNESS, DIAM)
    assert fresh["scores"] == {}, f"a changed {field} must discard saved scores"


def test_matching_stamps_keep_the_journal():
    """Positive control for the four above: if nothing kept a journal, they would
    all pass for a function that always returned empty."""
    saved = pool.new_merge_journal(GATE, WITNESS, DIAM)
    pool.record_score(saved, "k", 0.5)

    kept = pool.usable_merge_journal(saved, GATE, WITNESS, DIAM)
    assert kept["scores"] == {"k": 0.5}


@pytest.mark.parametrize("junk", [None, {}, {"scores": "not-a-dict"}, []])
def test_unusable_journals_degrade_to_empty(junk):
    """A missing or corrupt journal must cost the saved passes, never the merge."""
    assert pool.usable_merge_journal(junk, GATE, WITNESS, DIAM)["scores"] == {}


def test_record_score_evicts_oldest_first_at_the_cap():
    journal = pool.new_merge_journal(GATE, WITNESS, DIAM)
    for i in range(10):
        pool.record_score(journal, f"k{i}", i / 10, cap=4)

    assert list(journal["scores"]) == ["k6", "k7", "k8", "k9"]


# ── the behaviour the clause is about ────────────────────────────────────────

def _merge(members, journal, killed_at=None):
    """Replay aggregate()'s scoring sequence against a journal.

    Mirrors entry.py: baseline on both splits, the current head on gate, the
    best-first ordering pass (one per candidate), the greedy admission pass (one
    per candidate), then the new head on witness. Returns (scores_paid, result).
    ``killed_at`` raises once that many NEW scorings have been paid for, standing
    in for the OOM kill.
    """
    gate, wit = pool.split_identity(GATE), pool.split_identity(WITNESS)
    paid = []

    def score(paths, split):
        key = pool.score_key(split, paths)
        if key in journal["scores"]:
            return journal["scores"][key]
        if killed_at is not None and len(paid) == killed_at:
            raise RuntimeError("replica killed")
        # A real score; deterministic here so resume and restart are comparable.
        value = round(0.5 + len(paths) / 100, 6)
        paid.append(key)
        pool.record_score(journal, key, value)
        return value

    score([], gate)
    score([], wit)
    soup = []
    score(soup, gate)
    for m in members:
        score(soup + [m], gate)
    for m in members:
        score(soup + [m], gate)
        soup = soup + [m]
    head = score(soup, wit)
    return paid, head


def test_resume_after_a_kill_pays_only_for_the_passes_it_lost():
    """The clause: a kill mid-merge leaves a resumable checkpoint rather than
    returning nothing."""
    members = [f"m{i}.pt" for i in range(16)]

    journal = pool.new_merge_journal(GATE, WITNESS, DIAM)
    with pytest.raises(RuntimeError):
        _merge(members, journal, killed_at=11)
    assert len(journal["scores"]) == 11, "every pass paid for must be on disk"

    # Same pool, same splits — the second attempt reads the journal back.
    resumed = pool.usable_merge_journal(journal, GATE, WITNESS, DIAM)
    paid_on_resume, head = _merge(members, resumed)

    full_journal = pool.new_merge_journal(GATE, WITNESS, DIAM)
    paid_cold, head_cold = _merge(members, full_journal)

    assert head == head_cold, "resuming must not change the result"
    assert len(paid_on_resume) == len(paid_cold) - 11
    assert paid_on_resume == paid_cold[11:], "it must resume where it died, not elsewhere"


def test_without_a_journal_a_kill_costs_everything():
    """Discrimination control: the test above is only meaningful if the un-journaled
    merge really does lose its work. Restarting with a fresh journal re-pays every
    pass — that is the 62 minutes the issue is about."""
    members = [f"m{i}.pt" for i in range(16)]

    journal = pool.new_merge_journal(GATE, WITNESS, DIAM)
    with pytest.raises(RuntimeError):
        _merge(members, journal, killed_at=11)

    discarded = pool.new_merge_journal(GATE, WITNESS, DIAM)  # as if nothing persisted
    paid_restart, _ = _merge(members, discarded)
    paid_cold, _ = _merge(members, pool.new_merge_journal(GATE, WITNESS, DIAM))
    assert len(paid_restart) == len(paid_cold), "a lost journal re-pays every pass"


def test_the_ordering_pass_prepays_the_greedy_pass():
    """Both passes score ``current soup + candidate``; for the candidate admitted
    first those are the same set, so the second pass must be free. Measured here as
    the gap between the 2N+3 passes a naive merge runs and what the journal pays."""
    members = [f"m{i}.pt" for i in range(16)]
    paid, _ = _merge(members, pool.new_merge_journal(GATE, WITNESS, DIAM))
    assert len(paid) < 2 * len(members) + 3


# ── the wiring, pinned at source level ───────────────────────────────────────

def test_no_scoring_call_bypasses_the_journal():
    """``_score`` is the journal. A second call site reaching ``score_soup``
    directly would silently reintroduce an unresumable pass, and no unit test of
    ``pool`` could see it — so this reads entry.py.
    """
    source = (pathlib.Path(__file__).parent / "entry.py").read_text()
    calls = re.findall(r"runtime\.score_soup\(", source)
    assert len(calls) == 1, (
        f"expected exactly one runtime.score_soup call (inside _score, which "
        f"journals it); found {len(calls)}"
    )
    inside_score = source.split("async def _score(")[1].split("\n            async def ")[0]
    assert "runtime.score_soup(" in inside_score
    assert "pool.record_score(" in inside_score
    assert "_write_merge_journal(" in inside_score


# ── up-front admission (clause 1's second disjunct) ──────────────────────────

GIB = 1024 ** 3


def test_a_merge_that_fits_is_admitted():
    assert pool.soup_admission([2 * GIB, 1 * GIB], budget_bytes=12 * GIB,
                               base_bytes=3 * GIB) is None


def test_a_merge_that_cannot_fit_is_refused_with_the_numbers():
    reason = pool.soup_admission([3 * GIB], budget_bytes=8 * GIB, base_bytes=3 * GIB)
    assert reason is not None
    # The caller has to be able to act on it without reading this source.
    assert "12.0 GiB" in reason and "memory_mb" in reason


def test_refusal_sizes_the_largest_member_not_the_count():
    """Peak no longer depends on member count — the combine streams and the member
    cache is bounded — so sixteen small members must still be admitted where one
    large one is refused. A check that summed them would invert both."""
    many_small = [256 * 1024 ** 2] * 16
    assert pool.soup_admission(many_small, budget_bytes=12 * GIB, base_bytes=3 * GIB) is None
    assert pool.soup_admission([4 * GIB], budget_bytes=12 * GIB, base_bytes=3 * GIB) is not None


def test_the_base_footprint_is_counted():
    """Discrimination control: same members, same budget, only the replica's current
    RSS differs. A check that ignored it would admit both."""
    members = [3 * GIB]
    assert pool.soup_admission(members, budget_bytes=12 * GIB, base_bytes=0) is None
    assert pool.soup_admission(members, budget_bytes=12 * GIB, base_bytes=6 * GIB) is not None


@pytest.mark.parametrize("sizes", [[], [None], [None, None]])
def test_unknown_member_sizes_fail_open(sizes):
    """A pool that never recorded n_bytes tells us nothing about the peak. Refusing
    to merge on missing metadata would be a worse failure than the one this guards."""
    assert pool.soup_admission(sizes, budget_bytes=1, base_bytes=0) is None


def test_a_known_size_is_used_even_when_others_are_missing():
    """Positive control for failing open: partial metadata must not disable it."""
    assert pool.soup_admission([None, 8 * GIB], budget_bytes=12 * GIB,
                               base_bytes=0) is not None


def test_the_refusal_happens_before_any_scoring_pass():
    """"Rejected up front" is the clause. A check that ran after the baseline passes
    would already have spent GPU time, so the ORDER is the property — pinned at
    source because no unit test of pool can see it."""
    source = (pathlib.Path(__file__).parent / "entry.py").read_text()
    assert source.index("pool.soup_admission(") < source.index("await _score(")
