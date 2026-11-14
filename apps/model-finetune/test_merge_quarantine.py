"""#0139 end to end — what a merge DOES when a member cannot be vouched for.

``test_pool_quarantine.py`` pins the pure split. This file drives the real
``aggregate()`` coroutine against a stub artifact manager and a stub GPU runtime
that is allowed to DISAGREE with the pool index, which is the only way to reach the
mismatch case at all: no fixture can make a real sha256 come out wrong, so the
disagreement has to be injected at the one boundary where the bytes are hashed.

What is actually asserted here, rather than in the unit file:

- the merge proceeds and publishes — the ruled behaviour is quarantine, not refusal;
- a quarantined candidate reaches neither the soup nor the ``members`` record, and
  the published ``selected_contribution_ids`` does not contain it;
- it IS marked decided, so the next merge does not re-weigh it forever;
- a head member that fails verification stops the merge with NOTHING committed;
- the counts are stated on the result and in the manifest even when zero.

The stub runtime returns one constant score for every pass, so whether a candidate
is admitted is a pure function of that one number against the recorded baseline.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pool  # noqa: E402

POOL_ID = "bioimage-io/cpsam-soup-demo-pool"
GATE_SPLIT = {"artifact_id": "bioimage-io/val-gate", "sha256": "c" * 64}
WITNESS_SPLIT = {"artifact_id": "bioimage-io/val-witness", "sha256": "d" * 64}
HEAD_SHA = "e" * 64
SHA_GOOD = "1" * 64
SHA_OTHER = "2" * 64
SHA_HEAD_MEMBER = "3" * 64
WRONG = "9" * 64
T0 = 1_700_000_000.0

# Large enough that pool.soup_admission never refuses: a memory refusal would exit
# before the quarantine this file is about.
_BUDGET = {"declared_bytes": 64 * 1024 ** 3, "rss_bytes": 1024 ** 3}


class _Resp:
    def __init__(self, payload: Any) -> None:
        self.content = json.dumps(payload).encode()

    def raise_for_status(self) -> None:
        return None


class _AM:
    async def get_file(self, artifact_id: str, file_path: str) -> str:
        return "https://example.invalid/get/" + file_path

    async def put_file(self, artifact_id: str, file_path: str) -> str:
        return "https://example.invalid/put/" + file_path


class _Runtime:
    """The GPU deployment, reduced to the four calls ``aggregate()`` makes.

    ``actual`` maps a member's pool path to the digest hashing its bytes on the
    replica returns. A path missing from it falls back to the digest the spec
    carries — i.e. "the bytes are what the pool says they are" — so a test only has
    to name the members it wants to go wrong.
    """

    def __init__(self, f1: float, actual: Optional[Dict[str, str]] = None) -> None:
        self._f1 = f1
        self._actual = dict(actual or {})
        self.verified: List[str] = []

    async def soup_memory_budget(self) -> Dict[str, Any]:
        return dict(_BUDGET)

    async def verify_members(self, specs) -> Dict[str, Any]:
        self.verified.extend(s["key"] for s in specs)
        return {
            "digests": {s["key"]: self._actual.get(s["key"], s["sha256"])
                        for s in specs},
            "replica": "r0", "soup_dl_bytes": 0,
        }

    async def score_soup(self, specs, imgs, gts, diam_mean) -> Dict[str, Any]:
        return {"f1": self._f1, "replica": "r0", "soup_dl_bytes": 0}

    async def materialize_soup(self, specs, url) -> Dict[str, Any]:
        return {"sha256": HEAD_SHA, "n_bytes": 2048,
                "replica": "r0", "soup_dl_bytes": 0}


class _Merging:
    """Just enough of ``EntryApp`` for ``aggregate`` to run a merge end to end.

    Captures every PUT body by path, so the published manifest is read back as the
    bytes that were actually sent rather than as the dict that was built.
    """

    def __init__(self, index: Dict[str, Any], runtime: _Runtime,
                 manifests: Optional[Dict[str, Any]] = None) -> None:
        self._pool_locks: Dict[str, Any] = {}
        self._merges_in_flight = 0
        self.index = index
        self._runtime = runtime
        self.commits = 0
        self.put: Dict[str, Any] = {}
        self._manifests = manifests or {}

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

        return _AM(), _Client()

    async def _read_pool_index(self, am, pool_artifact_id: str) -> Dict[str, Any]:
        return self.index

    async def _http_retry(self, method: str, url: str, **kw: Any):
        path = url.split("/put/", 1)[-1] if "/put/" in url else url.split("/get/", 1)[-1]
        if method == "PUT":
            self.put[path] = json.loads(kw["content"].decode())
            return _Resp({})
        return _Resp(self._manifests.get(path, {}))

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
    import entry

    return entry.EntryApp.func_or_class.aggregate.__get__(instance)


def _contrib(cid: str, sha: Any, *, with_key: bool = True) -> Dict[str, Any]:
    rec = {"contribution_id": cid, "weights_path": f"contributions/{cid}/weights.pt",
           "n_bytes": 4096, "uploaded_at": T0}
    if with_key:
        rec["weights_sha256"] = sha
    return rec


def _index(contributions: List[Dict[str, Any]], **extra) -> Dict[str, Any]:
    """A cpsam pool with a pre-recorded baseline, so aggregate() does not spend two
    runtime passes measuring stock cpsam before reaching anything under test."""
    idx = {
        "model_type": "cpsam",
        "architecture_signature": "sig-1",
        "baseline": {
            "model": "stock-cpsam",
            "selection_metric": {"role": "selection", "aggregate": 0.50},
            "witness_metric": {"role": "witness", "aggregate": 0.40},
            "created_at": T0,
        },
        "contributions": contributions,
    }
    idx.update(extra)
    return idx


async def _merge(stub, **kw):
    return await _aggregate_of(stub)(
        pool_artifact_id=POOL_ID, pool_token="t",
        gate_val_images=[], gate_val_labels=[],
        witness_val_images=[], witness_val_labels=[],
        gate_split=dict(GATE_SPLIT), witness_split=dict(WITNESS_SPLIT), **kw)


# ── the ruled behaviour: exclude the member, publish the rest ────────────────

@pytest.mark.asyncio
async def test_a_digestless_candidate_is_excluded_and_the_merge_still_publishes():
    """The whole decision, in one assertion pair. The earlier attempt at this row
    aborted the merge on a digest-less member, which on a pool carrying one legacy
    record means the campaign never merges again."""
    stub = _Merging(_index([_contrib("c-good", SHA_GOOD),
                            _contrib("c-legacy", None, with_key=False)]),
                    _Runtime(0.90))
    result = await _merge(stub)

    assert result["published_version"] == "v2"
    assert result["admitted"] == ["c-good"]
    assert [q["contribution_id"] for q in result["quarantined"]] == ["c-legacy"]
    assert result["quarantine_counts"][pool.QUARANTINE_NO_DIGEST] == 1


@pytest.mark.asyncio
async def test_the_excluded_member_is_not_in_the_published_head():
    """The failure mode this row is about: a member whose bytes nothing checked
    ending up inside the published checkpoint. ``selected_contribution_ids`` is the
    permanent record of what was souped, in an immutable artifact."""
    stub = _Merging(_index([_contrib("c-good", SHA_GOOD),
                            _contrib("c-legacy", None)]),
                    _Runtime(0.90))
    await _merge(stub)
    manifest = stub.put["community/v2/manifest.json"]
    assert manifest["selected_contribution_ids"] == ["c-good"]
    assert "c-legacy" not in manifest["selected_contribution_ids"]


@pytest.mark.asyncio
async def test_the_excluded_member_is_not_reported_as_an_ordinary_rejection():
    """"Evaluated, not included" is a statement about a candidate that was scored.
    A quarantined member was never scored, so reporting it under ``rejected`` would
    tell a contributor their weights lost on merit."""
    stub = _Merging(_index([_contrib("c-good", SHA_GOOD),
                            _contrib("c-legacy", None)]),
                    _Runtime(0.90))
    result = await _merge(stub)
    assert "c-legacy" not in result["rejected"]
    assert "c-legacy" not in [m["contribution_id"]
                              for m in stub.put["community/v2/manifest.json"]["members"]]


@pytest.mark.asyncio
async def test_a_quarantined_candidate_is_marked_decided():
    """Otherwise it is re-fetched and re-flagged on every future merge, and because
    undecided_contributions is chronological and max_batch caps the batch, a growing
    head of permanently-quarantined rows starves real candidates out of every batch.
    The trade is deliberate: "kept in the pool" means the record and the weights
    stay in the artifact, not that the merge keeps re-weighing it."""
    stub = _Merging(_index([_contrib("c-good", SHA_GOOD),
                            _contrib("c-legacy", None)]),
                    _Runtime(0.90))
    await _merge(stub)
    assert set(stub.index["decided_contribution_ids"]) == {"c-good", "c-legacy"}


@pytest.mark.asyncio
async def test_a_digestless_member_is_never_downloaded_to_be_rejected():
    """It is quarantined from the index alone. Hashing it would move a gigabyte
    across the network to produce a number with nothing to compare against."""
    rt = _Runtime(0.90)
    stub = _Merging(_index([_contrib("c-good", SHA_GOOD),
                            _contrib("c-legacy", None)]), rt)
    await _merge(stub)
    assert rt.verified == ["contributions/c-good/weights.pt"]


# ── the mismatch is a different finding ──────────────────────────────────────

@pytest.mark.asyncio
async def test_bytes_that_do_not_hash_to_the_recorded_digest_are_an_incident():
    """The case that produced NO observable difference before this change: the
    digest is recorded, the bytes disagree with it, and the old merge souped them
    anyway because nothing ever read the digest back."""
    rt = _Runtime(0.90, actual={"contributions/c-bad/weights.pt": WRONG})
    stub = _Merging(_index([_contrib("c-good", SHA_GOOD),
                            _contrib("c-bad", SHA_OTHER)]), rt)
    result = await _merge(stub)

    bad = next(q for q in result["quarantined"] if q["contribution_id"] == "c-bad")
    assert bad["reason"] == pool.QUARANTINE_DIGEST_MISMATCH
    assert bad["expected_sha256"] == SHA_OTHER
    assert bad["actual_sha256"] == WRONG
    assert result["admitted"] == ["c-good"]


@pytest.mark.asyncio
async def test_a_mismatch_does_not_report_as_a_missing_digest():
    """Both land in ``quarantined``; only the reason tells them apart, and it is
    what a reader filtering an immutable manifest for integrity incidents has to
    filter on. One shared code would bury every incident among the legacy records."""
    rt = _Runtime(0.90, actual={"contributions/c-bad/weights.pt": WRONG})
    stub = _Merging(_index([_contrib("c-bad", SHA_OTHER),
                            _contrib("c-legacy", None)]), rt)
    result = await _merge(stub)
    by_id = {q["contribution_id"]: q["reason"] for q in result["quarantined"]}
    assert by_id == {"c-bad": pool.QUARANTINE_DIGEST_MISMATCH,
                     "c-legacy": pool.QUARANTINE_NO_DIGEST}
    assert result["quarantine_counts"] == {pool.QUARANTINE_NO_DIGEST: 1,
                                           pool.QUARANTINE_DIGEST_MISMATCH: 1}


# ── the head is checked too, and losing it is fatal ──────────────────────────

def _with_head(members: List[str], contributions: List[Dict[str, Any]]):
    idx = _index(contributions, current_community={
        "version": "v1", "path": "community/v1/weights.pt", "sha256": HEAD_SHA,
        "witness_metric": {"role": "witness", "aggregate": 0.60},
    }, decided_contribution_ids=list(members))
    manifests = {"community/v1/manifest.json": {
        "selected_contribution_ids": list(members)}}
    return idx, manifests


@pytest.mark.asyncio
async def test_a_head_member_that_fails_verification_stops_the_merge():
    """Not a thinner soup — a different one. Every number this merge would then
    report about "the current head" (its gate score, the bar a candidate must
    clear, the witness point carried forward) would describe a soup that is not the
    published checkpoint, and there is no honest way to continue."""
    idx, man = _with_head(["h-1", "h-2"],
                          [_contrib("h-1", SHA_HEAD_MEMBER),
                           _contrib("h-2", SHA_HEAD_MEMBER),
                           _contrib("c-good", SHA_GOOD)])
    rt = _Runtime(0.90, actual={"contributions/h-2/weights.pt": WRONG})
    stub = _Merging(idx, rt, man)
    with pytest.raises(ValueError, match="refused this merge"):
        await _merge(stub)


@pytest.mark.asyncio
async def test_that_refusal_publishes_nothing_and_commits_nothing():
    """A refusal that had already written half a merge would be worse than the bug.
    Checked by counting commits, not by reading the code path."""
    idx, man = _with_head(["h-1"], [_contrib("h-1", SHA_HEAD_MEMBER),
                                    _contrib("c-good", SHA_GOOD)])
    stub = _Merging(idx, _Runtime(0.90, actual={"contributions/h-1/weights.pt": WRONG}), man)
    with pytest.raises(ValueError):
        await _merge(stub)
    assert stub.commits == 0
    assert stub.put == {}
    # The pending candidate is left pending — the refusal does not burn it. The
    # pool already carried the head's own id in this list, so the assertion is on
    # what the merge ADDED, not on the key being absent.
    assert stub.index["decided_contribution_ids"] == ["h-1"]


@pytest.mark.asyncio
async def test_an_intact_head_merges_normally():
    """The control for the three tests above. Without it they are all satisfiable
    by a change that refuses every merge against any head at all."""
    idx, man = _with_head(["h-1"], [_contrib("h-1", SHA_HEAD_MEMBER),
                                    _contrib("c-good", SHA_GOOD)])
    stub = _Merging(idx, _Runtime(0.90), man)
    result = await _merge(stub)
    assert result["published_version"] == "v2"
    assert result["quarantined"] == []
    assert stub.commits > 0


# ── the floor ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_explicit_floor_refuses_rather_than_publishing_a_thin_soup():
    """The caller states how many members the published head must have; a merge
    that cannot reach it fails instead of publishing a smaller one quietly."""
    stub = _Merging(_index([_contrib("c-good", SHA_GOOD),
                            _contrib("c-legacy", None)]),
                    _Runtime(0.90))
    with pytest.raises(ValueError, match="floor of 2"):
        await _merge(stub, min_members=2)


@pytest.mark.asyncio
async def test_the_same_merge_publishes_at_the_floor_it_can_meet():
    """The discrimination control for the test above: the refusal has to be about
    the NUMBER, not about ``min_members`` having been passed at all."""
    stub = _Merging(_index([_contrib("c-good", SHA_GOOD),
                            _contrib("c-legacy", None)]),
                    _Runtime(0.90))
    result = await _merge(stub, min_members=1)
    assert result["published_version"] == "v2"


# ── a clean merge still says so ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_clean_merge_states_the_zero_rather_than_omitting_it():
    """Every live pool measured for this row is clean, so this is the shape almost
    every real merge produces. An absent field would read identically to a merge
    written by a build that predates the check — the exact ambiguity the row is
    about — so the counts are present and zero."""
    stub = _Merging(_index([_contrib("c-good", SHA_GOOD)]), _Runtime(0.90))
    result = await _merge(stub)
    assert result["quarantined"] == []
    assert result["n_quarantined"] == 0
    assert result["quarantine_counts"] == {pool.QUARANTINE_NO_DIGEST: 0,
                                           pool.QUARANTINE_DIGEST_MISMATCH: 0}
    manifest = stub.put["community/v2/manifest.json"]
    assert manifest["quarantined"] == []
    assert manifest["quarantine_counts"][pool.QUARANTINE_DIGEST_MISMATCH] == 0


@pytest.mark.asyncio
async def test_the_merge_record_carries_the_quarantine_too():
    """The manifest describes one published version; the merge log is where a
    reader reconstructs what happened over time, including merges that published
    nothing. A quarantine recorded in only one of the two is invisible in whichever
    the reader happens to open."""
    rt = _Runtime(0.90, actual={"contributions/c-bad/weights.pt": WRONG})
    stub = _Merging(_index([_contrib("c-good", SHA_GOOD),
                            _contrib("c-bad", SHA_OTHER)]), rt)
    await _merge(stub)
    merge = stub.index["merges"][-1]
    assert [q["contribution_id"] for q in merge["quarantined"]] == ["c-bad"]
    assert merge["quarantine_counts"][pool.QUARANTINE_DIGEST_MISMATCH] == 1


@pytest.mark.asyncio
async def test_a_merge_that_admits_nothing_still_records_the_quarantine():
    """``f1`` below the baseline admits nobody, so nothing is published and the
    manifest is never written. The quarantine still has to land somewhere, or a
    pool whose candidates all fail the gate accumulates unverifiable members with
    no record that anyone noticed."""
    rt = _Runtime(0.10, actual={"contributions/c-bad/weights.pt": WRONG})
    stub = _Merging(_index([_contrib("c-bad", SHA_OTHER)]), rt)
    result = await _merge(stub)
    assert result["published_version"] is None
    assert [q["contribution_id"] for q in result["quarantined"]] == ["c-bad"]
    assert stub.index["merges"][-1]["quarantine_counts"][
        pool.QUARANTINE_DIGEST_MISMATCH] == 1
