"""Re-depositing a checkpoint the pool already holds must not append a second
member — and must not wedge the pool artifact on the way out.

The defect (#0376): ``contribute`` appended unconditionally. A client whose
response was eaten by a gateway timeout retries, which is the correct thing for
it to do — 4 of 92 deposits on the r2 corpus needed it, with the 504 threshold
bracketed at 94.4–100.8 s — and the pool then held the same bytes twice. The
soup is a *uniform* key-wise mean, so that site is silently weighted 2:1, and
nothing on the page shows it: both members carry valid provenance and distinct
contribution ids.

Identity is the weights digest ALONE. Metadata cannot make two copies of one
checkpoint into two deposits; it can only hide the double weight. Comparing a
caller-supplied field as well (``dataset_fingerprint``) would let a retry defeat
de-duplication by accident.

Two things are asserted that the row's own clause does not ask for, because
"one member remains" is satisfied by a run that left the artifact unusable:

* the stage was DISCARDED, and
* no index PUT and no commit were issued at all.

``put_file`` mints a manifest row when it hands out the URL, not when the bytes
land, and ``_staged`` discards only on ``BaseException`` — so returning normally
from inside its block would leave a row with no object behind it, after which
every later ``commit`` raises ``File '<path>' does not exist in the artifact``
and ``_commit_with_retry`` spends ~62 s misreading that as the read-after-write
404. One retried deposit would take the whole pool down, which is strictly worse
than the double-weighting being fixed. That is why the duplicate path leaves by
raising a sentinel, and why these tests pin the discard rather than the member
count alone.

Run against the real ``contribute`` coroutine with its collaborators stubbed and
the REAL ``_staged`` bound to the stub. No GPU, no torch, no live Hypha.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import pytest

import pool

_METADATA_URL = "https://example.invalid/put/metadata"
_INDEX_URL = "https://example.invalid/put/index"
_WEIGHTS_URL = "https://example.invalid/put/weights"

_SESSION = "2026-10-25-020700"
_SIG = "sig-1"
_DIGEST = "a" * 64
_OTHER_DIGEST = "c" * 64
_FIRST_ID = "first-deposit-id"


def _member(contribution_id: str, digest: str, **over: Any) -> Dict[str, Any]:
    """An index member of the shape ``append_contribution`` writes."""
    entry = {
        "contribution_id": contribution_id,
        "weights_path": f"contributions/{contribution_id}/weights.pt",
        "metadata_path": f"contributions/{contribution_id}/metadata.json",
        "weights_sha256": digest,
        "architecture_signature": _SIG,
        "samples_seen": 10,
        "n_train_images": 5,
        "n_bytes": 1218617275,
        "uploaded_at": 1761350000.0,
    }
    entry.update(over)
    return entry


def _index(*members: Dict[str, Any]) -> Dict[str, Any]:
    idx = pool.empty_index("cpsam", _SIG)
    idx["contributions"] = list(members)
    return idx


# --------------------------------------------------------------------------
# The predicate, in isolation.
# --------------------------------------------------------------------------


def test_finds_the_member_holding_the_digest() -> None:
    first = _member(_FIRST_ID, _DIGEST)
    assert pool.find_contribution_by_digest(_index(first), _DIGEST) is first


def test_absent_digest_is_none() -> None:
    assert pool.find_contribution_by_digest(_index(_member(_FIRST_ID, _DIGEST)), _OTHER_DIGEST) is None


def test_empty_index_is_none() -> None:
    assert pool.find_contribution_by_digest(pool.empty_index("cpsam", _SIG), _DIGEST) is None
    assert pool.find_contribution_by_digest({}, _DIGEST) is None


def test_empty_digest_never_matches() -> None:
    """A member written without a digest must not be matched by a caller that also
    has none — that would de-duplicate two genuinely different checkpoints into
    one, which loses weights rather than double-counting them."""
    assert pool.find_contribution_by_digest(_index(_member(_FIRST_ID, "")), "") is None


def test_the_predicate_ignores_metadata_it_is_not_given() -> None:
    """All the predicate can see is the digest, which is the whole of the rule — so
    a member carrying any amount of extra metadata still matches on the digest.

    This is deliberately NOT the decision-1 test. The predicate takes no metadata
    argument, so no call of this shape can pin "identity is the digest ALONE": the
    claim is about what ``contribute`` compares, and it is pinned end to end by
    ``test_a_retry_reporting_different_metadata_is_still_one_deposit`` below. An
    earlier version of this case asserted the decision in its docstring while
    passing ``samples_seen=10, n_train_images=5`` — ``_member``'s own defaults —
    so its "differences" differed from nothing and it could not fail for the
    reason it gave."""
    first = _member(_FIRST_ID, _DIGEST, samples_seen=777, n_train_images=99, dataset_fingerprint="dataset-A")
    assert pool.find_contribution_by_digest(_index(first), _DIGEST) is first


def test_an_already_merged_member_is_still_recognised() -> None:
    """``decided_contribution_ids`` is deliberately not consulted: appending there
    is the WORST case, since the next merge compounds a weight that already
    shipped in a published community checkpoint."""
    first = _member(_FIRST_ID, _DIGEST)
    idx = _index(first)
    idx["decided_contribution_ids"] = [_FIRST_ID]
    assert pool.find_contribution_by_digest(idx, _DIGEST) is first


def test_earliest_match_wins_on_a_pool_that_already_holds_a_duplicate() -> None:
    """Pools written before this fix may already hold two members for one digest.
    The first is returned, so the answer does not depend on how many copies the
    pool happens to carry."""
    first = _member(_FIRST_ID, _DIGEST)
    second = _member("second-deposit-id", _DIGEST)
    assert pool.find_contribution_by_digest(_index(first, second), _DIGEST) is first


# --------------------------------------------------------------------------
# The backstop at the write site.
# --------------------------------------------------------------------------


def test_append_refuses_a_duplicate_digest() -> None:
    idx = _index(_member(_FIRST_ID, _DIGEST))
    with pytest.raises(ValueError) as err:
        pool.append_contribution(idx, _member("retry-id", _DIGEST))
    msg = str(err.value)
    assert _FIRST_ID in msg, "the refusal must name the member already holding the digest"
    assert len(idx["contributions"]) == 1, "the refused append must not have mutated the index"
    assert idx["contributions"][0]["contribution_id"] == _FIRST_ID


def test_append_still_accepts_a_different_checkpoint() -> None:
    """Paired with the refusal above so the guard cannot pass by refusing
    everything — a dedup that appends nothing would satisfy every member-count
    assertion in this file on its own."""
    idx = _index(_member(_FIRST_ID, _DIGEST))
    pool.append_contribution(idx, _member("second-deposit-id", _OTHER_DIGEST))
    assert [c["contribution_id"] for c in idx["contributions"]] == [_FIRST_ID, "second-deposit-id"]


def test_append_bootstraps_a_pool_with_no_contributions_key() -> None:
    idx: Dict[str, Any] = {}
    pool.append_contribution(idx, _member(_FIRST_ID, _DIGEST))
    assert len(idx["contributions"]) == 1


# --------------------------------------------------------------------------
# The whole contribute path.
# --------------------------------------------------------------------------


def _contribute_of(instance) -> Any:
    """The real ``contribute`` coroutine, bound to a stub instance.

    ``entry.EntryApp`` is a ``ray.serve.deployment.Deployment`` once
    ``@bioengine.app`` has run, so its methods are not reachable through it;
    ``func_or_class`` is the class the decorator wrapped.
    """
    import entry

    return entry.EntryApp.func_or_class.contribute.__get__(instance)


@pytest.fixture
def fake_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """A COMPLETED cpsam session. Stubbed rather than created on disk: this test is
    about what the pool index ends up holding, not about reading a real session."""
    mod = types.ModuleType("training")
    mod.get_status = lambda sid: {
        "status": "COMPLETED", "model_type": "cpsam",
        "n_epochs_completed": 2, "n_train_inputs": 5,
    }
    mod.session_backend = lambda sid: "cellpose"
    mod.read_training_params = lambda sid: {"model_type": "cpsam", "init_checkpoint": "stock"}
    monkeypatch.setitem(sys.modules, "training", mod)


class _Runtime:
    """Returns the digest of the bytes it just PUT — the real one computes it after
    the upload, which is why no check can run earlier than this call."""

    def __init__(self, digest: str) -> None:
        self._digest = digest

    async def prepare_contribution(self, session_id: str, weights_put_url: str) -> Dict[str, Any]:
        return {
            "architecture_signature": _SIG,
            "base_checkpoint_sha256": "b" * 64,
            "cellpose_version": "4.0.1",
            "l2_from_base": 0.5,
            "weights_sha256": self._digest,
            "n_bytes": 1218617275,
        }


class _Stub:
    """Just enough of ``EntryApp`` for ``contribute`` to reach a return, recording
    every artifact-mutating call so a wedged stage is visible to the assertions."""

    def __init__(self, index: Optional[Dict[str, Any]], digest: str) -> None:
        self.index = index
        self.digest = digest
        self.puts: List[Tuple[str, bytes]] = []
        self.edits: List[str] = []
        self.discards: List[str] = []
        self.commits: List[str] = []
        self.transport: List[Dict[str, int]] = []

    async def _check_runtime_available(self, model_type: str) -> None:
        return None

    def _runtime_for(self, model_type: str) -> object:
        return _Runtime(self.digest)

    async def _pool_service(self, pool_token: str):
        outer = self

        class _Client:
            async def disconnect(self) -> None:
                return None

        class _Am:
            async def edit(self, artifact_id: str, *a: Any, **k: Any) -> None:
                outer.edits.append(artifact_id)

            async def discard(self, artifact_id: str, *a: Any, **k: Any) -> None:
                outer.discards.append(artifact_id)

            async def put_file(self, artifact_id: str, file_path: str) -> str:
                if file_path.endswith("metadata.json"):
                    return _METADATA_URL
                if "weights" in file_path:
                    return _WEIGHTS_URL
                return _INDEX_URL

        return _Am(), _Client()

    async def _read_pool_index(self, am: Any, pool_artifact_id: str) -> Dict[str, Any]:
        if self.index is None:
            raise FileNotFoundError("fresh pool")
        return self.index

    async def _http_retry(self, method: str, url: str, content: bytes = b"", **kw: Any):
        self.puts.append((url, content))

        class _Resp:
            def raise_for_status(self) -> None:
                return None

        return _Resp()

    async def _commit_with_retry(self, am: Any, pool_artifact_id: str) -> None:
        self.commits.append(pool_artifact_id)

    def _staged(self, am: Any, artifact_id: str):
        """The REAL staging context manager, bound here rather than faked.

        A hand-rolled double would keep passing if it stopped discarding, which is
        exactly the failure this file exists to rule out.
        """
        import entry

        return entry.EntryApp.func_or_class._staged.__get__(self)(am, artifact_id)

    def _bump_transport(self, **kw: int) -> None:
        self.transport.append(dict(kw))


async def _deposit(stub: _Stub, **over: Any) -> Dict[str, Any]:
    """``**over`` so a case can vary what the CALLER reports — the optional
    ``dataset_fingerprint`` is the field a retrying client controls outright, and
    no case can pin decision 1 without being able to change it between deposits."""
    kwargs: Dict[str, Any] = {
        "session_id": _SESSION,
        "pool_artifact_id": "bioimage-io/cpsam-community-pool",
        "pool_token": "t",
    }
    kwargs.update(over)
    return await _contribute_of(stub)(**kwargs)


def _index_puts(stub: _Stub) -> List[bytes]:
    return [c for url, c in stub.puts if url == _INDEX_URL]


@pytest.mark.asyncio
async def test_a_redeposit_leaves_exactly_one_member(fake_training: None) -> None:
    """The row's clause 2, first half."""
    idx = _index(_member(_FIRST_ID, _DIGEST))
    stub = _Stub(idx, _DIGEST)
    await _deposit(stub)
    assert len(idx["contributions"]) == 1, (
        f"the pool gained a second member for one checkpoint: "
        f"{[c['contribution_id'] for c in idx['contributions']]}"
    )


@pytest.mark.asyncio
async def test_a_redeposit_leaves_the_first_member_intact(fake_training: None) -> None:
    """Clause 2's other half, paired with the count: a dedup that DELETED the
    member would also leave 'exactly one' false-green on an empty pool, and would
    satisfy the count assertion above on a pool of two."""
    first = _member(_FIRST_ID, _DIGEST)
    idx = _index(first, _member("unrelated", _OTHER_DIGEST))
    await _deposit(_Stub(idx, _DIGEST))
    held = {c["contribution_id"]: c for c in idx["contributions"]}
    assert set(held) == {_FIRST_ID, "unrelated"}
    assert held[_FIRST_ID]["weights_sha256"] == _DIGEST
    assert held[_FIRST_ID]["uploaded_at"] == first["uploaded_at"], (
        "the first deposit's record is immutable; a retry must not restamp it"
    )


@pytest.mark.asyncio
async def test_a_redeposit_returns_the_first_deposits_id(fake_training: None) -> None:
    """Returned, not raised: the client is retrying because a 504 ate its response,
    and a raise leaves it in exactly the position the 504 did."""
    returned = await _deposit(_Stub(_index(_member(_FIRST_ID, _DIGEST)), _DIGEST))
    assert returned["contribution_id"] == _FIRST_ID
    assert returned["deduplicated"] is True
    assert returned["weights_sha256"] == _DIGEST


@pytest.mark.asyncio
async def test_a_retry_reporting_different_metadata_is_still_one_deposit(
    fake_training: None,
) -> None:
    """Decision 1 — identity is the weights digest ALONE — pinned where the decision
    is actually made, in ``contribute``.

    The retry reports a different ``dataset_fingerprint`` than the deposit it is
    retrying, and the member the pool already holds reports a different
    ``samples_seen`` than this session computes (777 against ``5 * 2 = 10``). Both
    are metadata; neither may make these identical bytes into a second deposit,
    and the value handed back must be the pool's, not this call's.

    ``dataset_fingerprint`` is the field that makes this more than hypothetical: it
    is an optional RPC parameter the CALLER supplies, so a retrying client can
    report anything at all. Folding it into identity is the bypass this file's
    header warns about — a retry would defeat de-duplication by accident and the
    uniform soup would weight that site 2:1."""
    stored = _member(_FIRST_ID, _DIGEST, samples_seen=777, dataset_fingerprint="dataset-A")
    idx = _index(stored)
    stub = _Stub(idx, _DIGEST)

    returned = await _deposit(stub, dataset_fingerprint="dataset-B")

    assert len(idx["contributions"]) == 1, (
        f"a retry reporting different metadata appended a second member for one "
        f"checkpoint: {[c['contribution_id'] for c in idx['contributions']]}"
    )
    assert returned["deduplicated"] is True
    assert returned["contribution_id"] == _FIRST_ID, (
        "the retry must get the FIRST deposit's id back, not a fresh one"
    )
    assert returned["samples_seen"] == 777, (
        f"samples_seen must be read back off the stored member ({stored['samples_seen']}), "
        f"not recomputed from this call ({returned['samples_seen']}) — the stored member is "
        f"the one the soup will weight"
    )
    assert stub.discards == ["bioimage-io/cpsam-community-pool"]
    assert _index_puts(stub) == []


@pytest.mark.asyncio
async def test_two_real_deposits_of_one_checkpoint_leave_one_member(
    fake_training: None,
) -> None:
    """The same decision again with nothing hand-built: both deposits go through the
    real ``contribute``, so the member the second one matches against was written by
    the first rather than assembled by ``_member``.

    Every other case here models the retry by pre-loading an index, which cannot
    catch a mismatch between the shape ``append_contribution`` writes and the shape
    the predicate reads. This one can."""
    idx = _index()
    first = await _deposit(_Stub(idx, _DIGEST), dataset_fingerprint="dataset-A")
    assert first["deduplicated"] is False, "the first deposit into an empty pool is fresh"

    stub = _Stub(idx, _DIGEST)
    retry = await _deposit(stub, dataset_fingerprint="dataset-B")

    assert len(idx["contributions"]) == 1, (
        f"two deposits of one checkpoint left {len(idx['contributions'])} members"
    )
    assert retry["deduplicated"] is True
    assert retry["contribution_id"] == first["contribution_id"]
    assert retry["samples_seen"] == first["samples_seen"]
    assert stub.commits == [], "the retry must not commit a new pool version"


@pytest.mark.asyncio
async def test_a_fresh_deposit_says_deduplicated_false_rather_than_omitting_it(
    fake_training: None,
) -> None:
    """A key present only when true cannot be dated: a client holding a response
    without it cannot tell 'fresh deposit' from 'a release that predates
    de-duplication'. That is the defect ``app_version`` exists to avoid (#0116,
    #0140 — five of 92 saved responses were undatable)."""
    returned = await _deposit(_Stub(_index(_member(_FIRST_ID, _OTHER_DIGEST)), _DIGEST))
    assert "deduplicated" in returned, (
        "contribute's return value is the only record the depositing site keeps"
    )
    assert returned["deduplicated"] is False


@pytest.mark.asyncio
async def test_both_return_paths_have_the_same_shape(fake_training: None) -> None:
    """A retry must not hand the client a differently-shaped record than the
    deposit it is retrying."""
    fresh = await _deposit(_Stub(_index(_member(_FIRST_ID, _OTHER_DIGEST)), _DIGEST))
    dup = await _deposit(_Stub(_index(_member(_FIRST_ID, _DIGEST)), _DIGEST))
    assert set(fresh) == set(dup), f"fresh-only={set(fresh) - set(dup)} dup-only={set(dup) - set(fresh)}"


@pytest.mark.asyncio
async def test_a_redeposit_discards_the_stage_and_writes_nothing(fake_training: None) -> None:
    """The failure mode the row does not name, and the one that would be worse than
    the bug: ``put_file`` minted the duplicate's weights row inside the stage
    before the digest could be known, so leaving the block without discarding
    makes every later commit on this pool fail for ~62 s and then raise."""
    stub = _Stub(_index(_member(_FIRST_ID, _DIGEST)), _DIGEST)
    await _deposit(stub)
    assert stub.discards == ["bioimage-io/cpsam-community-pool"], (
        f"the stage was not discarded (discards={stub.discards}); a put_file row "
        f"with no object behind it leaves the pool permanently uncommittable"
    )
    assert stub.commits == [], "a recognised re-deposit must not commit a new version"
    assert _index_puts(stub) == [], "the index must not be rewritten at all on a re-deposit"
    assert [u for u, _ in stub.puts] == [], f"unexpected writes: {[u for u, _ in stub.puts]}"


@pytest.mark.asyncio
async def test_a_different_checkpoint_still_appends_and_commits(fake_training: None) -> None:
    """Clause 3's pair for the whole path: the dedup cannot pass by refusing every
    deposit. Read off the index bytes actually PUT, not off the in-memory dict."""
    idx = _index(_member(_FIRST_ID, _OTHER_DIGEST))
    stub = _Stub(idx, _DIGEST)
    returned = await _deposit(stub)

    puts = _index_puts(stub)
    assert len(puts) == 1, f"expected exactly one index PUT, got {len(puts)}"
    written = json.loads(puts[0])
    digests = [c["weights_sha256"] for c in written["contributions"]]
    assert digests == [_OTHER_DIGEST, _DIGEST]
    assert returned["contribution_id"] not in (_FIRST_ID, None)
    assert stub.commits == ["bioimage-io/cpsam-community-pool"]
    assert stub.discards == []


@pytest.mark.asyncio
async def test_the_first_deposit_into_an_uninitialised_pool_still_works(fake_training: None) -> None:
    """The bootstrap branch has no index to look in; the check must not break it."""
    stub = _Stub(None, _DIGEST)
    returned = await _deposit(stub)
    assert returned["deduplicated"] is False
    written = json.loads(_index_puts(stub)[0])
    assert [c["weights_sha256"] for c in written["contributions"]] == [_DIGEST]


@pytest.mark.asyncio
async def test_a_redeposit_counts_bytes_but_not_a_contribution(fake_training: None) -> None:
    """The bytes did cross the wire, so bytes_up counts them — pretending otherwise
    would understate the cost of exactly these retries. ``contributions`` is the
    number quoted as deposits, and the pool gained no member."""
    stub = _Stub(_index(_member(_FIRST_ID, _DIGEST)), _DIGEST)
    await _deposit(stub)
    assert len(stub.transport) == 1, f"expected one transport bump, got {stub.transport}"
    bumped = stub.transport[0]
    assert bumped.get("bytes_up") == 1218617275
    assert "contributions" not in bumped or bumped["contributions"] == 0, (
        f"a re-deposit must not increment the deposit count: {bumped}"
    )


@pytest.mark.asyncio
async def test_a_fresh_deposit_counts_a_contribution(fake_training: None) -> None:
    """The discriminator for the assertion above: it must be able to see the
    counter move."""
    stub = _Stub(_index(_member(_FIRST_ID, _OTHER_DIGEST)), _DIGEST)
    await _deposit(stub)
    assert stub.transport[0].get("contributions") == 1
    assert stub.transport[0].get("bytes_up") == 1218617275


@pytest.mark.asyncio
async def test_an_already_merged_checkpoint_is_recognised_on_the_whole_path(
    fake_training: None,
) -> None:
    """Clause 3 end to end: a re-deposit of a checkpoint already inside a published
    community soup is returned, never appended."""
    idx = _index(_member(_FIRST_ID, _DIGEST))
    idx["decided_contribution_ids"] = [_FIRST_ID]
    stub = _Stub(idx, _DIGEST)
    returned = await _deposit(stub)
    assert returned["contribution_id"] == _FIRST_ID
    assert returned["deduplicated"] is True
    assert len(idx["contributions"]) == 1
    assert stub.discards == ["bioimage-io/cpsam-community-pool"]


@pytest.mark.asyncio
async def test_the_sentinel_never_reaches_the_caller(fake_training: None) -> None:
    """The duplicate path leaves the staged block by raising, which is internal.
    An RPC caller must see a normal return — a leaked exception would read to a
    retrying client as another failed deposit."""
    import entry

    try:
        await _deposit(_Stub(_index(_member(_FIRST_ID, _DIGEST)), _DIGEST))
    except entry._RecognisedRedeposit as e:  # pragma: no cover - the failure
        pytest.fail(f"the internal sentinel escaped contribute: {e}")
