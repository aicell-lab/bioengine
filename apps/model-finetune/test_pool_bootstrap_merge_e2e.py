"""#0070 end-to-end: bootstrap a pool through the REAL `contribute`, then MERGE it.

This is the test whose absence let `d0679f8f` reach a `done` label with a dead production
path. Every pre-existing test built the base commitment itself, via the helper at
`test_soup_math.py:39`, and then handed it to `uniform_soup` — so the suite was green while
nothing in production wrote one, and a freshly created pool could not merge at all.

**That blind spot is easy to recreate, and my first version of this file did.** A test that
calls `pool.base_commitment(...)` and then `pool.empty_index(mt, sig, commitment)` passes
against the broken tree too, because `pool.py` was never the defect — `entry.py` was. So the
load-bearing arm here runs the actual `contribute` coroutine
(`entry.EntryApp.func_or_class.contribute.__get__(stub)`), lets IT bootstrap the index, reads
the `pool_index.json` bytes it PUT, and merges against that. Against the pre-fix entry the
written record carries `base_commitment: None` and the merge raises.

The remaining arms pin the merge contract itself — the base's diameter is carried rather than
averaged (#0070's original defect), the inventory is fail-closed, and a merge of zero members
raises rather than returning an empty checkpoint.

Synthetic tensors only: torch, no cellpose, no GPU, no pool artifact.
"""
import json
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import pytest

import pool

torch = pytest.importorskip("torch")

MODEL_TYPE = "cpsam"
_INDEX_URL = "https://example.invalid/put/index"
_WEIGHTS_URL = "https://example.invalid/put/weights"
_METADATA_URL = "https://example.invalid/put/metadata"
_SESSION = "2026-11-14-e2e"
_DIGEST = "a" * 64

# The base's diameter is NOT the mean of the members' below, so "carried the base's value"
# and "averaged the members" give different numbers and the assertions can tell them apart.
BASE_DIAM = 30.0
M1_DIAM, M2_DIAM = 41.0, 53.0          # mean 47.0, nowhere near 30.0


def _base_state_dict() -> Dict[str, Any]:
    return {
        "encoder.weight": torch.full((4,), 1.0, dtype=torch.float32),
        "decoder.bias": torch.full((2,), 2.0, dtype=torch.float32),
        "diam_labels": torch.tensor([BASE_DIAM], dtype=torch.float32),
        "diam_mean": torch.tensor([BASE_DIAM], dtype=torch.float32),
    }


def _member(weight: float, bias: float, diam: float) -> Dict[str, Any]:
    """A contribution with EXACTLY the base's key set — the merge requires that."""
    return {
        "encoder.weight": torch.full((4,), weight, dtype=torch.float32),
        "decoder.bias": torch.full((2,), bias, dtype=torch.float32),
        "diam_labels": torch.tensor([diam], dtype=torch.float32),
        "diam_mean": torch.tensor([diam], dtype=torch.float32),
    }


_BASE_SD = _base_state_dict()
_SIG = pool.architecture_signature(_BASE_SD)
# What the real runtime now returns: a commitment it built from the base it holds.
_COMMITMENT = pool.base_commitment(_BASE_SD, source="test:e2e-init-checkpoint")


# ── the real contribute path, with its collaborators stubbed ────────────────────────

@pytest.fixture
def fake_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """A COMPLETED cpsam session, stubbed rather than created on disk."""
    mod = types.ModuleType("training")
    mod.get_status = lambda sid: {
        "status": "COMPLETED", "model_type": MODEL_TYPE,
        "n_epochs_completed": 2, "n_train_inputs": 5,
    }
    mod.session_backend = lambda sid: "cellpose"
    mod.read_training_params = lambda sid: {"model_type": MODEL_TYPE,
                                            "init_checkpoint": "stock"}
    monkeypatch.setitem(sys.modules, "training", mod)


class _Runtime:
    """Stands in for the GPU runtime, returning what the real one now returns.

    The commitment is built by `pool.base_commitment` from a real state dict rather than
    hand-written, so this stub cannot drift from the contract `uniform_soup` enforces.
    """

    async def prepare_contribution(self, session_id: str, weights_put_url: str,
                                   model_type: str = MODEL_TYPE) -> Dict[str, Any]:
        return {
            "architecture_signature": _SIG,
            "base_checkpoint_sha256": "b" * 64,
            "cellpose_version": "4.0.1",
            "l2_from_base": 0.5,
            "weights_sha256": _DIGEST,
            "n_bytes": 1218617275,
            "base_commitment": _COMMITMENT,
        }


class _Stub:
    """Just enough of `EntryApp` for `contribute` to reach a return.

    `index=None` is the case under test: an uninitialized pool, which is the branch that
    calls `empty_index` and is where the missing writer lived.
    """

    def __init__(self) -> None:
        self.index: Optional[Dict[str, Any]] = None
        self.puts: List[Tuple[str, bytes]] = []
        self.commits: List[str] = []
        self.transport: List[Dict[str, int]] = []

    async def _check_runtime_available(self, model_type: str) -> None:
        return None

    def _runtime_for(self, model_type: str) -> object:
        return _Runtime()

    async def _pool_service(self, pool_token: str):
        class _Client:
            async def disconnect(self) -> None:
                return None

        class _Am:
            async def edit(self, artifact_id: str, *a: Any, **k: Any) -> None:
                return None

            async def discard(self, artifact_id: str, *a: Any, **k: Any) -> None:
                return None

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
        """The REAL staging context manager, bound rather than faked."""
        import entry

        return entry.EntryApp.func_or_class._staged.__get__(self)(am, artifact_id)

    def _bump_transport(self, **kw: int) -> None:
        self.transport.append(dict(kw))


async def _deposit(stub: _Stub) -> Dict[str, Any]:
    import entry

    contribute = entry.EntryApp.func_or_class.contribute.__get__(stub)
    return await contribute(session_id=_SESSION,
                            pool_artifact_id="bioimage-io/cpsam-e2e-pool",
                            pool_token="t")


def _written_index(stub: _Stub) -> Dict[str, Any]:
    """The pool_index.json the REAL contribute PUT — not one the test built."""
    bodies = [c for url, c in stub.puts if url == _INDEX_URL]
    assert bodies, "contribute PUT no pool index, so there is nothing to assert on"
    return json.loads(bodies[-1].decode("utf-8"))


# === THE load-bearing arm: production bootstrap, then a real merge ===================

@pytest.mark.asyncio
async def test_a_pool_bootstrapped_BY_CONTRIBUTE_can_merge(fake_training: None) -> None:
    """The arm that catches the defect: `entry.py` must WRITE the commitment it was given.

    Against `d0679f8f` this fails at the first assertion — `empty_index` was called with two
    positional arguments, so the record it PUT carried `base_commitment: None` and the merge
    below raises.
    """
    stub = _Stub()
    await _deposit(stub)
    index = _written_index(stub)

    assert index.get("base_commitment") is not None, (
        "contribute bootstrapped a pool with base_commitment: None — the record it wrote "
        "can never merge (#0070). The runtime supplied one; the entry dropped it."
    )
    # Complete, not merely present: a partial record raises later and further away.
    written = pool.assert_base_commitment(index["base_commitment"])
    assert written["key_count"] == len(_BASE_SD)
    assert written["source"] == "test:e2e-init-checkpoint"

    # ...and now MERGE against the record that contribute wrote.
    members = [_member(3.0, 5.0, M1_DIAM), _member(5.0, 9.0, M2_DIAM)]
    collected: List[Dict[str, float]] = []
    soup = pool.uniform_soup([lambda m=m: m for m in members],
                             base=index["base_commitment"],
                             member_non_learned=collected)

    assert float(soup["encoder.weight"][0]) == pytest.approx(4.0)   # (3+5)/2
    assert float(soup["decoder.bias"][0]) == pytest.approx(7.0)     # (5+9)/2
    # #0070's original defect: the base's diameter is carried, NOT averaged. 30.0 vs 47.0.
    assert float(soup["diam_labels"][0]) == pytest.approx(BASE_DIAM)
    assert float(soup["diam_labels"][0]) != pytest.approx((M1_DIAM + M2_DIAM) / 2)
    # Requirement 4: the contributors' own diameters survive, in loader order.
    assert [c["diam_labels"] for c in collected] == [M1_DIAM, M2_DIAM]
    assert set(soup) == set(_BASE_SD)


# === the merge contract itself ======================================================

def test_a_pool_bootstrapped_the_old_way_cannot_merge() -> None:
    """`empty_index(model_type, signature)` — the two-argument call `entry.py` used to make.

    No fallback to loading a stock base: the ruling is explicit that a fallback would make
    the commitment permanently optional.
    """
    index = pool.empty_index(MODEL_TYPE, _SIG)
    assert index["base_commitment"] is None
    with pytest.raises(ValueError, match="records no base commitment"):
        pool.uniform_soup([lambda: _member(3.0, 5.0, M1_DIAM)],
                          base=index["base_commitment"])


def test_a_member_with_an_extra_key_fails_the_merge_naming_the_key() -> None:
    rogue = _member(3.0, 5.0, M1_DIAM)
    rogue["encoder.some_future_statistic"] = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(ValueError) as ei:
        pool.uniform_soup([lambda: rogue], base=_COMMITMENT)
    assert "some_future_statistic" in str(ei.value), (
        "the refusal must name the offending key — a digest mismatch alone is not "
        "actionable: %s" % ei.value
    )


def test_a_member_missing_a_base_key_fails_the_merge() -> None:
    short = _member(3.0, 5.0, M1_DIAM)
    del short["decoder.bias"]
    with pytest.raises(ValueError):
        pool.uniform_soup([lambda: short], base=_COMMITMENT)


def test_the_merge_is_not_vacuously_green_on_an_empty_member_list() -> None:
    """Without this, every arm above could be satisfied by a soup that merged nothing."""
    with pytest.raises(ValueError, match="at least one state_dict"):
        pool.uniform_soup([], base=_COMMITMENT)
