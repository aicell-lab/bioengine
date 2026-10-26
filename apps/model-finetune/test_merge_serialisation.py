"""Regression tests for two concurrent aggregates publishing under one community version.

``aggregate`` chooses its community version by probing which ones already exist
(``_next_free_community_version``) and then spends minutes building and scoring the soup
before it publishes. Nothing underneath arbitrates that: measured against live Hypha,
two callers can hold ``edit(stage=True)`` on one artifact simultaneously with no refusal,
the first ``commit`` clears the single artifact-level staging flag, and a caller's commit
can return success while the committed path holds the OTHER caller's bytes. So the
check-then-act had to be closed in the app.

The span that is locked is the whole merge — index read to index commit — not just the
publish. A lock around the publish alone would still let the second caller build its soup
from the head it read before it waited, and committing that would silently drop the merge
that landed while it waited: a lost update, which is worse than a duplicate version.

Asserted here on the real ``aggregate`` coroutine with its collaborators stubbed, so the
property is pinned to the code rather than to the commit message. No GPU, no Ray, no live
Hypha — the defect is ordering, so it is provable without any of them.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import pytest


def _aggregate_of(instance) -> Any:
    """The real ``aggregate`` coroutine function, bound to a stub instance.

    ``entry.EntryApp`` is a ``ray.serve.deployment.Deployment`` once ``@bioengine.app``
    has run, so its methods are not reachable through it; ``func_or_class`` is the class
    the decorator wrapped. Imported lazily so a missing optional dependency surfaces as a
    test failure rather than a collection error.
    """
    import entry

    return entry.EntryApp.func_or_class.aggregate.__get__(instance)


class _Stub:
    """Just enough of ``EntryApp`` for ``aggregate`` to reach its critical region.

    ``_read_pool_index`` is the first call inside the locked span, so it is where the
    enter/exit marks go. It returns a pool whose ``model_type`` is not cpsam, which makes
    ``aggregate`` raise immediately afterwards — the merge exits through its ``finally``,
    which is also the only place the lock is released.
    """

    def __init__(self, log: List[Tuple[str, str]], hold: float = 0.05) -> None:
        self._pool_locks: Dict[str, asyncio.Lock] = {}
        self._merges_in_flight = 0
        self._log = log
        self._hold = hold

    async def _check_runtime_available(self, model_type: str) -> None:
        return None

    def _runtime_for(self, model_type: str) -> object:
        return object()

    async def _resolve_val_split(self, images, labels):
        return [], []

    async def _pool_service(self, pool_token: str):
        class _Client:
            async def disconnect(self_inner) -> None:
                return None

        return object(), _Client()

    async def _read_pool_index(self, am, pool_artifact_id: str) -> Dict[str, Any]:
        self._log.append(("enter", pool_artifact_id))
        await asyncio.sleep(self._hold)
        self._log.append(("exit", pool_artifact_id))
        return {"model_type": "not-cpsam"}


async def _merge(instance, pool_artifact_id: str) -> None:
    """One aggregate call, swallowing the sentinel refusal it is steered into."""
    with pytest.raises(ValueError, match="cpsam-only"):
        await _aggregate_of(instance)(
            pool_artifact_id=pool_artifact_id,
            pool_token="t",
            gate_val_images=[],
            gate_val_labels=[],
            witness_val_images=[],
            witness_val_labels=[],
            gate_split={},
            witness_split={},
        )


def _overlaps(log: List[Tuple[str, str]]) -> bool:
    """True iff any merge entered the span while another was still inside it."""
    depth = 0
    for mark, _ in log:
        depth += 1 if mark == "enter" else -1
        if depth > 1:
            return True
    return False


@pytest.mark.asyncio
async def test_two_merges_on_one_pool_do_not_overlap() -> None:
    """The defect: both callers were inside the probe-then-publish span at once."""
    log: List[Tuple[str, str]] = []
    stub = _Stub(log)

    await asyncio.gather(_merge(stub, "ws/p"), _merge(stub, "ws/p"))

    assert len(log) == 4, f"both merges must have run the span; got {log}"
    assert not _overlaps(log), f"two merges held one pool at once: {log}"
    assert log == [("enter", "ws/p"), ("exit", "ws/p"),
                   ("enter", "ws/p"), ("exit", "ws/p")]


@pytest.mark.asyncio
async def test_merges_on_different_pools_are_not_serialised() -> None:
    """Discrimination control: the lock is per pool, not one global merge lock.

    Without this, a lock that simply blocked every aggregate would pass the test above
    while needlessly serialising unrelated campaigns.
    """
    log: List[Tuple[str, str]] = []
    stub = _Stub(log)

    await asyncio.gather(_merge(stub, "ws/a"), _merge(stub, "ws/b"))

    assert len(log) == 4
    assert _overlaps(log), f"distinct pools were serialised against each other: {log}"


@pytest.mark.asyncio
async def test_a_failed_merge_releases_the_pool() -> None:
    """A merge that raises must not wedge its pool for the life of the replica.

    The release lives in the same ``finally`` as the in-flight counter precisely so the
    error path cannot skip it; a leaked lock would be a permanent deadlock on that pool,
    recoverable only by redeploying.
    """
    log: List[Tuple[str, str]] = []
    stub = _Stub(log)

    await _merge(stub, "ws/p")

    assert not stub._pool_locks["ws/p"].locked(), "the lock survived the exception path"
    assert stub._merges_in_flight == 0

    await _merge(stub, "ws/p")
    assert len(log) == 4, "a second merge could not acquire the pool after a failure"
