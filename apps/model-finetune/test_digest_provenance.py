"""Tests for the merge's refusal of members whose weights cannot be identified (#0139).

``weights_sha256`` is the only field that says WHICH weights a contribution was. The
writer has always recorded it correctly, but until this guard no read path looked at
it: a pool artifact is append-only and outlives the version that wrote each row, so a
production pool holds rows from every version it has served. A live one measured 8 of
92 deposit rows with the key absent or null, and those rows were loaded and averaged
into a published community checkpoint whose manifest then asserted a sha256 over
members whose own provenance had never been established.

Two levels, because the defect needs both:

* the helper, over a legacy / null / good three-row fixture — the shapes actually
  measured in the wild (key absent, key present and null, key good);
* the real ``aggregate`` coroutine with its collaborators stubbed, which is what pins
  the guard to the MERGE PATH rather than to a function nobody calls. Removing the
  call in ``entry.py`` leaves the helper tests passing and fails the entry-level ones.

No GPU, no Ray, no live Hypha: the property is a refusal before the first scoring
pass, so it is provable without any of them.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import pytest

import pool

GATE = {"artifact_id": "bioimage-io/splits", "sha256": "a" * 64}
WITNESS = {"artifact_id": "bioimage-io/splits", "sha256": "b" * 64}

# The three shapes the campaign measured. "legacy" pre-dates the field entirely;
# "null" carries the key with no value; "good" is what the current writer produces.
LEGACY = {"contribution_id": "legacy-row", "weights_path": "c/legacy.pt",
          "uploaded_at": 1.0, "n_bytes": 1024}
NULLED = {"contribution_id": "null-row", "weights_path": "c/null.pt",
          "weights_sha256": None, "uploaded_at": 2.0, "n_bytes": 1024}
GOOD = {"contribution_id": "good-row", "weights_path": "c/good.pt",
        "weights_sha256": "c" * 64, "uploaded_at": 3.0, "n_bytes": 1024}


# ── the helper, over the three-row fixture ───────────────────────────────────

def test_a_good_row_alone_is_accepted():
    """Discrimination control. A guard that refused everything would pass every
    test below while making the pool permanently unmergeable."""
    assert pool.digest_provenance_error([GOOD]) is None


def test_an_absent_digest_is_refused_and_named():
    err = pool.digest_provenance_error([LEGACY])
    assert err is not None
    assert "legacy-row" in err


def test_a_null_digest_is_refused_and_named():
    """A present-but-null key is the shape a bare ``.get()`` would wave through."""
    err = pool.digest_provenance_error([NULLED])
    assert err is not None
    assert "null-row" in err


def test_the_three_row_fixture_names_both_offenders_and_not_the_good_one():
    err = pool.digest_provenance_error([LEGACY, NULLED, GOOD])
    assert err is not None
    assert "legacy-row" in err and "null-row" in err
    assert "good-row" not in err, f"the acceptable row was named as an offender: {err}"


def test_an_empty_digest_string_is_refused():
    """``""`` is neither absent nor null and would pass an ``is None`` test."""
    assert pool.digest_provenance_error(
        [dict(GOOD, weights_sha256="   ")]) is not None


def test_a_row_with_no_contribution_id_still_produces_a_message():
    """The message is the only thing the operator gets, so it must not raise while
    building itself out of a row that is missing the id too."""
    err = pool.digest_provenance_error([{"weights_path": "c/x.pt"}])
    assert err is not None


def test_no_members_is_not_an_error():
    """An empty batch is an ordinary no-op merge, not a provenance failure."""
    assert pool.digest_provenance_error([]) is None


# ── the merge path: the real aggregate(), collaborators stubbed ───────────────

class _Budget(Exception):
    """Reached only if the guard did NOT refuse. Distinct from a ValueError so a
    test that expects the refusal fails loudly rather than matching the wrong raise."""


def _aggregate_of(instance) -> Any:
    """The real ``aggregate`` coroutine bound to a stub instance.

    ``entry.EntryApp`` is a ``ray.serve.deployment.Deployment`` once ``@bioengine.app``
    has run, so its methods are not reachable through it; ``func_or_class`` is the
    class the decorator wrapped. Imported lazily so a missing optional dependency is a
    test failure rather than a collection error.
    """
    import entry

    return entry.EntryApp.func_or_class.aggregate.__get__(instance)


class _Runtime:
    async def soup_memory_budget(self) -> Dict[str, Any]:
        raise _Budget("reached the memory budget: the digest guard did not refuse")


class _Stub:
    """Just enough of ``EntryApp`` for ``aggregate`` to reach the digest guard.

    The guard sits after ``batch`` is known and before the first scoring pass, so
    everything the merge does with a GPU or with large objects is past the point this
    exercises. ``soup_memory_budget`` is the next collaborator after the guard and it
    raises ``_Budget``, which is how a removed guard shows up as a wrong exception
    rather than as a silent pass.
    """

    def __init__(self, contributions: List[Dict[str, Any]]) -> None:
        self._pool_locks: Dict[str, asyncio.Lock] = {}
        self._merges_in_flight = 0
        self._contributions = contributions

    async def _check_runtime_available(self, model_type: str) -> None:
        return None

    def _runtime_for(self, model_type: str) -> object:
        return _Runtime()

    async def _resolve_val_split(self, images, labels) -> Tuple[List[Any], List[Any]]:
        return [], []

    async def _pool_service(self, pool_token: str):
        class _Client:
            async def disconnect(self_inner) -> None:
                return None

        return object(), _Client()

    async def _read_pool_index(self, am, pool_artifact_id: str) -> Dict[str, Any]:
        return {
            "model_type": "cpsam",
            "contributions": list(self._contributions),
            "decided_contribution_ids": [],
            "current_community": None,
            "baseline": {"selection_metric": {"aggregate": 0.5}},
        }

    async def _read_merge_journal(self, am, pool_artifact_id: str):
        return None

    async def _write_merge_journal(self, am, pool_artifact_id: str, journal) -> None:
        return None


async def _run_merge(contributions: List[Dict[str, Any]]):
    return await _aggregate_of(_Stub(contributions))(
        pool_artifact_id="ws/p",
        pool_token="t",
        gate_val_images=[],
        gate_val_labels=[],
        witness_val_images=[],
        witness_val_labels=[],
        gate_split=GATE,
        witness_split=WITNESS,
    )


@pytest.mark.asyncio
async def test_the_merge_refuses_the_three_row_pool_before_scoring():
    """THE mutation control. With the guard removed, ``aggregate`` runs on to
    ``soup_memory_budget`` and this raises ``_Budget`` instead of ``ValueError`` —
    i.e. the merge proceeded to score a pool holding two unidentifiable members."""
    with pytest.raises(ValueError, match="refused this merge"):
        await _run_merge([LEGACY, NULLED, GOOD])


@pytest.mark.asyncio
async def test_the_refusal_names_the_offending_contributions():
    """An operator has to know WHICH rows to re-deposit; a bare refusal would make
    them diff 92 rows by hand."""
    with pytest.raises(ValueError) as exc:
        await _run_merge([LEGACY, NULLED, GOOD])
    assert "legacy-row" in str(exc.value)
    assert "null-row" in str(exc.value)


@pytest.mark.asyncio
async def test_a_clean_pool_is_not_refused_by_the_guard():
    """Discrimination control at the merge level: the guard must let a good pool
    through to the next step. Without this, a guard that refused unconditionally
    would satisfy both tests above."""
    with pytest.raises(_Budget):
        await _run_merge([GOOD])
