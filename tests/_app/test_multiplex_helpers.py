"""Pin the ``bioengine.multiplex`` public helpers.

Ray Serve's ``_ModelMultiplexWrapper`` is where the LRU cache actually lives;
Ray stores the instance on each replica as ``self.__serve_multiplex_wrapper``
and does not expose a public way to reach it. The ``bioengine.multiplex``
submodule bridges that gap:

- ``evict_lru_model`` — evict the least-recently-used entry, return its id
- ``evict_all_models`` — drain the cache, return count evicted
- ``evict_model`` — evict a specific id
- ``cached_model_ids`` — list currently-cached ids in LRU→MRU order

These tests use a stub wrapper (mirroring the OrderedDict + async
``unload_model_lru`` contract Ray Serve exposes) so they don't need a live
Ray Serve deployment. If Ray changes the private attribute name in a future
release, ``_RAY_MULTIPLEX_ATTR`` in ``multiplex.py`` needs to update — the
"no wrapper found" tests catch that regression.
"""
from __future__ import annotations

from collections import OrderedDict

import pytest

import bioengine
from bioengine._app import multiplex
from bioengine._app.errors import ReservedMethodNameError


class _StubWrapper:
    """Mimics ``ray.serve.multiplex._ModelMultiplexWrapper`` just enough.

    Ray's ``load_model`` moves the entry to the end of the ``models``
    OrderedDict on access — so ``next(iter(models))`` is the LRU key, and
    ``models.popitem(last=False)`` in ``unload_model_lru`` pops it.
    """

    def __init__(self, initial: dict[str, object] | None = None) -> None:
        self.models: OrderedDict[str, object] = OrderedDict(initial or {})
        self.unload_calls: list[str] = []

    async def unload_model_lru(self) -> None:
        # Match Ray's semantics: pop the LRU (first) entry.
        model_id, _ = self.models.popitem(last=False)
        self.unload_calls.append(model_id)


def _attach(wrapper: _StubWrapper) -> object:
    """Return a bare object with the Ray-Serve-private attribute pointing at
    the stub — mirrors how a real deployment replica exposes its cache."""

    class _Replica:
        pass

    r = _Replica()
    setattr(r, multiplex._RAY_MULTIPLEX_ATTR, wrapper)
    return r


# ---------- cached_model_ids ----------


def test_cached_model_ids_returns_lru_to_mru_order() -> None:
    """Order of the returned list must match Ray's LRU→MRU convention so
    dashboards / status endpoints display the "next-to-be-evicted" entry
    first."""
    w = _StubWrapper({"a": 1, "b": 2, "c": 3})
    r = _attach(w)
    assert multiplex.cached_model_ids(r) == ["a", "b", "c"]


def test_cached_model_ids_returns_empty_when_no_wrapper() -> None:
    """A deployment without a ``@bioengine.multiplexed`` method — or one
    that hasn't been called yet — has no wrapper attribute. The helper
    must degrade to an empty list, not raise."""

    class _Bare:
        pass

    assert multiplex.cached_model_ids(_Bare()) == []


# ---------- evict_lru_model ----------


@pytest.mark.asyncio
async def test_evict_lru_returns_evicted_id_and_removes_it() -> None:
    w = _StubWrapper({"a": 1, "b": 2, "c": 3})
    r = _attach(w)
    evicted = await multiplex.evict_lru_model(r)
    assert evicted == "a"
    assert list(w.models.keys()) == ["b", "c"]
    assert w.unload_calls == ["a"]  # went through Ray's unload path


@pytest.mark.asyncio
async def test_evict_lru_returns_none_on_empty_cache() -> None:
    w = _StubWrapper({})
    r = _attach(w)
    assert await multiplex.evict_lru_model(r) is None
    assert w.unload_calls == []


@pytest.mark.asyncio
async def test_evict_lru_returns_none_when_no_wrapper() -> None:
    """No wrapper = no cache = nothing to evict. Must not raise."""

    class _Bare:
        pass

    assert await multiplex.evict_lru_model(_Bare()) is None


# ---------- evict_all_models ----------


@pytest.mark.asyncio
async def test_evict_all_drains_cache_and_returns_count() -> None:
    w = _StubWrapper({"a": 1, "b": 2, "c": 3})
    r = _attach(w)
    n = await multiplex.evict_all_models(r)
    assert n == 3
    assert list(w.models.keys()) == []
    # All three drained via Ray's LRU-unload path, LRU-first.
    assert w.unload_calls == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_evict_all_returns_zero_on_empty_cache() -> None:
    r = _attach(_StubWrapper({}))
    assert await multiplex.evict_all_models(r) == 0


@pytest.mark.asyncio
async def test_evict_all_returns_zero_when_no_wrapper() -> None:
    class _Bare:
        pass

    assert await multiplex.evict_all_models(_Bare()) == 0


# ---------- evict_model ----------


@pytest.mark.asyncio
async def test_evict_specific_removes_target_and_preserves_others_order() -> None:
    """Evicting a non-LRU model must not disturb the LRU order of the rest —
    otherwise the next natural eviction would target the wrong entry."""
    w = _StubWrapper({"a": 1, "b": 2, "c": 3, "d": 4})
    r = _attach(w)
    ok = await multiplex.evict_model(r, "c")
    assert ok is True
    assert list(w.models.keys()) == ["a", "b", "d"]
    # Ray's unload path fired for "c" specifically.
    assert w.unload_calls == ["c"]


@pytest.mark.asyncio
async def test_evict_specific_returns_false_when_not_cached() -> None:
    w = _StubWrapper({"a": 1})
    r = _attach(w)
    assert await multiplex.evict_model(r, "does-not-exist") is False
    # Nothing evicted.
    assert list(w.models.keys()) == ["a"]
    assert w.unload_calls == []


@pytest.mark.asyncio
async def test_evict_specific_returns_false_when_no_wrapper() -> None:
    class _Bare:
        pass

    assert await multiplex.evict_model(_Bare(), "anything") is False


# ---------- decorator "one per class" enforcement ----------


def test_multiple_multiplexed_methods_raises_at_scan_time() -> None:
    """Ray Serve stores its multiplex cache on ``self`` at a single
    hardcoded attribute. A second ``@bioengine.multiplexed`` method would
    silently share the first one's wrapper — mis-dispatching model_ids
    to the wrong loader. bioengine rejects the class outright.

    Note: ``pytest.raises(ReservedMethodNameError, ...)`` would be nicer
    but ``test_decorators_baseline_imports.py`` reloads ``bioengine._app``
    mid-suite — that gives ``ReservedMethodNameError`` a fresh class
    identity that mismatches whatever this file imported at collection
    time. Matching on the class name string is the same failure mode
    check without the identity trap.
    """
    with pytest.raises(Exception) as exc_info:

        @bioengine.app(num_cpus=1, num_gpus=0, memory_mb=128)
        class TwoCaches:  # noqa: D401 — test fixture
            @bioengine.multiplexed(max_models=2)
            async def load_a(self, model_id: str): ...

            @bioengine.multiplexed(max_models=2)
            async def load_b(self, model_id: str): ...

    assert type(exc_info.value).__name__ == "ReservedMethodNameError"
    assert "more than one @bioengine.multiplexed" in str(exc_info.value)


def test_single_multiplexed_method_is_accepted() -> None:
    """Regression fence for the enforcement check — a class with exactly
    one @bioengine.multiplexed method must still decorate cleanly."""

    @bioengine.app(num_cpus=1, num_gpus=0, memory_mb=128)
    class OneCache:  # noqa: D401 — test fixture
        @bioengine.multiplexed(max_models=5)
        async def load(self, model_id: str): ...

    # Decoration succeeded — nothing else to assert.
    assert OneCache is not None
