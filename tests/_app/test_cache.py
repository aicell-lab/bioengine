"""Unit tests for ``bioengine._app.cache``.

Covers ``PipelineCache`` semantics (LRU order, capacity, cleanup) and the
module-level helpers ``evict_all_models`` / ``evict_lru_model`` /
``evict_model`` / ``cached_model_ids``. Focuses on the cleanup guarantee
that made the framework refactor worth doing: every eviction path
(explicit and implicit overflow) calls ``_release_gpu_caches`` under the
same asyncio lock as the pop.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from bioengine._app import cache as bcache


class _FakePipeline:
    """Records its own __del__ so tests can assert cleanup order."""

    _log: List[str] = []

    def __init__(self, tag: str):
        self.tag = tag

    def __del__(self):  # pragma: no cover - order-dependent
        _FakePipeline._log.append(self.tag)


class TestPipelineCache:
    """Direct exercises against ``PipelineCache``."""

    def test_ctor_rejects_zero(self):
        with pytest.raises(ValueError):
            bcache.PipelineCache(max_models=0)

    async def test_hit_moves_to_mru(self):
        c = bcache.PipelineCache(max_models=3)
        calls = 0

        async def loader():
            nonlocal calls
            calls += 1
            return f"pipe-{calls}"

        # Load three
        for k in ("a", "b", "c"):
            await c.get_or_load(k, loader)
        assert c.keys() == ["a", "b", "c"]

        # Hit on "a" — moves to MRU end
        _ = await c.get_or_load("a", loader)
        assert c.keys() == ["b", "c", "a"]
        assert calls == 3  # loader NOT called on hit

    async def test_overflow_evicts_lru_before_load(self, monkeypatch):
        c = bcache.PipelineCache(max_models=2)
        released = []
        monkeypatch.setattr(bcache, "_release_gpu_caches", lambda: released.append(1))

        async def loader():
            return object()

        await c.get_or_load("a", loader)
        await c.get_or_load("b", loader)
        assert c.keys() == ["a", "b"]
        # Third overflows; "a" (LRU) evicted first
        await c.get_or_load("c", loader)
        assert c.keys() == ["b", "c"]
        # One eviction → one release call
        assert released == [1]

    async def test_evict_all_runs_release_once(self, monkeypatch):
        c = bcache.PipelineCache(max_models=3)
        released = []
        monkeypatch.setattr(bcache, "_release_gpu_caches", lambda: released.append(1))

        async def loader():
            return object()

        await c.get_or_load("a", loader)
        await c.get_or_load("b", loader)
        # No release yet — nothing was evicted
        assert released == []
        n = await c.evict_all()
        assert n == 2
        # Single release call regardless of how many entries dropped
        assert released == [1]
        assert c.keys() == []

    async def test_evict_all_empty_is_noop(self, monkeypatch):
        c = bcache.PipelineCache(max_models=3)
        released = []
        monkeypatch.setattr(bcache, "_release_gpu_caches", lambda: released.append(1))
        n = await c.evict_all()
        assert n == 0
        # Don't burn a torch.cuda.empty_cache when there's nothing to free
        assert released == []

    async def test_evict_specific(self, monkeypatch):
        c = bcache.PipelineCache(max_models=3)
        released = []
        monkeypatch.setattr(bcache, "_release_gpu_caches", lambda: released.append(1))

        async def loader():
            return object()

        await c.get_or_load("a", loader)
        await c.get_or_load("b", loader)
        assert await c.evict("a") is True
        assert c.keys() == ["b"]
        assert released == [1]
        # Missing key → False, no release
        assert await c.evict("nope") is False
        assert released == [1]

    async def test_evict_lru(self, monkeypatch):
        c = bcache.PipelineCache(max_models=3)
        released = []
        monkeypatch.setattr(bcache, "_release_gpu_caches", lambda: released.append(1))

        async def loader():
            return object()

        for k in ("a", "b"):
            await c.get_or_load(k, loader)
        evicted = await c.evict_lru()
        assert evicted == "a"
        assert c.keys() == ["b"]
        # Empty cache → None, no release
        await c.evict_all()
        released.clear()
        assert await c.evict_lru() is None
        assert released == []


class TestModuleHelpers:
    """Exercises the ``bioengine.cache.*`` helpers against a fake
    deployment carrying ``_bioengine_caches``.
    """

    async def test_no_caches_returns_zero(self):
        # A deployment instance that has no @bioengine.cached methods
        # is a valid shape — helpers no-op gracefully.
        class NoCache:
            pass

        instance = NoCache()
        assert await bcache.evict_all_models(instance) == 0
        assert await bcache.evict_lru_model(instance) is None
        assert await bcache.evict_model(instance, "x") is False
        assert bcache.cached_model_ids(instance) == []

    async def test_single_cache_no_method_name(self):
        class OneCache:
            pass

        instance = OneCache()
        instance._bioengine_caches = {"load_model": bcache.PipelineCache(3)}

        async def loader():
            return object()

        cache = instance._bioengine_caches["load_model"]
        await cache.get_or_load("m1", loader)
        await cache.get_or_load("m2", loader)
        assert bcache.cached_model_ids(instance) == ["m1", "m2"]
        assert await bcache.evict_all_models(instance) == 2

    async def test_multiple_caches_require_method_name(self):
        class TwoCaches:
            pass

        instance = TwoCaches()
        instance._bioengine_caches = {
            "load_model": bcache.PipelineCache(3),
            "load_dataset": bcache.PipelineCache(2),
        }
        # No method_name → LookupError from _single_cache
        with pytest.raises(LookupError):
            await bcache.evict_all_models(instance)
        # With method_name → routes correctly
        assert await bcache.evict_all_models(instance, method_name="load_model") == 0
