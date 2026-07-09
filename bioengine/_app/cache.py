"""Home-grown LRU model cache backing ``@bioengine.cached``.

Replaces the previous ``@bioengine.multiplexed`` → ``ray.serve.multiplexed``
delegation. The old path had two well-documented problems: (1) Ray Serve's
``_ModelMultiplexWrapper.unload_model_lru`` drops the Python ref to a
pipeline but does not call ``torch.cuda.empty_cache()`` — PyTorch's caching
allocator retains the freed blocks in its own pool, so ``pynvml`` continues
to report them as allocated even after eviction; (2) the framework reached
in via a private attribute (``__serve_multiplex_wrapper``) which meant
any Ray Serve upgrade could silently break us.

``PipelineCache`` here manages LRU + capacity in ~40 lines of Python and
always runs ``gc.collect()`` + ``torch.cuda.empty_cache()`` in the same
critical section as the eviction, so there is no observable
half-freed state. Framework doesn't hard-import ``torch``; apps that
don't use GPUs pay no import cost.

Ray Serve's cross-replica model routing (the one thing ``@serve.multiplexed``
uniquely provides) is not consumed by any current bioengine app — model
routing only kicks in when the caller sets a ``serve.get_multiplexed_model_id()``
hint on the request, which our deployments don't do. Foundation-model apps
that WANT that routing can still decorate a method with ``@serve.multiplexed``
directly.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, Awaitable, Callable, List, Optional


def _release_gpu_caches() -> None:
    """Force Python + PyTorch to actually return freed VRAM to the driver.

    ``gc.collect()`` first so any pipeline references still held by
    frames / weak refs / __del__ closures get walked. Then
    ``torch.cuda.empty_cache()`` returns the allocator's free blocks
    to the CUDA driver so ``pynvml.nvmlDeviceGetMemoryInfo`` reflects
    the eviction. No-op when ``torch`` isn't importable or CUDA isn't
    available — the framework doesn't hard-require torch and apps
    without GPUs don't pay for it here.
    """
    import gc

    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class PipelineCache:
    """Per-replica LRU cache of loaded model pipelines.

    Every eviction path — explicit ``evict_all`` / ``evict_lru`` /
    ``evict``, and implicit overflow inside ``get_or_load`` — calls
    ``_release_gpu_caches()`` under the same asyncio lock as the pop.
    ``pynvml`` reflects freed VRAM immediately.

    Not thread-safe — designed for asyncio, and Ray Serve replicas run
    everything on a single event loop. Concurrent async callers on
    the same cache serialise through the internal ``asyncio.Lock``.
    """

    def __init__(self, max_models: int):
        if max_models < 1:
            raise ValueError(
                f"PipelineCache max_models must be >= 1, got {max_models}"
            )
        self._max = max_models
        self._models: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = asyncio.Lock()

    @property
    def max_models(self) -> int:
        return self._max

    async def get_or_load(
        self, cache_key: str, loader: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Return the cached entry for ``cache_key``, or call ``loader``
        to build one.

        On cache miss with the cache at capacity, evicts the LRU entry
        (with cleanup) BEFORE calling ``loader`` — so torch's pool has
        clean blocks to reuse for the new pipeline instead of piling on
        top of the phantom-cached model. On cache hit, moves the key to
        MRU end.
        """
        async with self._lock:
            if cache_key in self._models:
                self._models.move_to_end(cache_key)
                return self._models[cache_key]

            while len(self._models) >= self._max:
                _key, evicted = self._models.popitem(last=False)
                del evicted
                _release_gpu_caches()

            pipeline = await loader()
            self._models[cache_key] = pipeline
            return pipeline

    async def evict_lru(self) -> Optional[str]:
        """Evict the least-recently-used entry. Returns its key, or
        ``None`` if the cache was empty.
        """
        async with self._lock:
            if not self._models:
                return None
            key, evicted = self._models.popitem(last=False)
            del evicted
            _release_gpu_caches()
            return key

    async def evict(self, cache_key: str) -> bool:
        """Evict a specific entry. Returns ``True`` if it was cached
        and got evicted, ``False`` otherwise.
        """
        async with self._lock:
            if cache_key not in self._models:
                return False
            evicted = self._models.pop(cache_key)
            del evicted
            _release_gpu_caches()
            return True

    async def evict_all(self) -> int:
        """Evict every entry. Returns the number of entries that were
        cached before the call. Runs ``_release_gpu_caches()`` once at
        the end — one collect + one empty_cache regardless of how many
        entries were dropped.
        """
        async with self._lock:
            n = len(self._models)
            if n == 0:
                return 0
            self._models.clear()
            _release_gpu_caches()
            return n

    def keys(self) -> List[str]:
        """LRU→MRU ordered snapshot of cached keys.

        Read-only. Doesn't take the lock — worst case is a stale view
        when another coroutine is mid-eviction, and the intended use
        (status endpoints, health checks, logging) tolerates that.
        """
        return list(self._models.keys())


# === Module-level helpers ===
#
# Mirror the public shape of the pre-0.11.22 ``bioengine.multiplex``
# submodule so migrating app code is a one-line find/replace:
#   ``bioengine.multiplex.evict_all_models(self)`` →
#   ``bioengine.cache.evict_all_models(self)``.

_CACHES_ATTR = "_bioengine_caches"


def _get_caches(app_instance: Any) -> dict:
    """Return the dict of ``PipelineCache`` instances installed by
    ``@bioengine.cached``, or ``{}`` if the deployment has no
    ``@bioengine.cached`` methods.
    """
    return getattr(app_instance, _CACHES_ATTR, {}) or {}


def _single_cache(app_instance: Any) -> Optional[PipelineCache]:
    """Convenience for the common case of one ``@bioengine.cached``
    method per class — the app doesn't need to name the cache. Returns
    ``None`` if there are zero caches; raises ``LookupError`` if there
    are multiple (caller must pick one by ``method_name``).
    """
    caches = _get_caches(app_instance)
    if not caches:
        return None
    if len(caches) > 1:
        names = ", ".join(sorted(caches))
        raise LookupError(
            f"App has multiple @bioengine.cached methods ({names}); "
            f"pass method_name= to select one."
        )
    return next(iter(caches.values()))


def _resolve_cache(
    app_instance: Any, method_name: Optional[str]
) -> Optional[PipelineCache]:
    if method_name is not None:
        return _get_caches(app_instance).get(method_name)
    return _single_cache(app_instance)


async def evict_lru_model(
    app_instance: Any, method_name: Optional[str] = None
) -> Optional[str]:
    """Evict the least-recently-used entry from a deployment's cache.

    ``method_name`` selects among multiple ``@bioengine.cached`` methods.
    Defaults to the single cache if the class has one.
    """
    cache = _resolve_cache(app_instance, method_name)
    if cache is None:
        return None
    return await cache.evict_lru()


async def evict_all_models(
    app_instance: Any, method_name: Optional[str] = None
) -> int:
    """Evict every entry from a deployment's cache. Returns the count
    that was present before the call.
    """
    cache = _resolve_cache(app_instance, method_name)
    if cache is None:
        return 0
    return await cache.evict_all()


async def evict_model(
    app_instance: Any, cache_key: str, method_name: Optional[str] = None
) -> bool:
    """Evict a specific cache entry by key."""
    cache = _resolve_cache(app_instance, method_name)
    if cache is None:
        return False
    return await cache.evict(cache_key)


def cached_model_ids(
    app_instance: Any, method_name: Optional[str] = None
) -> List[str]:
    """LRU→MRU ordered keys currently in a deployment's cache."""
    cache = _resolve_cache(app_instance, method_name)
    if cache is None:
        return []
    return cache.keys()
