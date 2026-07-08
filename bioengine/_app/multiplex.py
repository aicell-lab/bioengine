"""Public helpers for interacting with the multiplexed-model cache.

The ``@bioengine.multiplexed`` decorator wraps a class method so Ray Serve's
``_ModelMultiplexWrapper`` provides LRU-cached model loading with a per-replica
size cap. Ray keeps the wrapper on the deployment instance under the attribute
``__serve_multiplex_wrapper``, but does not expose a user-facing way to reach
in — no manual eviction API, no way to inspect what's currently loaded, no way
to free the GPU on demand before a heavy call.

This module bridges that gap. Every helper accepts the deployment instance
(``self``) and calls into the same ``_ModelMultiplexWrapper.unload_model_lru``
Ray already uses internally when it needs to evict on cache overflow — so
cleanup semantics (calling the model's ``__del__``, releasing GPU memory,
metrics accounting) are identical to Ray's own eviction path. That means a
foundation-model app can drop its cached model right before running an
expensive ``test()`` and the GPU frees the same way it would if the cache had
naturally overflowed.

Usage inside a deployment method:

    import bioengine

    class RuntimeApp:
        @bioengine.multiplexed(max_models=3)
        async def load_model(self, model_id: str): ...

        @bioengine.method
        async def free_gpu_for_test(self) -> int:
            return await bioengine.multiplex.evict_all_models(self)

Ray Serve enforces one ``@serve.multiplexed`` per class (attribute name is
hardcoded); bioengine surfaces that as a hard rejection at ``_scan_class`` time
so authors don't hit silently-shared caches. These helpers therefore always
operate on the single wrapper.
"""

from __future__ import annotations

from typing import Any, List, Optional

# Ray stores the multiplex wrapper on the deployment instance under this
# hardcoded attribute name. See ray/serve/api.py:_multiplex_wrapper — the
# constant lives inline there rather than being exported, so we mirror it
# here. If a Ray Serve upgrade ever renames it, this is the only place
# that needs a change.
_RAY_MULTIPLEX_ATTR = "__serve_multiplex_wrapper"


def _get_wrapper(app_instance: Any) -> Optional[Any]:
    """Return Ray Serve's ``_ModelMultiplexWrapper`` on ``app_instance``.

    ``None`` when the deployment has no ``@bioengine.multiplexed`` method,
    or when a decorated method exists but has not yet been called on this
    replica (Ray creates the wrapper lazily on first call).
    """
    return getattr(app_instance, _RAY_MULTIPLEX_ATTR, None)


async def evict_lru_model(app_instance: Any) -> Optional[str]:
    """Evict the least-recently-used model from the multiplexed cache.

    Args:
        app_instance: the ``self`` of a ``@bioengine.app`` deployment that
            has a ``@bioengine.multiplexed`` method.

    Returns:
        The evicted ``model_id``, or ``None`` if the cache is empty or the
        multiplexed method hasn't been called yet on this replica.
    """
    wrapper = _get_wrapper(app_instance)
    if wrapper is None or not wrapper.models:
        return None
    # ``models`` is an OrderedDict ordered LRU → MRU (Ray moves entries to
    # the end on access in ``load_model``). Peek at the LRU key BEFORE
    # unload_model_lru mutates the dict so we can return it.
    lru_id = next(iter(wrapper.models))
    await wrapper.unload_model_lru()
    return lru_id


async def evict_all_models(app_instance: Any) -> int:
    """Evict every model from the multiplexed cache.

    Uses the same LRU-unload code path Ray uses for cache-overflow eviction,
    so per-model cleanup (``__del__``, GPU-memory release) is identical.

    Returns:
        The number of models evicted (0 if the cache was empty or no
        wrapper exists yet on this replica).
    """
    wrapper = _get_wrapper(app_instance)
    if wrapper is None:
        return 0
    count = len(wrapper.models)
    while wrapper.models:
        await wrapper.unload_model_lru()
    return count


async def evict_model(app_instance: Any, model_id: str) -> bool:
    """Evict a specific model from the multiplexed cache by id.

    Preserves the LRU order of the remaining models. Implemented by moving
    the target to the LRU end of the ``OrderedDict`` and calling Ray's
    ``unload_model_lru`` — that path drives the same ``__del__``-hook
    cleanup Ray uses on natural eviction.

    Args:
        app_instance: the ``self`` of a ``@bioengine.app`` deployment.
        model_id: the model to evict.

    Returns:
        ``True`` if the model was in cache and got evicted; ``False`` if
        no wrapper exists yet or the model wasn't in cache.
    """
    wrapper = _get_wrapper(app_instance)
    if wrapper is None or model_id not in wrapper.models:
        return False
    # Move the target to position 0 (LRU end) so ``unload_model_lru``
    # picks it. move_to_end(last=False) is O(1) on OrderedDict.
    wrapper.models.move_to_end(model_id, last=False)
    await wrapper.unload_model_lru()
    return True


def cached_model_ids(app_instance: Any) -> List[str]:
    """Return the ids currently in the multiplexed cache, LRU → MRU order.

    Useful for a status endpoint or a health-check that wants to report
    what's warm. Returns an empty list when no wrapper exists yet.
    """
    wrapper = _get_wrapper(app_instance)
    if wrapper is None:
        return []
    return list(wrapper.models.keys())
