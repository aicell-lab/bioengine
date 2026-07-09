"""DEPRECATED — use ``bioengine.cache`` instead.

This module is preserved as a thin delegation layer so existing app
code keeps working through the ``@bioengine.multiplexed`` →
``@bioengine.cached`` migration. Every function here forwards to its
``bioengine.cache`` counterpart and emits ``DeprecationWarning``.

Historical context: this module used to reach into Ray Serve's
``_ModelMultiplexWrapper.unload_model_lru`` to expose manual eviction
control. That path dropped Python refs to the cached pipeline but did
not call ``torch.cuda.empty_cache()`` — PyTorch's caching allocator
retained the freed blocks, so ``pynvml`` continued to report them as
allocated. The home-grown ``bioengine.cache`` (backed by
``@bioengine.cached``) fixes that; delegating here means callers on
the deprecated name get the fix too.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional

# Ray stores the multiplex wrapper on the deployment instance under this
# hardcoded attribute name. Kept as a constant for the very old
# ``@bioengine.multiplexed`` deployments that were built + serialised
# before this file was rewritten as a shim — if the running deployment
# still has a Ray Serve multiplex wrapper attached rather than one of
# our ``PipelineCache`` instances, the helpers below fall back to the
# old behavior with a louder warning. See ``_legacy_ray_serve_wrapper``.
_RAY_MULTIPLEX_ATTR = "__serve_multiplex_wrapper"


def _warn_once(msg: str) -> None:
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


def _legacy_ray_serve_wrapper(app_instance: Any) -> Optional[Any]:
    """Return the Ray Serve ``_ModelMultiplexWrapper`` on ``app_instance``
    if the deployment still carries one (i.e. it was built with a
    pre-refactor bioengine that decorated via ``serve.multiplexed``).
    ``None`` for anything decorated with the current
    ``@bioengine.cached`` / ``@bioengine.multiplexed``.
    """
    return getattr(app_instance, _RAY_MULTIPLEX_ATTR, None)


async def evict_lru_model(app_instance: Any) -> Optional[str]:
    """DEPRECATED — use ``bioengine.cache.evict_lru_model``."""
    _warn_once(
        "bioengine.multiplex.evict_lru_model is deprecated; use "
        "bioengine.cache.evict_lru_model."
    )
    from bioengine._app import cache as _cache

    if cache_ids := _cache.cached_model_ids(app_instance):
        # New-style deployment; delegate.
        return await _cache.evict_lru_model(app_instance)

    # Old-style deployment (Ray Serve wrapper still on the instance).
    wrapper = _legacy_ray_serve_wrapper(app_instance)
    if wrapper is None or not wrapper.models:
        return None
    lru_id = next(iter(wrapper.models))
    await wrapper.unload_model_lru()
    return lru_id


async def evict_all_models(app_instance: Any) -> int:
    """DEPRECATED — use ``bioengine.cache.evict_all_models``."""
    _warn_once(
        "bioengine.multiplex.evict_all_models is deprecated; use "
        "bioengine.cache.evict_all_models."
    )
    from bioengine._app import cache as _cache

    if _cache._get_caches(app_instance):
        return await _cache.evict_all_models(app_instance)

    wrapper = _legacy_ray_serve_wrapper(app_instance)
    if wrapper is None:
        return 0
    count = len(wrapper.models)
    while wrapper.models:
        await wrapper.unload_model_lru()
    return count


async def evict_model(app_instance: Any, model_id: str) -> bool:
    """DEPRECATED — use ``bioengine.cache.evict_model``."""
    _warn_once(
        "bioengine.multiplex.evict_model is deprecated; use "
        "bioengine.cache.evict_model."
    )
    from bioengine._app import cache as _cache

    if _cache._get_caches(app_instance):
        return await _cache.evict_model(app_instance, model_id)

    wrapper = _legacy_ray_serve_wrapper(app_instance)
    if wrapper is None or model_id not in wrapper.models:
        return False
    wrapper.models.move_to_end(model_id, last=False)
    await wrapper.unload_model_lru()
    return True


def cached_model_ids(app_instance: Any) -> List[str]:
    """DEPRECATED — use ``bioengine.cache.cached_model_ids``."""
    from bioengine._app import cache as _cache

    ids = _cache.cached_model_ids(app_instance)
    if ids:
        return ids
    wrapper = _legacy_ray_serve_wrapper(app_instance)
    if wrapper is None:
        return []
    return list(wrapper.models.keys())
