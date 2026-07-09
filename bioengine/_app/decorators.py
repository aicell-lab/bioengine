"""The ``@bioengine.app`` decorator and the lifecycle/method markers.

Apply ``@bioengine.app(...)`` to a class to make it a BioEngine app deployment:

    @bioengine.app(num_cpus=1, memory_mb=512, pip=["pandas"])
    class DemoApp:
        @bioengine.async_init
        async def load(self): ...

        @bioengine.method
        async def ping(self) -> dict: ...

The decorator mutates the user class in place — adding lifecycle wiring,
collecting method schemas, attaching ``_bioengine_*`` metadata — and then
returns the class wrapped in ``@serve.deployment``. Mutating in place
(rather than building a subclass) preserves the class's ``__module__`` and
``__qualname__``, which lets cloudpickle resolve it by reference on
Ray Serve replicas via the ``py_modules`` upload.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from bioengine._app.errors import ReservedMethodNameError
from bioengine._app.mixin import _make_check_health, wrap_init

# Top-level imports of hypha_rpc and ray.serve were moved inside the
# decorator function bodies. The introspection task in
# bioengine._app.bootstrap is pickled-by-reference; unpickling it on the
# Ray client server (head-pod side, upstream of any runtime_env) re-runs
# this module's import block in whatever base environment that pod has.
# That pod is not guaranteed to ship hypha_rpc or ray[serve] — anything
# the decorators need at *application* time stays a lazy import.

# Method names the framework owns. User code cannot define them as plain
# methods — the new contract is to mark a method with @bioengine.<hook>
# and let the name be whatever the user wants.
_RESERVED_NAMES = ("check_health", "async_init", "test_deployment")

# Tag attached to functions by the marker decorators below.
_KIND_ATTR = "_bioengine_kind"

# Flag attached by ``@bioengine.method(context=True)``; read by the
# introspect task and surfaced to ProxyDeployment so the proxy knows to
# forward the Hypha caller's context as a kwarg.
_WANTS_CONTEXT_ATTR = "_bioengine_wants_context"


# ─────────────────────────── method markers ──────────────────────────────


def method(
    fn: Optional[Callable[..., Any]] = None,
    *,
    context: bool = False,
) -> Callable[..., Any]:
    """Expose an instance method as a BioEngine API method.

    Equivalent to ``hypha_rpc``'s ``@schema_method(arbitrary_types_allowed=True)``
    plus a tag the ``@bioengine.app`` decorator reads at class-creation time
    to assemble ``_bioengine_method_schemas``. Arbitrary types are allowed by
    default so methods can accept ``numpy.ndarray``, ``PIL.Image``, etc.
    without users having to wire a custom decorator.

    Set ``context=True`` to receive the Hypha caller's context dict as a
    ``context`` kwarg::

        @bioengine.method(context=True)
        async def whoami(self, context):
            return context["user"]["id"]

    The ``context`` parameter is **not** exposed in the schema Hypha
    publishes to clients — it is auto-injected by Hypha at the service
    boundary and forwarded by the ProxyDeployment so the user method
    receives the live caller identity on every call.
    """

    def _wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        from hypha_rpc.utils.schema import schema_method

        if not context:
            wrapped = schema_method(arbitrary_types_allowed=True)(f)
            setattr(wrapped, _KIND_ATTR, "method")
            setattr(wrapped, _WANTS_CONTEXT_ATTR, False)
            return wrapped

        # context=True: the user method must declare a ``context``
        # parameter. We build the schema from the full signature first
        # (so the parameter types still pick up annotations), then strip
        # ``context`` out of the published parameters dict and wrap the
        # original function in a pass-through that forwards everything —
        # including the proxy-injected ``context`` kwarg — to the user
        # code. The wrapped function intentionally bypasses pydantic
        # validation on the replica side because ``context`` reaches the
        # replica as an extra kwarg that the public schema does not
        # describe; the Hypha service boundary still validates the public
        # parameters before the call ever leaves the proxy.
        sig = inspect.signature(f)
        if "context" not in sig.parameters:
            raise TypeError(
                f"@bioengine.method(context=True) on "
                f"'{getattr(f, '__qualname__', f.__name__)}': the method "
                f"must declare a 'context' parameter (e.g. "
                f"'async def {f.__name__}(self, ..., context): ...')."
            )

        full_schema = schema_method(arbitrary_types_allowed=True)(f).__schema__
        params_schema = dict(full_schema.get("parameters") or {})
        properties = dict(params_schema.get("properties") or {})
        properties.pop("context", None)
        params_schema["properties"] = properties
        required = [
            r for r in (params_schema.get("required") or []) if r != "context"
        ]
        if required:
            params_schema["required"] = required
        else:
            params_schema.pop("required", None)
        public_schema = {
            "name": full_schema.get("name") or f.__name__,
            "description": full_schema.get("description") or f.__doc__,
            "parameters": params_schema,
        }

        if inspect.iscoroutinefunction(f):

            @wraps(f)
            async def wrapper(self, *args, **kwargs):
                return await f(self, *args, **kwargs)

        else:

            @wraps(f)
            def wrapper(self, *args, **kwargs):
                return f(self, *args, **kwargs)

        wrapper.__schema__ = public_schema
        setattr(wrapper, _KIND_ATTR, "method")
        setattr(wrapper, _WANTS_CONTEXT_ATTR, True)
        return wrapper

    if fn is not None:
        return _wrap(fn)
    return _wrap


def async_init(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method as the one-shot async initialiser.

    Ray Serve replicas call ``check_health`` repeatedly; the framework
    invokes the marked method once on the first ``check_health`` call,
    before traffic is admitted.
    """
    setattr(fn, _KIND_ATTR, "async_init")
    return fn


def smoke_test(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method as the startup smoke test.

    Runs once after ``async_init``. If it raises, the replica is marked
    unhealthy and Ray Serve never admits traffic.
    """
    setattr(fn, _KIND_ATTR, "smoke_test")
    return fn


def health_check(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method as the periodic health check.

    Invoked on every ``check_health`` call after one-shot init has
    completed. Raise to signal unhealthy.
    """
    setattr(fn, _KIND_ATTR, "health_check")
    return fn


def cached(*, max_models: int = 3) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a method as an LRU-cached model loader.

    The first non-``self`` positional argument is the cache key. The
    method body is the loader — called on cache miss to produce the
    entry. On cache overflow the least-recently-used entry is evicted
    BEFORE the new load, and every eviction path runs
    ``gc.collect()`` + ``torch.cuda.empty_cache()`` under the same
    asyncio lock as the pop, so ``pynvml`` reflects freed VRAM
    immediately.

    Manual cache control is available via the ``bioengine.cache``
    submodule — useful when a downstream call needs the full GPU
    (e.g. running ``bioimageio.core.test_model`` on a foundation
    model that won't fit alongside cached siblings):

    .. code-block:: python

        import bioengine

        class RuntimeApp:
            @bioengine.cached(max_models=3)
            async def load_model(self, model_id: str): ...

            @bioengine.method
            async def free_gpu_for_test(self) -> int:
                # Drop every cached model right now. GPU memory is
                # returned to the CUDA driver in the same critical
                # section, so pynvml reflects the free immediately.
                return await bioengine.cache.evict_all_models(self)

    ``bioengine.cache`` also exposes ``evict_lru_model``,
    ``evict_model``, and ``cached_model_ids``.

    Multiple ``@bioengine.cached`` methods per class are allowed —
    each gets its own independent ``PipelineCache`` under its method
    name. Pass ``method_name=`` to the ``bioengine.cache`` helpers to
    disambiguate when there is more than one.

    This decorator replaces the older ``@bioengine.multiplexed``,
    which forwarded to Ray Serve's ``@serve.multiplexed``. Ray's
    ``unload_model_lru`` dropped Python refs but did not call
    ``torch.cuda.empty_cache()`` — pipelines evicted from the
    multiplex cache left ~200 MB of VRAM stuck per model, observable
    via ``pynvml`` across subsequent calls. The home-grown cache
    fixes that; ``@bioengine.multiplexed`` is preserved as a
    deprecated alias.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, _KIND_ATTR, "cached")
        fn._bioengine_cache_max_models = max_models  # type: ignore[attr-defined]
        return fn

    return decorator


def multiplexed(*, max_models: int = 3) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """DEPRECATED — use ``@bioengine.cached`` instead.

    Previously forwarded to Ray Serve's
    ``@serve.multiplexed(max_num_models_per_replica=N)``. Now aliases
    ``@bioengine.cached``, which uses a home-grown LRU cache that
    properly releases GPU memory on every eviction (Ray Serve's
    path leaks torch's caching allocator blocks — pipelines evicted
    from the multiplex cache left ~200 MB per model stuck in VRAM,
    observable via ``pynvml``).

    Emits ``DeprecationWarning`` at decorator time. Existing app code
    keeps working; migrate at leisure by renaming to
    ``@bioengine.cached``.
    """
    import warnings

    warnings.warn(
        "@bioengine.multiplexed is deprecated; use @bioengine.cached. "
        "Same semantics, but the new home-grown cache properly runs "
        "gc.collect() + torch.cuda.empty_cache() on every eviction — "
        "Ray Serve's multiplex path leaks torch's allocator blocks.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cached(max_models=max_models)


# ───────────────────────────── @bioengine.app ────────────────────────────


def app(
    *,
    num_cpus: float = 1,
    num_gpus: float = 0,
    memory_mb: Optional[int] = None,
    pip: Optional[List[str]] = None,
    env_vars: Optional[Dict[str, str]] = None,
    max_ongoing_requests: int = 10,
    ray_actor_options: Optional[Dict[str, Any]] = None,
    **serve_deployment_kwargs: Any,
) -> Callable[[type], Any]:
    """Mark a class as a BioEngine app deployment.

    Args:
        num_cpus: CPU cores per replica.
        num_gpus: GPU devices per replica (the worker may force this to 0
            when ``disable_gpu`` is set at deploy time).
        memory_mb: Soft memory cap in megabytes. Converted to Ray's bytes
            convention internally.
        pip: Additional pip requirements for the replica's runtime_env on
            top of BioEngine's baseline (``hypha-rpc``, ``pydantic``,
            ``httpx``, and the ``bioengine[worker]`` package).
        env_vars: Static env vars baked into the replica's runtime_env.
            Secrets passed at deploy time are layered on top.
        max_ongoing_requests: Concurrent request cap per replica.
        ray_actor_options: Escape hatch for advanced Ray Serve options
            (e.g. ``resources={"custom": 1}``). Deep-merged on top of the
            flat args above.
        **serve_deployment_kwargs: Forwarded to ``serve.deployment(...)``
            (e.g. ``num_replicas``, ``autoscaling_config``).
    """

    def decorator(cls: type) -> Any:
        from ray import serve

        if not inspect.isclass(cls):
            raise TypeError(
                "@bioengine.app must be applied to a class, not "
                f"{type(cls).__name__}"
            )

        _reject_reserved_names(cls)
        lifecycle, method_schemas = _scan_class(cls)
        composition_params = _extract_composition_params(cls)

        cls._bioengine_app_marker = True
        cls._bioengine_method_schemas = method_schemas
        cls._bioengine_lifecycle = lifecycle
        cls._bioengine_composition_params = composition_params

        orig_init = cls.__init__
        cls.__init__ = wrap_init(cls, orig_init)
        cls.check_health = _make_check_health(cls, lifecycle)

        # Wrap every @bioengine.cached method with a thin dispatcher
        # that resolves the per-method ``PipelineCache`` instance from
        # the deployment and drives ``get_or_load`` around the raw
        # loader function. Cache instances are created lazily on first
        # call so the ``asyncio.Lock`` binds to whatever loop Ray Serve
        # runs the replica on.
        for mname in lifecycle["cached"]:
            raw = getattr(cls, mname)
            max_models = getattr(raw, "_bioengine_cache_max_models", 3)
            setattr(cls, mname, _make_cached_wrapper(raw, mname, max_models))

        opts = _build_ray_actor_options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory_mb=memory_mb,
            pip=pip,
            env_vars=env_vars,
            extra=ray_actor_options,
        )
        return serve.deployment(
            ray_actor_options=opts,
            max_ongoing_requests=max_ongoing_requests,
            **serve_deployment_kwargs,
        )(cls)

    return decorator


# ────────────────────────── internal helpers ─────────────────────────────


def _make_cached_wrapper(
    raw: Callable[..., Any], method_name: str, max_models: int
) -> Callable[..., Any]:
    """Wrap a ``@bioengine.cached`` method so calls go through the
    per-replica ``PipelineCache``.

    On first call for a given deployment instance, lazily instantiates
    the cache (or picks up one already installed by ``wrap_init``) and
    stashes it on ``self._bioengine_caches[method_name]``. Subsequent
    calls hit the same cache. The first positional arg after ``self``
    is the cache key; the raw function is the loader.
    """
    from bioengine._app.cache import PipelineCache

    @wraps(raw)
    async def wrapped(self: Any, cache_key: Any, *args: Any, **kwargs: Any) -> Any:
        caches = getattr(self, "_bioengine_caches", None)
        if caches is None:
            caches = {}
            self._bioengine_caches = caches
        cache = caches.get(method_name)
        if cache is None:
            cache = PipelineCache(max_models=max_models)
            caches[method_name] = cache

        async def _loader() -> Any:
            return await raw(self, cache_key, *args, **kwargs)

        return await cache.get_or_load(str(cache_key), _loader)

    # Preserve the marker so downstream introspection still sees this
    # method was ``@bioengine.cached``, and re-attach the max_models
    # attribute in case someone inspects it.
    setattr(wrapped, _KIND_ATTR, "cached")
    wrapped._bioengine_cache_max_models = max_models  # type: ignore[attr-defined]
    return wrapped


def _reject_reserved_names(cls: type) -> None:
    for name in _RESERVED_NAMES:
        member = cls.__dict__.get(name)
        if member is None:
            continue
        kind = getattr(member, _KIND_ATTR, None)
        if kind is None:
            raise ReservedMethodNameError(
                f"{cls.__module__}.{cls.__qualname__} defines a method "
                f"named '{name}', which is reserved by the BioEngine "
                f"framework. Use the corresponding decorator instead: "
                f"@bioengine.async_init / @bioengine.smoke_test / "
                f"@bioengine.health_check — the method itself can be "
                f"named anything."
            )


def _scan_class(cls: type) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Collect lifecycle hooks and ``@method`` schemas from the class body."""
    lifecycle: Dict[str, Any] = {
        "async_init": None,
        "smoke_test": None,
        "health_check": None,
        # Every ``@bioengine.cached`` method name — allowed multiple
        # per class (each gets its own ``PipelineCache``). The now-
        # deprecated ``@bioengine.multiplexed`` decorator aliases
        # ``@bioengine.cached`` so its methods land in the same bucket.
        "cached": [],
    }
    method_schemas: List[Dict[str, Any]] = []

    for attr_name in dir(cls):
        member = getattr(cls, attr_name, None)
        kind = getattr(member, _KIND_ATTR, None)
        if kind is None:
            continue
        if kind == "method":
            # Carry the wants_context flag in the schema dict so the
            # introspect task ships it to the proxy, which uses it to
            # decide whether to inject the Hypha caller context on each
            # call. The flag survives JSON serialisation via _sanitise_schemas.
            entry = dict(member.__schema__)
            entry["wants_context"] = bool(
                getattr(member, _WANTS_CONTEXT_ATTR, False)
            )
            method_schemas.append(entry)
        elif kind == "cached":
            lifecycle["cached"].append(attr_name)
        elif kind in lifecycle:
            if lifecycle[kind] is not None:
                raise ReservedMethodNameError(
                    f"{cls.__module__}.{cls.__qualname__} has more than "
                    f"one @bioengine.{kind} method: "
                    f"'{lifecycle[kind]}' and '{attr_name}'. Only one is allowed."
                )
            lifecycle[kind] = attr_name

    return lifecycle, method_schemas


def _extract_composition_params(cls: type) -> Dict[str, str]:
    """Identify ``__init__`` params whose annotation is another ``@bioengine.app`` class.

    Returns a mapping ``{param_name: "module:qualname"}``. The runtime
    handle wrapping (``_wrap_runtime_handles`` in mixin.py) reads this to
    decide which incoming ``DeploymentHandle`` kwargs to wrap.

    Annotations referencing another ``@bioengine.app`` class actually
    resolve to a Ray Serve ``Deployment`` wrapper (the return value of
    ``serve.deployment(...)(cls)``), so we reach through ``.func_or_class``
    to read the marker and identify the underlying user class.

    Forward references (string annotations) cannot be resolved at decoration
    time without the full module namespace; they fall back to "skip" — the
    introspection Ray task will re-walk the hints with ``typing.get_type_hints``
    and surface a clearer error if it can't resolve them.
    """
    if cls.__init__ is object.__init__:
        return {}
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return {}

    params: Dict[str, str] = {}
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        ann = p.annotation
        if ann is inspect.Parameter.empty:
            continue
        if isinstance(ann, str):
            continue
        user_cls = _resolve_app_user_class(ann)
        if user_cls is not None:
            params[name] = f"{user_cls.__module__}:{user_cls.__qualname__}"
    return params


def _resolve_app_user_class(annotation: Any) -> Optional[type]:
    """Return the user class if ``annotation`` references a ``@bioengine.app`` class.

    Accepts either the Ray Serve ``Deployment`` wrapper (which is what's
    bound in the user's module namespace after decoration) or the raw user
    class. Returns ``None`` otherwise.
    """
    inner = getattr(annotation, "func_or_class", None)
    if inner is not None and getattr(inner, "_bioengine_app_marker", False):
        return inner
    if inspect.isclass(annotation) and getattr(annotation, "_bioengine_app_marker", False):
        return annotation
    return None


def _build_ray_actor_options(
    *,
    num_cpus: float,
    num_gpus: float,
    memory_mb: Optional[int],
    pip: Optional[List[str]],
    env_vars: Optional[Dict[str, str]],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Translate the flat decorator args into Ray Serve's nested options shape."""
    opts: Dict[str, Any] = {"num_cpus": num_cpus, "num_gpus": num_gpus}
    if memory_mb is not None:
        opts["memory"] = int(memory_mb) * 1024 * 1024

    runtime_env: Dict[str, Any] = {}
    if pip:
        runtime_env["pip"] = list(pip)
    if env_vars:
        runtime_env["env_vars"] = dict(env_vars)
    if runtime_env:
        opts["runtime_env"] = runtime_env

    if extra:
        opts = _deep_merge(opts, extra)
    return opts


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
