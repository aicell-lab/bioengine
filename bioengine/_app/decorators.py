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
from typing import Any, Callable, Dict, List, Optional

from hypha_rpc.utils.schema import schema_method
from ray import serve

from bioengine._app.errors import ReservedMethodNameError
from bioengine._app.mixin import _make_check_health, wrap_init

# Method names the framework owns. User code cannot define them as plain
# methods — the new contract is to mark a method with @bioengine.<hook>
# and let the name be whatever the user wants.
_RESERVED_NAMES = ("check_health", "async_init", "test_deployment")

# Tag attached to functions by the marker decorators below.
_KIND_ATTR = "_bioengine_kind"


# ─────────────────────────── method markers ──────────────────────────────


def method(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Expose an instance method as a BioEngine API method.

    Equivalent to ``hypha_rpc``'s ``@schema_method`` plus a tag the
    ``@bioengine.app`` decorator reads at class-creation time to assemble
    ``_bioengine_method_schemas``.
    """
    wrapped = schema_method(fn)
    setattr(wrapped, _KIND_ATTR, "method")
    return wrapped


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


def multiplexed(*, max_models: int = 3) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a method as a Ray Serve multiplexed model loader.

    Equivalent to ``@serve.multiplexed(max_num_models_per_replica=N)`` —
    ``@bioengine.app`` applies the underlying Ray Serve decorator at the
    right point in the pipeline.
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, _KIND_ATTR, "multiplexed")
        fn._bioengine_multiplexed_max_models = max_models  # type: ignore[attr-defined]
        return fn

    return decorator


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

        for mname in lifecycle["multiplexed"]:
            raw = getattr(cls, mname)
            max_models = getattr(raw, "_bioengine_multiplexed_max_models", 3)
            setattr(
                cls,
                mname,
                serve.multiplexed(max_num_models_per_replica=max_models)(raw),
            )

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
        "multiplexed": [],
    }
    method_schemas: List[Dict[str, Any]] = []

    for attr_name in dir(cls):
        member = getattr(cls, attr_name, None)
        kind = getattr(member, _KIND_ATTR, None)
        if kind is None:
            continue
        if kind == "method":
            method_schemas.append(member.__schema__)
        elif kind == "multiplexed":
            lifecycle["multiplexed"].append(attr_name)
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
