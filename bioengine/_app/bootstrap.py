"""Bootstrap functions that run inside the app's Ray ``runtime_env``.

The BioEngine worker never imports user app code directly. Instead it ships
the user's package as ``runtime_env.py_modules`` and runs these two functions
as Ray tasks in that env:

* :func:`introspect_app` — imports the entry class, walks the type-hint
  composition graph, and returns a JSON-compatible ``AppSpec`` dict.
* :func:`build_application` — reconstructs the Ray Serve bind graph from
  the spec, wraps the entry in ``ProxyDeployment``, and returns the
  ``serve.Application`` for the worker to ``serve.run``.

Both functions live in the ``bioengine[worker]`` package which is installed
in every replica's runtime_env, so they're reachable wherever the user's
package is reachable.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from typing import Any, Dict, List, Optional

from bioengine._app.errors import (
    BioEngineUserError,
    CompositionCycleError,
)

#: Manifest format the worker and bootstrap agree on. Bumped together
#: whenever the spec shape changes.
SPEC_FORMAT_VERSION = "0.6.0"


# ───────────────────────────── introspection ─────────────────────────────


def introspect_app(entry_id: str) -> Dict[str, Any]:
    """Walk the type-hint composition graph rooted at ``entry_id``.

    Returns a JSON-compatible dict the worker uses for resource accounting,
    method-schema discovery, and kwargs validation. The shape:

    .. code-block:: jsonc

        {
          "format_version": "0.6.0",
          "entry_id": "demo_app.deployment:DemoApp",
          "classes": {
            "demo_app.deployment:DemoApp": {
              "module": "demo_app.deployment",
              "qualname": "DemoApp",
              "ray_actor_options": {...},
              "max_ongoing_requests": 20,
              "method_schemas": [...],
              "lifecycle_methods": {...},
              "init_params": [
                {"name": "runtime_a", "kind": "deployment_handle",
                 "target": "demo_app.runtime:RuntimeA", "required": true},
                {"name": "batch_size", "kind": "value",
                 "annotation": "int", "default": 32, "required": false},
              ]
            }
          }
        }

    Args:
        entry_id: Fully qualified ``module.path:ClassName``.

    Raises:
        BioEngineUserError: The entry module can't be imported, the class
            doesn't exist, isn't ``@bioengine.app``-decorated, or has a
            type-hint annotation the framework can't resolve.
        CompositionCycleError: The type-hint graph contains a cycle.
    """
    entry_cls = _load_app_class(entry_id)
    user_cls = _user_class_of(entry_cls)

    classes: Dict[str, Dict[str, Any]] = {}
    # DFS with a visiting set so cycles surface as a clear error.
    _walk(entry_id, user_cls, entry_cls, classes, visiting=[])

    return {
        "format_version": SPEC_FORMAT_VERSION,
        "entry_id": entry_id,
        "classes": classes,
    }


def _walk(
    cid: str,
    user_cls: type,
    deployment: Any,
    classes: Dict[str, Dict[str, Any]],
    visiting: List[str],
) -> None:
    if cid in visiting:
        cycle = " → ".join(visiting + [cid])
        raise CompositionCycleError(f"Composition cycle: {cycle}")
    if cid in classes:
        return

    visiting.append(cid)
    try:
        module_name, qualname = cid.split(":", 1)
        init_params, child_ids = _describe_init(user_cls)
        classes[cid] = {
            "module": module_name,
            "qualname": qualname,
            "ray_actor_options": _ensure_jsonable(deployment.ray_actor_options),
            "max_ongoing_requests": getattr(deployment, "max_ongoing_requests", 10),
            "method_schemas": _sanitise_schemas(
                getattr(user_cls, "_bioengine_method_schemas", [])
            ),
            "lifecycle_methods": dict(
                getattr(user_cls, "_bioengine_lifecycle", {})
            ),
            "init_params": init_params,
        }
        for child_id, child_user_cls, child_deployment in child_ids:
            _walk(child_id, child_user_cls, child_deployment, classes, visiting)
    finally:
        visiting.pop()


def _describe_init(user_cls: type) -> tuple[List[Dict[str, Any]], List[tuple]]:
    """Extract init parameter info and collect child @bioengine.app references."""
    if user_cls.__init__ is object.__init__:
        return [], []

    sig = inspect.signature(user_cls.__init__)
    # ``user_cls.__init__`` was replaced by ``wrap_init`` from mixin.py,
    # so its ``__globals__`` point at the mixin module, not the user's.
    # Eval forward refs against the user class's defining module instead.
    eval_globals = getattr(
        sys.modules.get(user_cls.__module__), "__dict__", {}
    )

    init_params: List[Dict[str, Any]] = []
    child_refs: List[tuple] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        ann = _resolve_annotation(param.annotation, eval_globals)
        if ann is inspect.Parameter.empty:
            ann = None

        from bioengine._app.decorators import _resolve_app_user_class

        target_user_cls = _resolve_app_user_class(ann) if ann is not None else None
        required = param.default is inspect.Parameter.empty

        if target_user_cls is not None:
            target_id = f"{target_user_cls.__module__}:{target_user_cls.__qualname__}"
            init_params.append(
                {
                    "name": name,
                    "kind": "deployment_handle",
                    "target": target_id,
                    "required": required,
                }
            )
            # The annotation actually resolves to the Ray Serve Deployment
            # wrapper around target_user_cls — that's what carries
            # ray_actor_options. Find it from the module namespace.
            child_deployment = _load_app_class(target_id)
            child_refs.append((target_id, target_user_cls, child_deployment))
        else:
            init_params.append(
                {
                    "name": name,
                    "kind": "value",
                    "annotation": _stringify_annotation(ann),
                    "default": _safe_default(param.default),
                    "required": required,
                }
            )

    return init_params, child_refs


def _resolve_annotation(annotation: Any, eval_globals: Dict[str, Any]) -> Any:
    """Resolve a string forward reference against ``eval_globals``.

    Replaces ``typing.get_type_hints``, which crashes on Ray Serve
    ``Deployment`` objects (their ``__eq__`` accesses ``other._version``
    on typing sentinels that lack that attribute). We only need to
    *resolve* annotations to their classes, not run typing's full
    validation pipeline.
    """
    if annotation is inspect.Parameter.empty:
        return inspect.Parameter.empty
    if not isinstance(annotation, str):
        return annotation
    try:
        return eval(annotation, eval_globals, {})
    except Exception:
        return annotation  # leave as string; treated as a value param


def _load_app_class(class_id: str) -> Any:
    """Import ``module:qualname`` and return the bound attribute."""
    if ":" not in class_id:
        raise BioEngineUserError(
            f"Invalid class id {class_id!r}: expected 'module.path:ClassName'"
        )
    module_name, qualname = class_id.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BioEngineUserError(
            f"Cannot import {module_name!r} for app entry {class_id!r}: {exc}. "
            f"This usually means a top-of-file ``import`` in your package "
            f"references a dependency that isn't in your "
            f"``@bioengine.app(pip=[...])`` declaration. Either add it to "
            f"the pip list, or move the import inside a method body."
        ) from exc

    # Support nested classes by walking the qualname segments.
    obj: Any = module
    for segment in qualname.split("."):
        if not hasattr(obj, segment):
            raise BioEngineUserError(
                f"{module_name!r} has no attribute {qualname!r} "
                f"(missing segment {segment!r}). Check the manifest's "
                f"``entry`` field matches the @bioengine.app class."
            )
        obj = getattr(obj, segment)

    user_cls = _user_class_of(obj)
    if not getattr(user_cls, "_bioengine_app_marker", False):
        raise BioEngineUserError(
            f"{class_id!r} is not decorated with @bioengine.app — "
            f"manifest entry and composition handles must point at a "
            f"BioEngine app class."
        )
    return obj


def _user_class_of(deployment: Any) -> type:
    """Return the user class underlying a Ray Serve ``Deployment``."""
    inner = getattr(deployment, "func_or_class", None)
    return inner if inner is not None else deployment


# ──────────────────────── JSON-safe normalisation ────────────────────────


def _ensure_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` to JSON-friendly primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _ensure_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_ensure_jsonable(v) for v in obj]
    return repr(obj)


def _sanitise_schemas(schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip non-serialisable defaults from ``schema_method`` output.

    The ``hypha-rpc`` schema dicts already contain pydantic JSON schemas;
    they should round-trip through JSON but defensive normalisation keeps
    the worker robust against future schema-shape changes.
    """
    return [_ensure_jsonable(s) for s in schemas]


def _stringify_annotation(ann: Any) -> Optional[str]:
    if ann is None:
        return None
    if isinstance(ann, str):
        return ann
    name = getattr(ann, "__name__", None)
    if name is not None:
        return name
    return repr(ann)


_DEFAULT_SENTINEL = object()


def _safe_default(default: Any) -> Any:
    """Convert a parameter default to a JSON-safe representation.

    Returns the marker ``"__bioengine_no_default__"`` if no default exists
    (so consumers can distinguish "default is None" from "no default").
    """
    if default is inspect.Parameter.empty:
        return "__bioengine_no_default__"
    return _ensure_jsonable(default)


# ───────────────────────── application builder ───────────────────────────


def build_and_run_application(
    spec: Dict[str, Any],
    application_kwargs: Dict[str, Dict[str, Any]],
    proxy_args: Dict[str, Any],
    application_id: str,
    route_prefix: str,
    pkg_uri: str,
    proxy_memory_in_gb: float,
) -> Dict[str, Any]:
    """Assemble the Ray Serve bind graph AND call ``serve.run`` in-process.

    Returns a status dict — never a ``serve.Application``. Ray Serve's
    ``Deployment.__getattr__`` falls into unbounded recursion when an
    unpickled deployment is re-accessed, so the graph must stay inside
    the process that built it.

    ``pkg_uri`` is the Ray-internal ``gcs://`` URI of the user's app
    package. Every deployment's ``runtime_env`` is augmented with this
    URI so the replica venv can ``import`` the user's modules — without
    this, cloudpickle on the replica fails with ``ModuleNotFoundError``
    for deployments whose own ``pip=`` forced a fresh venv that didn't
    inherit ``py_modules`` from the calling task.

    ``proxy_memory_in_gb`` overrides ``ProxyDeployment.ray_actor_options
    .memory`` so the scheduler is biased toward a node with at least
    that much free memory for the WebSocket/WebRTC bridge. Ray treats
    ``memory`` as a placement hint, not a runtime cap.
    """
    from ray import serve

    from bioengine.apps.proxy_deployment import ProxyDeployment

    handles: Dict[str, Any] = {}

    def _with_pkg(cls: Any) -> Any:
        opts = dict(cls.ray_actor_options or {})
        runtime_env = dict(opts.get("runtime_env") or {})
        py_modules = list(runtime_env.get("py_modules") or [])
        if pkg_uri not in py_modules:
            py_modules.append(pkg_uri)
        runtime_env["py_modules"] = py_modules
        opts["runtime_env"] = runtime_env
        return cls.options(ray_actor_options=opts)

    def bind(cid: str) -> Any:
        if cid in handles:
            return handles[cid]
        meta = spec["classes"][cid]
        cls = _with_pkg(_load_app_class(cid))
        bind_kwargs: Dict[str, Any] = dict(application_kwargs.get(cid, {}))
        for param in meta["init_params"]:
            if param["kind"] == "deployment_handle":
                bind_kwargs[param["name"]] = bind(param["target"])
        handles[cid] = cls.bind(**bind_kwargs)
        return handles[cid]

    entry_handle = bind(spec["entry_id"])

    # Override the proxy's ray_actor_options.memory at deployment time.
    # The default 0 reservation lets Ray place the proxy on any node,
    # including resource-tight ones (e.g. the head). A real reservation
    # biases scheduling toward nodes with headroom for the
    # WebSocket/WebRTC payloads the proxy terminates.
    proxy_actor_options = dict(ProxyDeployment.ray_actor_options or {})
    proxy_actor_options["memory"] = int(proxy_memory_in_gb * (1024**3))
    proxy_cls = ProxyDeployment.options(ray_actor_options=proxy_actor_options)

    app = proxy_cls.bind(
        entry_deployment_handle=entry_handle,
        method_schemas=spec["classes"][spec["entry_id"]]["method_schemas"],
        **proxy_args,
    )

    serve.run(
        app,
        name=application_id,
        route_prefix=route_prefix,
        blocking=False,
    )
    return {"status": "submitted", "application_id": application_id}


# ───────────────────────── kwargs validation ─────────────────────────────


def validate_kwargs_against_spec(
    spec: Dict[str, Any],
    application_kwargs: Dict[str, Dict[str, Any]],
) -> None:
    """Validate user-provided kwargs against the AppSpec's ``init_params``.

    This runs on the *worker* (not in the Ray task) so users get fast,
    obvious errors before the build proceeds. We check presence of
    required params and reject unknown ones; deep type validation is
    deferred to the replica side, where actual types are reachable.

    Raises:
        BioEngineUserError: A required param is missing, or an unknown
            param was supplied.
    """
    for cid, meta in spec["classes"].items():
        provided = application_kwargs.get(cid, {})
        param_specs = {p["name"]: p for p in meta["init_params"]}

        for name in provided:
            if name not in param_specs:
                raise BioEngineUserError(
                    f"Unexpected init kwarg {name!r} for {cid}. "
                    f"Expected one of {sorted(param_specs)}."
                )

        for name, p in param_specs.items():
            if p["kind"] == "deployment_handle":
                continue  # filled in by the bind graph, not by the user
            if p["required"] and name not in provided:
                raise BioEngineUserError(
                    f"Missing required init kwarg {name!r} for {cid} "
                    f"(annotated as {p.get('annotation') or 'unknown'})."
                )
