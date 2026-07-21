"""Bootstrap functions that run inside the app's Ray ``runtime_env``.

The BioEngine worker never imports user app code directly. The flow is:

1. Worker submits :func:`introspect_app_in_ray_task` as a short Ray task. The
   task materialises the user package into ``<app_dir>/source/`` (Hypha
   download, short-TTL token), walks the type-hint composition graph via
   :func:`introspect_app`, packages the source dir into Ray's internal
   GCS package store, and returns ``{spec, app_source_uri}``.
2. Worker checks resources against the spec.
3. Worker submits :func:`build_and_run_application` as a second Ray task. The
   task re-materialises source (no-op on shared-FS clusters; a Ray-GCS pull
   via ``BIOENGINE_APP_SOURCE_URI`` on non-shared ones — no Hypha auth at
   this point) and calls ``cls.bind`` + ``serve.run(blocking=False)``.

Replicas don't run any function in this module directly; they boot with
``BIOENGINE_APP_SOURCE_URI`` in their env_vars and call the same
``_ensure_source`` to materialise their copy of the source, bypassing
Hypha entirely.
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


def introspect_app_in_ray_task(
    entry_id: str,
    env_vars: Dict[str, str],
) -> Dict[str, Any]:
    """Phase-1 Ray task: download user source, introspect, package to Ray-GCS.

    Returns ``{"spec": …, "app_source_uri": "gcs://_ray_pkg_<hash>.zip"}``.

    The download uses the Hypha ``BIOENGINE_ARTIFACT_DOWNLOAD_URL`` +
    ``_TOKEN`` env vars (short-TTL). Once source is on disk, we hash and
    upload it to Ray's internal package store; the resulting URI is what
    every replica's ``runtime_env`` ships so they can re-materialise the
    same bytes without going back through Hypha (token expires before
    most replicas finish their pip install).
    """
    import logging
    import os
    from pathlib import Path

    from bioengine._app.replica_init import _ensure_source

    # Apply env_vars from runtime_env onto the process so _ensure_source
    # reads the same keys as the eventual replicas. Overwrite (not
    # setdefault): this build's version/URI must win even on a worker whose
    # process env somehow already carries a prior build's values.
    for key, value in env_vars.items():
        os.environ[key] = value

    app_dir = Path(env_vars["BIOENGINE_APP_DIR"])
    version = env_vars.get("BIOENGINE_ARTIFACT_VERSION", "")
    if not version:
        raise BioEngineUserError(
            "BIOENGINE_ARTIFACT_VERSION not in task env_vars; the worker "
            "is expected to populate it before submitting the introspect "
            "task."
        )

    logger = logging.getLogger("ray.serve")
    source = _ensure_source(app_dir, version, logger)

    src_str = str(source)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    spec = introspect_app(entry_id)
    app_source_uri = _package_source_to_ray_gcs(source)
    return {"spec": spec, "app_source_uri": app_source_uri}


def _package_source_to_ray_gcs(source_dir: "Path") -> str:  # type: ignore[name-defined]
    """Hash + upload ``source_dir`` to Ray's internal package store.

    Returns the ``gcs://_ray_pkg_<hash>.zip`` URI replicas pass to
    :func:`ray._private.runtime_env.packaging.download_and_unpack_package`
    to re-materialise the same bytes.

    Why not use ``runtime_env.py_modules=[<local_path>]`` and let Ray
    Serve upload for us? Because the bootstrap puts source at a stable
    ``<app_dir>/source/`` (so cellpose's ``home/`` checkpoints survive
    version bumps), not Ray's per-process ``runtime_resources/`` dir.
    Explicit upload here decouples the GCS URI from runtime_env handling.

    Ray patch versions disagree on whether ``include_gitignore`` is a
    required argument on these packaging APIs; the inspect gates keep us
    cross-compatible.
    """
    from ray._private.runtime_env.packaging import (
        get_uri_for_directory,
        upload_package_if_needed,
    )

    src = str(source_dir.resolve())

    get_kwargs: Dict[str, Any] = {}
    if "include_gitignore" in inspect.signature(get_uri_for_directory).parameters:
        get_kwargs["include_gitignore"] = False
    uri = get_uri_for_directory(src, **get_kwargs)

    upload_kwargs: Dict[str, Any] = {}
    upload_sig = inspect.signature(upload_package_if_needed)
    if "include_gitignore" in upload_sig.parameters:
        upload_kwargs["include_gitignore"] = False
    # Belt-and-braces: exclude any leftover Ray package zip from the
    # source tree. Earlier dev iterations of v0.11.4 wrote temp zips
    # into ``src`` itself (when ``base_directory == src``), Ray's walker
    # then included its own output back into the package and looped to
    # 360 GiB on shared NFS before the task was killed. The fix below
    # uses a scratch ``base_directory``, but if any artifact / pod still
    # has a stale ``_ray_pkg_*.zip`` sitting in ``source/`` from history,
    # this exclude keeps it out of the new package.
    upload_kwargs["excludes"] = ["*.zip", "_ray_pkg_*.zip"]
    # ``base_directory`` is a scratch dir where Ray writes the temporary
    # zip file before pushing to GCS. Must not point at the source dir
    # itself.
    import tempfile
    base_dir = tempfile.mkdtemp(prefix="bioengine-pkg-")
    try:
        # Ray patches disagree on the third positional name:
        # Ray 2.5x ships ``(pkg_uri, base_directory, module_path, ...)``;
        # earlier patches expose ``(pkg_uri, base_directory, ...)`` only.
        if "module_path" in upload_sig.parameters:
            upload_package_if_needed(uri, base_dir, src, **upload_kwargs)
        elif "package_path" in upload_sig.parameters:
            upload_package_if_needed(uri, base_dir, src, **upload_kwargs)
        else:
            upload_package_if_needed(uri, base_dir, **upload_kwargs)
    finally:
        import shutil as _shutil
        _shutil.rmtree(base_dir, ignore_errors=True)
    return uri


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


_REQ_NAME_SPLIT = ("==", ">=", "<=", "~=", ">", "<", "[")


def _requirement_name(req: str) -> str:
    """Extract the package name from a pip requirement string.

    ``pandas==2.2.0`` → ``pandas``; ``hypha-rpc>=0.21`` → ``hypha-rpc``;
    ``httpx[http2]==0.28.1`` → ``httpx``.
    """
    out = req.strip()
    for sep in _REQ_NAME_SPLIT:
        if sep in out:
            out = out.split(sep, 1)[0]
    return out.strip().lower()


def _merge_pip_lists(base: List[str], to_add: List[str]) -> List[str]:
    """Append entries from ``to_add`` to ``base`` unless an entry with the
    same package name already exists. User-declared entries (in ``base``)
    win on name collision so the framework never silently rewrites a
    pinned version the user set."""
    existing_names = {_requirement_name(r) for r in base}
    merged = list(base)
    for req in to_add:
        if _requirement_name(req) not in existing_names:
            merged.append(req)
            existing_names.add(_requirement_name(req))
    return merged


#: Kept for back-compat with tests that pin the hook name; Ray Serve does
#: NOT honour ``runtime_env.worker_process_setup_hook`` for its replicas
#: (no references under ``ray/serve/``) so the actual replica bootstrap
#: is invoked from ``bioengine/__init__.py`` as a ``sys.meta_path`` finder
#: that fires the source-materialise step on the first unresolved import
#: (the cellpose-finetuning case where cloudpickle reaches the user
#: module before any ``import bioengine``). See PR #119 commit log for
#: the empirical sequence of attempts that led here.
_REPLICA_SETUP_HOOK = "bioengine._app.replica_init:setup_replica_environment"


def build_and_run_application(
    spec: Dict[str, Any],
    application_kwargs: Dict[str, Dict[str, Any]],
    proxy_args: Dict[str, Any],
    application_id: str,
    route_prefix: str,
    bioengine_uri: str,
    app_source_uri: str,
    proxy_pip: List[str],
    user_replica_framework_pip: List[str],
    replica_env_vars: Dict[str, str],
    proxy_memory_in_gb: float,
    scaling: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble the Ray Serve bind graph AND call ``serve.run`` in-process.

    Returns a status dict — never a ``serve.Application``. Ray Serve's
    ``Deployment.__getattr__`` falls into unbounded recursion when an
    unpickled deployment is re-accessed, so the graph must stay inside
    the process that built it.

    Runs in a Ray task on whatever node Ray schedules — no head-node pin.
    ``bioengine`` is available via the task's ``runtime_env.py_modules =
    [bioengine_uri]``. The user's app source is materialised here by
    :func:`bioengine._app.replica_init._ensure_source` using the Ray-GCS
    branch (``BIOENGINE_APP_SOURCE_URI``) which was just uploaded by the
    introspect Ray task — the Hypha short-TTL token is not needed and
    isn't in ``replica_env_vars``.

    Each deployment's ``runtime_env`` is augmented with
    ``py_modules=[bioengine_uri]`` and ``env_vars`` carrying
    ``BIOENGINE_APP_SOURCE_URI`` + ``BIOENGINE_APP_DIR`` + secrets. The
    actual source download on each replica runs via the
    ``sys.meta_path`` finder installed in :mod:`bioengine.__init__`,
    which triggers the same ``_ensure_source`` call before the user
    module's import is retried.

    ``proxy_memory_in_gb`` overrides ``ProxyDeployment.ray_actor_options
    .memory`` so the scheduler is biased toward a node with at least
    that much free memory for the WebSocket/WebRTC bridge. Ray treats
    ``memory`` as a placement hint, not a runtime cap.
    """
    import logging
    import os
    from pathlib import Path

    from ray import serve

    from bioengine._app.replica_init import _ensure_source
    from bioengine.apps.proxy_deployment import ProxyDeployment

    # Re-apply env_vars from the task's runtime_env so the source loader
    # below sees the same BIOENGINE_* keys as the eventual replicas.
    # Overwrite (not setdefault) so this build's values win.
    for key, value in replica_env_vars.items():
        os.environ[key] = value
    # The Ray-GCS source URI lives on the task env explicitly; replicas
    # get it via per-deployment env_vars injected in `_with_pkg`.
    os.environ["BIOENGINE_APP_SOURCE_URI"] = app_source_uri

    head_app_dir = Path(replica_env_vars["BIOENGINE_APP_DIR"])
    head_version = replica_env_vars.get("BIOENGINE_ARTIFACT_VERSION") or spec.get(
        "version", ""
    )
    head_source = _ensure_source(
        head_app_dir, head_version, logging.getLogger("ray.serve")
    )
    src_str = str(head_source)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    handles: Dict[str, Any] = {}

    def _with_pkg(cls: Any) -> Any:
        opts = dict(cls.ray_actor_options or {})
        runtime_env = dict(opts.get("runtime_env") or {})
        # Ray Serve replicas do NOT inherit job-level py_modules (observed
        # empirically on KTH). The bioengine package has to ride in the
        # deployment's runtime_env to be on sys.path at
        # ``cloudpickle.loads`` time, otherwise the replica dies at
        # ``__init__`` with ``ModuleNotFoundError: No module named
        # 'bioengine'``. ``bioengine_uri`` is a content-hashed gcs URI
        # already pushed by the worker's ``ray.init`` upload, so handing
        # it back here re-resolves to the cached package — no second
        # GCS upload, no ray-client gRPC bridge wedge.
        py_modules = list(runtime_env.get("py_modules") or [])
        if bioengine_uri not in py_modules:
            py_modules.append(bioengine_uri)
        # Ship the user source via py_modules too. Ray extracts py_modules
        # to ``/tmp/ray/.../runtime_resources/py_modules_files/<hash>/``
        # and adds that dir to PYTHONPATH BEFORE the replica's Python
        # interpreter runs cloudpickle.loads — so ``import main``
        # (cellpose) or ``import entry`` (model-runner) resolves natively
        # without any hook / finder / pickle-by-value trick. The source
        # is content-hashed in Ray's GCS package store by the introspect
        # task; replicas pull from there.
        if app_source_uri not in py_modules:
            py_modules.append(app_source_uri)
        runtime_env["py_modules"] = py_modules
        # Ray's default_worker.py honours ``worker_process_setup_hook``
        # *before* Ray Serve's ServeReplica.__init__ runs cloudpickle.loads.
        # Setting it to ``bioengine._app.replica_init:setup_replica_environment``
        # forces ``import bioengine._app.replica_init`` → ``import bioengine``
        # → ``_install_replica_bootstrap_finder`` BEFORE the user class is
        # loaded. The hook itself populates ``<app_dir>/source/`` and adds
        # it to ``sys.path`` so cloudpickle.loads can resolve user
        # modules like ``main`` (cellpose) or ``entry`` (model-runner)
        # without by-value vs by-ref pickling tricks.
        runtime_env["worker_process_setup_hook"] = _REPLICA_SETUP_HOOK
        # Merge the framework-required pip deps (hypha-rpc, pydantic)
        # into whatever the user declared via ``@bioengine.app(pip=…)``.
        # The replica needs them at cloudpickle.loads time to resolve
        # references that the ``@bioengine.method`` wrapping created
        # via ``hypha_rpc.utils.schema.schema_method``. User-declared
        # entries take precedence on package name.
        runtime_env["pip"] = _merge_pip_lists(
            list(runtime_env.get("pip") or []),
            user_replica_framework_pip,
        )
        # Merge the worker-side env_vars + the app-source URI into
        # whatever static ``env_vars`` the author declared via
        # ``@bioengine.app(env_vars=…)``. The author's entries take
        # precedence on key collision.
        deploy_env_vars = {
            **replica_env_vars,
            "BIOENGINE_APP_SOURCE_URI": app_source_uri,
            **(runtime_env.get("env_vars") or {}),
        }
        runtime_env["env_vars"] = deploy_env_vars
        opts["runtime_env"] = runtime_env
        return cls.options(ray_actor_options=opts)

    scaling_map = dict(scaling or {})

    def bind(cid: str) -> Any:
        if cid in handles:
            return handles[cid]
        meta = spec["classes"][cid]
        cls = _with_pkg(_load_app_class(cid))
        # Per-deployment scaling: the user keys the dict by class name
        # (matching what get_app_status reports). Multi-class apps can
        # mix fixed and autoscaling configs across deployments.
        class_name = meta["qualname"].split(".")[-1]
        per_class = scaling_map.get(class_name) or {}
        per_class_autoscale = per_class.get("autoscaling_config")
        per_class_replicas = per_class.get("num_replicas")
        if per_class_autoscale:
            cls = cls.options(autoscaling_config=per_class_autoscale)
        elif per_class_replicas is not None and per_class_replicas != 1:
            cls = cls.options(num_replicas=per_class_replicas)
        bind_kwargs: Dict[str, Any] = dict(application_kwargs.get(cid, {}))
        for param in meta["init_params"]:
            if param["kind"] == "deployment_handle":
                bind_kwargs[param["name"]] = bind(param["target"])
        handles[cid] = cls.bind(**bind_kwargs)
        return handles[cid]

    entry_handle = bind(spec["entry_id"])

    # Override the proxy's ray_actor_options.memory at deployment time.
    # ``num_cpus=0`` is set on the decorator (see proxy_deployment.py); a
    # real memory reservation biases scheduling toward nodes with headroom
    # for the WebSocket/WebRTC payloads the proxy terminates. ``proxy_pip``
    # is computed on the worker (where bioengine has dist-info); the proxy
    # class deliberately ships with no static ``runtime_env`` because
    # resolving the pip list at import time crashed the actor pod —
    # see :mod:`bioengine.apps.proxy_deployment` for the rationale.
    proxy_actor_options = dict(ProxyDeployment.ray_actor_options or {})
    proxy_actor_options["memory"] = int(proxy_memory_in_gb * (1024**3))
    proxy_runtime_env = dict(proxy_actor_options.get("runtime_env") or {})
    proxy_runtime_env["pip"] = proxy_pip
    # Same as user deployments above: replicas don't inherit job-level
    # py_modules, so include the bioengine gcs URI.
    proxy_py_modules = list(proxy_runtime_env.get("py_modules") or [])
    if bioengine_uri not in proxy_py_modules:
        proxy_py_modules.append(bioengine_uri)
    if app_source_uri not in proxy_py_modules:
        proxy_py_modules.append(app_source_uri)
    proxy_runtime_env["py_modules"] = proxy_py_modules
    proxy_runtime_env["worker_process_setup_hook"] = _REPLICA_SETUP_HOOK
    proxy_runtime_env["env_vars"] = {
        **replica_env_vars,
        "BIOENGINE_APP_SOURCE_URI": app_source_uri,
        **(proxy_runtime_env.get("env_vars") or {}),
    }
    proxy_actor_options["runtime_env"] = proxy_runtime_env
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
