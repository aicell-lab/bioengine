"""Unit tests for ``bioengine._app.decorators`` and friends.

Covers the class-decoration behaviour: marker scanning, lifecycle
detection, composition-param extraction, reserved-name protection, and
``BioEngineRuntimeHandle`` proxy semantics.

None of these tests require a running Ray cluster — they exercise pure
metadata manipulation and proxy attribute access.
"""

import asyncio

import pytest
from pydantic import Field

import bioengine
from bioengine._app.errors import ReservedMethodNameError
from bioengine._app.runtime_handle import BioEngineRuntimeHandle


class _FakeMethod:
    """A fake Ray Serve method exposing ``.remote()`` and ``.options()``."""

    def __init__(self, name):
        self.name = name

    def remote(self, *args, **kwargs):
        return ("remote", self.name, args, kwargs)

    def options(self, **opts):
        method = self

        class _Configured:
            def remote(self, *a, **kw):
                return ("configured", method.name, a, kw, opts)

            def options(self, **more):
                merged = {**opts, **more}
                return method.options(**merged)

        return _Configured()


class _FakeHandle:
    def __getattr__(self, name):
        return _FakeMethod(name)


# ─────────────────────────── @bioengine.app ──────────────────────────────


def test_app_preserves_module_and_qualname():
    """Critical for cloudpickle by-reference: decorator must not subclass."""

    @bioengine.app(num_cpus=1)
    class MyApp:
        @bioengine.method
        async def ping(self) -> str:
            return "pong"

    user_cls = MyApp.func_or_class
    assert user_cls.__qualname__.endswith("MyApp")
    assert user_cls.__module__ == __name__


def test_app_marker_set_on_user_class():
    @bioengine.app()
    class App:
        pass

    assert App.func_or_class._bioengine_app_marker is True


def test_method_schemas_collected():
    @bioengine.app()
    class App:
        @bioengine.method
        async def alpha(self, x: int = Field(..., description="x")) -> dict:
            """Alpha."""
            return {"x": x}

        @bioengine.method
        async def beta(self) -> str:
            """Beta."""
            return "b"

        async def not_exposed(self):
            return "secret"

    names = {s["name"] for s in App.func_or_class._bioengine_method_schemas}
    assert names == {"alpha", "beta"}


def test_lifecycle_detection():
    @bioengine.app()
    class App:
        @bioengine.async_init
        async def setup(self):
            pass

        @bioengine.smoke_test
        async def verify(self):
            pass

        @bioengine.health_check
        async def liveness(self):
            pass

        @bioengine.cached(max_models=5)
        async def load_model(self, model_id: str):
            return None

    lc = App.func_or_class._bioengine_lifecycle
    assert lc["async_init"] == "setup"
    assert lc["smoke_test"] == "verify"
    assert lc["health_check"] == "liveness"
    assert "load_model" in lc["cached"]


def test_reserved_method_name_rejected():
    with pytest.raises(ReservedMethodNameError, match="check_health"):

        @bioengine.app()
        class Bad:
            def check_health(self):
                pass


def test_multiple_async_init_rejected():
    with pytest.raises(ReservedMethodNameError, match="more than one"):

        @bioengine.app()
        class Bad:
            @bioengine.async_init
            async def a(self):
                pass

            @bioengine.async_init
            async def b(self):
                pass


def test_ray_actor_options_translation():
    @bioengine.app(num_cpus=2, num_gpus=1, memory_mb=512, pip=["pandas"])
    class App:
        pass

    opts = App.ray_actor_options
    assert opts["num_cpus"] == 2
    assert opts["num_gpus"] == 1
    assert opts["memory"] == 512 * 1024 * 1024
    assert opts["runtime_env"]["pip"] == ["pandas"]


def test_ray_actor_options_extra_deep_merged():
    @bioengine.app(
        num_cpus=1,
        pip=["pandas"],
        ray_actor_options={"resources": {"custom": 1}, "runtime_env": {"pip": ["numpy"]}},
    )
    class App:
        pass

    opts = App.ray_actor_options
    assert opts["resources"] == {"custom": 1}
    assert opts["runtime_env"]["pip"] == ["numpy"]


# ─────────────────────── composition params ──────────────────────────────

# Module-level fixtures for composition tests — nested classes get
# ``<locals>`` in their qualname which prevents the introspection task
# from re-importing them; classes a user would write are always at module
# scope, so we test that shape here too.


@bioengine.app(num_cpus=1)
class _CompositionRuntimeA:
    @bioengine.method
    async def ping(self) -> str:
        return "pong"


@bioengine.app(num_cpus=0)
class _CompositionEntry:
    def __init__(self, runtime_a: _CompositionRuntimeA, batch_size: int = 32):
        self.runtime_a = runtime_a



def test_composition_params_extracted_from_type_hints():
    # Use module-level classes (defined below) — nested classes get a
    # ``<locals>`` qualname that the introspection task could not reimport.
    params = _CompositionEntry.func_or_class._bioengine_composition_params
    assert "runtime_a" in params
    assert params["runtime_a"] == f"{__name__}:_CompositionRuntimeA"
    assert "batch_size" not in params  # primitive value, not a deployment handle


def test_composition_params_empty_when_no_init():
    @bioengine.app()
    class App:
        pass

    assert App.func_or_class._bioengine_composition_params == {}


# ──────────────────── BioEngineRuntimeHandle proxy ───────────────────────


def test_runtime_handle_hides_remote():
    runtime = BioEngineRuntimeHandle(_FakeHandle(), "pkg:RuntimeA")
    result = runtime.ping(1, 2)
    assert result == ("remote", "ping", (1, 2), {})


def test_runtime_handle_options_forwarded():
    runtime = BioEngineRuntimeHandle(_FakeHandle())
    result = runtime.predict.options(stream=True)("x")
    assert result == ("configured", "predict", ("x",), {}, {"stream": True})


def test_runtime_handle_explicit_remote_still_accepted():
    runtime = BioEngineRuntimeHandle(_FakeHandle())
    assert runtime.method.remote("y") == ("remote", "method", ("y",), {})


def test_runtime_handle_raw_escape_hatch():
    handle = _FakeHandle()
    runtime = BioEngineRuntimeHandle(handle, "pkg:Runtime")
    assert runtime._raw is handle


def test_runtime_handle_rejects_underscore_attrs():
    runtime = BioEngineRuntimeHandle(_FakeHandle())
    with pytest.raises(AttributeError):
        runtime._private_method  # noqa: B018


def test_runtime_handle_repr_mentions_target():
    runtime = BioEngineRuntimeHandle(_FakeHandle(), "demo_app.runtimes:RuntimeA")
    assert "demo_app.runtimes:RuntimeA" in repr(runtime)


# ─────────────────────── module-level accessors ──────────────────────────


def test_datasets_raises_without_server_url(monkeypatch):
    """User code that touches bioengine.datasets in module scope should get a clear error."""
    from bioengine._app import accessors

    monkeypatch.delenv("BIOENGINE_DATA_SERVER_URL", raising=False)
    accessors._reset_for_tests()

    from bioengine._app.errors import MissingDataServerError

    with pytest.raises(MissingDataServerError, match="BIOENGINE_DATA_SERVER_URL"):
        bioengine.datasets.list_datasets  # noqa: B018


def test_datasets_caches_singleton(monkeypatch):
    from bioengine._app import accessors
    from bioengine.datasets import BioEngineDatasets

    monkeypatch.setenv("BIOENGINE_DATA_SERVER_URL", "auto")
    accessors._reset_for_tests()

    first = bioengine.datasets.list_datasets.__self__
    second = bioengine.datasets.list_datasets.__self__
    assert first is second
    assert isinstance(first, BioEngineDatasets)


def test_logger_lazy_attribute():
    assert bioengine.logger is not None
    assert hasattr(bioengine.logger, "info")


# ─────────────────────── ``__init__`` wrapping ───────────────────────────


def test_user_init_runs_after_framework_setup(monkeypatch):
    """The wrapped __init__ should call user code AND populate framework state."""
    monkeypatch.setenv("BIOENGINE_DATA_SERVER_URL", "auto")

    init_args = []

    @bioengine.app(num_cpus=0)
    class App:
        def __init__(self, value: str = "default"):
            init_args.append(value)
            self.value = value

    user_cls = App.func_or_class
    instance = user_cls(value="hello")
    assert init_args == ["hello"]
    assert instance.value == "hello"
    # Framework state present:
    assert instance._bioengine_replica_initialized is False
    assert instance._bioengine_replica_test_failed is False
    assert isinstance(instance._bioengine_health_check_lock, asyncio.Lock)


def test_runtime_handle_wraps_deployment_handle_in_init(monkeypatch):
    """Composition params get wrapped automatically; user sees BioEngineRuntimeHandle."""
    monkeypatch.setenv("BIOENGINE_DATA_SERVER_URL", "auto")

    @bioengine.app(num_cpus=0)
    class RuntimeA:
        @bioengine.method
        async def ping(self) -> str:
            return "pong"

    @bioengine.app(num_cpus=0)
    class Entry:
        def __init__(self, runtime_a: RuntimeA):
            self.runtime_a = runtime_a

    # Construct directly with a fake DeploymentHandle subclass.
    from ray.serve.handle import DeploymentHandle

    class _Fake(DeploymentHandle):  # type: ignore[misc]
        def __init__(self):
            pass

        def __getattr__(self, name):
            return _FakeMethod(name)

    user_cls = Entry.func_or_class
    instance = user_cls(runtime_a=_Fake())
    assert isinstance(instance.runtime_a, BioEngineRuntimeHandle)
