"""A user-facing proxy around a Ray Serve ``DeploymentHandle`` that hides ``.remote()``.

In the v0.6 authoring model, composition is declared by type hint
(``def __init__(self, runtime_a: RuntimeA)``) and authors call methods as
plain async functions (``await self.runtime_a.ping(text)``). Ray Serve's
real interface is ``handle.ping.remote(text)``; this proxy bridges the gap.

The proxy returns the underlying ``DeploymentResponse`` unwrapped, so:

* ``await self.runtime_a.predict(x)`` → final result
* ``async for chunk in self.runtime_a.stream(prompt): ...`` → streaming works
* ``self.runtime_a.predict.options(multiplexed_model_id="m").remote(x)`` → still works

Power users who need raw Ray Serve behaviour can reach the underlying
``DeploymentHandle`` via ``runtime._raw``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ray.serve.handle import DeploymentHandle


class _BoundMethod:
    """A ``runtime.method_name`` accessor that forwards calls to ``.remote()``."""

    __slots__ = ("_handle", "_name")

    def __init__(self, handle: "DeploymentHandle", name: str) -> None:
        self._handle = handle
        self._name = name

    def __call__(self, *args: Any, **kwargs: Any):
        return getattr(self._handle, self._name).remote(*args, **kwargs)

    def options(self, **opts: Any) -> "_BoundMethod":
        configured = getattr(self._handle, self._name).options(**opts)
        return _BoundMethodFromHandle(configured)

    def remote(self, *args: Any, **kwargs: Any):
        """Explicit ``.remote()`` is accepted for parity with raw handles."""
        return self(*args, **kwargs)


class _BoundMethodFromHandle:
    """Variant returned by ``.options(...)`` — wraps an already-configured method."""

    __slots__ = ("_method",)

    def __init__(self, method: Any) -> None:
        self._method = method

    def __call__(self, *args: Any, **kwargs: Any):
        return self._method.remote(*args, **kwargs)

    def remote(self, *args: Any, **kwargs: Any):
        return self._method.remote(*args, **kwargs)

    def options(self, **opts: Any) -> "_BoundMethodFromHandle":
        return _BoundMethodFromHandle(self._method.options(**opts))


class BioEngineRuntimeHandle:
    """Proxy that exposes a ``@bioengine.app`` composition handle without ``.remote()``."""

    __slots__ = ("_handle", "_target_cls_id")

    def __init__(self, handle: "DeploymentHandle", target_cls_id: str = "") -> None:
        self._handle = handle
        self._target_cls_id = target_cls_id

    def __getattr__(self, name: str) -> _BoundMethod:
        if name.startswith("_"):
            raise AttributeError(name)
        return _BoundMethod(self._handle, name)

    @property
    def _raw(self) -> "DeploymentHandle":
        """Escape hatch: the underlying Ray Serve ``DeploymentHandle``."""
        return self._handle

    def __repr__(self) -> str:
        target = self._target_cls_id or "unknown"
        return f"<BioEngineRuntimeHandle target={target}>"
