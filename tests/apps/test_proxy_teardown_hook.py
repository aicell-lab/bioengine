"""Pin how ``ProxyDeployment``'s teardown hook behaves in and out of Ray Serve.

Serve gives a deployment class exactly one per-replica shutdown hook — the
destructor — and awaits it (``call_destructor`` in
``ray/serve/_private/replica.py``, whose comment says "Make sure to accept
``async def __del__(self)`` as well"). That is why the production destructor is
a coroutine function, and why making it synchronous would silently drop the
Hypha deregistration and client_id release the registration record depends on.

Nothing outside Serve awaits it, so every proxy a test hand-builds and drops
used to hand CPython's collector a coroutine it discarded with a
``RuntimeWarning``, charged to whichever unrelated test was running at the time.
``ProxyDouble`` exists to keep that off the suite.
"""

from __future__ import annotations

import gc
import inspect
import warnings

from tests.apps._proxy_double import PROXY_CLS, ProxyDouble


def _unawaited_destructor_warnings(cls) -> list[str]:
    """Build an instance of ``cls``, drop it, and collect the GC's complaints."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        instance = object.__new__(cls)
        del instance
        gc.collect()
        gc.collect()
    return [str(w.message) for w in caught if "never awaited" in str(w.message)]


def test_the_production_destructor_stays_awaitable() -> None:
    """Serve awaits ``__del__``; a synchronous one would return before the
    deregistration and disconnect it contains ever ran."""
    assert inspect.iscoroutinefunction(PROXY_CLS.__del__)


def test_dropping_the_production_class_warns() -> None:
    """Positive control for the test below — without it, a double that stopped
    suppressing anything would still look clean."""
    assert _unawaited_destructor_warnings(PROXY_CLS) == [
        "coroutine 'ProxyDeployment.__del__' was never awaited"
    ]


def test_dropping_the_test_double_is_silent() -> None:
    assert _unawaited_destructor_warnings(ProxyDouble) == []
