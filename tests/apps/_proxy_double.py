"""A ``ProxyDeployment`` stand-in for tests that no collector has to finalise.

``ProxyDeployment.__del__`` is ``async def`` because a deployment class's only
per-replica shutdown hook in Ray Serve *is* the destructor, and Serve awaits it
(``call_destructor`` in ``ray/serve/_private/replica.py``). CPython's collector
does not await it: it calls ``__del__``, gets a coroutine object back and drops
it. Every hand-built proxy a test leaves behind therefore turns into
``RuntimeWarning: coroutine 'ProxyDeployment.__del__' was never awaited``,
reported against whichever unrelated test the collector happened to run in.

Tests build their stand-ins from :class:`ProxyDouble`, which is the production
class minus that Serve-only hook. Nothing else differs, so source assertions and
behaviour tests should keep using ``PROXY_CLS``.
"""

from __future__ import annotations

from bioengine.apps import proxy_deployment as pd_module

PROXY_CLS = pd_module.ProxyDeployment.func_or_class


class ProxyDouble(PROXY_CLS):
    """``ProxyDeployment`` without the async destructor Ray Serve drives."""

    def __del__(self) -> None:
        """Drop the inherited ``async def __del__``.

        No Serve replica owns a test instance, so there is nobody to await the
        production destructor — leaving it in place only hands the collector a
        coroutine it will discard with a warning.
        """
