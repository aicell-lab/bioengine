"""Report which serialising bound is held, on the surface an operator polls.

A caller starved behind a long call cannot see why, and neither can an
operator: ``get_app_status`` reports Ray Serve's view — application
``RUNNING``, every deployment ``HEALTHY`` at its replica target, every
``message`` empty — and Serve gives an app no field to write to, so "busy with
a 62-minute merge" and "healthy and idle" are the same response (#0072).

The replica *logs* are in that same response and are the one part a deployment
can write. They are also the right channel rather than merely the available
one: logs are read from the replica's log files, not through the replica's
request queue, and a probe that travels that queue cannot report that the queue
is blocked.

Two things this must not rely on, both measured on the europa worker
(bioengine 0.16.1, 2026-10-26):

* **Not the health check.** Ray Serve called ``check_health`` exactly once per
  replica — a hook on it logged nothing over three minutes with
  ``health_check_period_s`` at 30 — so a signal riding it would be silent for
  exactly the window it exists to describe. The reporter owns its own cadence
  instead, started from ``@bioengine.async_init`` (which that one call runs).
* **Not ``bioengine.logger``.** There it is a concrete ``logging.Logger``
  resolved when the app module is imported, which happens in the build task;
  Ray ships the deployment class by value, so the replica inherits a handler
  belonging to a dead process and every app record is dropped. ``ray.serve`` is
  the logger whose handlers write the replica log files, and it is what
  BioEngine's own replica-side code uses.
"""

import asyncio
import logging
import time
from typing import Callable, Dict, Optional


class BusyLog:
    """Logs ``BUSY <name> held for <n>s`` while ``probes[name]()`` is true.

    One instance per replica, started once from that replica's
    ``@bioengine.async_init``. ``probes`` maps a human-readable bound name to a
    predicate — ``asyncio.Lock.locked`` or ``asyncio.Semaphore.locked`` for a
    bound, or any callable for a long call the deployment tracks itself.

    The hold is re-stated every ``interval`` rather than logged once because the
    reachable log is a moving tail: cellpose's own progress output floods the
    replica's stderr during a long segmentation and Ray rotates the file, so a
    single line written at the start of a 330 s hold is gone from
    ``get_app_status``'s window long before the hold ends.
    """

    def __init__(
        self,
        probes: Dict[str, Callable[[], bool]],
        interval: float = 30.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._probes = probes
        self._interval = interval
        self._logger = logger or logging.getLogger("ray.serve")
        self._since: Dict[str, float] = {}
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                self.tick()
            except Exception as e:
                # The reporter is the only in-flight signal there is; dying
                # silently would read exactly like an idle replica.
                self._logger.warning(f"BUSY reporter tick failed: {e}")

    def tick(self) -> None:
        now = time.time()
        for name, held in self._probes.items():
            if held():
                self._logger.info(
                    f"BUSY {name} held for "
                    f"{now - self._since.setdefault(name, now):.0f}s - concurrent "
                    "callers are queued behind it, not failing"
                )
            elif name in self._since:
                self._logger.info(
                    f"BUSY {name} released after {now - self._since.pop(name):.0f}s"
                )
