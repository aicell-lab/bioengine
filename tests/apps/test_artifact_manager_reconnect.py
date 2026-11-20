"""The worker re-resolves its artifact-manager proxy instead of wedging forever.

A hypha_rpc service proxy is pinned to one *client instance* of the remote
service. The artifact manager can re-register under a new client id with nothing
having restarted — observed at KTH, hypha-server at restartCount 0 and two days
old — and hypha_rpc's websocket reconnect does not re-resolve cached proxies. The
worker resolves this one once in ``initialize()`` and holds it for the process
lifetime, so every artifact-backed call (``deploy_app``, ``list_apps``,
``get_app_manifest``) then hung to timeout while every local call stayed green.
Nothing recovered it: ``auto_redeploy`` needs the artifact read that is broken,
and ``run_code`` executes in Ray tasks so it cannot reach the in-process handle.
model-runner was down ~62 minutes and only a pod restart cleared it.

hypha_rpc's own stale-service retry does not reach this handle: it is installed by
``RPC.get_remote_service``, and a service resolved by name through
``server.get_service`` never goes through that path. This wrapper is the only
retry layer these calls have.

The retry rule is the delicate part, and it is not "retry transport errors":

* a **send-side** failure provably never reached the server, so retrying it is
  safe even for a write;
* a **timeout or mid-call disconnect** may already have executed, so retrying
  would risk double-running a ``create`` or ``commit``. Those re-raise, and only
  the handle is refreshed — one visible failure instead of an endless outage.
"""
from __future__ import annotations

import asyncio
import logging

import pytest
from hypha_rpc.rpc import RemoteException

from bioengine.apps.manager import _ReconnectingArtifactManager
from bioengine.utils.stale_service import stale_service_kind

logger = logging.getLogger("test-artifact-manager")


class _Proxy:
    """One resolved client instance of the artifact manager.

    The signatures mirror the real service: ``read``/``list``/``get_file`` take
    ``silent``, ``read_file``/``list_files`` do not.
    """

    id = "public/artifact-manager"

    def __init__(self, name: str, fails_with: BaseException | None = None):
        self.name = name
        self._fails_with = fails_with
        self.calls: list[tuple] = []
        self.kwargs_seen: list[dict] = []

    async def _answer(self, method, artifact_id, kwargs):
        await asyncio.sleep(0)
        self.calls.append((method, artifact_id))
        self.kwargs_seen.append(kwargs)
        if self._fails_with is not None:
            raise self._fails_with
        return {"id": artifact_id, "served_by": self.name}

    async def read(self, artifact_id, silent=False, version=None):
        return await self._answer(
            "read", artifact_id, {"silent": silent, "version": version}
        )

    async def list(self, parent_id=None, silent=False):
        return await self._answer("list", parent_id, {"silent": silent})

    async def read_file(self, artifact_id, file_path, version=None):
        return await self._answer(
            "read_file", artifact_id, {"file_path": file_path, "version": version}
        )

    async def list_files(self, artifact_id, dir_path=None):
        return await self._answer("list_files", artifact_id, {"dir_path": dir_path})

    async def commit(self, artifact_id, **kwargs):
        return await self._answer("commit", artifact_id, kwargs)


class _RecordingProxy(_Proxy):
    """Records the exact ``(args, kwargs)`` each method was dispatched with."""

    def __init__(self, name, fails_with=None):
        super().__init__(name, fails_with)
        self.dispatched: list[tuple] = []

    async def read(self, *args, **kwargs):
        self.dispatched.append((args, kwargs))
        return await super().read(*args, **kwargs)


class _Server:
    """Hands out a fresh proxy each time the service is resolved."""

    def __init__(self, *proxies: _Proxy):
        self._proxies = list(proxies)
        self.resolutions = 0

    async def get_service(self, service_id):
        assert service_id == "public/artifact-manager"
        await asyncio.sleep(0)
        self.resolutions += 1
        return self._proxies.pop(0)


def _wrap(server, proxy: _Proxy) -> _ReconnectingArtifactManager:
    return _ReconnectingArtifactManager(server=server, proxy=proxy, logger=logger)


def _remote_exception(rvalue: str, rtrace: str) -> RemoteException:
    """A remote failure exactly as ``hypha_rpc`` reconstructs it locally."""
    return RemoteException("RemoteError:" + rvalue + "\n" + rtrace)


# ===== the classifier =====


def test_send_side_failures_are_known_to_have_never_landed():
    assert (
        stale_service_kind(
            RuntimeError("Failed to send the request when calling method")
        )
        == "never_sent"
    )
    assert (
        stale_service_kind(RuntimeError("WebSocket reconnection timed out"))
        == "never_sent"
    )


def test_failures_that_may_have_landed_are_classified_separately():
    assert stale_service_kind(asyncio.TimeoutError()) == "maybe_sent"
    assert (
        stale_service_kind(ConnectionError("Client disconnected: ws/abc"))
        == "maybe_sent"
    )
    assert (
        stale_service_kind(
            RuntimeError("Method call timed out: ws/abc:services.x.read")
        )
        == "maybe_sent"
    )


def test_ordinary_errors_are_not_stale_proxy_failures():
    assert stale_service_kind(ValueError("artifact not found")) is None
    assert stale_service_kind(PermissionError("denied")) is None


def test_a_remote_failure_is_never_treated_as_unsent():
    """The remote traceback is part of the message the markers are scanned in.

    ``RemoteException`` carries ``"RemoteError:" + value + "\\n" + traceback``, and
    the hypha server runs hypha_rpc itself, so a server-side handler that fails
    while making its *own* RPC call produces a traceback containing a send-side
    marker verbatim. Matching it would classify a request that demonstrably
    reached the server — and ran — as one that never left this process.
    """
    exc = _remote_exception(
        "KeyError: 'manifest'",
        'File "/hypha/rpc.py", line 641, in handle_result\n'
        "Exception: Failed to send the request when calling method "
        "(ws/other:services.x.notify)\n",
    )

    assert stale_service_kind(exc) is None


def test_the_servers_own_routing_refusal_is_classified():
    """Measured against a live server: the shape this module exists for.

    A handle resolved by name whose target client has gone fails neither
    send-side nor by timeout — the server answers with an error envelope, so it
    arrives as a ``RemoteException`` like the test above. It carries no remote
    traceback, because nothing ran, and that is what tells the two apart.
    """
    refusal = RemoteException(
        "RemoteError:Target peer bioimage-io/bdeNa83tqUgQbsxJLsSP2m is not connected\n"
    )

    assert stale_service_kind(refusal) == "never_sent"


# ===== pass-through =====


@pytest.mark.asyncio
async def test_a_healthy_call_is_forwarded_untouched():
    proxy = _Proxy("first")
    server = _Server()
    am = _wrap(server, proxy)

    result = await am.read("ws/my-app")

    assert result["served_by"] == "first"
    assert server.resolutions == 0, "a healthy call must not re-resolve"


@pytest.mark.asyncio
async def test_an_application_error_propagates_without_re_resolving():
    """Only transport failures mean the handle is stale."""
    proxy = _Proxy("first", fails_with=ValueError("artifact not found"))
    server = _Server()
    am = _wrap(server, proxy)

    with pytest.raises(ValueError):
        await am.read("ws/missing")

    assert server.resolutions == 0


@pytest.mark.asyncio
async def test_a_remote_write_failure_is_not_replayed():
    """End to end for the classifier hole: a ``commit`` that already ran.

    The remote handler raised, so the version snapshot may exist. Replaying it
    against a fresh proxy would create a second one.
    """
    exc = _remote_exception(
        "RuntimeError: staging is empty",
        "Exception: Failed to send the request when calling method (ws/x:y)\n",
    )
    dead = _Proxy("dead", fails_with=exc)
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    with pytest.raises(RemoteException):
        await am.commit("ws/my-app")

    assert server.resolutions == 0, "a remote error says nothing about the handle"
    assert fresh.calls == [], "a write that reached the server must never be replayed"


# ===== the repair =====


@pytest.mark.asyncio
async def test_a_send_side_failure_re_resolves_and_retries():
    dead = _Proxy("dead", fails_with=RuntimeError("Failed to send the request"))
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    result = await am.read("ws/my-app")

    assert server.resolutions == 1
    assert result["served_by"] == "fresh", "the retry must use the new proxy"


@pytest.mark.asyncio
async def test_the_servers_routing_refusal_re_resolves_and_retries():
    """On ``origin/main`` this raised instead, and nothing was re-resolved.

    The live counterpart was run against ``https://hypha.aicell.io``:
    the same refusal, the real ``public/artifact-manager`` re-resolved, and the
    retry returned the artifact. This pins it without a server.
    """
    dead = _Proxy(
        "dead",
        fails_with=RemoteException(
            "RemoteError:Target peer bioimage-io/bdeNa83tqUgQbsxJLsSP2m is not "
            "connected\n"
        ),
    )
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    result = await am.read("ws/my-app")

    assert server.resolutions == 1
    assert result["served_by"] == "fresh"


@pytest.mark.asyncio
async def test_a_routing_refusal_on_a_write_is_retried_because_nothing_ran():
    """The counterpart to the timed-out-write guard below.

    A refusal is ``never_sent``, so unlike a timeout it carries no risk of
    double-execution — measured from an invocation log that did not
    move. Refusing to retry here would leave a write failing for a handle the
    wrapper had already replaced.
    """
    dead = _Proxy(
        "dead",
        fails_with=RemoteException(
            "RemoteError:Target peer bioimage-io/dead-client is not connected\n"
        ),
    )
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    result = await am.commit("ws/my-app")

    assert server.resolutions == 1
    assert result["served_by"] == "fresh"
    assert fresh.calls == [("commit", "ws/my-app")]


@pytest.mark.asyncio
async def test_a_timed_out_write_refreshes_the_handle_but_is_never_replayed():
    """The double-execute guard.

    A timeout cancels nothing at the far end, so a ``commit`` that timed out may
    still be running. Replaying it would create a second version snapshot.
    """
    dead = _Proxy("dead", fails_with=asyncio.TimeoutError())
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    with pytest.raises(asyncio.TimeoutError):
        await am.commit("ws/my-app")

    assert server.resolutions == 1, "the handle must still be refreshed"
    assert fresh.calls == [], "a write that may have landed must not be replayed"


@pytest.mark.asyncio
async def test_a_timed_out_read_is_retried_against_the_fresh_proxy():
    """The observed outage's own signature, end to end.

    The call that hung was ``read`` — the ``upload_app`` write before it had
    already succeeded, and ``deploy_app`` then failed at manifest load. Repeating
    a read is safe however the first attempt failed, so this is the case that
    takes the observed fault to zero failed calls rather than one.
    """
    dead = _Proxy("dead", fails_with=asyncio.TimeoutError())
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    result = await am.read("ws/my-app")

    assert server.resolutions == 1
    assert result["served_by"] == "fresh"


@pytest.mark.asyncio
async def test_a_timed_out_list_is_retried_and_silenced():
    """``list`` is the call behind ``list_apps``, and it takes ``silent`` too."""
    dead = _Proxy("dead", fails_with=asyncio.TimeoutError())
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    result = await am.list("ws/bioengine-apps")

    assert result["served_by"] == "fresh"
    assert fresh.kwargs_seen[-1]["silent"] is True


@pytest.mark.asyncio
async def test_a_timed_out_list_files_is_retried_unsilenced():
    dead = _Proxy("dead", fails_with=asyncio.TimeoutError())
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    result = await am.list_files("ws/my-app")

    assert result["served_by"] == "fresh"


# ===== the ``silent`` injection =====


@pytest.mark.asyncio
async def test_a_retried_read_does_not_count_a_second_view():
    """Reads are repeatable but not side-effect-free.

    ``silent`` defaults to False and ``read`` increments a view count, so a
    naive retry would double-count. The first attempt still counts normally;
    only the retry is silenced.
    """
    dead = _Proxy("dead", fails_with=asyncio.TimeoutError())
    fresh = _RecordingProxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    await am.read("ws/my-app")

    assert fresh.dispatched[-1] == (("ws/my-app",), {"silent": True})


@pytest.mark.asyncio
async def test_a_send_side_read_retry_is_not_silenced():
    """A call that never reached the server counted nothing, so the retry is
    the *first* real view and must be recorded as one."""
    dead = _Proxy("dead", fails_with=RuntimeError("Failed to send the request"))
    fresh = _RecordingProxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    await am.read("ws/my-app")

    assert "silent" not in fresh.dispatched[-1][1]


@pytest.mark.asyncio
async def test_a_read_without_a_silent_parameter_is_retried_unsilenced():
    """``read_file`` and ``list_files`` do not accept ``silent``; passing it
    would turn a recoverable timeout into a TypeError."""
    dead = _Proxy("dead", fails_with=asyncio.TimeoutError())
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    result = await am.read_file("ws/my-app", file_path="manifest.yaml")

    assert result["served_by"] == "fresh"


@pytest.mark.asyncio
async def test_an_explicit_silent_false_is_not_overridden():
    """The caller asked for the view to be counted; the retry is the only view
    that will ever happen for this call, so honour it."""
    dead = _Proxy("dead", fails_with=asyncio.TimeoutError())
    fresh = _RecordingProxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    await am.read("ws/my-app", silent=False)

    assert fresh.dispatched[-1][1]["silent"] is False


@pytest.mark.asyncio
async def test_a_positional_silent_does_not_collide_with_the_injected_one():
    """``read(aid, False)`` passes ``silent`` positionally. Injecting the keyword
    as well raises TypeError, turning a recoverable timeout into a hard failure.
    """
    dead = _Proxy("dead", fails_with=asyncio.TimeoutError())
    fresh = _RecordingProxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    result = await am.read("ws/my-app", False)

    assert result["served_by"] == "fresh"
    assert fresh.dispatched[-1] == (("ws/my-app", False), {})


# ===== concurrency =====


@pytest.mark.asyncio
async def test_the_next_call_after_a_timed_out_write_succeeds():
    """The outage is one failed call, not an indefinite wedge."""
    dead = _Proxy("dead", fails_with=asyncio.TimeoutError())
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    with pytest.raises(asyncio.TimeoutError):
        await am.commit("ws/my-app")

    result = await am.read("ws/my-app")
    assert result["served_by"] == "fresh"
    assert server.resolutions == 1, "the handle was already replaced"


@pytest.mark.asyncio
async def test_concurrent_failures_re_resolve_the_service_once():
    """Every in-flight call fails against the same dead proxy.

    Without the generation guard each one would resolve its own replacement,
    turning a single eviction into a burst of ``get_service`` calls. The fakes
    yield so the gathered coroutines genuinely interleave.
    """
    dead = _Proxy("dead", fails_with=RuntimeError("Failed to send the request"))
    fresh = _Proxy("fresh")
    server = _Server(fresh)
    am = _wrap(server, dead)

    results = await asyncio.gather(*(am.read(f"ws/app-{i}") for i in range(5)))

    assert server.resolutions == 1
    assert all(r["served_by"] == "fresh" for r in results)


@pytest.mark.asyncio
async def test_a_call_issued_during_a_re_resolve_waits_for_the_new_handle():
    """A write starting mid-repair must not be sent to the handle known to be dead.

    Without the gate it is dispatched to the dead proxy, burns a full method
    timeout and is re-raised — ~30s of avoidable outage per call in the window,
    and every write in it fails for nothing.
    """
    resolving = asyncio.Event()
    release = asyncio.Event()

    dead = _Proxy("dead", fails_with=asyncio.TimeoutError())
    fresh = _Proxy("fresh")

    class _SlowServer(_Server):
        async def get_service(self, service_id):
            resolving.set()
            await release.wait()
            return await super().get_service(service_id)

    server = _SlowServer(fresh)
    am = _wrap(server, dead)

    read = asyncio.create_task(am.read("ws/my-app"))
    await resolving.wait()

    commit = asyncio.create_task(am.commit("ws/my-app"))
    await asyncio.sleep(0)
    assert dead.calls == [("read", "ws/my-app")], "the commit must not reach the dead proxy"

    release.set()
    assert (await read)["served_by"] == "fresh"
    assert (await commit)["served_by"] == "fresh"
    assert server.resolutions == 1


@pytest.mark.asyncio
async def test_a_failed_re_resolve_surfaces_rather_than_being_swallowed():
    """If the service genuinely cannot be resolved, say so."""

    class _BrokenServer:
        resolutions = 0

        async def get_service(self, service_id):
            raise RuntimeError("Hypha is returning 500")

    dead = _Proxy("dead", fails_with=RuntimeError("Failed to send the request"))
    am = _wrap(_BrokenServer(), dead)

    with pytest.raises(RuntimeError, match="500"):
        await am.read("ws/my-app")

    fresh = _Proxy("fresh")
    am._server = _Server(fresh)
    assert (await am.read("ws/my-app"))["served_by"] == "fresh", (
        "a failed re-resolve must not leave callers blocked forever"
    )


# ===== attribute delegation =====


def test_a_non_callable_attribute_comes_from_the_underlying_proxy():
    """The proxy is a dict-backed ``ObjectProxy``: ``.id`` is the service id, not
    a method. Returning a coroutine function for it would silently corrupt any
    caller that reads service metadata."""
    am = _wrap(_Server(), _Proxy("first"))

    assert am.id == "public/artifact-manager"


def test_an_attribute_the_service_does_not_have_is_not_invented():
    am = _wrap(_Server(), _Proxy("first"))

    assert not hasattr(am, "totally_made_up")


# ===== the fix reaches the cached handle =====


@pytest.mark.asyncio
async def test_the_manager_wraps_the_proxy_it_caches(monkeypatch):
    """The wrapper is useless unless the manager actually installs it.

    This is the step that makes the fix reach ``AppBuilder`` and every
    ``artifact_utils`` helper for free, since they are all handed this object.
    """
    from hypha_rpc.rpc import RemoteService

    from bioengine.apps import manager as manager_module
    from bioengine.apps.manager import AppsManager

    class _Builder:
        received = None

        def complete_initialization(self, server, artifact_manager, worker_service_id):
            _Builder.received = artifact_manager

    collection_received = []

    async def ensure_collection(artifact_manager, workspace, logger):
        collection_received.append(artifact_manager)

    monkeypatch.setattr(
        manager_module, "ensure_applications_collection", ensure_collection
    )

    async def get_service(service_id):
        assert service_id == "public/artifact-manager"
        return _Proxy("first")

    server = RemoteService.fromDict(
        {
            "id": "ws/hypha:built-in",
            "config": {"workspace": "ws"},
            "get_service": get_service,
        }
    )

    manager = AppsManager.__new__(AppsManager)
    manager.logger = logger
    manager.app_builder = _Builder()

    await manager.complete_initialization(
        server=server, admin_users=[], worker_service_id="ws/worker"
    )

    assert isinstance(manager.artifact_manager, _ReconnectingArtifactManager)
    assert _Builder.received is manager.artifact_manager
    assert collection_received == [manager.artifact_manager]


# ---------------------------------------------------------------------------
# The FIRST resolution, in complete_initialization(), had no retry at all.
#
# The wrapper above can only help once it exists. Measured 2026-11-19 on a
# production Kubernetes deployment, both halves within one hour: the worker pod
# already running rode out a
# `public:artifact-manager` outage and re-registered unaided, while a pod started
# during the same window reached CrashLoopBackOff with restartCount 5 on
# `KeyError: 'Service not found: public/*:artifact-manager@*'`. A routine rollout
# landing in a platform blip was terminal; doing nothing survived it.
# ---------------------------------------------------------------------------
import ast
import inspect
import textwrap
import time as _time

from bioengine.apps import manager as _manager_module
from bioengine.apps.manager import AppsManager


class _FlakyServer:
    """Fails the first ``fail_times`` resolutions, then hands over a proxy."""

    def __init__(self, fail_times: int, error: BaseException | None = None):
        self._remaining = fail_times
        self._error = error or KeyError(
            "Service not found: public/*:artifact-manager@*"
        )
        self.attempts = 0

    async def get_service(self, service_id):
        assert service_id == "public/artifact-manager"
        await asyncio.sleep(0)
        self.attempts += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise self._error
        return _Proxy("after-retry")


def _manager_with(server) -> AppsManager:
    """An AppsManager with only the two attributes this method touches.

    ``__init__`` builds a RayCluster and a workspace on disk, none of which this
    resolution path reads, so constructing one would test the fixture instead of
    the method.
    """
    manager = object.__new__(AppsManager)
    manager.server = server
    manager.logger = logger
    return manager


@pytest.fixture
def fast_grace(monkeypatch):
    """Shrink the real schedule so the test exercises it rather than mocking it."""
    monkeypatch.setattr(_manager_module, "_ARTIFACT_MANAGER_RESOLVE_GRACE_S", 0.40)
    monkeypatch.setattr(
        _manager_module, "_ARTIFACT_MANAGER_RESOLVE_BACKOFF_INITIAL_S", 0.02
    )
    monkeypatch.setattr(
        _manager_module, "_ARTIFACT_MANAGER_RESOLVE_BACKOFF_MAX_S", 0.05
    )
    return 0.40


@pytest.mark.asyncio
async def test_a_transient_resolution_failure_at_boot_is_survived(fast_grace):
    """THE DEFECT: before the fix, one failure here killed the worker."""
    server = _FlakyServer(fail_times=3)
    proxy = await _manager_with(server)._resolve_artifact_manager_proxy(grace_s=_manager_module._ARTIFACT_MANAGER_RESOLVE_GRACE_S)

    assert proxy.name == "after-retry"
    assert server.attempts == 4, "should have retried exactly until it succeeded"


@pytest.mark.asyncio
async def test_a_healthy_boot_is_not_delayed(monkeypatch):
    """The retry must cost nothing when the service is there.

    A fix that sleeps before its first attempt would slow every normal start,
    which is a regression nobody would attribute to this change.

    The backoff is set far LARGER than the assertion window, so a single
    pre-attempt sleep would blow it. Real time is used rather than a patched
    ``asyncio.sleep``: patching it reaches the whole event loop, including this
    file's own fakes and pytest-asyncio, which is what made an earlier version
    of this test fail against a correct fix.
    """
    monkeypatch.setattr(_manager_module, "_ARTIFACT_MANAGER_RESOLVE_GRACE_S", 30.0)
    monkeypatch.setattr(
        _manager_module, "_ARTIFACT_MANAGER_RESOLVE_BACKOFF_INITIAL_S", 5.0
    )

    server = _FlakyServer(fail_times=0)
    started = _time.monotonic()
    proxy = await _manager_with(server)._resolve_artifact_manager_proxy(grace_s=_manager_module._ARTIFACT_MANAGER_RESOLVE_GRACE_S)
    elapsed = _time.monotonic() - started

    assert proxy.name == "after-retry"
    assert server.attempts == 1
    assert elapsed < 1.0, (
        f"a healthy boot took {elapsed:.2f}s against a 5s initial backoff — "
        "the first attempt is being delayed"
    )


@pytest.mark.asyncio
async def test_a_service_that_never_returns_still_RAISES_rather_than_hanging(
    fast_grace,
):
    """CONTROL. The fix bounds the wait; it must not swallow the failure.

    Without this, 'the worker no longer crash-loops' could be satisfied by a
    worker that hangs forever looking healthy — a worse outcome than exiting,
    and one the crash-loop at least made visible.
    """
    sentinel = KeyError("Service not found: public/*:artifact-manager@*")
    server = _FlakyServer(fail_times=10**9, error=sentinel)

    started = _time.monotonic()
    with pytest.raises(KeyError) as raised:
        await _manager_with(server)._resolve_artifact_manager_proxy(grace_s=_manager_module._ARTIFACT_MANAGER_RESOLVE_GRACE_S)
    elapsed = _time.monotonic() - started

    assert raised.value is sentinel, "the operator must see the real error"
    assert server.attempts > 1, "it should have retried before giving up"
    assert elapsed < fast_grace * 3, (
        f"gave up after {elapsed:.2f}s against a {fast_grace:.2f}s grace — "
        "the wait is not bounded by the grace"
    )


@pytest.mark.asyncio
async def test_the_wait_never_overshoots_the_grace(monkeypatch):
    """The backoff must be clamped to the remaining budget, not just doubled.

    ``delay`` grows independently of the deadline, so an unclamped sleep parks
    past the grace and silently widens the window this is meant to bound.

    The clamp is isolated by making the backoff LARGER than the whole grace:
    unclamped, the very first sleep would be 5 s against a 0.2 s budget, so the
    call could not return anywhere near the deadline. Clamped, it sleeps only
    the remaining budget and gives up on time. Real time again — see the note
    in the healthy-boot test about why patching ``asyncio.sleep`` is wrong here.
    """
    monkeypatch.setattr(_manager_module, "_ARTIFACT_MANAGER_RESOLVE_GRACE_S", 0.20)
    monkeypatch.setattr(
        _manager_module, "_ARTIFACT_MANAGER_RESOLVE_BACKOFF_INITIAL_S", 5.0
    )
    monkeypatch.setattr(
        _manager_module, "_ARTIFACT_MANAGER_RESOLVE_BACKOFF_MAX_S", 5.0
    )

    server = _FlakyServer(fail_times=10**9)
    started = _time.monotonic()
    with pytest.raises(KeyError):
        await _manager_with(server)._resolve_artifact_manager_proxy(grace_s=_manager_module._ARTIFACT_MANAGER_RESOLVE_GRACE_S)
    elapsed = _time.monotonic() - started

    assert server.attempts >= 1
    assert elapsed < 1.0, (
        f"took {elapsed:.2f}s to give up on a 0.20s grace with a 5s backoff — "
        "the sleep is not clamped to the remaining budget"
    )


def test_the_boot_path_resolves_THROUGH_the_retry_and_not_around_it():
    """TRIPWIRE: the call site, not the helper.

    Every test above would still pass if `complete_initialization` went back to
    calling `get_service("public/artifact-manager")` directly — the helper would
    simply be dead code. This asserts the wiring, so a future refactor cannot
    reintroduce the unprotected resolution while the suite stays green.
    """
    source = inspect.getsource(AppsManager.complete_initialization)
    tree = ast.parse(textwrap.dedent(source))

    bare_resolutions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(
            isinstance(arg, ast.Constant) and arg.value == "public/artifact-manager"
            for arg in node.args
        )
    ]
    assert not bare_resolutions, (
        "complete_initialization resolves 'public/artifact-manager' directly "
        "again; it must go through _resolve_artifact_manager_proxy, which is the "
        "only thing that survives a Hypha gap at boot"
    )

    retry_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_resolve_artifact_manager_proxy"
    ]
    assert len(retry_calls) == 1, (
        f"expected exactly one call to _resolve_artifact_manager_proxy, "
        f"found {len(retry_calls)}"
    )


# ── The grace must NOT reach the running worker's repair path ────────────────


@pytest.mark.asyncio
async def test_the_repair_path_fails_fast_instead_of_waiting_the_boot_grace():
    """The regression this guards against.

    `complete_initialization` is NOT boot-only: a running worker repairs an evicted
    registration through `_connect_to_server`, whose entire budget is RECONNECT_BUDGET_S
    (15 s) so the repair "fits inside one pass". With the grace applied unconditionally
    that repair blocked for up to 300 s, which stalled every later step in the tick,
    staled the 120 s heartbeat so the host census aged the worker out, and pushed
    `unreachable_for` past the condemn threshold on the FIRST attempt -- making the
    absorb-silently branch unreachable.

    The assertion is on ELAPSED TIME against the caller's own budget, not on a flag,
    because the flag is only interesting insofar as it bounds the wait.
    """
    from bioengine.utils.network import RECONNECT_BUDGET_S

    server = _FlakyServer(fail_times=10_000)   # never succeeds
    started = asyncio.get_event_loop().time()
    with pytest.raises(Exception):
        await _manager_with(server)._resolve_artifact_manager_proxy()   # default grace = 0
    elapsed = asyncio.get_event_loop().time() - started

    assert elapsed < RECONNECT_BUDGET_S, (
        "the repair path took %.2fs, which does not fit inside its own %.0fs budget"
        % (elapsed, RECONNECT_BUDGET_S)
    )
    assert server.attempts == 1, (
        "expected ONE attempt on the fail-fast path, got %d" % server.attempts
    )


@pytest.mark.asyncio
async def test_the_boot_path_still_retries_when_it_opts_in(fast_grace):
    """The control: the fix must not have simply deleted the retry.

    Without this, the test above would pass against a version that never waits at all --
    i.e. the original defect, reintroduced.
    """
    server = _FlakyServer(fail_times=3)
    proxy = await _manager_with(server)._resolve_artifact_manager_proxy(
        grace_s=_manager_module._ARTIFACT_MANAGER_RESOLVE_GRACE_S
    )
    assert proxy.name == "after-retry"
    assert server.attempts == 4, server.attempts


# ── The two call sites, pinned separately — not just the default ───────────────


def _enclosing_source(method):
    """The source of ONE method, so a match proves location and not mere presence.

    A test that greps the whole module passes against a mutant that deletes the
    opt-in from the boot path and adds it anywhere else in the file. Scoping each
    assertion to `inspect.getsource(<the method>)` is what makes it a statement
    about *where* the argument is passed.
    """
    import inspect

    return inspect.getsource(method)


def test_the_default_grace_is_FAIL_FAST_at_both_layers():
    """A caller that has not thought about the grace must get one attempt.

    Both the repair path (`_check_service_registration` -> `_connect_to_server()`
    with no arguments) and any future caller inherit these defaults, so the
    defaults are the actual safety property -- not the boot path's explicit value.
    """
    import inspect

    from bioengine.apps.manager import AppsManager
    from bioengine.worker.worker import BioEngineWorker

    checked = {}
    for label, fn in (
        ("AppsManager._resolve_artifact_manager_proxy", AppsManager._resolve_artifact_manager_proxy),
        ("AppsManager.complete_initialization", AppsManager.complete_initialization),
        ("BioEngineWorker._connect_to_server", BioEngineWorker._connect_to_server),
    ):
        params = inspect.signature(fn).parameters
        name = "grace_s" if "grace_s" in params else "artifact_manager_grace_s"
        assert name in params, f"{label} has no grace parameter at all"
        checked[label] = params[name].default

    assert checked, "nothing was checked"
    for label, default in checked.items():
        assert default == 0, (
            f"{label} defaults its grace to {default!r}; the repair path calls it "
            f"with no arguments, so a non-zero default puts the 300 s wait back on "
            f"the monitoring tick"
        )


def test_the_BOOT_call_site_opts_IN_and_the_REPAIR_call_site_does_NOT():
    """Where the opt-in lives, asserted per-method rather than per-file.

    The grace itself is not wrong -- it is right at boot. The problem is that
    `complete_initialization` is reached from TWO places and only one of them can
    afford to wait. So the fix is a property of the two call sites, and this is the
    arm that fails if someone later "simplifies" it back to one path.
    """
    from bioengine.worker.worker import BioEngineWorker

    boot = _enclosing_source(BioEngineWorker.start)
    repair = _enclosing_source(BioEngineWorker._check_service_registration)

    # CONTROL: both methods must actually call _connect_to_server, or the two
    # assertions below are vacuous -- an empty haystack satisfies "not in".
    assert "_connect_to_server" in boot, "boot path no longer calls _connect_to_server"
    assert "_connect_to_server" in repair, "repair path no longer calls _connect_to_server"

    assert "artifact_manager_grace_s" in boot, (
        "the boot path no longer opts into the grace, so a Hypha gap at boot "
        "crash-loops the worker again, which is the defect this module exists for"
    )
    assert "artifact_manager_grace_s" not in repair, (
        "the repair path now passes a grace; one monitoring tick can again block "
        "for the full grace, stale the heartbeat and condemn the registration on "
        "its first attempt"
    )
