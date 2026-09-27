"""Pin the framework-level usage counter.

BioEngine could not say how many calls it had ever served: the only per-call
record was a line in a Ray Serve replica log, and those die with the Ray
session. The contract these tests hold in place:

* A count survives the replica that took it — a restart adds a shard, it does
  not reset the total.
* Attempted, answered, failed and rejected are separate claims. A deployment
  that wedges mid-request leaves attempted pinned at max_ongoing_requests,
  answered at zero and rejected rising, which is exactly the shape that made
  the original log-derived figure unusable.
* Two writers in the same directory cannot lose each other's increments; a
  reader never sees a half-written shard; and two writes racing on the *same*
  shard — a stats read against the maintenance tick, in one process — do not
  tear it either.
* Traffic is attributed to a caller *class*, biased so the external figure is a
  floor. No caller identity reaches disk — an unattributed total is mostly the
  project's own traffic and is not publishable.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from bioengine.apps import proxy_deployment as pd_module
from bioengine.apps._cache_fs_tasks import clear_dir_on_node, list_dirs_on_node
from bioengine.apps.usage_ledger import (
    ANONYMOUS,
    EVENTS,
    EXTERNAL,
    INTERNAL,
    LEDGER_DIRNAME,
    UNKNOWN,
    UsageLedger,
    classify_caller,
    ledger_dir_for_app,
    read_usage,
)
from tests.apps._proxy_double import ProxyDouble


# ===== helpers =====


def _ledger(tmp_path: Path, writer_id: str, **metadata) -> UsageLedger:
    return UsageLedger(tmp_path / "ledger", writer_id, metadata or None)


class _Method:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    async def remote(self, *args, **kwargs):
        return await self._behaviour(*args, **kwargs)


class _Handle:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    def __getattr__(self, name):
        return _Method(self._behaviour)


def _bare_proxy(tmp_path: Path, behaviour, *, authorized_users=None, slots=4):
    inst = object.__new__(ProxyDouble)
    inst.application_id = "counted-app"
    inst.workspace = "host-ws"
    inst.authorized_users = authorized_users or {"*": ["*"]}
    inst.max_ongoing_requests = slots
    inst.service_semaphore = asyncio.Semaphore(slots)
    inst.entry_deployment_handle = _Handle(behaviour)
    inst._usage_flush_lock = asyncio.Lock()
    inst._usage_ledger = UsageLedger(
        tmp_path / "ledger", "replica-a", {"app_version": "1.0.0"}
    )
    return inst


def _totals(**overrides) -> dict:
    return {**{event: 0 for event in EVENTS}, **overrides}


def _method(proxy, monkeypatch, name="infer", wants_context=False):
    monkeypatch.setattr(pd_module, "schema_function", lambda func, **kw: func)
    return proxy._create_deployment_function(
        {
            "name": name,
            "description": "",
            "parameters": {"type": "object", "properties": {}},
            "wants_context": wants_context,
        }
    )


def _user_context(**user):
    return {"user": {"id": "u-1", "email": "u@example.org", **user}}


# ===== monotonicity across the life of a replica =====


def test_count_survives_a_replica_restart(tmp_path: Path) -> None:
    before = _ledger(tmp_path, "replica-1")
    for _ in range(3):
        before.record("infer", EXTERNAL, "attempted")
        before.record("infer", EXTERNAL, "answered")
    before.flush()

    after = _ledger(tmp_path, "replica-2")
    after.record("infer", EXTERNAL, "attempted")
    after.record("infer", EXTERNAL, "answered")
    after.flush()

    stats = read_usage(tmp_path / "ledger")
    assert stats["shards"] == 2
    assert stats["totals"] == _totals(attempted=4, answered=4)


def test_count_survives_a_version_bump(tmp_path: Path) -> None:
    old = _ledger(tmp_path, "replica-1", app_version="1.0.0")
    old.record("infer", EXTERNAL, "attempted")
    old.flush()

    new = _ledger(tmp_path, "replica-2", app_version="2.0.0")
    new.record("infer", EXTERNAL, "attempted")
    new.flush()

    stats = read_usage(tmp_path / "ledger")
    assert stats["totals"]["attempted"] == 2
    assert stats["app_versions"] == ["1.0.0", "2.0.0"]


def test_a_reused_writer_id_resumes_instead_of_overwriting(tmp_path: Path) -> None:
    """Ray may hand a restarted replica the same id; its predecessor's totals
    must not be clobbered."""
    first = _ledger(tmp_path, "replica-1")
    for _ in range(5):
        first.record("infer", EXTERNAL, "attempted")
    first.flush()

    second = _ledger(tmp_path, "replica-1")
    second.record("infer", EXTERNAL, "attempted")
    second.flush()

    stats = read_usage(tmp_path / "ledger")
    assert stats["shards"] == 1
    assert stats["totals"]["attempted"] == 6


def test_shards_outlive_the_app_cache_directory(tmp_path: Path) -> None:
    """Clearing an app's cache frees disk; it must not reset the counter."""
    apps_workdir = tmp_path / "apps"
    app_dir = apps_workdir / "host-ws-counted-app"
    app_dir.mkdir(parents=True)

    ledger = UsageLedger(ledger_dir_for_app(app_dir), "replica-1")
    ledger.record("infer", EXTERNAL, "attempted")
    ledger.flush()

    result = clear_dir_on_node(str(apps_workdir), "host-ws", "counted-app", "node-1")
    assert result["ok"] is True
    assert not app_dir.exists()

    stats = read_usage(ledger_dir_for_app(app_dir))
    assert stats["totals"]["attempted"] == 1


async def test_clear_app_directory_refuses_to_delete_the_ledger_root() -> None:
    """Hiding the directory from the listing is not the same as closing the
    API: the clear task falls back to a bare directory name, so
    application_id='.bioengine-usage' would rmtree every app's counters."""
    from bioengine.apps.manager import AppsManager

    manager = object.__new__(AppsManager)
    manager._check_initialized = lambda: None
    manager.admin_users = ["admin@lab.org"]
    context = {"user": {"id": "admin-id", "email": "admin@lab.org"}}

    for bad in (LEDGER_DIRNAME, ".", "..", ".hidden"):
        with pytest.raises(ValueError, match="plain name"):
            await manager.clear_app_directory(application_id=bad, context=context)


def test_ledger_directory_is_not_offered_as_an_app_cache(tmp_path: Path) -> None:
    apps_workdir = tmp_path / "apps"
    app_dir = apps_workdir / "host-ws-counted-app"
    app_dir.mkdir(parents=True)
    ledger = UsageLedger(ledger_dir_for_app(app_dir), "replica-1")
    ledger.record("infer", EXTERNAL, "attempted")
    ledger.flush()

    listed = list_dirs_on_node(str(apps_workdir), "host-ws-", [], "node-1")
    assert [entry["name"] for entry in listed] == ["host-ws-counted-app"]


# ===== attempted vs answered =====


def test_attempted_and_answered_diverge_when_the_call_fails(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, "replica-1")
    ledger.record("infer", EXTERNAL, "attempted")
    ledger.record("infer", EXTERNAL, "answered")
    ledger.record("infer", EXTERNAL, "attempted")
    ledger.record("infer", EXTERNAL, "failed")
    ledger.flush()

    totals = read_usage(tmp_path / "ledger")["totals"]
    assert totals == _totals(attempted=2, answered=1, failed=1)


async def test_proxy_counts_an_answered_call(tmp_path: Path, monkeypatch) -> None:
    async def ok(*args, **kwargs):
        return "done"

    proxy = _bare_proxy(tmp_path, ok)
    call = _method(proxy, monkeypatch)

    assert await call(context=_user_context()) == "done"
    await proxy._flush_usage()

    stats = read_usage(tmp_path / "ledger")
    assert stats["totals"] == _totals(attempted=1, answered=1)
    assert stats["by_method"]["infer"][EXTERNAL]["answered"] == 1


async def test_proxy_counts_a_raising_call_as_failed(tmp_path: Path, monkeypatch) -> None:
    async def boom(*args, **kwargs):
        raise RuntimeError("CUDA error")

    proxy = _bare_proxy(tmp_path, boom)
    call = _method(proxy, monkeypatch)

    with pytest.raises(RuntimeError):
        await call(context=_user_context())
    await proxy._flush_usage()

    assert read_usage(tmp_path / "ledger")["totals"] == _totals(
        attempted=1, failed=1
    )


async def test_a_call_that_never_returns_counts_as_attempted_only(
    tmp_path: Path, monkeypatch
) -> None:
    """The 110-queued / 0-completed shape: the deployment wedged mid-request,
    so neither an answer nor a failure was ever recorded."""
    entered = asyncio.Event()

    async def hang(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    proxy = _bare_proxy(tmp_path, hang)
    call = _method(proxy, monkeypatch)

    task = asyncio.create_task(call(context=_user_context()))
    await asyncio.wait_for(entered.wait(), timeout=5)
    await proxy._flush_usage()

    assert read_usage(tmp_path / "ledger")["totals"] == _totals(attempted=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_an_unauthorized_call_is_not_counted_at_all(
    tmp_path: Path, monkeypatch
) -> None:
    """Unauthorized callers never reach the deployment, so they are not
    attempts on it — and they were not turned away for capacity either."""

    async def ok(*args, **kwargs):
        return "done"

    proxy = _bare_proxy(tmp_path, ok, authorized_users={"*": ["someone@else.org"]})
    call = _method(proxy, monkeypatch)

    with pytest.raises(PermissionError):
        await call(context=_user_context())
    await proxy._flush_usage()

    assert read_usage(tmp_path / "ledger")["totals"] == _totals()


async def test_calls_turned_away_at_capacity_are_counted_as_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    """The wedge signal has to stay quantitative.

    Once a deployment hangs, attempted pins at max_ongoing_requests and every
    further real call is refused at the semaphore. Without this counter the
    110-queued incident would have read as max_ongoing_requests of demand
    rather than 110.
    """
    entered = asyncio.Event()

    async def hang(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    proxy = _bare_proxy(tmp_path, hang, slots=1)
    call = _method(proxy, monkeypatch)

    wedged = asyncio.create_task(call(context=_user_context()))
    await asyncio.wait_for(entered.wait(), timeout=5)

    for _ in range(9):
        with pytest.raises(RuntimeError, match="maximum"):
            await call(context=_user_context())
    await proxy._flush_usage()

    totals = read_usage(tmp_path / "ledger")["totals"]
    assert totals == _totals(attempted=1, rejected=9)

    wedged.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wedged


async def test_a_rejection_is_attributed_to_its_caller_class(
    tmp_path: Path, monkeypatch
) -> None:
    entered = asyncio.Event()

    async def hang(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    proxy = _bare_proxy(tmp_path, hang, slots=1)
    call = _method(proxy, monkeypatch)

    wedged = asyncio.create_task(
        call(context=_user_context(scope={"workspaces": {"host-ws": "a"}}))
    )
    await asyncio.wait_for(entered.wait(), timeout=5)

    with pytest.raises(RuntimeError):
        await call(context=_user_context())
    with pytest.raises(RuntimeError):
        await call(context={"user": {"id": "anon-1", "is_anonymous": True}})
    await proxy._flush_usage()

    by_class = read_usage(tmp_path / "ledger")["by_caller_class"]
    assert by_class[INTERNAL]["attempted"] == 1
    assert by_class[EXTERNAL]["rejected"] == 1
    assert by_class[ANONYMOUS]["rejected"] == 1

    wedged.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wedged


async def test_reading_the_stats_does_not_inflate_them(
    tmp_path: Path, monkeypatch
) -> None:
    async def ok(*args, **kwargs):
        return "done"

    proxy = _bare_proxy(tmp_path, ok)
    call = _method(proxy, monkeypatch)
    await call(context=_user_context())

    first = await proxy.get_usage_stats(context=_user_context())
    second = await proxy.get_usage_stats(context=_user_context())

    assert first["totals"]["answered"] == 1
    assert second["totals"] == first["totals"]


# ===== concurrency =====


def test_two_writers_do_not_lose_increments(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    per_writer = 500
    start = threading.Barrier(2)

    def drive(writer_id: str) -> None:
        ledger = UsageLedger(directory, writer_id)
        start.wait()
        for i in range(per_writer):
            ledger.record("infer", EXTERNAL, "attempted")
            ledger.record("infer", EXTERNAL, "answered")
            if i % 10 == 0:
                ledger.flush()
        ledger.flush()

    threads = [
        threading.Thread(target=drive, args=(name,))
        for name in ("replica-a", "replica-b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Distinct shards is the mechanism, not an incidental: a single file with
    # two writers is what loses increments.
    assert {path.name for path in directory.glob("*.json")} == {
        "replica-a.json",
        "replica-b.json",
    }
    stats = read_usage(directory)
    assert stats["shards"] == 2
    assert stats["totals"]["attempted"] == 2 * per_writer
    assert stats["totals"]["answered"] == 2 * per_writer


def test_a_reader_never_sees_a_half_written_shard(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    stop = threading.Event()
    torn: list = []
    reads: list = []

    def write() -> None:
        ledger = UsageLedger(directory, "replica-a")
        for _ in range(400):
            ledger.record("infer", EXTERNAL, "attempted")
            ledger.flush()
        stop.set()

    def read() -> None:
        while not stop.is_set():
            stats = read_usage(directory)
            reads.append(stats["shards"])
            if stats["unreadable_shards"]:
                torn.append(True)
                return

    writer = threading.Thread(target=write)
    reader = threading.Thread(target=read)
    writer.start()
    reader.start()
    writer.join()
    reader.join()

    assert not torn
    assert len(reads) > 10, "the reader never overlapped the writer"
    assert max(reads) == 1


def test_two_overlapping_writes_to_one_shard_leave_it_whole(tmp_path: Path) -> None:
    """The damaging race, and the one the single-writer test above cannot see.

    ``get_usage_stats`` and the maintenance tick flush the same ledger from the
    same process, so a staging file discriminated only by pid is shared: one
    writer renames the file the other is still filling. On shutdown the torn
    shard is never rewritten and the replica's whole lifetime is lost, so this
    hammers differing payloads — identical ones overwrite byte-for-byte and
    hide the defect.
    """
    directory = tmp_path / "ledger"
    trials = 60
    unparseable = 0
    errors: list = []

    for trial in range(trials):
        ledger = UsageLedger(directory / f"t{trial}", "replica-a")
        for i in range(2000):
            ledger.record(f"m{i}", EXTERNAL, "attempted")
        early = ledger.serialize()
        for i in range(2000, 3000):
            ledger.record(f"m{i}", EXTERNAL, "attempted")
        late = ledger.serialize()

        start = threading.Barrier(2)

        def race(snapshot) -> None:
            start.wait()
            try:
                ledger.write(*snapshot)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=race, args=(snapshot,))
            for snapshot in (early, late)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        try:
            json.loads(ledger.path.read_text())
        except (OSError, ValueError):
            unparseable += 1

    assert not errors, f"write() raised {len(errors)}/{2 * trials} times: {errors[:3]}"
    assert unparseable == 0, f"{unparseable}/{trials} shards left torn"
    assert read_usage(directory / "t0")["unreadable_shards"] == 0


async def test_concurrent_flushes_serialize(tmp_path: Path, monkeypatch) -> None:
    """A read racing the maintenance tick must not put two writes on one shard."""

    async def ok(*args, **kwargs):
        return "done"

    proxy = _bare_proxy(tmp_path, ok)
    proxy.entry_deployment_ready = False
    call = _method(proxy, monkeypatch)
    await call(context=_user_context())

    overlap = []
    inflight = 0
    real_write = proxy._usage_ledger.write

    def slow_write(revision, payload):
        nonlocal inflight
        inflight += 1
        overlap.append(inflight)
        time.sleep(0.05)
        real_write(revision, payload)
        inflight -= 1

    monkeypatch.setattr(proxy._usage_ledger, "write", slow_write)

    await asyncio.gather(
        proxy._flush_usage(),
        proxy._maintenance_tick(),
        proxy.get_usage_stats(context=_user_context()),
    )

    assert max(overlap) == 1, f"writes overlapped: {overlap}"
    assert read_usage(tmp_path / "ledger")["unreadable_shards"] == 0


def test_read_usage_tolerates_a_corrupt_shard(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    good = UsageLedger(directory, "replica-a")
    good.record("infer", EXTERNAL, "attempted")
    good.flush()
    (directory / "replica-b.json").write_text("{not json")

    stats = read_usage(directory)
    assert stats["totals"]["attempted"] == 1
    assert stats["shards"] == 1
    assert stats["unreadable_shards"] == 1


def test_read_usage_on_a_directory_that_was_never_written(tmp_path: Path) -> None:
    stats = read_usage(tmp_path / "nothing-here")
    assert stats["shards"] == 0
    assert stats["totals"] == _totals()


# ===== caller attribution =====


def test_anonymous_caller_is_its_own_class() -> None:
    context = {"user": {"id": "anon-1", "is_anonymous": True}}
    assert classify_caller(context, "host-ws") == ANONYMOUS


def test_any_grant_on_the_hosting_workspace_is_internal() -> None:
    """Hypha's UserPermission is r/rw/a. All three are keys to the workspace
    that nobody outside the project is issued, so all three are internal."""
    for grant in ("r", "rw", "a"):
        context = _user_context(scope={"workspaces": {"host-ws": grant}})
        assert classify_caller(context, "host-ws") == INTERNAL, grant


def test_a_read_scoped_operator_is_not_counted_as_an_outside_user() -> None:
    """The bias must run one way. A worker mints read-scoped tokens on its own
    workspace, so bucketing read as external would inflate the one figure this
    feature exists to publish."""
    context = _user_context(scope={"workspaces": {"host-ws": "r"}})
    assert classify_caller(context, "host-ws") == INTERNAL


def test_global_access_is_internal() -> None:
    for grant in ("r", "rw", "a"):
        context = _user_context(scope={"workspaces": {"*": grant}})
        assert classify_caller(context, "host-ws") == INTERNAL, grant


def test_a_caller_with_no_key_to_the_hosting_workspace_is_external() -> None:
    """What a genuine outside caller looks like: Hypha's get_user_info grants
    admin on their own ws-user-<id> and nothing else, and update_user_scope
    never adds a workspace they had no permission on — so the hosting
    workspace is simply absent from their scope."""
    context = _user_context(scope={"workspaces": {"ws-user-u-1": "a"}})
    assert classify_caller(context, "host-ws") == EXTERNAL

    assert classify_caller(_user_context(scope={"workspaces": {}}), "host-ws") == EXTERNAL
    assert classify_caller(_user_context(), "host-ws") == EXTERNAL


def test_a_grant_on_some_other_workspace_is_external() -> None:
    context = _user_context(scope={"workspaces": {"unrelated-ws": "a"}})
    assert classify_caller(context, "host-ws") == EXTERNAL


def test_a_named_authorized_user_is_internal() -> None:
    context = _user_context()
    assert (
        classify_caller(context, "host-ws", {"*": ["u@example.org"]}) == INTERNAL
    )


def test_a_token_account_is_matched_through_its_parent() -> None:
    context = {"user": {"id": "blossom-account-1", "parent": "real-account"}}
    assert (
        classify_caller(context, "host-ws", {"*": ["real-account"]}) == INTERNAL
    )


def test_public_authorization_does_not_make_every_caller_internal() -> None:
    assert classify_caller(_user_context(), "host-ws", {"*": ["*"]}) == EXTERNAL


def test_on_a_public_app_the_authorized_users_leg_carries_nothing() -> None:
    """Why the workspace grant has to be sufficient on its own.

    Neither the builder nor the manager injects the deploying user or the
    worker's admins into a rule that already contains "*", so a public app's
    authorized_users is exactly ["*"] and classification rests entirely on the
    scope check. Pinned here so the docstring cannot drift back into promising
    a backstop that does not exist.
    """
    from bioengine.apps.builder import AppBuilder

    resolved = AppBuilder._resolve_authorized_users(
        manifest={"authorized_users": ["*"]},
        override=None,
        deploying_user=("deployer-id", "deployer@lab.org"),
        admin_users=["admin@lab.org"],
    )
    assert resolved == {"*": ["*"]}
    assert classify_caller(_user_context(), "host-ws", resolved) == EXTERNAL

    private = AppBuilder._resolve_authorized_users(
        manifest={"authorized_users": ["someone@else.org"]},
        override=None,
        deploying_user=("deployer-id", "deployer@lab.org"),
        admin_users=["admin@lab.org"],
    )
    assert "admin@lab.org" in private["*"]
    admin = {"user": {"id": "admin-id", "email": "admin@lab.org"}}
    assert classify_caller(admin, "host-ws", private) == INTERNAL


def test_a_caller_without_identity_is_unknown() -> None:
    assert classify_caller(None, "host-ws") == UNKNOWN
    assert classify_caller({}, "host-ws") == UNKNOWN
    assert classify_caller({"user": {}}, "host-ws") == UNKNOWN


async def test_unknown_is_unreachable_from_the_request_path(
    tmp_path: Path, monkeypatch
) -> None:
    """The bucket exists to keep a direct caller of classify_caller honest, not
    because the proxy can produce it: _check_permissions rejects a context with
    neither id nor email before anything is counted."""

    async def ok(*args, **kwargs):
        return "done"

    proxy = _bare_proxy(tmp_path, ok)
    call = _method(proxy, monkeypatch)

    for context in (None, {}, {"user": {}}):
        assert classify_caller(context, "host-ws") == UNKNOWN
        with pytest.raises(PermissionError):
            await call(context=context)
    await proxy._flush_usage()

    assert UNKNOWN not in read_usage(tmp_path / "ledger")["by_caller_class"]


async def test_internal_and_external_traffic_are_counted_apart(
    tmp_path: Path, monkeypatch
) -> None:
    async def ok(*args, **kwargs):
        return "done"

    proxy = _bare_proxy(tmp_path, ok)
    call = _method(proxy, monkeypatch)

    for _ in range(3):
        await call(
            context={
                "user": {
                    "id": "operator",
                    "email": "ops@lab.org",
                    "scope": {"workspaces": {"host-ws": "a"}},
                }
            }
        )
    await call(context=_user_context())
    await call(context={"user": {"id": "anon-9", "is_anonymous": True}})
    await proxy._flush_usage()

    by_class = read_usage(tmp_path / "ledger")["by_caller_class"]
    assert by_class[INTERNAL]["answered"] == 3
    assert by_class[EXTERNAL]["answered"] == 1
    assert by_class[ANONYMOUS]["answered"] == 1


# ===== what reaches disk =====


async def test_no_caller_identity_reaches_disk(tmp_path: Path, monkeypatch) -> None:
    async def ok(*args, **kwargs):
        return "done"

    proxy = _bare_proxy(tmp_path, ok)
    call = _method(proxy, monkeypatch)
    await call(
        context={
            "user": {
                "id": "distinctive-user-id",
                "email": "distinctive@example.org",
                "parent": "distinctive-parent",
                "roles": ["distinctive-role"],
                "scope": {"workspaces": {"ws-user-distinctive": "a"}},
            },
            "ws": "ws-user-distinctive",
            "from": "ws-user-distinctive/distinctive-client",
        }
    )
    await proxy._flush_usage()

    written = "\n".join(
        path.read_text() for path in (tmp_path / "ledger").glob("*.json")
    )
    assert read_usage(tmp_path / "ledger")["by_caller_class"][EXTERNAL]["answered"] == 1
    for secret in (
        "distinctive-user-id",
        "distinctive@example.org",
        "distinctive-parent",
        "distinctive-role",
        "ws-user-distinctive",
        "distinctive-client",
    ):
        assert secret not in written, secret


def test_a_shard_records_only_declared_fields(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path,
        "replica-1",
        application_id="counted-app",
        workspace="host-ws",
        artifact_id="host-ws/counted-app",
        app_version="1.0.0",
        worker_service_id="host-ws/client:bioengine-worker",
    )
    ledger.record("infer", EXTERNAL, "attempted")
    ledger.flush()

    shard = json.loads(ledger.path.read_text())
    assert set(shard) == {
        "format",
        "writer_id",
        "opened_at",
        "updated_at",
        "application_id",
        "workspace",
        "artifact_id",
        "app_version",
        "worker_service_id",
        "counts",
    }
    assert shard["counts"] == {"infer": {EXTERNAL: _totals(attempted=1)}}


# ===== wiring =====


async def test_the_maintenance_tick_persists_pending_counts(
    tmp_path: Path, monkeypatch
) -> None:
    """Counts are flushed by the existing upkeep loop, and before its
    readiness gate, so an app whose sibling just dropped out still persists
    what it served."""

    async def ok(*args, **kwargs):
        return "done"

    proxy = _bare_proxy(tmp_path, ok)
    proxy.entry_deployment_ready = False
    call = _method(proxy, monkeypatch)
    await call(context=_user_context())

    assert read_usage(tmp_path / "ledger")["totals"]["answered"] == 0
    await proxy._maintenance_tick()
    assert read_usage(tmp_path / "ledger")["totals"]["answered"] == 1


async def test_a_stalled_shard_write_does_not_stall_the_maintenance_loop(
    tmp_path: Path, monkeypatch
) -> None:
    """The flush is the first statement of the tick, and the loop is strictly
    sequential — so an unbounded wait on a hung PVC would stop the reachability
    probe and the re-registration recovery for the life of the replica. The
    executor thread cannot be cancelled; the wait on it must be bounded.
    """
    released = threading.Event()
    entered = threading.Event()

    async def ok(*args, **kwargs):
        return "done"

    proxy = _bare_proxy(tmp_path, ok)
    proxy.entry_deployment_ready = True
    proxy.server = None
    proxy.websocket_service_id = None
    proxy._connection_lost = False
    proxy._next_register_at = 0.0
    proxy._registration_lock = asyncio.Lock()
    proxy._registration_failure = None
    proxy._probe_due_at = 0.0

    registered = []

    async def _register():
        registered.append(True)

    monkeypatch.setattr(proxy, "_register_services", _register)
    monkeypatch.setattr(pd_module, "_USAGE_FLUSH_TIMEOUT_S", 0.2)

    def hang(revision, payload):
        entered.set()
        released.wait(timeout=30)

    call = _method(proxy, monkeypatch)
    await call(context=_user_context())
    monkeypatch.setattr(proxy._usage_ledger, "write", hang)

    started = time.monotonic()
    await asyncio.wait_for(proxy._maintenance_tick(), timeout=10)
    elapsed = time.monotonic() - started

    assert entered.is_set(), "the write never started"
    assert elapsed < 5, f"the tick waited {elapsed:.1f}s on a hung write"
    assert registered, "the tick gave up before doing its actual work"

    released.set()


async def test_a_broken_ledger_never_breaks_a_call(tmp_path: Path, monkeypatch) -> None:
    async def ok(*args, **kwargs):
        return "done"

    proxy = _bare_proxy(tmp_path, ok)
    call = _method(proxy, monkeypatch)

    def explode(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(proxy._usage_ledger, "record", explode)
    monkeypatch.setattr(proxy._usage_ledger, "write", explode)

    assert await call(context=_user_context()) == "done"
    await proxy._flush_usage()


async def test_a_hypha_client_can_call_usage_stats_on_the_registered_service(
    tmp_path: Path, monkeypatch
) -> None:
    """Registered and callable, not merely mentioned in the source."""
    registered: dict = {}

    class _FakeServer:
        config = {"workspace": "host-ws", "client_id": "proxy-1"}
        rpc = None

        def __getitem__(self, key):
            return self.config

        async def register_service(self, payload):
            registered.update(payload)
            return {"id": "host-ws/proxy-1:counted-app"}

    async def ok(*args, **kwargs):
        return "done"

    proxy = _bare_proxy(tmp_path, ok)
    proxy.server_url = "https://hypha.example"
    proxy.proxy_service_token = "t"
    proxy.client_id = "proxy-1"
    proxy.application_name = "Counted App"
    proxy.application_description = ""
    proxy.method_schemas = [
        {"name": "infer", "description": "", "parameters": {"type": "object"}}
    ]
    proxy.websocket_service_id = None
    proxy.rtc_service_id = None

    async def _noop(*args, **kwargs):
        return None

    async def _connect(config):
        return _FakeServer()

    async def _register_rtc(server, service_id, config):
        return {"id": f"host-ws/proxy-1:{service_id}"}

    monkeypatch.setattr(pd_module, "schema_function", lambda func, **kw: func)
    monkeypatch.setattr(pd_module, "connect_to_server", _connect)
    monkeypatch.setattr(pd_module, "register_rtc_service", _register_rtc)
    monkeypatch.setattr(proxy, "_reset_server_connection", _noop)
    monkeypatch.setattr(proxy, "_fetch_ice_servers", _noop)

    await proxy._register_services()

    assert "get_usage_stats" in registered
    stats = await registered["get_usage_stats"](context=_user_context())
    assert stats["available"] is True
    assert stats["application_id"] == "counted-app"
    assert set(stats["totals"]) == set(EVENTS)


def test_an_app_without_durable_storage_still_deploys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("BIOENGINE_APP_DIR", raising=False)
    inst = object.__new__(ProxyDouble)
    inst.application_id = "counted-app"
    inst.workspace = "host-ws"
    inst.app_data = {}
    assert inst._open_usage_ledger("replica-1") is None
