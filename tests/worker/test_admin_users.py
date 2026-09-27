"""Editing the worker's admin users on a live worker.

Permissions are checked per call against ``self.admin_users``, so the list is
editable at runtime without re-registering the service — the only thing that
was missing is an API. Three properties have to hold for that API to be safe:
only a named admin can change who the admins are, nobody can lock the worker
(or themselves) out, and a runtime change must not silently revert on the next
restart, when the ``--admin-users`` startup flag is replayed verbatim.
"""

import inspect
import json

import pytest

from bioengine.utils import create_context
from bioengine.worker.worker import BioEngineWorker

ADMIN = create_context("admin-id", "admin@example.org")
NEWCOMER = create_context("new-id", "new@example.org")
OUTSIDER = create_context("outsider-id", "outsider@example.org")


def _bare_worker(tmp_path, admin_users, **attrs):
    """A worker with only the state the admin-user methods touch."""
    worker = BioEngineWorker.__new__(BioEngineWorker)
    worker.start_time = None  # silences __del__ on a half-built instance
    worker.logger = _Logger()
    worker.admin_users = list(admin_users)
    worker._admin_users_file = tmp_path / "admin_users.json"
    worker._worker_user_email = "worker@service.internal"
    for key, value in attrs.items():
        setattr(worker, key, value)
    return worker


class _Logger:
    def __init__(self):
        self.records = []
        self.warnings = []

    def _record(self, message):
        self.records.append(str(message))

    def warning(self, message):
        self.warnings.append(str(message))
        self._record(message)

    info = error = debug = _record


async def test_a_granted_admin_can_immediately_call_admin_methods(tmp_path):
    worker = _bare_worker(tmp_path, ["admin@example.org"])

    with pytest.raises(PermissionError):
        await worker.list_admin_users(context=OUTSIDER)

    await worker.add_admin_user(user="outsider@example.org", context=ADMIN)

    assert await worker.list_admin_users(context=OUTSIDER) == [
        "admin@example.org",
        "outsider@example.org",
    ]


async def test_the_change_reaches_the_component_managers(tmp_path):
    """The managers hold the same list object, so mutation must stay in place."""

    class _Holder:
        pass

    worker = _bare_worker(tmp_path, ["admin@example.org"])
    apps_manager = _Holder()
    code_executor = _Holder()
    apps_manager.admin_users = worker.admin_users
    code_executor.admin_users = worker.admin_users

    await worker.add_admin_user(user="new@example.org", context=ADMIN)
    await worker.remove_admin_user(user="admin@example.org", context=NEWCOMER)

    assert apps_manager.admin_users == ["new@example.org"]
    assert code_executor.admin_users == ["new@example.org"]


async def test_a_caller_cannot_revoke_their_own_admin_permissions(tmp_path):
    worker = _bare_worker(tmp_path, ["admin@example.org", "other@example.org"])

    with pytest.raises(ValueError, match="cannot revoke their own"):
        await worker.remove_admin_user(user="admin@example.org", context=ADMIN)

    assert worker.admin_users == ["admin@example.org", "other@example.org"]


async def test_a_caller_cannot_revoke_themselves_by_user_id(tmp_path):
    worker = _bare_worker(tmp_path, ["admin-id", "other@example.org"])

    with pytest.raises(ValueError, match="cannot revoke their own"):
        await worker.remove_admin_user(user="admin-id", context=ADMIN)


async def test_the_last_admin_cannot_be_removed(tmp_path):
    worker = _bare_worker(tmp_path, ["admin@example.org"])

    with pytest.raises(ValueError, match="last admin user"):
        await worker.remove_admin_user(user="admin@example.org", context=ADMIN)

    assert worker.admin_users == ["admin@example.org"]


async def test_wildcard_access_does_not_confer_the_right_to_edit_the_admin_list(
    tmp_path,
):
    """``--admin-users '*'`` makes every caller an admin over the worker's
    operations. It must not also make every caller able to rewrite who the
    admins are: a wildcard caller who could grant themselves a named entry and
    then revoke the wildcard would end up the only admin, on a list that
    overrides the startup flag the operator would use to take it back."""
    worker = _bare_worker(tmp_path, ["worker@service.internal", "*"])

    with pytest.raises(PermissionError):
        await worker.add_admin_user(user="outsider@example.org", context=OUTSIDER)
    with pytest.raises(PermissionError):
        await worker.remove_admin_user(user="*", context=OUTSIDER)

    assert worker.admin_users == ["worker@service.internal", "*"]
    assert not (tmp_path / "admin_users.json").exists()


async def test_a_named_admin_still_edits_the_list_on_a_wildcard_worker(tmp_path):
    worker = _bare_worker(tmp_path, ["admin@example.org", "*"])

    await worker.add_admin_user(user="new@example.org", context=ADMIN)

    assert await worker.remove_admin_user(user="*", context=ADMIN) == [
        "admin@example.org",
        "new@example.org",
    ]
    assert json.loads((tmp_path / "admin_users.json").read_text()) == [
        "admin@example.org",
        "new@example.org",
    ]


async def test_the_workers_own_identity_cannot_be_removed(tmp_path):
    """_connect_to_server re-inserts it, so removal would revert silently."""
    worker = _bare_worker(
        tmp_path, ["worker@service.internal", "admin@example.org"]
    )

    with pytest.raises(ValueError, match="restored on the next reconnect"):
        await worker.remove_admin_user(user="worker@service.internal", context=ADMIN)

    assert "worker@service.internal" in worker.admin_users


async def test_the_wildcard_cannot_be_granted_over_rpc(tmp_path):
    worker = _bare_worker(tmp_path, ["admin@example.org"])

    with pytest.raises(ValueError, match="wildcard"):
        await worker.add_admin_user(user="*", context=ADMIN)

    assert worker.admin_users == ["admin@example.org"]


async def test_an_empty_identifier_is_refused(tmp_path):
    worker = _bare_worker(tmp_path, ["admin@example.org"])

    with pytest.raises(ValueError, match="must not be empty"):
        await worker.add_admin_user(user="   ", context=ADMIN)


async def test_repeated_grants_and_revocations_are_idempotent(tmp_path):
    worker = _bare_worker(tmp_path, ["admin@example.org"])

    await worker.add_admin_user(user="new@example.org", context=ADMIN)
    await worker.add_admin_user(user="new@example.org", context=ADMIN)
    assert worker.admin_users == ["admin@example.org", "new@example.org"]

    await worker.remove_admin_user(user="new@example.org", context=ADMIN)
    result = await worker.remove_admin_user(user="new@example.org", context=ADMIN)
    assert result == ["admin@example.org"]


async def test_a_non_admin_cannot_read_or_change_the_admin_list(tmp_path):
    worker = _bare_worker(tmp_path, ["admin@example.org"])

    for call in (
        worker.list_admin_users(context=OUTSIDER),
        worker.add_admin_user(user="outsider@example.org", context=OUTSIDER),
        worker.remove_admin_user(user="admin@example.org", context=OUTSIDER),
    ):
        with pytest.raises(PermissionError):
            await call

    assert worker.admin_users == ["admin@example.org"]


async def test_a_grant_is_written_before_it_takes_effect(tmp_path):
    worker = _bare_worker(tmp_path, ["admin@example.org"])

    await worker.add_admin_user(user="new@example.org", context=ADMIN)

    assert json.loads((tmp_path / "admin_users.json").read_text()) == [
        "admin@example.org",
        "new@example.org",
    ]


async def test_an_unwritable_store_leaves_the_admin_list_unchanged(tmp_path):
    """A grant that only exists in memory would revert on restart without ever failing."""
    worker = _bare_worker(tmp_path, ["admin@example.org"])
    worker._admin_users_file = tmp_path / "not-a-dir" / "admin_users.json"
    (tmp_path / "not-a-dir").write_text("this is a file, not a directory")

    with pytest.raises(RuntimeError, match="Failed to persist admin users"):
        await worker.add_admin_user(user="new@example.org", context=ADMIN)

    assert worker.admin_users == ["admin@example.org"]


def test_the_persisted_list_overrides_the_startup_seed(tmp_path):
    """The seed is replayed on every restart; the runtime overlay has to win."""
    worker = _bare_worker(tmp_path, ["seeded@example.org"])
    (tmp_path / "admin_users.json").write_text(
        json.dumps(["seeded@example.org", "added-at-runtime@example.org"])
    )

    worker._load_persisted_admin_users()

    assert worker.admin_users == [
        "seeded@example.org",
        "added-at-runtime@example.org",
    ]


def test_a_runtime_revocation_is_not_undone_by_the_seed(tmp_path):
    worker = _bare_worker(tmp_path, ["seeded@example.org", "revoked@example.org"])
    (tmp_path / "admin_users.json").write_text(json.dumps(["seeded@example.org"]))

    worker._load_persisted_admin_users()

    assert worker.admin_users == ["seeded@example.org"]


def test_the_divergence_from_the_seed_is_reported(tmp_path):
    worker = _bare_worker(tmp_path, ["seeded@example.org", "revoked@example.org"])
    (tmp_path / "admin_users.json").write_text(
        json.dumps(["seeded@example.org", "added@example.org"])
    )

    worker._load_persisted_admin_users()

    reported = "\n".join(worker.logger.records)
    assert "added@example.org" in reported
    assert "revoked@example.org" in reported


def test_a_corrupt_store_falls_back_to_the_seed(tmp_path):
    worker = _bare_worker(tmp_path, ["seeded@example.org"])
    (tmp_path / "admin_users.json").write_text("{not json")

    worker._load_persisted_admin_users()

    assert worker.admin_users == ["seeded@example.org"]
    assert any("Ignoring unreadable" in r for r in worker.logger.records)


def test_a_store_of_the_wrong_shape_falls_back_to_the_seed(tmp_path):
    worker = _bare_worker(tmp_path, ["seeded@example.org"])
    (tmp_path / "admin_users.json").write_text(json.dumps({"admins": ["a@b.c"]}))

    worker._load_persisted_admin_users()

    assert worker.admin_users == ["seeded@example.org"]


def test_no_store_leaves_the_seed_alone(tmp_path):
    worker = _bare_worker(tmp_path, ["seeded@example.org"])

    worker._load_persisted_admin_users()

    assert worker.admin_users == ["seeded@example.org"]
    assert worker.logger.records == []


async def test_the_admin_methods_are_exposed_on_the_worker_service(tmp_path):
    registered = {}

    class _Server:
        async def register_service(self, service):
            registered.update(service)
            return type("Info", (), {"id": "ws/client:bioengine-worker"})()

    class _Cluster:
        mode = "single-machine"

    class _Component:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    worker = _bare_worker(
        tmp_path,
        ["admin@example.org"],
        server=_Server(),
        ray_cluster=_Cluster(),
        apps_manager=_Component(),
        code_executor=_Component(),
        service_id="bioengine-worker",
        worker_name="test-worker",
        full_service_id="ws/client:bioengine-worker",
    )

    await worker._register_bioengine_worker_service()

    assert registered["list_admin_users"] == worker.list_admin_users
    assert registered["add_admin_user"] == worker.add_admin_user
    assert registered["remove_admin_user"] == worker.remove_admin_user


async def test_a_wildcard_admin_list_warns_that_anyone_can_run_code(tmp_path):
    """The wildcard reads as a permissions shortcut; it is open remote execution."""
    worker = _bare_worker(tmp_path, ["admin@example.org", "*"])

    worker._warn_if_wildcard_admin_users()

    assert len(worker.logger.warnings) == 1
    warning = worker.logger.warnings[0]
    assert "run_code" in warning
    assert "anonymous" in warning


async def test_a_named_admin_list_does_not_warn(tmp_path):
    worker = _bare_worker(tmp_path, ["admin@example.org"])

    worker._warn_if_wildcard_admin_users()

    assert worker.logger.warnings == []


def test_the_wildcard_warning_is_wired_into_startup():
    """A warning nobody calls is the same as no warning."""
    assert "_warn_if_wildcard_admin_users" in inspect.getsource(
        BioEngineWorker.start
    )
