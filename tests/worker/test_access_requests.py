"""The starting-user lockout guard, and the access-request flow that replaces
the wildcard admin list.

Three separate properties are pinned here.

The worker's admin list no longer honours a ``"*"`` entry. It used to authorize
every caller that could reach the Hypha server — including unauthenticated ones
— for ``run_code``, ``deploy_app`` and ``upload_app``, which is arbitrary
execution on the Ray cluster. The entry is dropped where the list is built
rather than refused at each of the twenty call sites that read it, so no call
site can be the one that was missed.

The users a worker is started with can never lose admin permissions. The
persisted list at ``<workspace_dir>/admin_users.json`` overrides the startup
seed, so removing one of them was a lockout that re-rolling the pod with the
same ``--admin-users`` did not undo. The guard has to key on the seed captured
before that overlay is loaded, and the overlay has to be repaired on the way in,
or the lockout outlives the fix for it.

Asking for access is a public method, and it is the only one. It is off unless
the operator turns it on, one request per account, and a denial stands until an
admin clears it.
"""

import json
import tempfile
from pathlib import Path

import pytest

from bioengine.utils import create_context
from bioengine.worker.worker import BioEngineWorker

ADMIN = create_context("admin-id", "admin@example.org")
FOUNDER = create_context("founder-id", "founder@example.org")
FIRST = create_context("first-id", "first@example.org")
REQUESTER = create_context("requester-id", "requester@example.org")
OTHER = create_context("other-id", "other@example.org")

# Hypha reports email=None and flags is_anonymous for a caller with no token,
# and mints a fresh random id per anonymous connection.
ANONYMOUS = {
    "user": {"id": "anonymouz-torpid-lobster-52790260", "email": None, "is_anonymous": True}
}


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


def _worker(tmp_path, admin_users, founding=None, enabled=True, **attrs):
    """A worker carrying only the state these methods touch."""
    worker = BioEngineWorker.__new__(BioEngineWorker)
    worker.start_time = None  # silences __del__ on a half-built instance
    worker.logger = _Logger()
    worker.admin_users = list(admin_users)
    worker._admin_users_file = tmp_path / "admin_users.json"
    worker._worker_user_email = "worker@service.internal"
    worker._founding_admin_users = tuple(
        admin_users if founding is None else founding
    )
    worker._wildcard_admin_users_warned = False
    worker.enable_access_requests = enabled
    worker._access_requests_file = tmp_path / "access_requests.json"
    worker._access_requests = {}
    for key, value in attrs.items():
        setattr(worker, key, value)
    return worker


@pytest.fixture
def short_workspace_dir():
    """A workspace directory short enough for Ray's plasma socket path.

    RayCluster derives ray_temp_dir from workspace_dir and refuses a socket path
    over 107 bytes, which pytest's own tmp_path exceeds. Nothing here starts a
    Ray cluster; the constructor only validates the path.
    """
    with tempfile.TemporaryDirectory(prefix="be", dir="/tmp") as short_dir:
        yield Path(short_dir)


def _real_worker(workspace_dir, admin_users):
    """A worker built through the real __init__, to prove the wiring there."""
    return BioEngineWorker(
        mode="single-machine",
        admin_users=admin_users,
        workspace_dir=workspace_dir,
        token="not-a-real-token",
        log_file="off",
        ray_cluster_config={"head_num_cpus": 1, "head_num_gpus": 0},
    )


def _requested(worker, tmp_path):
    """The stored request for the requester, read back off disk."""
    return json.loads((tmp_path / "access_requests.json").read_text())[
        "requester@example.org"
    ]


# ---------------------------------------------------------------------------
# The starting user cannot be removed
# ---------------------------------------------------------------------------


async def test_a_starting_user_cannot_be_removed_by_another_admin(tmp_path):
    """The sharpest invariant: not the last admin, not the caller, still refused."""
    worker = _worker(
        tmp_path,
        ["founder@example.org", "admin@example.org"],
        founding=("founder@example.org",),
    )

    with pytest.raises(ValueError, match="worker was started with this user"):
        await worker.remove_admin_user(user="founder@example.org", context=ADMIN)

    assert worker.admin_users == ["founder@example.org", "admin@example.org"]
    assert not (tmp_path / "admin_users.json").exists()


async def test_a_starting_user_cannot_be_removed_by_a_later_granted_admin(tmp_path):
    """A granted admin is the realistic attacker: the grant is what gets them in."""
    worker = _worker(
        tmp_path, ["founder@example.org"], founding=("founder@example.org",)
    )

    await worker.add_admin_user(user="other@example.org", context=FOUNDER)

    with pytest.raises(ValueError, match="worker was started with this user"):
        await worker.remove_admin_user(user="founder@example.org", context=OTHER)

    assert "founder@example.org" in worker.admin_users


async def test_every_user_named_at_startup_is_protected_not_just_the_first(tmp_path):
    worker = _worker(
        tmp_path,
        ["first@example.org", "second@example.org", "admin@example.org"],
        founding=("first@example.org", "second@example.org"),
    )

    for founder in ("first@example.org", "second@example.org"):
        with pytest.raises(ValueError, match="worker was started with this user"):
            await worker.remove_admin_user(user=founder, context=ADMIN)

    # A user who is not a starting user is still removable, so the guard is not
    # simply refusing everything.
    assert await worker.remove_admin_user(
        user="admin@example.org", context=FIRST
    ) == ["first@example.org", "second@example.org"]


async def test_the_starting_user_guard_outranks_the_wildcard_caller(tmp_path):
    worker = _worker(
        tmp_path,
        ["founder@example.org", "*"],
        founding=("founder@example.org",),
    )

    with pytest.raises(PermissionError):
        await worker.remove_admin_user(
            user="founder@example.org", context=create_context("x", "x@example.org")
        )

    assert "founder@example.org" in worker.admin_users


def test_the_starting_users_are_captured_before_the_overlay_is_loaded(
    short_workspace_dir,
):
    """Read after the overlay, the founder set would be whatever the file says.

    That is the failure mode this ordering exists for: a runtime-added admin
    would become unremovable, and a founder a previous version had already
    removed would stay unprotected.
    """
    (short_workspace_dir / "admin_users.json").write_text(
        json.dumps(["added-at-runtime@example.org"])
    )

    worker = _real_worker(short_workspace_dir, ["founder@example.org"])

    assert worker._founding_admin_users == ("founder@example.org",)
    assert "added-at-runtime@example.org" not in worker._founding_admin_users


def test_a_store_that_dropped_a_starting_user_is_repaired_on_load(tmp_path):
    """Otherwise the lockout survives the upgrade that closed it.

    The overlay wins over the seed, so a worker that was locked out by an older
    version would come back up still locked out.
    """
    worker = _worker(
        tmp_path,
        ["founder@example.org"],
        founding=("founder@example.org",),
    )
    (tmp_path / "admin_users.json").write_text(
        json.dumps(["usurper@example.org"])
    )

    worker._load_persisted_admin_users()

    assert worker.admin_users == ["founder@example.org", "usurper@example.org"]
    assert any("Restoring admin user" in r for r in worker.logger.warnings)


def test_a_store_holding_every_starting_user_is_left_in_its_own_order(tmp_path):
    worker = _worker(
        tmp_path,
        ["founder@example.org"],
        founding=("founder@example.org",),
    )
    (tmp_path / "admin_users.json").write_text(
        json.dumps(["granted@example.org", "founder@example.org"])
    )

    worker._load_persisted_admin_users()

    assert worker.admin_users == ["granted@example.org", "founder@example.org"]
    assert worker.logger.warnings == []


# ---------------------------------------------------------------------------
# The wildcard is not honoured on the execution path
# ---------------------------------------------------------------------------


def test_a_wildcard_seed_never_reaches_the_admin_list(short_workspace_dir):
    """run_code, deploy_app and upload_app all read this list unqualified.

    Dropping the entry here is what makes every one of them refuse an
    anonymous caller, rather than each call site needing its own flag.
    """
    worker = _real_worker(short_workspace_dir, ["*"])

    assert worker.admin_users == []
    assert worker._founding_admin_users == ()


def test_a_wildcard_in_the_persisted_store_is_dropped_too(tmp_path):
    """The store overrides the seed, so it is a second way in."""
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    (tmp_path / "admin_users.json").write_text(
        json.dumps(["admin@example.org", "*"])
    )

    worker._load_persisted_admin_users()

    assert worker.admin_users == ["admin@example.org"]


async def test_a_wildcard_caller_is_refused_by_the_shared_permission_check(
    short_workspace_dir,
):
    """What the execution methods actually call, on the list they actually read."""
    from bioengine.utils import check_permissions

    worker = _real_worker(short_workspace_dir, ["*"])

    with pytest.raises(PermissionError):
        check_permissions(
            context=ANONYMOUS,
            authorized_users=worker.admin_users,
            resource_name="running code",
        )


async def test_check_access_reports_no_access_on_a_former_wildcard_worker(tmp_path):
    worker = _worker(tmp_path, [], founding=())

    assert await worker.check_access(context=ANONYMOUS) is False


# ---------------------------------------------------------------------------
# The toggle
# ---------------------------------------------------------------------------


async def test_the_toggle_off_removes_the_request_surface_from_the_service(tmp_path):
    registered = await _register(tmp_path, enabled=False)

    for method in (
        "request_admin_access",
        "get_admin_access_request",
        "list_access_requests",
        "resolve_access_request",
    ):
        assert method not in registered


async def test_the_toggle_on_exposes_the_request_surface_on_the_service(tmp_path):
    registered = await _register(tmp_path, enabled=True)

    for method in (
        "request_admin_access",
        "get_admin_access_request",
        "list_access_requests",
        "resolve_access_request",
    ):
        assert method in registered


async def test_the_toggle_off_also_refuses_a_direct_call(tmp_path):
    """The registration gate is the surface; this is the method refusing anyway."""
    worker = _worker(tmp_path, ["admin@example.org"], enabled=False)

    for call in (
        worker.request_admin_access(reason="", context=REQUESTER),
        worker.get_admin_access_request(context=REQUESTER),
        worker.list_access_requests(context=ADMIN),
        worker.resolve_access_request(
            user="requester@example.org", decision="grant", context=ADMIN
        ),
    ):
        with pytest.raises(RuntimeError, match="Access requests are disabled"):
            await call


async def _register(tmp_path, enabled):
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

    worker = _worker(
        tmp_path,
        ["admin@example.org"],
        enabled=enabled,
        server=_Server(),
        ray_cluster=_Cluster(),
        apps_manager=_Component(),
        code_executor=_Component(),
        service_id="bioengine-worker",
        worker_name="test-worker",
        full_service_id="ws/client:bioengine-worker",
    )
    await worker._register_bioengine_worker_service()
    return registered


# ---------------------------------------------------------------------------
# One request per user
# ---------------------------------------------------------------------------


async def test_a_non_admin_can_request_access_once(tmp_path):
    worker = _worker(tmp_path, ["admin@example.org"], founding=())

    request = await worker.request_admin_access(reason="new group member", context=REQUESTER)

    assert request["email"] == "requester@example.org"
    assert request["status"] == "pending"
    assert request["reason"] == "new group member"
    # Requesting is not granting.
    assert worker.admin_users == ["admin@example.org"]


async def test_a_second_request_from_the_same_user_is_refused(tmp_path):
    worker = _worker(tmp_path, ["admin@example.org"], founding=())

    await worker.request_admin_access(reason="first", context=REQUESTER)

    with pytest.raises(ValueError, match="already has an access request"):
        await worker.request_admin_access(reason="second", context=REQUESTER)

    # Refused, not queued and not replacing the first.
    assert len(worker._access_requests) == 1
    assert _requested(worker, tmp_path)["reason"] == "first"


async def test_a_second_request_is_refused_across_a_restart(tmp_path):
    """A request that only lived in memory would be re-filable after a pod roll."""
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    await worker.request_admin_access(reason="first", context=REQUESTER)

    restarted = _worker(tmp_path, ["admin@example.org"], founding=())
    restarted._access_requests = restarted._load_access_requests()

    with pytest.raises(ValueError, match="already has an access request"):
        await restarted.request_admin_access(reason="second", context=REQUESTER)


async def test_a_denied_user_cannot_request_again(tmp_path):
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    await worker.request_admin_access(reason="first", context=REQUESTER)

    await worker.resolve_access_request(
        user="requester@example.org", decision="deny", context=ADMIN
    )

    with pytest.raises(ValueError, match="already has an access request"):
        await worker.request_admin_access(reason="again", context=REQUESTER)
    assert worker.admin_users == ["admin@example.org"]


async def test_an_admin_can_clear_a_denial_so_the_user_may_ask_again(tmp_path):
    """Otherwise a denial is permanent and there is no way back."""
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    await worker.request_admin_access(reason="first", context=REQUESTER)
    await worker.resolve_access_request(
        user="requester@example.org", decision="deny", context=ADMIN
    )

    assert (
        await worker.resolve_access_request(
            user="requester@example.org", decision="clear", context=ADMIN
        )
        is None
    )

    again = await worker.request_admin_access(reason="second try", context=REQUESTER)
    assert again["status"] == "pending"


async def test_the_request_key_is_the_email_not_the_client_id(tmp_path):
    """generate_token mints a fresh client id per token but keeps the email.

    Keyed on the id, refreshing a token would buy an unlimited supply of
    requests from one person.
    """
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    await worker.request_admin_access(reason="first", context=REQUESTER)

    refreshed_token = create_context("a-brand-new-client-id", "requester@example.org")

    with pytest.raises(ValueError, match="already has an access request"):
        await worker.request_admin_access(reason="second", context=refreshed_token)


async def test_an_existing_admin_has_nothing_to_request(tmp_path):
    worker = _worker(tmp_path, ["admin@example.org"], founding=())

    with pytest.raises(ValueError, match="already an admin"):
        await worker.request_admin_access(reason="", context=ADMIN)

    assert worker._access_requests == {}


async def test_an_anonymous_caller_cannot_file_a_request(tmp_path):
    """Hypha mints a fresh random id per anonymous connection and no email.

    Accepting these would let one caller fill the store by reconnecting, and
    granting one would re-admit every anonymous caller at once.
    """
    worker = _worker(tmp_path, ["admin@example.org"], founding=())

    with pytest.raises(PermissionError, match="Log in before requesting"):
        await worker.request_admin_access(reason="let me in", context=ANONYMOUS)

    assert worker._access_requests == {}
    assert not (tmp_path / "access_requests.json").exists()


async def test_the_placeholder_email_create_context_substitutes_is_refused(tmp_path):
    worker = _worker(tmp_path, ["admin@example.org"], founding=())

    with pytest.raises(PermissionError, match="Log in before requesting"):
        await worker.request_admin_access(reason="", context=create_context())


# ---------------------------------------------------------------------------
# Reading and resolving
# ---------------------------------------------------------------------------


async def test_a_requester_can_read_their_own_request(tmp_path):
    """A request nobody can observe is indistinguishable from a dropped one."""
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    await worker.request_admin_access(reason="please", context=REQUESTER)

    assert (await worker.get_admin_access_request(context=REQUESTER))[
        "status"
    ] == "pending"
    assert await worker.get_admin_access_request(context=OTHER) is None


async def test_reading_your_own_request_does_not_expose_anyone_elses(tmp_path):
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    await worker.request_admin_access(reason="please", context=REQUESTER)

    assert await worker.get_admin_access_request(context=OTHER) is None
    with pytest.raises(PermissionError):
        await worker.list_access_requests(context=OTHER)


async def test_granting_makes_the_requester_an_admin_and_persists_it(tmp_path):
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    await worker.request_admin_access(reason="please", context=REQUESTER)

    resolved = await worker.resolve_access_request(
        user="requester@example.org", decision="grant", context=ADMIN
    )

    assert resolved["status"] == "granted"
    assert resolved["resolved_by"] == "admin@example.org"
    assert worker.admin_users == ["admin@example.org", "requester@example.org"]
    assert json.loads((tmp_path / "admin_users.json").read_text()) == [
        "admin@example.org",
        "requester@example.org",
    ]


async def test_a_granted_user_is_not_a_starting_user(tmp_path):
    """A grant must not make someone unremovable; only --admin-users does that."""
    worker = _worker(
        tmp_path, ["founder@example.org"], founding=("founder@example.org",)
    )
    await worker.request_admin_access(reason="please", context=REQUESTER)
    await worker.resolve_access_request(
        user="requester@example.org", decision="grant", context=FOUNDER
    )

    assert await worker.remove_admin_user(
        user="requester@example.org", context=FOUNDER
    ) == ["founder@example.org"]


async def test_a_non_admin_cannot_resolve_or_list_requests(tmp_path):
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    await worker.request_admin_access(reason="please", context=REQUESTER)

    for call in (
        worker.list_access_requests(context=OTHER),
        worker.resolve_access_request(
            user="requester@example.org", decision="grant", context=OTHER
        ),
        # Nor may the requester grant their own request.
        worker.resolve_access_request(
            user="requester@example.org", decision="grant", context=REQUESTER
        ),
    ):
        with pytest.raises(PermissionError):
            await call

    assert worker.admin_users == ["admin@example.org"]


async def test_a_wildcard_covered_caller_cannot_resolve_a_request(tmp_path):
    """Same reason add_admin_user requires a named admin: a wildcard caller
    turning the wildcard into a named grant ends up the only admin."""
    worker = _worker(tmp_path, ["admin@example.org", "*"], founding=())
    worker._access_requests = {
        "requester@example.org": {
            "email": "requester@example.org",
            "status": "pending",
            "requested_at": 1.0,
        }
    }

    with pytest.raises(PermissionError):
        await worker.resolve_access_request(
            user="requester@example.org", decision="grant", context=OTHER
        )
    with pytest.raises(PermissionError):
        await worker.list_access_requests(context=OTHER)


async def test_an_admin_sees_every_request_oldest_first(tmp_path):
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    await worker.request_admin_access(reason="first", context=REQUESTER)
    await worker.request_admin_access(reason="second", context=OTHER)

    listed = await worker.list_access_requests(context=ADMIN)

    assert [r["email"] for r in listed] == [
        "requester@example.org",
        "other@example.org",
    ]


async def test_resolving_a_request_that_does_not_exist_is_refused(tmp_path):
    worker = _worker(tmp_path, ["admin@example.org"], founding=())

    with pytest.raises(ValueError, match="No access request"):
        await worker.resolve_access_request(
            user="nobody@example.org", decision="grant", context=ADMIN
        )


async def test_an_unknown_decision_is_refused_rather_than_treated_as_a_denial(tmp_path):
    """The Literal annotation only documents the schema; it does not validate."""
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    await worker.request_admin_access(reason="please", context=REQUESTER)

    with pytest.raises(ValueError, match="Unknown decision"):
        await worker.resolve_access_request(
            user="requester@example.org", decision="approve", context=ADMIN
        )

    assert worker._access_requests["requester@example.org"]["status"] == "pending"


async def test_an_unwritable_store_leaves_the_requests_unchanged(tmp_path):
    """A request that only exists in memory reverts on restart without failing."""
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    worker._access_requests_file = tmp_path / "not-a-dir" / "access_requests.json"
    (tmp_path / "not-a-dir").write_text("this is a file, not a directory")

    with pytest.raises(RuntimeError, match="Failed to persist access requests"):
        await worker.request_admin_access(reason="please", context=REQUESTER)

    assert worker._access_requests == {}


def test_a_corrupt_request_store_does_not_stop_the_worker(tmp_path):
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    (tmp_path / "access_requests.json").write_text("{not json")

    assert worker._load_access_requests() == {}
    assert any("Ignoring unreadable" in r for r in worker.logger.records)


def test_a_request_store_of_the_wrong_shape_is_ignored(tmp_path):
    worker = _worker(tmp_path, ["admin@example.org"], founding=())
    (tmp_path / "access_requests.json").write_text(json.dumps(["not", "a", "map"]))

    assert worker._load_access_requests() == {}
