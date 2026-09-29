"""Unit tests for per-dataset access requests on the data server.

Every token in this file is a fake string matched by a stubbed ``parse_token``;
no Hypha server is contacted and no real credential appears anywhere.

Deliberately imports only ``proxy_server`` and ``access_requests`` — not
``HttpZarrStore`` — so these run in the standard suite, which uses the worker
image and has no ``zarr``.
"""

import asyncio
import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

# Imported by full path on purpose: ``from bioengine.datasets import proxy_server``
# goes through the package's __getattr__ delegation and raises MissingDependency
# instead of importing the submodule.
import bioengine.datasets.proxy_server as proxy_server
from bioengine.datasets.access_requests import (
    AccessRequestStore,
    is_worker_admin,
    requester_identity,
)
from bioengine.datasets.proxy_server import _build_app, load_datasets

OWNER = "owner@lab.org"
STRANGER = "stranger@elsewhere.org"
MEMBER = "member@lab.org"

# Fake bearer tokens; the stub below is the only thing that reads them.
USERS = {
    "tok-owner": {"id": "user:owner", "email": OWNER},
    "tok-stranger": {"id": "user:stranger", "email": STRANGER},
    "tok-member": {"id": "user:member", "email": MEMBER},
    "tok-mixedcase": {"id": "user:mixed", "email": "Mixed.Case@Lab.org"},
    "tok-anonymous": {"id": "anonymouz-abc", "email": None, "is_anonymous": True},
}


# Which fake tokens the fake worker calls admins. Mutated mid-test to stand in
# for add_admin_user / remove_admin_user on a live worker.
WORKER_ADMINS = {"tok-owner"}
WORKER_REACHABLE = {"value": True}
ADMIN_CHECKS = []


@pytest.fixture(autouse=True)
def stub_token_parsing(monkeypatch):
    """Resolve fake bearer tokens locally instead of calling Hypha."""

    async def fake_parse_token(token, cached_user_info):
        if token is None:
            return {"id": "anonymous-user", "email": "no-email"}
        if token not in USERS:
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="Invalid token")
        return dict(USERS[token])

    monkeypatch.setattr(proxy_server, "parse_token", fake_parse_token)


@pytest.fixture(autouse=True)
def stub_worker_admin_check(monkeypatch):
    """Stand in for the worker's public check_access over Hypha RPC.

    The real call carries the caller's own token and asks the worker about
    whoever is calling; this mirrors that by answering from the token alone.
    """
    WORKER_ADMINS.clear()
    WORKER_ADMINS.add("tok-owner")
    WORKER_REACHABLE["value"] = True
    ADMIN_CHECKS.clear()

    async def fake_is_worker_admin(token, worker_service_id, server_url, logger=None):
        ADMIN_CHECKS.append((token, worker_service_id))
        if not token:
            return False
        if not WORKER_REACHABLE["value"]:
            return False
        return token in WORKER_ADMINS

    monkeypatch.setattr(proxy_server, "is_worker_admin", fake_is_worker_admin)


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def make_data_dir(tmp_path: Path, authorized_users) -> Path:
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "blood-atlas"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.yaml").write_text(
        yaml.dump({"id": "blood-atlas", "authorized_users": authorized_users})
    )
    (dataset_dir / "notes.txt").write_text("payload")

    other_dir = data_dir / "other-set"
    other_dir.mkdir()
    (other_dir / "manifest.yaml").write_text(
        yaml.dump({"id": "other-set", "authorized_users": []})
    )
    (other_dir / "notes.txt").write_text("other payload")
    return data_dir


def build(
    tmp_path: Path,
    authorized_users=None,
    enabled=True,
    worker_service_id="ws/worker:bioengine-worker",
    store_file=None,
):
    data_dir = make_data_dir(
        tmp_path, [MEMBER] if authorized_users is None else authorized_users
    )
    store = AccessRequestStore(
        store_file=store_file or (tmp_path / "state" / "access_requests.json"),
        worker_service_id=worker_service_id,
        enabled=enabled,
    )
    app = _build_app(
        data_dir=data_dir,
        datasets=load_datasets(data_dir),
        cached_user_info={},
        access_requests=store,
    )
    return TestClient(app), store


# ---------------------------------------------------------------------------
# The toggle: two independent gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/datasets/blood-atlas/access-request"),
        ("get", "/datasets/blood-atlas/access-request"),
        ("get", "/access-requests"),
        ("post", "/access-requests/blood-atlas/resolve"),
    ],
)
def test_disabled_server_registers_no_request_routes(tmp_path, method, path):
    client, _ = build(tmp_path, enabled=False)
    response = getattr(client, method)(path, headers=auth("tok-owner"))
    assert response.status_code == 404


def test_disabled_store_refuses_a_direct_call(tmp_path):
    """The second gate: reachable even by wiring that forgets the route gate."""
    _, store = build(tmp_path, enabled=False)
    with pytest.raises(RuntimeError, match="disabled"):
        store.submit("blood-atlas", STRANGER, STRANGER, "user:stranger", "")


@pytest.mark.parametrize("worker_service_id", [None, "", "   "])
def test_enabled_without_a_worker_stays_off(tmp_path, worker_service_id):
    """Nobody could authorize a decision, and a public write to disk that
    nobody can drain is worse than no surface at all."""
    client, store = build(tmp_path, enabled=True, worker_service_id=worker_service_id)
    assert store.enabled is False
    assert (
        client.post(
            "/datasets/blood-atlas/access-request", headers=auth("tok-stranger")
        ).status_code
        == 404
    )


# ---------------------------------------------------------------------------
# Who may file a request
# ---------------------------------------------------------------------------


def test_unauthenticated_caller_is_refused(tmp_path):
    """Anonymous identities are free and unlimited, so one-per-user bounds nothing."""
    client, _ = build(tmp_path)
    response = client.post("/datasets/blood-atlas/access-request")
    assert response.status_code == 401
    assert "Log in" in response.json()["detail"]


def test_anonymous_hypha_identity_is_refused(tmp_path):
    client, _ = build(tmp_path)
    response = client.post(
        "/datasets/blood-atlas/access-request", headers=auth("tok-anonymous")
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "user_info",
    [
        {"id": "x", "email": None},
        {"id": "x", "email": "no-email"},
        {"id": "x", "email": "anonymous@example.com"},
        {"id": "x", "email": "not-an-address"},
        {"id": "x", "email": "real@lab.org", "is_anonymous": True},
    ],
)
def test_requester_identity_refuses_unusable_identities(user_info):
    with pytest.raises(PermissionError):
        requester_identity(user_info)


def test_requester_identity_splits_key_from_stored_email():
    """Lowercased key for dedup, verbatim address for what a grant must match."""
    assert requester_identity({"email": "Mixed.Case@Lab.org"}) == (
        "mixed.case@lab.org",
        "Mixed.Case@Lab.org",
    )


def test_already_authorized_user_has_nothing_to_request(tmp_path):
    client, _ = build(tmp_path)
    response = client.post(
        "/datasets/blood-atlas/access-request", headers=auth("tok-member")
    )
    assert response.status_code == 409
    assert "nothing to request" in response.json()["detail"]


def test_request_for_an_unknown_dataset_is_refused(tmp_path):
    client, _ = build(tmp_path)
    assert (
        client.post(
            "/datasets/no-such-set/access-request", headers=auth("tok-stranger")
        ).status_code
        == 404
    )


# ---------------------------------------------------------------------------
# One request per account per dataset
# ---------------------------------------------------------------------------


def test_first_request_is_recorded_pending(tmp_path):
    client, _ = build(tmp_path)
    record = client.post(
        "/datasets/blood-atlas/access-request",
        params={"reason": "collaborating on the atlas"},
        headers=auth("tok-stranger"),
    ).json()
    assert record["status"] == "pending"
    assert record["email"] == STRANGER
    assert record["dataset_id"] == "blood-atlas"
    assert record["reason"] == "collaborating on the atlas"


def test_second_request_is_refused_not_queued(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    response = client.post(
        "/datasets/blood-atlas/access-request", headers=auth("tok-stranger")
    )
    assert response.status_code == 409
    assert "already has an access request" in response.json()["detail"]


def test_a_request_is_scoped_to_one_dataset(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))

    assert (
        client.get(
            "/datasets/other-set/access-request", headers=auth("tok-stranger")
        ).json()
        is None
    )
    # ... and the same account may still file for the other dataset.
    assert (
        client.post(
            "/datasets/other-set/access-request", headers=auth("tok-stranger")
        ).status_code
        == 200
    )


def test_case_variants_cannot_hold_two_requests(tmp_path):
    client, store = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-mixedcase"))
    response = client.post(
        "/datasets/blood-atlas/access-request", headers=auth("tok-mixedcase")
    )
    assert response.status_code == 409
    assert list(store._requests["blood-atlas"]) == ["mixed.case@lab.org"]


def test_requester_reads_only_their_own_request(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))

    own = client.get(
        "/datasets/blood-atlas/access-request", headers=auth("tok-stranger")
    ).json()
    assert own["email"] == STRANGER
    assert (
        client.get(
            "/datasets/blood-atlas/access-request", headers=auth("tok-mixedcase")
        ).json()
        is None
    )


# ---------------------------------------------------------------------------
# Only an approver decides. Split by decision: 'grant' would look guarded even
# without the check, because a granted stranger is authorized anyway — 'deny'
# and 'clear' touch nothing downstream and are what prove the gate exists.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decision", ["grant", "deny", "clear"])
def test_non_approver_cannot_resolve(tmp_path, decision):
    client, store = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))

    response = client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": decision},
        headers=auth("tok-member"),
    )
    assert response.status_code == 403
    assert store._requests["blood-atlas"][STRANGER]["status"] == "pending"


def test_unauthenticated_caller_cannot_resolve(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    assert (
        client.post(
            "/access-requests/blood-atlas/resolve",
            params={"user": STRANGER, "decision": "deny"},
        ).status_code
        == 403
    )


def test_non_approver_cannot_list_requests(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    assert (
        client.get("/access-requests", headers=auth("tok-member")).status_code == 403
    )


# ---------------------------------------------------------------------------
# The approver gate is the worker's live admin list, asked per call
# ---------------------------------------------------------------------------


def test_the_caller_is_asked_about_as_themselves(tmp_path):
    """The caller's own token goes to the worker — this server asserts no
    identity on anyone's behalf, and cannot ask about a third party."""
    client, _ = build(tmp_path)
    client.get("/access-requests", headers=auth("tok-owner"))
    assert ADMIN_CHECKS[-1] == ("tok-owner", "ws/worker:bioengine-worker")


def test_an_admin_added_at_runtime_can_decide_immediately(tmp_path):
    """The gate (A) failed: a list snapshotted at startup would still refuse."""
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    assert (
        client.get("/access-requests", headers=auth("tok-member")).status_code == 403
    )

    WORKER_ADMINS.add("tok-member")  # stands in for the worker's add_admin_user

    assert (
        client.get("/access-requests", headers=auth("tok-member")).status_code == 200
    )
    assert (
        client.post(
            "/access-requests/blood-atlas/resolve",
            params={"user": STRANGER, "decision": "grant"},
            headers=auth("tok-member"),
        ).status_code
        == 200
    )


def test_an_admin_removed_at_runtime_stops_deciding_immediately(tmp_path):
    """A cached answer would let a revoked admin keep deciding."""
    client, _ = build(tmp_path)
    assert client.get("/access-requests", headers=auth("tok-owner")).status_code == 200

    WORKER_ADMINS.discard("tok-owner")  # stands in for remove_admin_user

    assert client.get("/access-requests", headers=auth("tok-owner")).status_code == 403


def test_every_decision_re_asks_the_worker(tmp_path):
    """No caching: the answer must track the worker's list, not a snapshot."""
    client, _ = build(tmp_path)
    ADMIN_CHECKS.clear()
    client.get("/access-requests", headers=auth("tok-owner"))
    client.get("/access-requests", headers=auth("tok-owner"))
    assert len(ADMIN_CHECKS) == 2


@pytest.mark.parametrize("decision", ["grant", "deny", "clear"])
def test_an_unreachable_worker_refuses_decisions(tmp_path, decision):
    """Fail closed: an outage must not wave decisions through."""
    client, store = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))

    WORKER_REACHABLE["value"] = False

    response = client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": decision},
        headers=auth("tok-owner"),
    )
    assert response.status_code == 403
    assert store._requests["blood-atlas"][STRANGER]["status"] == "pending"


def test_an_unreachable_worker_leaves_the_public_half_up(tmp_path):
    """Filing and reading your own request never touch the worker."""
    client, _ = build(tmp_path)
    WORKER_REACHABLE["value"] = False

    assert (
        client.post(
            "/datasets/blood-atlas/access-request", headers=auth("tok-stranger")
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/datasets/blood-atlas/access-request", headers=auth("tok-stranger")
        ).json()["status"]
        == "pending"
    )


def test_an_unreachable_worker_does_not_revoke_dataset_access(tmp_path):
    """The overlay is a local file; a worker outage must not lock readers out."""
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "grant"},
        headers=auth("tok-owner"),
    )

    WORKER_REACHABLE["value"] = False

    assert (
        client.get(
            "/datasets/blood-atlas/files", headers=auth("tok-stranger")
        ).status_code
        == 200
    )
    assert (
        client.get("/datasets/blood-atlas/files", headers=auth("tok-member")).status_code
        == 200
    )


def test_an_unauthenticated_caller_cannot_list(tmp_path):
    client, _ = build(tmp_path)
    assert client.get("/access-requests").status_code == 403


# --- the real is_worker_admin, with the Hypha connection stubbed out ---


def connect_recorder(monkeypatch, check_access_returns=True, raises=None):
    """Replace hypha_rpc.connect_to_server and record whether it was used."""
    import hypha_rpc

    calls = []

    class FakeWorker:
        async def check_access(self):
            return check_access_returns

    class FakeClient:
        async def get_service(self, service_id):
            calls.append(("get_service", service_id))
            if raises:
                raise raises
            return FakeWorker()

    class FakeConnect:
        def __init__(self, config):
            calls.append(("connect", config.get("token")))

        async def __aenter__(self):
            return FakeClient()

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(hypha_rpc, "connect_to_server", FakeConnect)
    return calls


def test_is_worker_admin_skips_the_connection_without_a_token(monkeypatch):
    """An anonymous caller is never an admin, so don't spend a websocket on it."""
    calls = connect_recorder(monkeypatch)
    assert asyncio.run(is_worker_admin(None, "ws/w:bioengine-worker", "http://h")) is False
    assert calls == []


def test_is_worker_admin_sends_the_callers_own_token(monkeypatch):
    calls = connect_recorder(monkeypatch, check_access_returns=True)
    assert asyncio.run(
        is_worker_admin("tok-caller", "ws/w:bioengine-worker", "http://h")
    ) is True
    assert ("connect", "tok-caller") in calls
    assert ("get_service", "ws/w:bioengine-worker") in calls


def test_is_worker_admin_relays_a_negative_verdict(monkeypatch):
    connect_recorder(monkeypatch, check_access_returns=False)
    assert asyncio.run(
        is_worker_admin("tok-caller", "ws/w:bioengine-worker", "http://h")
    ) is False


def test_is_worker_admin_fails_closed_when_the_worker_is_unreachable(monkeypatch):
    connect_recorder(monkeypatch, raises=RuntimeError("Service not found"))
    assert asyncio.run(
        is_worker_admin("tok-caller", "ws/w:bioengine-worker", "http://h")
    ) is False


def test_approver_lists_every_request_oldest_first(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post("/datasets/other-set/access-request", headers=auth("tok-mixedcase"))

    listed = client.get("/access-requests", headers=auth("tok-owner")).json()
    assert [record["dataset_id"] for record in listed] == ["blood-atlas", "other-set"]


def test_resolving_an_absent_request_is_refused(tmp_path):
    client, _ = build(tmp_path)
    response = client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "deny"},
        headers=auth("tok-owner"),
    )
    assert response.status_code == 400
    assert "No access request" in response.json()["detail"]


def test_unknown_decision_is_refused(tmp_path):
    client, store = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    response = client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "approve"},
        headers=auth("tok-owner"),
    )
    assert response.status_code == 400
    assert store._requests["blood-atlas"][STRANGER]["status"] == "pending"


# ---------------------------------------------------------------------------
# A grant is the record, and it has to actually open the door
# ---------------------------------------------------------------------------


def test_stranger_is_refused_before_a_grant(tmp_path):
    client, _ = build(tmp_path)
    assert (
        client.get(
            "/datasets/blood-atlas/files", headers=auth("tok-stranger")
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/data/blood-atlas/notes.txt", headers=auth("tok-stranger")
        ).status_code
        == 403
    )


def test_grant_opens_file_listing_and_file_bytes(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "grant"},
        headers=auth("tok-owner"),
    )

    assert "notes.txt" in client.get(
        "/datasets/blood-atlas/files", headers=auth("tok-stranger")
    ).json()
    bytes_response = client.get(
        "/data/blood-atlas/notes.txt", headers=auth("tok-stranger")
    )
    assert bytes_response.status_code == 200
    assert bytes_response.text == "payload"


def test_grant_is_scoped_to_the_dataset_it_was_made_on(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "grant"},
        headers=auth("tok-owner"),
    )
    assert (
        client.get("/datasets/other-set/files", headers=auth("tok-stranger")).status_code
        == 403
    )


def test_grant_stores_the_address_hypha_reports_not_the_key(tmp_path):
    """check_permissions compares case-sensitively; a lowercased grant would
    store an entry the caller's own requests never match."""
    client, store = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-mixedcase"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        # The approver names the requester in a third casing on purpose.
        params={"user": "MIXED.CASE@lab.ORG", "decision": "grant"},
        headers=auth("tok-owner"),
    )

    assert store.granted_users("blood-atlas") == ["Mixed.Case@Lab.org"]
    assert (
        client.get(
            "/datasets/blood-atlas/files", headers=auth("tok-mixedcase")
        ).status_code
        == 200
    )


def test_grant_does_not_displace_the_manifest_allowlist(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "grant"},
        headers=auth("tok-owner"),
    )
    assert (
        client.get("/datasets/blood-atlas/files", headers=auth("tok-member")).status_code
        == 200
    )


def test_a_dataset_with_an_empty_allowlist_can_still_be_granted(tmp_path):
    client, _ = build(tmp_path, authorized_users=[])
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "grant"},
        headers=auth("tok-owner"),
    )
    assert (
        client.get(
            "/datasets/blood-atlas/files", headers=auth("tok-stranger")
        ).status_code
        == 200
    )


# ---------------------------------------------------------------------------
# Denial is terminal until cleared; clear revokes a grant
# ---------------------------------------------------------------------------


def test_denial_blocks_a_second_request(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "deny"},
        headers=auth("tok-owner"),
    )

    response = client.post(
        "/datasets/blood-atlas/access-request", headers=auth("tok-stranger")
    )
    assert response.status_code == 409
    assert "denied" in response.json()["detail"]


def test_denial_is_visible_to_the_requester(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "deny"},
        headers=auth("tok-owner"),
    )
    own = client.get(
        "/datasets/blood-atlas/access-request", headers=auth("tok-stranger")
    ).json()
    assert own["status"] == "denied"
    assert own["resolved_by"] == OWNER


def test_denial_grants_nothing(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "deny"},
        headers=auth("tok-owner"),
    )
    assert (
        client.get(
            "/datasets/blood-atlas/files", headers=auth("tok-stranger")
        ).status_code
        == 403
    )


def test_clear_lifts_a_denial_and_lets_them_ask_again(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "deny"},
        headers=auth("tok-owner"),
    )

    cleared = client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "clear"},
        headers=auth("tok-owner"),
    )
    assert cleared.json() is None
    assert (
        client.post(
            "/datasets/blood-atlas/access-request", headers=auth("tok-stranger")
        ).status_code
        == 200
    )


def test_clear_revokes_a_grant(tmp_path):
    client, _ = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "grant"},
        headers=auth("tok-owner"),
    )
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "clear"},
        headers=auth("tok-owner"),
    )
    assert (
        client.get(
            "/datasets/blood-atlas/files", headers=auth("tok-stranger")
        ).status_code
        == 403
    )


def test_clear_leaves_other_requests_in_place(tmp_path):
    client, store = build(tmp_path)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-mixedcase"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "clear"},
        headers=auth("tok-owner"),
    )
    assert list(store._requests["blood-atlas"]) == ["mixed.case@lab.org"]


# ---------------------------------------------------------------------------
# Persistence — the restart path
# ---------------------------------------------------------------------------


def test_a_grant_survives_a_restart(tmp_path):
    store_file = tmp_path / "state" / "access_requests.json"
    client, _ = build(tmp_path, store_file=store_file)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-mixedcase"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": "Mixed.Case@Lab.org", "decision": "grant"},
        headers=auth("tok-owner"),
    )

    fresh_client, fresh_store = build(
        tmp_path / "second", store_file=store_file
    )
    assert fresh_store.granted_users("blood-atlas") == ["Mixed.Case@Lab.org"]
    assert (
        fresh_client.get(
            "/datasets/blood-atlas/files", headers=auth("tok-mixedcase")
        ).status_code
        == 200
    )


def test_a_denial_survives_a_restart(tmp_path):
    store_file = tmp_path / "state" / "access_requests.json"
    client, _ = build(tmp_path, store_file=store_file)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "deny"},
        headers=auth("tok-owner"),
    )

    fresh_client, _ = build(tmp_path / "second", store_file=store_file)
    assert (
        fresh_client.post(
            "/datasets/blood-atlas/access-request", headers=auth("tok-stranger")
        ).status_code
        == 409
    )


def test_persistence_happens_before_the_decision_takes_effect(tmp_path):
    """A grant that only exists in memory would revert on restart without ever
    failing, so a store that cannot be written must refuse the decision."""
    store_file = tmp_path / "state" / "access_requests.json"
    client, store = build(tmp_path, store_file=store_file)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))

    # A regular file where the store's directory should be: mkdir fails, so the
    # write fails, without depending on file permissions (the suite runs as root).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    store.store_file = blocker / "access_requests.json"

    with pytest.raises(RuntimeError, match="Failed to persist"):
        store.resolve("blood-atlas", STRANGER, "grant", OWNER)
    assert store._requests["blood-atlas"][STRANGER]["status"] == "pending"


def test_an_unreadable_store_starts_empty_rather_than_crashing(tmp_path):
    store_file = tmp_path / "state" / "access_requests.json"
    store_file.parent.mkdir(parents=True)
    store_file.write_text("{ not json")
    _, store = build(tmp_path, store_file=store_file)
    assert store.list_all() == []


def test_a_store_of_the_wrong_shape_is_ignored(tmp_path):
    store_file = tmp_path / "state" / "access_requests.json"
    store_file.parent.mkdir(parents=True)
    store_file.write_text(json.dumps({"blood-atlas": ["not", "a", "mapping"]}))
    _, store = build(tmp_path, store_file=store_file)
    assert store.list_all() == []


def test_turning_the_feature_off_stops_honouring_its_grants(tmp_path):
    """An overlay grant is invisible in the manifest, so a server whose request
    surface is off must not keep opening the door with one."""
    store_file = tmp_path / "state" / "access_requests.json"
    client, _ = build(tmp_path, store_file=store_file)
    client.post("/datasets/blood-atlas/access-request", headers=auth("tok-stranger"))
    client.post(
        "/access-requests/blood-atlas/resolve",
        params={"user": STRANGER, "decision": "grant"},
        headers=auth("tok-owner"),
    )

    off_client, off_store = build(
        tmp_path / "second", store_file=store_file, enabled=False
    )
    assert off_store.granted_users("blood-atlas") == []
    assert (
        off_client.get(
            "/datasets/blood-atlas/files", headers=auth("tok-stranger")
        ).status_code
        == 403
    )
