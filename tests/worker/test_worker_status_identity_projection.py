"""The worker's public status surface must not publish anyone's identity.

The worker's Hypha service is registered ``"visibility": "public"`` and
``get_status`` carries no permission check — deliberately, because it is the
health surface: the deployed Kubernetes *startup* probe curls it unauthenticated
and greps ``is_ready``, the website's worker list renders ``geo_location`` and
``bioengine_version`` on each card, and the CLI's cluster view and the KTH
gpu-cuda-watch script read ``ray_cluster``. But the payload also carried ``admin_users``, the
worker's admin allowlist, to every anonymous caller — the same list
``list_admin_users`` requires admin to return, and the same one permission
denials were changed to stop naming.

So the health half stays anonymous and ``admin_users`` goes only to a worker
admin, who is on it already.

Asserted on the response *shape*, not on the one field we know about today: a
denylist would leave every field nobody has thought of on the public side.

Every address here is a fake at example.invalid, which by RFC 2606 can never be
a real domain.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from bioengine.utils import create_context
from bioengine.worker.worker import (
    PUBLIC_WORKER_STATUS_FIELDS,
    BioEngineWorker,
    public_worker_status,
)

ADMIN_EMAIL = "admin@example.invalid"
ADMIN_ID = "user:admin"
OUTSIDER_EMAIL = "outsider@example.invalid"

ADMIN = create_context(ADMIN_ID, ADMIN_EMAIL)
OUTSIDER = create_context("user:outsider", OUTSIDER_EMAIL)
ANONYMOUS = create_context()

# Planted in a field that IS public, so "the marker is absent" means the
# projection dropped it rather than that searching the response never worked.
PUBLIC_MARKER = "PLANTED-PUBLIC-GEO-MARKER"


def bare_worker(admin_users=None, **attrs) -> BioEngineWorker:
    """A worker with only the state ``get_status`` reads."""
    worker = BioEngineWorker.__new__(BioEngineWorker)
    worker.logger = logging.getLogger("test")  # __del__ reaches for it
    worker.start_time = 1_700_000_000.0
    worker.workspace = "example-workspace"
    worker.client_id = "worker-abc"
    worker.admin_users = list(
        admin_users if admin_users is not None else [ADMIN_EMAIL, ADMIN_ID]
    )
    worker.geo_location = {"country_name": "Sweden", "region": PUBLIC_MARKER}
    worker.is_ready = asyncio.Event()
    worker.is_ready.set()
    worker._monitor_consecutive_errors = 0
    worker._monitor_degraded_threshold = 5
    worker.ray_cluster = SimpleNamespace(
        mode="slurm",
        status={
            "head_address": "10.0.0.7:6379",
            "mode": "slurm",
            "geo_location": {"country_name": "Sweden"},
            "cluster": {"total_cpu": 64, "total_gpu": 4},
            "nodes": {"node-1": {"total_gpu": 4}},
            "slurm_jobs": {"queued": [], "running": []},
        },
        slurm_workers=None,
    )
    for key, value in attrs.items():
        setattr(worker, key, value)
    return worker


# ---------------------------------------------------------------------------
# What an unauthenticated caller must not receive
# ---------------------------------------------------------------------------


async def test_the_anonymous_response_carries_no_field_outside_the_allowlist():
    """The point of the allowlist: a field added to the status payload next
    year must not become public by nobody having thought about it."""
    status = await bare_worker().get_status(context=ANONYMOUS)
    unexpected = set(status) - set(PUBLIC_WORKER_STATUS_FIELDS)
    assert not unexpected, f"published {sorted(unexpected)}"


async def test_every_field_in_the_payload_has_been_decided():
    """The half of the allowlist a projection cannot enforce by itself: this
    fails when a *new* field appears in the admin payload, so whoever adds one
    has to say out loud whether it is public."""
    full = await bare_worker().get_status(context=ADMIN)
    undecided = set(full) - set(PUBLIC_WORKER_STATUS_FIELDS) - {"admin_users"}
    assert not undecided, (
        f"{sorted(undecided)} is in the get_status payload but neither in "
        "PUBLIC_WORKER_STATUS_FIELDS nor knowingly withheld"
    )


async def test_no_planted_identity_survives_into_the_anonymous_response():
    """Searched over the whole serialized response rather than field by field,
    so a value that reappears somewhere unexpected is still caught."""
    rendered = repr(await bare_worker().get_status(context=ANONYMOUS))
    for planted in (ADMIN_EMAIL, ADMIN_ID):
        assert planted not in rendered, planted


async def test_the_search_finds_a_value_that_is_meant_to_come_through():
    """Positive control for the assertion above: absence means the projection
    dropped it, not that searching the response never worked."""
    rendered = repr(await bare_worker().get_status(context=ANONYMOUS))
    assert PUBLIC_MARKER in rendered


async def test_a_logged_in_outsider_sees_only_the_public_view():
    status = await bare_worker().get_status(context=OUTSIDER)
    assert "admin_users" not in status


async def test_a_malformed_context_gets_the_public_view():
    """Fails closed: a caller the gate cannot authorize is not an admin."""
    for context in ({}, {"user": "not-a-dict"}, {"user": {}}, None):
        status = await bare_worker().get_status(context=context)
        assert "admin_users" not in status, context


async def test_a_caller_covered_only_by_a_wildcard_does_not_earn_the_list():
    """A '*' entry authorizes *using* the worker, not reading who runs it.
    ``_forbid_wildcard_admin_users`` strips '*' from admin_users at startup, so
    this is the second barrier rather than the first — but reading the list is
    exactly the operation that must not inherit a wildcard grant."""
    status = await bare_worker(admin_users=["*"]).get_status(context=OUTSIDER)
    assert "admin_users" not in status


# ---------------------------------------------------------------------------
# What the anonymous half is for: health, versions, location, hardware
# ---------------------------------------------------------------------------


async def test_every_public_field_reaches_an_anonymous_caller():
    status = await bare_worker().get_status(context=ANONYMOUS)
    missing = set(PUBLIC_WORKER_STATUS_FIELDS) - set(status)
    assert not missing, f"withheld {sorted(missing)}"


async def test_the_kubernetes_startup_probe_still_reads_readiness_unauthenticated():
    """The deployed startup probe is a plain ``curl | grep '"is_ready": true'``
    with no token, and it is the one probe that calls this method — the
    deployed liveness probe is a local ``kill -0 1``. Asserted on the serialized
    bytes because that is what the probe actually matches."""
    status = await bare_worker().get_status(context=ANONYMOUS)
    assert '"is_ready": true' in json.dumps(status)


async def test_the_anonymous_caller_still_sees_versions_and_mode():
    """What a dashboard and the worker-list cards render."""
    status = await bare_worker().get_status(context=ANONYMOUS)
    assert status["worker_mode"] == "slurm"
    assert status["bioengine_version"]
    assert status["ray_version"]
    assert status["hypha_rpc_version"]
    assert status["service_uptime"] >= 0


async def test_ray_cluster_and_geo_location_stay_public():
    """A deliberate decision, pinned so it is reopened rather than drifted
    into: neither names a third party. ``geo_location`` is the operator's own
    advertisement of where the worker runs and is what the website's worker
    list draws on the map; ``ray_cluster`` carries cluster-internal addresses
    and a resource inventory that the CLI's ``cluster status``, the custom
    dashboard template, the federated-run scripts and the KTH gpu-cuda-watch
    script all read.

    ``ray_cluster`` is published *whole*. The allowlist is top level only, so
    anything nested under it is public the day it lands — deliberately, and
    stated in PUBLIC_WORKER_STATUS_FIELDS so the next reader does not assume
    the guarantee reaches further than it does."""
    status = await bare_worker().get_status(context=ANONYMOUS)
    assert status["geo_location"]["country_name"] == "Sweden"
    assert status["ray_cluster"]["head_address"] == "10.0.0.7:6379"
    assert status["ray_cluster"]["cluster"]["total_gpu"] == 4


# ---------------------------------------------------------------------------
# The list still goes to the people already on it
# ---------------------------------------------------------------------------


async def test_a_worker_admin_still_receives_the_admin_users_list():
    status = await bare_worker().get_status(context=ADMIN)
    assert status["admin_users"] == [ADMIN_EMAIL, ADMIN_ID]


async def test_an_admin_matched_by_user_id_also_receives_it():
    worker = bare_worker(admin_users=[ADMIN_ID])
    status = await worker.get_status(context=create_context(ADMIN_ID, None))
    assert status["admin_users"] == [ADMIN_ID]


@pytest.mark.parametrize(
    "context, expected",
    [(ADMIN, True), (OUTSIDER, False), (ANONYMOUS, False)],
)
async def test_the_websites_is_worker_admin_computation_still_works(context, expected):
    """``BioEngineWorker.tsx`` derives admin-only UI from the status payload:
    ``const admins = status?.admin_users ?? []`` then checks whether the
    logged-in user's id or email is in it. Transcribed here so a projection
    that breaks the dashboard's admin gate fails in this repo."""
    status = await bare_worker().get_status(context=context)

    admins = status.get("admin_users") or []
    user = context["user"]
    is_worker_admin = "*" in admins or any(
        identifier in admins for identifier in (user["id"], user["email"])
    )

    assert is_worker_admin is expected


# ---------------------------------------------------------------------------
# check_access, on the same public service
# ---------------------------------------------------------------------------


async def test_check_access_answers_only_about_the_caller():
    """Read in the same pass as get_status because it is registered on the same
    public service. It takes no argument naming anyone else and returns a bare
    bool, so it tells a caller about their own entitlement and nothing about
    who else has it."""
    worker = bare_worker()
    assert await worker.check_access(context=ADMIN) is True
    for context in (OUTSIDER, ANONYMOUS):
        answer = await worker.check_access(context=context)
        assert answer is False
        assert ADMIN_EMAIL not in repr(answer)


# ---------------------------------------------------------------------------
# The projection on its own
# ---------------------------------------------------------------------------


def test_the_projection_copies_every_public_field_not_just_the_first():
    payload = {field: f"value-{field}" for field in PUBLIC_WORKER_STATUS_FIELDS}
    payload["admin_users"] = [ADMIN_EMAIL]
    assert public_worker_status(payload) == {
        field: f"value-{field}" for field in PUBLIC_WORKER_STATUS_FIELDS
    }


def test_the_projection_omits_a_public_field_the_payload_lacks():
    """A partial payload must not gain a KeyError or a None-valued field."""
    assert public_worker_status({"is_ready": True}) == {"is_ready": True}
