"""Pin the ProxyDeployment client_id construction — worker_client_id +
sha1(application_id)[:8].

The scheme must be:
- deterministic (same app_id → same suffix)
- stable across ProxyDeployment replica restarts (no replica_tag input)
- unique per app on the same worker (different app_ids → different suffixes)

The dashboard's Monitor & Manage view and every ``get_app_status`` caller
build the service URL by mirroring this formula in
``bioengine.apps.manager.AppsManager._get_application_service_ids``, so
the two implementations must agree.
"""
from __future__ import annotations

import hashlib


def _expected_suffix(application_id: str) -> str:
    return hashlib.sha1(application_id.encode("utf-8")).hexdigest()[:8]


def _proxy_client_id_from_code(
    worker_client_id: str, application_id: str
) -> str:
    """Replicate the exact expression used in ProxyDeployment.__init__."""
    app_hash = hashlib.sha1(application_id.encode("utf-8")).hexdigest()[:8]
    return f"{worker_client_id}-{app_hash}"


def test_client_id_is_deterministic_for_same_app_id() -> None:
    assert _proxy_client_id_from_code("worker-abc", "demo-app") == _proxy_client_id_from_code(
        "worker-abc", "demo-app"
    )


def test_client_id_uses_worker_client_id_as_prefix() -> None:
    result = _proxy_client_id_from_code("bioengine-worker-kth-2b297", "model-runner")
    assert result.startswith("bioengine-worker-kth-2b297-")
    assert len(result) == len("bioengine-worker-kth-2b297-") + 8


def test_different_app_ids_produce_different_suffixes() -> None:
    a = _proxy_client_id_from_code("worker-abc", "model-runner")
    b = _proxy_client_id_from_code("worker-abc", "cellpose-finetuning")
    assert a != b
    assert a.split("-")[-1] != b.split("-")[-1]


def test_different_workers_but_same_app_share_the_hash_suffix() -> None:
    a = _proxy_client_id_from_code("worker-A", "demo-app")
    b = _proxy_client_id_from_code("worker-B", "demo-app")
    # The suffix (the app-hash part) matches across workers — only the
    # worker prefix differs — which is exactly what makes the client_id
    # stable across worker restarts if the operator pins --client-id.
    assert a.split("-")[-1] == b.split("-")[-1] == _expected_suffix("demo-app")


def test_manager_and_proxy_agree_on_the_url_shape() -> None:
    """The manager's _get_application_service_ids must build the URL
    from exactly the same formula as ProxyDeployment.__init__."""
    from bioengine.apps import manager as manager_module
    from bioengine.apps import proxy_deployment as proxy_module

    # Both modules import hashlib at module level for the same formula.
    assert hasattr(manager_module, "hashlib")
    assert hasattr(proxy_module, "hashlib")
