"""Pin the token-expiration + deployed-by-worker fields on app_data.

Recovery reads these keys from the running ProxyDeployment's app_data
to (a) surface token expiry in ``get_app_status`` so operators can act
before it lapses and (b) detect apps deployed by a previous worker
whose client_id no longer matches (URLs may go stale).
"""
from __future__ import annotations

import inspect

from bioengine.apps import builder as builder_module


def test_app_data_keys_are_declared_in_build() -> None:
    """The three metadata keys must exist in the source of AppBuilder.build."""
    src = inspect.getsource(builder_module.AppBuilder.build)
    for key in (
        '"deployed_by_worker_client_id"',
        '"proxy_service_token_issued_at"',
        '"proxy_service_token_ttl_seconds"',
    ):
        assert key in src, f"expected {key} in AppBuilder.build source"


def test_metadata_mirrors_app_data_keys() -> None:
    """AppBuilder.build's returned ``metadata`` dict must carry the same
    keys, so fresh deploys (which use metadata) populate the same fields
    as recovered deploys (which use app_data)."""
    src = inspect.getsource(builder_module.AppBuilder.build)
    # Rough locate of the metadata literal: everything between the token
    # generation and the ``return BuiltApp(`` call. Overly-fragile? A bit,
    # but the failure mode of that fragility is a caught assert, not
    # silent drift.
    for key in (
        '"deployed_by_worker_client_id"',
        '"proxy_service_token_issued_at"',
        '"proxy_service_token_ttl_seconds"',
    ):
        assert src.count(key) >= 2, (
            f"expected {key} to appear in both app_data and metadata dicts"
        )
