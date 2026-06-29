"""Pin the ``ProxyDeployment`` actor's CPU reservation at 0.

Ray Serve's ``@serve.deployment(...)`` decorator backfills
``ray_actor_options = {"num_cpus": 1}`` when the argument is omitted, which
silently turned every proxy into a 1-CPU reservation despite the worker
code intending 0. The decorator now sets ``num_cpus=0`` explicitly; this
test pins that invariant so the regression can't reappear unnoticed.
"""
from __future__ import annotations

from bioengine.apps.proxy_deployment import ProxyDeployment


def test_proxy_deployment_reserves_zero_cpus() -> None:
    assert ProxyDeployment.ray_actor_options.get("num_cpus") == 0
