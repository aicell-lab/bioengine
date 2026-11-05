"""Ray version-compat shims.

These imports access Ray internals because no public Ray API exposes the
same information across the 2.33 - 2.55 release range that BioEngine
supports:

  * ``GlobalState.{total,available}_resources_per_node`` — required by
    ``proxy_actor.get_cluster_state`` to report per-node availability.
    ``ray.nodes()`` and ``ray.util.state.list_nodes()`` return per-node
    *totals* only, and the dashboard ``/nodes`` endpoint only populates
    per-node availability when autoscaler V2 is running (empty in
    single-machine and SLURM modes).
  * ``GcsClientOptions`` — needed to construct ``GlobalState``.
  * ``global_worker.node.address_info["webui_url"]`` — fallback dashboard
    URL lookup, used only when the caller does not supply a URL (i.e. in
    external-cluster mode where BioEngine does not own the dashboard).
  * ``RAY_PROCESSES`` and ``psutil`` — what ``ray stop`` itself matches on.
    ``RayCluster._shutdown_head_node`` cannot shell out to ``ray stop``: it
    takes no cluster selector and kills every Ray process the invoking user
    owns on the node, so on a shared HPC login node stopping one worker
    takes down every other worker running under the same account. Reusing
    Ray's own table keeps the set of process kinds — and the order they
    must die in — identical to ``ray stop``. ``psutil`` is re-exported from
    Ray's vendored copy because it is not a direct BioEngine dependency and
    is absent from the worker image's site-packages.

This module is the single grep target for Ray private-API access in
BioEngine; if Ray ever ships a public equivalent, swap the imports here
and the call sites do not change.
"""
from typing import Optional

from ray._private.state import GlobalState  # noqa: F401
from ray._private.utils import psutil  # noqa: F401
from ray._raylet import GcsClientOptions  # noqa: F401
from ray.autoscaler._private.constants import RAY_PROCESSES  # noqa: F401


def get_dashboard_url_fallback() -> Optional[str]:
    """Return the dashboard URL via private worker state, or None on failure.

    Only used by ``BioEngineProxyActor`` when no URL was passed in by the
    caller — i.e. external-cluster mode, where BioEngine did not start the
    dashboard itself.
    """
    import ray

    try:
        return ray._private.worker.global_worker.node.address_info.get("webui_url")
    except Exception:
        return None
