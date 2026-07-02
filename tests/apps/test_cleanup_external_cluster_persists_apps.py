"""Pin the external-cluster cleanup invariant.

In external-cluster mode the Ray Serve state (including every deployed
ProxyDeployment + entry deployment) lives on the shared KubeRay cluster,
outside the worker pod. When the worker pod terminates (SIGTERM, helm
uninstall, node reschedule), apps MUST persist so the next worker's
recover_deployed_applications can adopt them — otherwise pod restarts
silently wipe out running services.

Before this pin, BioEngineWorker._cleanup unconditionally called
apps_manager.stop_all_apps → serve.delete(app_id) for every deployed
app. Combined with the SIGTERM handler added in the same PR, k8s pod
termination would delete Ray Serve apps from the shared cluster and
defeat recovery.

Source-inspection test rather than an end-to-end integration test —
_cleanup composes Ray + Hypha + AppsManager in a way that's not
practical to reconstruct in a unit test. The contract we're pinning is
that the guard exists.
"""
from __future__ import annotations

import inspect

from bioengine.worker import worker as worker_module


def test_cleanup_skips_stop_all_apps_in_external_cluster_mode() -> None:
    src = inspect.getsource(worker_module.BioEngineWorker._cleanup)
    # The guard must check for external-cluster before calling stop_all_apps.
    assert "external-cluster" in src, (
        "expected _cleanup to branch on ray_cluster.mode == 'external-cluster'"
    )
    # stop_all_apps must be under the non-external-cluster branch, not
    # unconditional as before.
    external_guard = src.find("external-cluster")
    stop_all_apps = src.find("stop_all_apps")
    assert external_guard != -1 and stop_all_apps != -1
    assert external_guard < stop_all_apps, (
        "the external-cluster guard must precede the stop_all_apps call — "
        "otherwise external-cluster pod restarts wipe the shared Ray Serve state."
    )
