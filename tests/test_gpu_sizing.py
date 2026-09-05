"""Unit tests for deploy-time GPU reservation sizing.

Exercises the pure static helpers on ``AppBuilder`` directly so no Ray cluster
or Hypha server is needed.
"""

import logging

import pytest

from bioengine.apps.builder import _GPU_HANDLE_EPSILON, AppBuilder
from bioengine._app.errors import BioEngineUserError

logger = logging.getLogger("test")


def _spec(gpu_memory_mb, ray_actor_options=None):
    return {
        "classes": {
            "App": {
                "gpu_memory_mb": gpu_memory_mb,
                "ray_actor_options": ray_actor_options or {},
            }
        }
    }


def _opts(spec):
    return spec["classes"]["App"]["ray_actor_options"]


def test_vram_advertised_uses_custom_resource():
    spec = _spec(8000)
    AppBuilder._apply_gpu_memory(
        spec,
        disable_gpu=False,
        gpu_sizing={"vram_resource_advertised": True, "min_gpu_total_mb": 40960},
        logger=logger,
    )
    assert _opts(spec)["num_gpus"] == _GPU_HANDLE_EPSILON
    assert _opts(spec)["resources"]["VRAM_MB"] == 8000


def test_no_vram_resource_falls_back_to_fraction():
    spec = _spec(8000)
    AppBuilder._apply_gpu_memory(
        spec,
        disable_gpu=False,
        gpu_sizing={"vram_resource_advertised": False, "min_gpu_total_mb": 40960},
        logger=logger,
    )
    assert _opts(spec)["num_gpus"] == pytest.approx(0.2, abs=0.01)
    assert "resources" not in _opts(spec)


def test_elastic_cluster_without_gpu_node_requests_vram():
    """A SLURM cluster scaled to zero has no GPU node to size against, but every
    worker it launches advertises VRAM_MB — so request VRAM, not a whole GPU."""
    spec = _spec(8000)
    AppBuilder._apply_gpu_memory(
        spec,
        disable_gpu=False,
        gpu_sizing={
            "vram_resource_advertised": False,
            "min_gpu_total_mb": None,
            "elastic_gpu_cluster": True,
        },
        logger=logger,
    )
    assert _opts(spec)["num_gpus"] == _GPU_HANDLE_EPSILON
    assert _opts(spec)["resources"]["VRAM_MB"] == 8000


def test_static_cluster_without_gpu_node_reserves_whole_gpu():
    spec = _spec(8000)
    AppBuilder._apply_gpu_memory(
        spec,
        disable_gpu=False,
        gpu_sizing={"vram_resource_advertised": False, "min_gpu_total_mb": None},
        logger=logger,
    )
    assert _opts(spec)["num_gpus"] == 1.0


def test_whole_gpu_sentinel_ignores_vram_packing():
    spec = _spec(-1)
    AppBuilder._apply_gpu_memory(
        spec,
        disable_gpu=False,
        gpu_sizing={
            "vram_resource_advertised": True,
            "min_gpu_total_mb": 40960,
            "elastic_gpu_cluster": True,
        },
        logger=logger,
    )
    assert _opts(spec)["num_gpus"] == 1.0
    assert "resources" not in _opts(spec)


def test_disable_gpu_wins():
    spec = _spec(8000)
    AppBuilder._apply_gpu_memory(
        spec,
        disable_gpu=True,
        gpu_sizing={"vram_resource_advertised": True, "min_gpu_total_mb": 40960},
        logger=logger,
    )
    assert _opts(spec)["num_gpus"] == 0


def test_reapplying_clears_previous_reservation():
    """Re-derivation must be idempotent: a VRAM reservation from an earlier
    cluster view must not linger once the fraction path is taken."""
    spec = _spec(8000)
    vram_sizing = {"vram_resource_advertised": True, "min_gpu_total_mb": 40960}
    AppBuilder._apply_gpu_memory(spec, False, vram_sizing, logger)
    AppBuilder._apply_gpu_memory(
        spec,
        False,
        {"vram_resource_advertised": False, "min_gpu_total_mb": 40960},
        logger,
    )
    assert "resources" not in _opts(spec)
    assert _opts(spec)["num_gpus"] == pytest.approx(0.2, abs=0.01)


def test_request_larger_than_smallest_gpu_rejected():
    spec = _spec(80000)
    with pytest.raises(BioEngineUserError, match="does not fit the smallest GPU"):
        AppBuilder._apply_gpu_memory(
            spec,
            disable_gpu=False,
            gpu_sizing={"vram_resource_advertised": False, "min_gpu_total_mb": 40960},
            logger=logger,
        )


def test_sum_resources_keeps_fractional_num_gpus():
    """Epsilon and fractional reservations must not truncate to zero, or the
    SLURM autoscaler sees an app with no GPU demand at all."""
    spec = _spec(8000, {"num_cpus": 1, "num_gpus": _GPU_HANDLE_EPSILON,
                        "resources": {"VRAM_MB": 8000}})
    totals = AppBuilder._sum_resources(spec, proxy_memory_in_gb=1)
    assert totals["num_gpus"] > 0
    assert totals["VRAM_MB"] == 8000
