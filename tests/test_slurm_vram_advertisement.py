"""The generated SLURM batch script must not advertise VRAM it cannot back.

``VRAM_MB`` is a promise: the AppBuilder packs replicas onto a GPU by that
number, so a node advertising more than one device can address overcommits the
GPU and the replica OOMs at startup. Two cases where the honest answer is to
advertise nothing are pinned here, since both were found on a live cluster
rather than in review.

These assert on the generated script TEXT because the detection runs on the
compute node, far from where the script is written.
"""

import re

import pytest

from bioengine.cluster.slurm_workers import SlurmWorkers


class _RayCluster:
    address = "ray://head.example:10001"


def _script(tmp_path, num_gpus):
    workers = SlurmWorkers(
        ray_cluster=_RayCluster(),
        worker_workspace_dir=str(tmp_path),
        image="ghcr.io/aicell-lab/bioengine-worker:test",
    )
    path = workers._create_sbatch_script(
        num_gpus=num_gpus,
        num_cpus=4,
        mem_in_gb_per_cpu=8,
        time_limit="1:00:00",
        further_slurm_args=[],
    )
    return open(path, encoding="utf-8").read()


@pytest.fixture
def one_gpu_script(tmp_path):
    return _script(tmp_path, num_gpus=1)


def test_a_mig_node_advertises_no_vram(one_gpu_script):
    # nvidia-smi reports the PARENT device's memory on a MIG allocation — an
    # A100-80GB host hands out a 1g.10gb slice and still answers 81920 — and no
    # --query-gpu field exposes the slice, so the figure cannot be corrected.
    assert "mig.mode.current" in one_gpu_script
    assert '"$MIG_MODE" = "Enabled"' in one_gpu_script


def test_the_mig_check_precedes_the_advertisement(one_gpu_script):
    # Order is the whole fix: reading memory.total first and advertising it
    # before testing MIG mode would publish the parent's figure anyway.
    assert one_gpu_script.index('"$MIG_MODE" = "Enabled"') < one_gpu_script.index(
        'VRAM_RESOURCE=", \\"VRAM_MB\\"'
    )


def test_a_plain_single_gpu_node_still_advertises(one_gpu_script):
    # The MIG gate must not cost non-MIG nodes their VRAM packing.
    assert "memory.total" in one_gpu_script
    assert 'VRAM_RESOURCE=", \\"VRAM_MB\\": $VRAM_PER_GPU"' in one_gpu_script


@pytest.mark.parametrize("num_gpus", [0, 2])
def test_only_single_gpu_workers_detect_vram(tmp_path, num_gpus):
    # A node-level VRAM_MB on a multi-GPU worker is a sum Ray can satisfy across
    # devices, which overcommits any one of them.
    script = _script(tmp_path, num_gpus=num_gpus)
    assert f"[ {num_gpus} -eq 1 ]" in script


def test_the_job_id_reaches_the_container_expanded(one_gpu_script):
    # Single quotes forward the literal "${SLURM_JOB_ID}", which the container
    # runtime passes through as an empty value.
    assert '--env=SLURM_JOB_ID="${SLURM_JOB_ID}"' in one_gpu_script
    assert not re.search(r"--env=SLURM_JOB_ID='", one_gpu_script)
