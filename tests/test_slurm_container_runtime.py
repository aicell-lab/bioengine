"""The generated SLURM batch script must not pin the container runtime.

``scripts/start_hpc_worker.sh`` has always fallen back apptainer then
singularity, and the deployment guide's SLURM prerequisites say "Singularity or
Apptainer installed on compute nodes". The generated batch script hardcoded
``apptainer exec``, so a site whose compute nodes carry only singularity passed
every check the launcher makes and then failed per-job, inside a script the
operator never wrote.

These assert on the generated script TEXT rather than on a resolution helper,
because the script is the artefact that actually runs and it is produced far
from where the choice is made.
"""

import re

import pytest

from bioengine.cluster.slurm_workers import SlurmWorkers


class _RayCluster:
    """The only thing the script builder reads off the cluster."""

    address = "ray://head.example:10001"


@pytest.fixture
def sbatch_script(tmp_path):
    """The batch script SlurmWorkers generates, read back off disk."""
    workers = SlurmWorkers(
        ray_cluster=_RayCluster(),
        worker_workspace_dir=str(tmp_path),
        image="ghcr.io/aicell-lab/bioengine-worker:test",
    )
    path = workers._create_sbatch_script(
        num_gpus=1,
        num_cpus=4,
        mem_in_gb_per_cpu=8,
        time_limit="1:00:00",
        further_slurm_args=[],
    )
    return open(path, encoding="utf-8").read()


def test_the_runtime_is_resolved_on_the_compute_node(sbatch_script):
    # The head node that generates this script cannot see the compute node's
    # toolchain, so the choice has to happen at job time.
    assert "command -v apptainer" in sbatch_script
    assert "command -v singularity" in sbatch_script
    assert "CONTAINER_CMD=apptainer" in sbatch_script
    assert "CONTAINER_CMD=singularity" in sbatch_script


def test_the_exec_line_does_not_name_a_runtime(sbatch_script):
    # The defect: `apptainer exec` written literally into the script.
    assert not re.search(r"^\s*apptainer exec\b", sbatch_script, re.M)
    assert not re.search(r"^\s*singularity exec\b", sbatch_script, re.M)
    assert '"$CONTAINER_CMD" exec' in sbatch_script


def test_apptainer_is_preferred_when_both_are_present(sbatch_script):
    # Same order as scripts/start_hpc_worker.sh. Checked by position, since both
    # names appear and mere presence would not distinguish the order.
    assert sbatch_script.index("command -v apptainer") < sbatch_script.index(
        "command -v singularity"
    )


def test_neither_runtime_present_fails_the_job_loudly(sbatch_script):
    # A job that silently proceeds without a container runtime would fail later
    # and further from the cause.
    assert "Neither apptainer nor singularity found" in sbatch_script
    assert "exit 1" in sbatch_script


def test_both_cache_and_bind_variables_stay_covered(sbatch_script):
    # These were already runtime-agnostic before the fix; a change that pinned
    # the runtime again would likely drop one of them too.
    for var in (
        "APPTAINER_CACHEDIR",
        "SINGULARITY_CACHEDIR",
        "APPTAINER_BIND",
        "SINGULARITY_BIND",
    ):
        assert var in sbatch_script, var
