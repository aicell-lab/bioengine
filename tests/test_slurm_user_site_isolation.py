"""The generated SLURM batch script must hide the host user site from pip.

Ray resolves a runtime_env by running pip inside the container, where apptainer
has mounted the submitting user's home. ``site.getusersitepackages()`` is
derived from HOME, so that pip sees ``~/.local`` and reports packages there as
already satisfied — while the replica that later imports them runs with HOME
rebound per app by ``bioengine._app.replica_init`` to a directory with no
``.local`` at all. Build time and run time compute different user sites, and the
package is missing exactly where it is needed.

Asserts on the generated script TEXT, like test_slurm_container_runtime.py, for
the same reason: the script is the artefact that runs.
"""

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


def test_the_user_site_is_hidden_from_the_container(sbatch_script):
    assert "--env=PYTHONNOUSERSITE=1" in sbatch_script


def test_it_is_passed_explicitly_because_the_environment_is_cleaned(sbatch_script):
    # --cleanenv drops the launcher's environment, so exporting the variable
    # outside the container would not reach pip. Dropping --cleanenv instead
    # would let the whole host environment back in.
    assert "--cleanenv" in sbatch_script
    assert sbatch_script.index("--cleanenv") < sbatch_script.index(
        "--env=PYTHONNOUSERSITE=1"
    )
