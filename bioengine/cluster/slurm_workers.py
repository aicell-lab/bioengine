import asyncio
import logging
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import ray

from bioengine import __version__
from bioengine.utils import create_logger


class SlurmWorkers:
    """
    Manages SLURM workers for BioEngine with intelligent autoscaling capabilities.

    This class provides comprehensive management of Ray worker nodes running as SLURM jobs
    within containerized environments using Apptainer/Singularity. It includes intelligent
    autoscaling that automatically adjusts the number of workers based on pending tasks,
    resource utilization, and configurable policies.

    The autoscaling system continuously monitors the Ray cluster state and makes scaling decisions based on:
    - Number of pending tasks requiring node assignment
    - Current worker node utilization (CPU/GPU usage patterns)
    - Configurable cooldown periods to prevent oscillations
    - Min/max worker limits for cost control
    - Historical idle time patterns for efficient scale-down decisions

    Key Features:
    - Automatic worker scaling with intelligent demand prediction
    - Containerized worker execution with Apptainer/Singularity support
    - Graceful worker shutdown with proper job lifecycle management
    - Comprehensive error handling with retry mechanisms
    - Customizable SLURM job parameters and resource allocation
    - Real-time monitoring and logging for operational visibility
    """

    def __init__(
        self,
        ray_cluster,
        # Slurm job configuration parameters
        worker_workspace_dir: str,
        image: str = f"ghcr.io/aicell-lab/bioengine-worker:{__version__}",
        default_num_gpus: int = 1,
        default_num_cpus: int = 8,
        default_mem_in_gb_per_cpu: int = 16,
        default_time_limit: str = "4:00:00",
        further_slurm_args: Optional[List[str]] = None,
        gpu_slurm_flag: str = "--gpus={n}",
        further_apptainer_args: Optional[List[str]] = None,
        # Autoscaling configuration parameters
        min_workers: int = 0,
        max_workers: int = 4,
        scale_up_cooldown_seconds: int = 60,  # 1 minute between scale ups
        scale_down_check_interval_seconds: int = 60,  # Check for idle workers every 60 seconds
        scale_down_threshold_seconds: int = 300,  # 5 minutes of idleness before scale down
        grace_period: int = 60,  # Grace period for job cancellation
        # Logger configuration
        log_file: Optional[Union[str, Path]] = None,
        debug: bool = False,
    ):
        """
        Initialize the SlurmWorkers with the given configuration.

        Args:
            ray_cluster: Ray cluster manager instance
            worker_workspace_dir: Workspace directory mounted to the container when starting a worker
            image: BioEngine remote docker image or path to the image file
            default_num_gpus: Default number of GPUs to allocate per worker
            default_num_cpus: Default number of CPUs to allocate per worker
            default_mem_in_gb_per_cpu: Default memory (GB) to allocate per CPU
            default_time_limit: Default SLURM job time limit (HH:MM:SS format)
            further_slurm_args: Additional SLURM arguments to include in job submissions
                (e.g. ["--account=my-acct", "--partition=gpu", "-C", "thin"]).
            gpu_slurm_flag: Template for the SLURM GPU directive. The literal token
                "{n}" is replaced with the requested GPU count. Common values:
                "--gpus={n}" (default; Berzelius, generic clusters) or
                "--gres=gpu:{n}" (clusters that require the gres syntax). Set to an
                empty string to omit the directive (e.g. when GPUs are requested via
                ``further_slurm_args``).
            further_apptainer_args: Additional CLI flags passed to ``apptainer exec``
                between the built-in flags and the container image. Use for extra
                bind mounts (e.g. ["--bind", "/proj/aicell:/proj/aicell"]) or to
                forward host env vars.
            min_workers: Minimum number of workers to maintain
            max_workers: Maximum number of workers allowed
            scale_up_cooldown_seconds: Cooldown period in seconds between scale-ups
            scale_down_check_interval_seconds: Interval in seconds between scaling checks
            scale_down_threshold_seconds: Idle time in seconds before scaling down
            grace_period: Grace period in seconds for job cancellation
            log_file: Optional log file to write logs to
            debug: Enable debug logging

        Raises:
            ValueError: If worker_workspace_dir is None, or if gpu_slurm_flag is
                non-empty but does not contain the "{n}" placeholder.
        """
        # Set up logging
        self.logger = create_logger(
            name="SlurmWorkers",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        if worker_workspace_dir is None:
            raise ValueError(
                "Mountable worker workspace directory ('--worker-workspace-dir') must be set in 'SLURM' mode"
            )

        self.ray_cluster = ray_cluster

        # SLURM job configuration parameters
        self.image = (
            f"docker://{image}"
            if not image.startswith("docker://") and not image.endswith(".sif")
            else image
        )
        self.job_name = "ray_worker"
        self.worker_workspace_dir = str(worker_workspace_dir)
        self.default_num_gpus = default_num_gpus
        self.default_num_cpus = default_num_cpus
        self.default_mem_in_gb_per_cpu = default_mem_in_gb_per_cpu
        self.default_time_limit = default_time_limit
        self.further_slurm_args = further_slurm_args if further_slurm_args else []
        if gpu_slurm_flag and "{n}" not in gpu_slurm_flag:
            raise ValueError(
                "gpu_slurm_flag must be empty or contain the '{n}' placeholder "
                f"(got: {gpu_slurm_flag!r})"
            )
        self.gpu_slurm_flag = gpu_slurm_flag
        self.further_apptainer_args = (
            further_apptainer_args if further_apptainer_args else []
        )

        # Autoscaling configuration parameters
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_cooldown = scale_up_cooldown_seconds
        self.scale_down_check_interval = scale_down_check_interval_seconds
        self.scale_down_threshold = scale_down_threshold_seconds
        self.grace_period = grace_period

        # Initialize state variables
        self.last_scale_up_time = 0
        self.last_scale_down_check = 0
        self.worker_creation_task = None
        self.worker_deletion_tasks = {}

    def _create_sbatch_script(
        self,
        num_gpus: int,
        num_cpus: int,
        mem_in_gb_per_cpu: int,
        time_limit: str,
        further_slurm_args: List[str],
    ) -> str:
        # Ray reports resources as floats; sbatch directives need integers.
        num_gpus = int(num_gpus)
        num_cpus = int(num_cpus)
        mem_in_gb_per_cpu = max(1, int(mem_in_gb_per_cpu))
        """
        Create a SLURM batch script for launching a Ray worker in a container.

        Generates a temporary batch script with SLURM directives and Apptainer commands
        to launch a containerized Ray worker with specified resource allocation.

        Args:
            num_gpus: Number of GPUs to allocate
            num_cpus: Number of CPUs to allocate
            mem_in_gb_per_cpu: Memory (GB) to allocate per CPU
            time_limit: SLURM job time limit (HH:MM:SS format)
            further_slurm_args: Additional SLURM arguments to include

        Returns:
            Path to the created temporary batch script file

        Raises:
            Exception: If script creation fails
        """
        try:
            # Define the apptainer command with the Ray worker command
            apptainer_args = [
                "apptainer exec",
                "--nv",
                "--cleanenv",
                "--env=SLURM_JOB_ID='${SLURM_JOB_ID}'",
                "--pwd /app",
                f"--bind {self.worker_workspace_dir}:${{HOME}}/.bioengine",
            ]
            apptainer_args.extend(arg for arg in self.further_apptainer_args if arg)
            apptainer_cmd = " ".join(apptainer_args)
            self.logger.info(
                f"Binding workspace directory '{self.worker_workspace_dir}' to container directory '{os.environ['HOME']}/.bioengine'"
            )
            if self.further_apptainer_args:
                self.logger.info(
                    f"Additional apptainer args: {self.further_apptainer_args}"
                )

            # Define the Ray worker command that will run inside the container and add it to the command
            ray_worker_cmd = (
                "ray start "
                f"--address={self.ray_cluster.address} "
                f"--num-cpus={num_cpus} "
                f"--num-gpus={num_gpus} "
                "--resources='{\"slurm_job_id:'${SLURM_JOB_ID}'\": 1}' "
                "--block"
            )
            # Example: ray start --address='10.81.254.11:6379' --num-cpus=8 --num-gpus=1 --resources='{"slurm_job_id:${SLURM_JOB_ID}": 1}' --block

            # Add the container image and Ray worker command to the apptainer command
            apptainer_cmd += f" {self.image} {ray_worker_cmd}"

            # Define further SLURM arguments if provided
            further_slurm_args = (
                "\n".join([f"#SBATCH {arg}" for arg in further_slurm_args if arg])
                if further_slurm_args
                else ""
            )

            # Optional GPU directive — some clusters use "--gres=gpu:N" instead of "--gpus=N"
            gpu_directive = (
                f"#SBATCH {self.gpu_slurm_flag.format(n=num_gpus)}"
                if self.gpu_slurm_flag and num_gpus > 0
                else ""
            )

            # Define content of batch script with SLURM directives
            batch_script = f"""#!/bin/bash
            #SBATCH --job-name={self.job_name}
            #SBATCH --ntasks=1
            #SBATCH --nodes=1
            {gpu_directive}
            #SBATCH --cpus-per-task={num_cpus}
            #SBATCH --mem-per-cpu={mem_in_gb_per_cpu}G
            #SBATCH --time={time_limit}
            #SBATCH --chdir={self.worker_workspace_dir}
            #SBATCH --output={self.worker_workspace_dir}/slurm_logs/%x_%j.out
            #SBATCH --error={self.worker_workspace_dir}/slurm_logs/%x_%j.err
            {further_slurm_args}

            # Print some diagnostic information
            echo "Host: $(hostname)"
            echo "Date: $(date)"
            echo "GPUs: {num_gpus}, CPUs: {num_cpus}"
            echo "GPU info: $(nvidia-smi -L)"
            echo "Job ID: $SLURM_JOB_ID"
            echo "Working directory: $(pwd)"
            echo "Running command: {apptainer_cmd}"
            echo ""
            echo "========================================"
            echo ""

            if [ -z "$SLURM_JOB_ID" ]; then
                echo "SLURM_JOB_ID is not set. This script may not be running in a SLURM job."
                exit 1
            fi

            # Create the log directory if it doesn't exist
            mkdir -p {self.worker_workspace_dir}/slurm_logs

            # Reset bound in paths in case of submission from a container
            APPTAINER_BIND=
            SINGULARITY_BIND=

            # Set the cache directory for Apptainer
            export APPTAINER_CACHEDIR="{self.worker_workspace_dir}/images"
            export SINGULARITY_CACHEDIR="{self.worker_workspace_dir}/images"

            {apptainer_cmd}

            # Print completion status
            echo "Job completed with status $?"
            """

            # Create a temporary batch script
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False
            ) as batch_file:
                for line in batch_script.split("\n"):
                    # Remove leading whitespace
                    line = line.strip()
                    batch_file.write(line + "\n")

                self.logger.debug(
                    f"Created sbatch script '{batch_file.name}':\n{batch_script}"
                )

            return batch_file.name
        except Exception as e:
            self.logger.error(f"Failed to create sbatch script: {e}")
            raise e

    async def _submit_job(self, sbatch_script: str, delete_script: bool = False) -> str:
        """
        Submit a SLURM job using the provided batch script.

        Executes the sbatch command to submit a job to the SLURM scheduler and parses
        the job ID from the output.

        Args:
            sbatch_script: Path to the batch script file to submit
            delete_script: Whether to delete the script file after submission

        Returns:
            The SLURM job ID of the submitted job

        Raises:
            subprocess.CalledProcessError: If sbatch command fails
            RuntimeError: If job ID cannot be parsed from output
            Exception: If job submission fails for any other reason
        """
        try:
            # Submit the job with sbatch
            proc = await asyncio.create_subprocess_exec(
                "sbatch",
                sbatch_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self.logger.error(
                    f"sbatch returned exit {proc.returncode}: {error_msg.strip()}"
                )
                raise subprocess.CalledProcessError(
                    proc.returncode, "sbatch", stderr=error_msg
                )

            output = stdout.decode().strip()
            self.logger.info(output)

            # Clean up the temporary file
            if delete_script:
                try:
                    await asyncio.to_thread(os.remove, sbatch_script)
                except Exception:
                    self.logger.warning(
                        f"Failed to remove temporary script: {sbatch_script}"
                    )

            # Parse job ID from SlurmManager output (usually "Submitted batch job 12345")
            if output and "Submitted batch job" in output:
                job_id = output.split()[-1]
            else:
                raise RuntimeError(f"Failed to get job ID - stdout: {output}")

            return job_id
        except Exception as e:
            self.logger.error(f"Failed to submit job: {e}")
            raise e

    async def _get_jobs(self) -> Dict[str, Dict[str, str]]:
        """
        Query SLURM for status of all BioEngine worker jobs.

        Executes squeue command to retrieve information about all jobs for the current user,
        filters for BioEngine worker jobs, and returns their status information.

        Returns:
            Dict mapping job_id to dict containing name, state, runtime, time_limit

        Raises:
            subprocess.CalledProcessError: If squeue command fails
            Exception: If job status retrieval fails for any other reason
        """
        try:
            # Run squeue command to get all jobs for the current user
            # Format: JobID, Name, State, Time, TimeLimit
            proc = await asyncio.create_subprocess_exec(
                "squeue",
                "-u",
                os.environ["USER"],
                "-o",
                "%i %j %T %M %L",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(
                    proc.returncode, "squeue", stderr=error_msg
                )

            # Parse the output
            lines = stdout.decode().strip().split("\n")

            # Skip the header line if present
            if len(lines) > 0 and not lines[0][0].isdigit():
                lines = lines[1:]

            jobs = {}
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        job_id, job_name, state, runtime, time_limit = (
                            parts[0],
                            parts[1],
                            parts[2],
                            parts[3],
                            parts[4],
                        )

                        # Only include BioEngine worker jobs
                        if job_name == self.job_name:
                            jobs[job_id] = {
                                "name": job_name,
                                "state": state,
                                "runtime": runtime,
                                "time_limit": time_limit,
                            }

            return jobs
        except Exception as e:
            self.logger.error(f"Failed to get jobs: {e}")
            raise e

    async def _get_job_ids(self) -> List[str]:
        """
        Get a list of all running BioEngine worker job IDs.

        Returns:
            List of job IDs for all running BioEngine worker jobs

        Raises:
            Exception: If job retrieval fails for any reason
        """
        jobs = await self._get_jobs()
        return list(jobs.keys())

    async def _cancel_jobs(self, job_ids: Optional[List[str]] = None) -> List[str]:
        """
        Cancel running SLURM jobs.

        Executes the scancel command to cancel SLURM jobs and waits for them
        to be properly terminated within the grace period.

        Args:
            job_ids: Specific job IDs to cancel. If None, cancels all BioEngine worker jobs

        Returns:
            List of job IDs that were cancelled

        Raises:
            ValueError: If no matching jobs found to cancel when job_ids are specified
            subprocess.CalledProcessError: If scancel command fails
            Exception: If job cancellation fails for any other reason
        """
        # First get all ray worker jobs
        jobs_to_cancel = await self._get_job_ids()

        if not jobs_to_cancel:
            self.logger.info("No running BioEngine worker jobs found")
            return []

        try:
            # Filter jobs if specific job_ids were provided
            if job_ids:
                jobs_to_cancel = [
                    str(job_id) for job_id in job_ids if str(job_id) in jobs_to_cancel
                ]
                if not jobs_to_cancel:
                    self.logger.warning(f"No matching jobs found to cancel: {job_ids}")
                    raise ValueError(
                        f"No matching jobs found to cancel. Provided job IDs: {job_ids} - Running jobs: {jobs_to_cancel}"
                    )

            # Cancel the jobs
            self.logger.info(
                f"Cancelling {len(jobs_to_cancel)} job(s): {jobs_to_cancel}"
            )
            proc = await asyncio.create_subprocess_exec(
                "scancel",
                *jobs_to_cancel,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(
                    proc.returncode, "scancel", stderr=error_msg
                )

            # Check if jobs were successfully cancelled
            start_time = time.time()
            while time.time() - start_time < self.grace_period:
                await asyncio.sleep(1)
                running_jobs = await self._get_job_ids()
                if not any(job_id in running_jobs for job_id in jobs_to_cancel):
                    self.logger.info(
                        f"Successfully cancelled {len(jobs_to_cancel)} job(s)"
                    )
                    return jobs_to_cancel

            cancelled_jobs = [
                job_id for job_id in jobs_to_cancel if job_id not in running_jobs
            ]
            self.logger.warning(
                f"Failed to cancel all jobs within {self.grace_period} seconds. Cancelled jobs: {cancelled_jobs}"
            )
            return cancelled_jobs
        except Exception as e:
            self.logger.error(f"Failed to cancel jobs: {e}")
            raise e

    async def _add_worker(
        self,
        num_gpus: Optional[int] = None,
        num_cpus: Optional[int] = None,
        mem_in_gb_per_cpu: Optional[int] = None,
        time_limit: Optional[str] = None,
        further_slurm_args: Optional[List[str]] = None,
    ) -> str:
        """
        Start a new SLURM worker as a Ray worker node.

        Creates a temporary batch script to launch a containerized Ray worker
        with the specified resources. Submits the job to SLURM and waits for
        the worker to join the Ray cluster.

        Args:
            num_gpus: Number of GPUs to allocate per worker. Uses default if None
            num_cpus: Number of CPUs to allocate per worker. Uses default if None
            mem_in_gb_per_cpu: Memory (GB) to allocate per CPU. Uses default if None
            time_limit: SLURM job time limit (HH:MM:SS format). Uses default if None
            further_slurm_args: Additional SLURM arguments. Uses default if None

        Returns:
            The Ray node ID of the started worker

        Raises:
            RuntimeError: If Ray is not initialized or worker fails to start
            Exception: If worker creation fails for any other reason
        """
        try:
            submitted_job_id = None

            # Set default values if not provided. Ray reports resources as
            # floats (e.g. 1.0); coerce here so log lines, the sbatch GPU
            # directive, and downstream consumers all see plain ints.
            num_gpus = int(num_gpus if num_gpus is not None else self.default_num_gpus)
            num_cpus = int(num_cpus if num_cpus is not None else self.default_num_cpus)
            mem_in_gb_per_cpu = int(
                mem_in_gb_per_cpu
                if mem_in_gb_per_cpu is not None
                else self.default_mem_in_gb_per_cpu
            )
            time_limit = (
                time_limit if time_limit is not None else self.default_time_limit
            )
            further_slurm_args = (
                further_slurm_args
                if further_slurm_args is not None
                else self.further_slurm_args
            )

            # Create sbatch script using SlurmManager
            sbatch_script = await asyncio.to_thread(
                self._create_sbatch_script,
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                mem_in_gb_per_cpu=mem_in_gb_per_cpu,
                time_limit=time_limit,
                further_slurm_args=further_slurm_args,
            )

            # Submit the job
            submitted_job_id = await self._submit_job(sbatch_script, delete_script=True)
            self.logger.info(
                f"Worker job submitted successfully. Worker & Job ID: {submitted_job_id}, Resources: {num_gpus} GPU(s), "
                f"{num_cpus} CPU(s), {mem_in_gb_per_cpu}G mem/CPU, {time_limit} time limit"
            )

            # Wait for the worker to be added to the Ray cluster
            node_is_pending = True
            while node_is_pending:
                await asyncio.sleep(5)

                # Check if the job is still running
                job_ids = await self._get_job_ids()
                if submitted_job_id not in job_ids:
                    raise RuntimeError(
                        f"Job {submitted_job_id} is no longer running. Worker may not have started successfully."
                    )

                # Check if the worker node has already been added to the Ray cluster
                cluster_state = (
                    await self.ray_cluster.proxy_actor_handle.get_cluster_state.remote()
                )
                node_id = None
                for candidate_id, node_resources in cluster_state["nodes"].items():
                    if node_resources["slurm_job_id"] == submitted_job_id:
                        node_id = candidate_id
                        self.logger.info(
                            f"Worker node with ID '{node_id}' is now running in the Ray cluster"
                        )
                        node_is_pending = False
                        break

            return node_id

        except asyncio.CancelledError:
            self.logger.info("Worker creation task was cancelled")
            if submitted_job_id:
                await self._cancel_jobs([submitted_job_id])
        except Exception as e:
            self.logger.error(f"Error starting worker: {e}")
            if submitted_job_id:
                await self._cancel_jobs([submitted_job_id])
        finally:
            self.worker_creation_task = None

    async def _close_worker(self, node_id: str, job_id: str) -> None:
        """
        Shut down an existing SLURM worker node from the Ray cluster.

        Gracefully stops the worker by sending a stop command to the node,
        then cancels the corresponding SLURM job.

        Args:
            node_id: The Ray node ID of the worker to stop

        Raises:
            RuntimeError: If Ray is not initialized, worker not found, or shutdown fails
            Exception: If worker shutdown fails for any other reason
        """
        try:
            self.logger.info(
                f"Removing worker '{node_id}' (job_id='{job_id}') from cluster status..."
            )

            self.logger.info(f"Stopping all processes on worker '{node_id}'...")

            @ray.remote(resources={f"slurm_job_id:{job_id}": 0.1})
            def stop_worker():
                """Stops the worker node by executing 'ray stop'."""
                import subprocess

                subprocess.Popen(
                    ["ray", "stop"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # This should only effect a single worker node if the worker runs in a container
            obj_ref = stop_worker.remote()
            try:
                await asyncio.wait_for(obj_ref, timeout=self.grace_period)
            except TimeoutError:
                self.logger.error(
                    f"Failed to send shutdown command to worker '{node_id}' within {self.grace_period} seconds. Cancelling job..."
                )
                raise

            # Wait for worker node to be removed from the cluster
            start_time = time.time()
            while time.time() - start_time < self.grace_period:
                await asyncio.sleep(3)
                worker_nodes = self.ray_cluster.status["nodes"]
                if node_id not in worker_nodes:
                    break

            cancelled_jobs = await self._cancel_jobs([job_id])
            if cancelled_jobs == [job_id]:
                self.logger.info(f"Successfully removed worker '{node_id}'")
            else:
                raise RuntimeError(
                    f"Failed to cancel job '{job_id}' from worker '{node_id}'"
                )

        except Exception as e:
            self.logger.error(f"Error shutting down worker {node_id}: {e}")
            raise e
        finally:
            self.worker_deletion_tasks.pop(node_id, None)

    async def _check_scale_up(self) -> None:
        """
        Attempt to scale up the cluster by adding a new worker.

        Checks if scaling up is allowed based on cooldown periods and worker limits.
        If conditions are met, submits a new SLURM job with default resource allocation.

        Args:
            n_worker_jobs: Current number of worker jobs in SLURM

        Returns:
            None

        Raises:
            Exception: If worker creation fails
        """
        # SCALE UP LOGIC
        self.logger.debug("Checking scale up conditions...")
        current_time = time.time()
        n_worker_jobs = await self.get_num_worker_jobs()

        can_scale_up = (
            # No worker creation task in progress
            self.worker_creation_task is None
            # Cooldown period has passed
            and (current_time - self.last_scale_up_time) > self.scale_up_cooldown
            # Not at max worker limit
            and n_worker_jobs < self.max_workers
        )
        if not can_scale_up:
            return

        # Get the required resources (some pending items — e.g. jobs — do not
        # carry a `required_resources` field, so probe each candidate).
        required_resources = None
        for resource_type in ("actors", "tasks", "jobs"):
            pending_items = self.ray_cluster.status["cluster"][
                "pending_resources"
            ].get(resource_type, [])
            for item in reversed(pending_items):
                candidate = item.get("required_resources") if item else None
                if candidate:
                    required_resources = candidate
                    break
            if required_resources:
                break
        if not required_resources:
            self.logger.info(
                "No pending resources with explicit requirements found; "
                "scaling up with defaults."
            )
            required_resources = {}

        num_cpus = required_resources.get("CPU", 1)
        num_gpus = required_resources.get("GPU", 0)
        memory = required_resources.get("memory", 0) / 1024**3  # Convert bytes to GB
        mem_in_gb_per_cpu = math.ceil(
            memory / num_cpus
        )  # Calculate memory in GB per CPU

        if num_cpus < self.default_num_cpus:
            num_cpus = self.default_num_cpus
        if num_gpus < self.default_num_gpus:
            num_gpus = self.default_num_gpus
        if mem_in_gb_per_cpu < self.default_mem_in_gb_per_cpu:
            mem_in_gb_per_cpu = self.default_mem_in_gb_per_cpu

        # TODO: Check how to pass time limit and further SLURM args
        time_limit = self.default_time_limit
        further_slurm_args = self.further_slurm_args

        # Update last scale up time
        self.last_scale_up_time = current_time
        self.logger.info(
            f"Scaling up with {num_gpus} GPU(s) and {num_cpus} CPU(s) due to pending resources"
        )

        self.worker_creation_task = asyncio.create_task(
            self._add_worker(
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                mem_in_gb_per_cpu=mem_in_gb_per_cpu,
                time_limit=time_limit,
                further_slurm_args=further_slurm_args,
            ),
            name="WorkerScaleUpTask",
        )

    async def _check_is_idle(self, node_resources: Dict) -> bool:
        """
        Check if a worker node is idle based on its GPU and CPU utilization.

        A node is considered idle if it has no active CPU or GPU usage.

        Args:
            node_resources: Dictionary containing node resource information with
                keys 'total_cpu', 'used_cpu', 'total_gpu', 'used_gpu'
                (as produced by ``BioEngineProxyActor.get_cluster_state``).

        Returns:
            True if the node is completely idle (no used CPUs or GPUs), False otherwise
        """
        used_cpus = node_resources["used_cpu"]
        used_gpus = node_resources["used_gpu"] if node_resources["total_gpu"] > 0 else 0

        # Node is idle if no GPUs or CPUs are used
        return used_gpus == 0 and used_cpus == 0

    async def _find_idle_nodes(self, worker_nodes: Dict[str, dict]) -> Set[str]:
        """
        Find idle worker nodes based on GPU and CPU utilization metrics.

        Iterates through all running nodes and identifies those that are completely idle
        (no GPU or CPU usage).

        Args:
            running_nodes: List of node information dictionaries from Ray cluster

        Returns:
            Set of node_ids that are currently idle
        """
        idle_nodes = set()
        for node_id, node_resources in worker_nodes.items():
            is_idle = await self._check_is_idle(node_resources)
            if is_idle:
                idle_nodes.add(node_id)

        return idle_nodes

    async def _check_scale_down(self) -> None:
        """
        Attempt to scale down the cluster by removing idle workers.

        Checks if scaling down is allowed based on cooldown periods and worker limits.
        If conditions are met, identifies the longest idle node and removes it.

        Raises:
            Exception: If worker removal fails
        """
        # SCALE DOWN LOGIC
        current_time = time.time()
        n_worker_jobs = await self.get_num_worker_jobs()

        # Check if we can scale down based on cooldown and worker limits
        can_scale_down = (
            # Ray cluster has status history available
            self.ray_cluster.cluster_status_history is not None
            # Idle node check interval has passed
            and (current_time - self.last_scale_down_check)
            > self.scale_down_check_interval
            # Not at min worker limit
            and n_worker_jobs > self.min_workers
        )
        if not can_scale_down:
            return

        self.logger.debug("Checking for idle workers to scale down...")
        self.last_scale_down_check = current_time

        # Find nodes that have been idle for at least the threshold duration
        nodes_to_remove = None
        max_idle_time = 0

        for i, (timepoint, cluster_status) in enumerate(
            reversed(self.ray_cluster.cluster_status_history.items())
        ):
            idle_nodes = await self._find_idle_nodes(cluster_status["nodes"])
            idle_time = current_time - timepoint

            # If no idle nodes found at this timepoint, stop checking further back
            if not idle_nodes:
                break

            # If this is the first iteration, initialize nodes_to_remove
            if i == 0:
                nodes_to_remove = idle_nodes
                max_idle_time = idle_time
            else:
                # Find intersection of nodes to remove and current idle nodes
                overlap = nodes_to_remove & idle_nodes
                if len(overlap) == 0:
                    # No overlap means no persistently idle nodes, stop checking
                    break

                nodes_to_remove = overlap
                max_idle_time = idle_time

                # If we've found nodes idle for longer than threshold, we can stop
                if idle_time > self.scale_down_threshold:
                    break

        # Remove all nodes that have been idle for at least the threshold
        if nodes_to_remove and max_idle_time > self.scale_down_threshold:
            workers_removed = False
            for idle_node_id in list(nodes_to_remove):
                # Double-check current worker status to ensure it's still idle
                worker_resources = self.ray_cluster.status["nodes"].get(idle_node_id)
                if worker_resources and await self._check_is_idle(worker_resources):
                    self.logger.info(
                        f"Scaling down worker '{idle_node_id}' due to inactivity for at least {max_idle_time:.1f}s"
                    )
                    job_id = worker_resources["slurm_job_id"]
                    self.worker_deletion_tasks[idle_node_id] = asyncio.create_task(
                        self._close_worker(node_id=idle_node_id, job_id=job_id),
                        name=f"WorkerScaleDownTask_{idle_node_id}",
                    )
                    workers_removed = True

            if not workers_removed:
                self.logger.debug(
                    "No workers met the criteria for removal after final validation"
                )
        else:
            self.logger.debug(
                f"No idle workers found to scale down. Max idle time: {max_idle_time:.1f}s (threshold: {self.scale_down_threshold}s)"
            )

    async def get_num_worker_jobs(self) -> int:
        """
        Get number of SLURM workers in the cluster (configuring, pending or running).

        Queries SLURM for all BioEngine worker jobs regardless of their current state.
        This includes jobs that are pending in the queue, configuring, or actively running.

        Returns:
            Total count of BioEngine worker jobs in all states

        Raises:
            Exception: If job query fails due to SLURM connection issues
        """
        return len(await self._get_jobs()) - len(self.worker_deletion_tasks)

    @staticmethod
    def _parse_slurm_duration(value: str) -> Optional[int]:
        """Parse a SLURM duration string into seconds.

        SLURM emits durations as ``[D-]HH:MM:SS`` or ``MM:SS`` (the leading
        ``D-`` is days). ``UNLIMITED`` / ``INVALID`` / ``NOT_SET`` collapse to
        ``None`` so callers can skip "remaining time" calculations gracefully.
        """
        if not value or value in {"UNLIMITED", "INVALID", "NOT_SET", "N/A"}:
            return None
        days = 0
        rest = value
        if "-" in value:
            d, _, rest = value.partition("-")
            try:
                days = int(d)
            except ValueError:
                return None
        parts = rest.split(":")
        try:
            parts_int = [int(p) for p in parts]
        except ValueError:
            return None
        if len(parts_int) == 3:
            h, m, s = parts_int
        elif len(parts_int) == 2:
            h, m, s = 0, parts_int[0], parts_int[1]
        else:
            return None
        return days * 86400 + h * 3600 + m * 60 + s

    @staticmethod
    def _parse_slurm_timestamp(value: str) -> Optional[float]:
        """Parse a SLURM ISO-ish timestamp (``YYYY-MM-DDTHH:MM:SS``) → epoch seconds."""
        if not value or value in {"N/A", "Unknown", "None"}:
            return None
        try:
            return time.mktime(time.strptime(value, "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, OverflowError):
            return None

    async def get_job_status(self) -> Dict[str, List[Dict[str, Any]]]:
        """Snapshot of BioEngine worker SLURM jobs split into queued vs running.

        Each entry carries the timing information needed by the dashboard to
        render "waited for / remaining in" badges:

        - **queued** (pending in the SLURM queue, state ``PD``/``CF``):
          ``job_id``, ``state``, ``submit_time`` (epoch s, may be ``None``),
          ``pending_seconds`` (int, may be ``None``), ``reason`` (str).
        - **running** (state ``R``): ``job_id``, ``state``, ``node``,
          ``start_time`` (epoch s, may be ``None``), ``time_limit_seconds``
          (int, may be ``None``), ``elapsed_seconds`` (int),
          ``remaining_seconds`` (int, may be ``None`` if no time limit).

        All durations are accurate to the second, but the UI should round to
        the minute when displaying — per-second updates would be churny.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "squeue",
                "-u", os.environ["USER"],
                "-n", self.job_name,
                "-h",
                "-o", "%i|%T|%V|%S|%l|%M|%R|%N",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                err = (stderr.decode() if stderr else "").strip()
                self.logger.warning(f"squeue failed (exit {proc.returncode}): {err}")
                return {"queued": [], "running": []}
        except FileNotFoundError:
            # squeue not available (e.g. running unit tests without SLURM)
            return {"queued": [], "running": []}

        now = time.time()
        queued: List[Dict[str, Any]] = []
        running: List[Dict[str, Any]] = []
        for line in stdout.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 8:
                continue
            job_id, state, submit_s, start_s, time_limit_s, elapsed_s, reason, node = parts
            submit_t = self._parse_slurm_timestamp(submit_s)
            start_t = self._parse_slurm_timestamp(start_s)
            time_limit = self._parse_slurm_duration(time_limit_s)
            elapsed = self._parse_slurm_duration(elapsed_s) or 0

            if state == "RUNNING":
                remaining: Optional[int]
                if time_limit is not None:
                    remaining = max(0, time_limit - elapsed)
                else:
                    remaining = None
                running.append({
                    "job_id": job_id,
                    "state": state,
                    "node": node or None,
                    "start_time": start_t,
                    "time_limit_seconds": time_limit,
                    "elapsed_seconds": elapsed,
                    "remaining_seconds": remaining,
                })
            else:  # PENDING, CONFIGURING, COMPLETING, etc — treat as queued
                pending_secs: Optional[int]
                if submit_t is not None:
                    pending_secs = max(0, int(now - submit_t))
                else:
                    pending_secs = None
                queued.append({
                    "job_id": job_id,
                    "state": state,
                    "submit_time": submit_t,
                    "pending_seconds": pending_secs,
                    "reason": reason,
                })
        return {"queued": queued, "running": running}

    async def check_scaling(self) -> None:
        """
        Make scaling decisions based on resource utilization and pending tasks.

        Evaluates the current cluster state including pending tasks and worker jobs,
        then decides whether to scale up (if there are pending tasks) or scale down
        (if workers are idle). Updates the last scaling decision timestamp.

        Raises:
            Exception: If Ray is not initialized or scaling decision fails
        """
        # Logic: If at least one task needs resources, check scale up, otherwise check scale down
        if self.ray_cluster.status["cluster"]["pending_resources"]["total"] > 0:
            await self._check_scale_up()
        else:
            await self._check_scale_down()

    async def close_all(self):
        """
        Shut down all SLURM worker nodes from the Ray cluster.

        Gracefully stops each worker by sending a stop command to the node,
        then cancels the corresponding SLURM job.

        Raises:
            RuntimeError: If Ray is not initialized or shutdown fails
            Exception: If worker shutdown fails for any other reason
        """
        try:
            # Stop current worker creation task if it exists
            if self.worker_creation_task and not self.worker_creation_task.done():
                self.worker_creation_task.cancel()

            # Get all worker nodes
            worker_nodes = self.ray_cluster.status["nodes"]

            if not worker_nodes:
                self.logger.info("No active worker nodes found")
                return

            self.logger.info(f"Found {len(worker_nodes)} active worker nodes")

            # Stop each worker node
            for node_id, node_resources in worker_nodes.items():
                job_id = node_resources["slurm_job_id"]
                self.worker_deletion_tasks[node_id] = asyncio.create_task(
                    self._close_worker(node_id=node_id, job_id=job_id),
                    name=f"WorkerScaleDownTask_{node_id}",
                )

            # Make sure no jobs are left running
            remaining_jobs = await self._get_job_ids()
            if remaining_jobs:
                self.logger.warning(
                    f"Some jobs are still running after shutdown: {remaining_jobs}. Cancelling them..."
                )
                cancelled_jobs = await self._cancel_jobs(remaining_jobs)
                if cancelled_jobs:
                    self.logger.warning(
                        f"Cancelled jobs after shutdown: {cancelled_jobs}"
                    )
                else:
                    self.logger.info("All jobs cancelled successfully")

        except Exception as e:
            self.logger.error(f"Error shutting down all workers: {e}")
            raise e
