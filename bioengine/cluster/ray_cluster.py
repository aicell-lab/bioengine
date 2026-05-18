import asyncio
import logging
import os
import re
import subprocess
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import ray
from ray import serve

import bioengine
from bioengine.cluster.proxy_actor import BioEngineProxyActor
from bioengine.cluster.slurm_workers import SlurmWorkers
from bioengine.utils import (
    acquire_free_port,
    create_logger,
    date_format,
    get_internal_ip,
    stream_logging_format,
)


def _is_stale_handle_error(exc: BaseException) -> bool:
    """Detect a Ray Client stale-handle error.

    When a detached actor we hold a handle for (e.g. the Ray Serve controller)
    is restarted on the cluster side, the server no longer knows about the
    handle registered for our gRPC session. The next remote call surfaces a
    plain ``Exception`` whose message contains "doesn't have a handle for".
    Ray Client does not auto-recover this; we have to reconnect to register
    fresh handles. Match on substring because the exception type is generic.
    """
    return "doesn't have a handle for" in str(exc)


class RayCluster:
    """
    Manages Ray cluster lifecycle across different deployment environments.

    This class provides a unified interface for managing Ray clusters in various environments:
    - SLURM-managed HPC systems with automatic worker scaling
    - Single-machine deployments for local development
    - Connection to existing Ray clusters

    The class handles the complete lifecycle from cluster initialization through
    worker management to graceful shutdown, with robust error handling and
    logging throughout. It includes intelligent autoscaling for SLURM environments,
    automatic port allocation to avoid conflicts, and comprehensive monitoring.

    Key Features:
    - Multi-environment support (SLURM, single-machine, external-cluster)
    - Dynamic port allocation with conflict resolution
    - Container-based worker deployment via SLURM with Apptainer/Singularity
    - Automatic scaling based on resource utilization and task demand
    - Ray Serve integration for model serving capabilities
    - Comprehensive cluster monitoring with historical status tracking
    - Graceful shutdown with proper resource cleanup
    - Robust error handling with automatic reconnection

    Attributes:
        mode (str): Deployment mode ('slurm', 'single-machine', 'external-cluster')
        ray_cluster_config (dict): Configuration for Ray head node
        slurm_worker_config (dict): Configuration for SLURM workers (if applicable)
        is_ready (asyncio.Event): Whether the cluster is currently active
        start_time (float): Timestamp when cluster was started
        max_status_history_length (int): Maximum entries in status history
        cluster_status_history (OrderedDict): Historical status of cluster
        slurm_workers (SlurmWorkers): SLURM worker manager instance
        logger: Logger instance for cluster operations
    """

    def __init__(
        self,
        mode: Literal["single-machine", "slurm", "external-cluster"],
        # Ray Head Node Configuration parameters
        head_node_address: Optional[str] = None,
        head_node_port: int = 6379,
        node_manager_port: int = 6700,
        object_manager_port: int = 6701,
        redis_shard_port: int = 6702,
        serve_port: int = 8000,
        dashboard_port: int = 8265,
        client_server_port: int = 10001,
        redis_password: Optional[str] = None,
        ray_temp_dir: Union[str, Path] = f"{os.environ['HOME']}/.bioengine/ray",
        head_num_cpus: int = 0,
        head_num_gpus: int = 0,
        head_memory_in_gb: Optional[int] = None,
        runtime_env_pip_cache_size_gb: int = 30,  # Ray default is 10 GB
        force_clean_up: bool = True,
        # SLURM Worker Configuration parameters
        image: str = f"ghcr.io/aicell-lab/bioengine-worker:{bioengine.__version__}",
        worker_workspace_dir: Optional[str] = None,
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
        scale_up_cooldown_seconds: int = 60,
        scale_down_check_interval_seconds: int = 60,
        scale_down_threshold_seconds: int = 300,
        # Logger configuration
        log_file: Optional[Union[str, Path]] = None,
        debug: bool = False,
    ):
        """Initialize cluster manager with networking and resource configurations.

        Ray ports configuration: https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations
        SLURM networking caveats: https://github.com/ray-project/ray/blob/1000ae9671967994f7bfdf7b1e1399223ad4fc61/doc/source/cluster/vms/user-guides/community/slurm.rst#id22

        Args:
            mode: Mode of operation ('single-machine', 'slurm', or 'external-cluster').
            head_node_address: IP address for head node. Uses first system IP if None.
            head_node_port: Port for Ray head node and GCS server. Default 6379.
            node_manager_port: Base port for Ray node manager services. Default 6700.
            object_manager_port: Port for object manager service. Default 6701.
            redis_shard_port: Port for Redis sharding. Default 6702.
            serve_port: Port for Ray Serve HTTP server. Default 8000.
            dashboard_port: Port for Ray dashboard. Default 8265.
            client_server_port: Base port for Ray client services. Default 10001.
            redis_password: Password for Redis server. Generated randomly if None.
            ray_temp_dir: Temporary directory for Ray. Default '/home/<user>/.bioengine/ray'.
            head_num_cpus: Number of CPUs for head node (single-machine mode). Default 0.
            head_num_gpus: Number of GPUs for head node (single-machine mode). Default 0.
            head_memory_in_gb: Memory limit for head node in GB. If not set, Ray will auto-detect available memory.
            runtime_env_pip_cache_size_gb: Size of pip cache for runtime environments in GB. Default 30.
            force_clean_up: Force cleanup of previous Ray cluster on start. Default True.
            image: Container image for workers (SLURM mode). Default bioengine-worker.
            worker_workspace_dir: Workspace directory mounted to worker containers (SLURM mode).
            default_num_gpus: Default GPU count per worker. Default 1.
            default_num_cpus: Default CPU count per worker. Default 8.
            default_mem_in_gb_per_cpu: Default memory per CPU in GB. Default 16.
            default_time_limit: Default SLURM job time limit. Default '4:00:00'.
            further_slurm_args: Additional SLURM arguments for job submission.
            gpu_slurm_flag: Template for the GPU sbatch directive (default
                "--gpus={n}"). Use "--gres=gpu:{n}" on clusters that require gres,
                or "" to omit and supply via ``further_slurm_args``.
            further_apptainer_args: Extra CLI flags forwarded to ``apptainer exec``
                inside each SLURM worker job (e.g. additional ``--bind`` mounts).
            min_workers: Minimum number of workers for autoscaling. Default 0.
            max_workers: Maximum number of workers for autoscaling. Default 4.
            scale_up_cooldown_seconds: Cooldown between scale-up operations. Default 60.
            scale_down_check_interval_seconds: Interval between scale-down checks. Default 60.
            scale_down_threshold_seconds: Idle time before scaling down. Default 300.
            log_file: File path for logging output. Uses console if None.
            debug: Enable debug-level logging. Default False.

        Raises:
            ValueError: If mode is invalid or configuration is inconsistent.
            RuntimeError: If SLURM is required but not available.
            FileNotFoundError: If Ray executable is not found.
        """
        # Set up logging
        self.logger = create_logger(
            name="RayCluster",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        # Check mode
        if mode not in ["single-machine", "slurm", "external-cluster"]:
            raise ValueError(
                f"Unsupported Ray cluster mode: '{mode}'. "
                "Supported modes are 'slurm', 'single-machine' and 'external-cluster'."
            )
        self.mode = mode

        # Initialize cluster state and monitoring attributes
        self.address = None
        self.serve_http_url = None
        self.proxy_actor_handle = None
        self.proxy_actor_name = "BIOENGINE_PROXY_ACTOR"
        self.cluster_status_history = OrderedDict()
        self.max_status_history_length = 100
        self.is_ready = asyncio.Event()

        self.start_time = None
        self.lock_file = None
        self._owns_lock = False
        self.slurm_workers = None

        # Base configuration for connecting to Ray cluster
        self.ray_cluster_config = {
            "head_node_address": str(head_node_address or get_internal_ip()),
            "ports": {
                "head_node": int(head_node_port),  # GCS server port
                "client_server": int(client_server_port),
                "serve": int(serve_port),
            },
            "slurm": {},
        }

        # Additional configuration for starting a Ray cluster
        if mode in ["single-machine", "slurm"]:
            # Find and store Ray executable path
            self.ray_exec_path = self._find_ray_executable()
            self.serve_exec_path = self._find_serve_executable()

            # Check if Ray temp dir does not exceed length of 107 bytes
            ray_temp_dir = Path(ray_temp_dir).resolve()
            self._check_ray_temp_dir_length(ray_temp_dir)

            # Check number of CPUs if in single-machine mode
            if self.mode == "single-machine" and head_num_cpus <= 0:
                raise ValueError(
                    "When running on a single machine, 'head_num_cpus' must be greater than 0"
                )

            # Set cluster configuration
            self.ray_cluster_config.update(
                {
                    "ray_temp_dir": ray_temp_dir,
                    "head_num_cpus": int(head_num_cpus),
                    "head_num_gpus": int(head_num_gpus),
                    "head_memory_in_gb": (
                        int(head_memory_in_gb)
                        if head_memory_in_gb is not None
                        else None
                    ),
                    "redis_password": str(redis_password or os.urandom(16).hex()),
                    "force_clean_up": bool(force_clean_up),
                }
            )

            # Set cluster ports
            self.ray_cluster_config["ports"].update(
                {
                    "node_manager": int(node_manager_port),
                    "object_manager": int(object_manager_port),
                    "redis_shard": int(redis_shard_port),
                    "dashboard": int(dashboard_port),
                }
            )

            # Set runtime environment pip cache size
            if runtime_env_pip_cache_size_gb <= 0:
                raise ValueError("runtime_env_pip_cache_size_gb must be greater than 0")
            os.environ["RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB"] = str(
                runtime_env_pip_cache_size_gb
            )
            self.logger.debug(
                f"Setting RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB to {runtime_env_pip_cache_size_gb}"
            )

            # Set SLURM worker configuration if in slurm mode
            if self.mode == "slurm":
                self._check_slurm_available()

                if worker_workspace_dir is None:
                    raise ValueError(
                        "worker_workspace_dir must be provided when mode is 'slurm'"
                    )
                self.ray_cluster_config["slurm"].update(
                    {
                        "image": image,
                        "worker_workspace_dir": str(worker_workspace_dir),
                        "default_num_gpus": int(default_num_gpus),
                        "default_num_cpus": int(default_num_cpus),
                        "default_mem_in_gb_per_cpu": int(default_mem_in_gb_per_cpu),
                        "default_time_limit": str(default_time_limit),
                        "further_slurm_args": further_slurm_args or [],
                        "gpu_slurm_flag": str(gpu_slurm_flag),
                        "further_apptainer_args": further_apptainer_args or [],
                        "min_workers": int(min_workers),
                        "max_workers": int(max_workers),
                        "scale_up_cooldown_seconds": int(scale_up_cooldown_seconds),
                        "scale_down_check_interval_seconds": int(
                            scale_down_check_interval_seconds
                        ),
                        "scale_down_threshold_seconds": int(
                            scale_down_threshold_seconds
                        ),
                        "log_file": log_file,
                        "debug": bool(debug),
                    }
                )

    def __del__(self):
        """Cleanup lock file when the RayCluster instance is destroyed."""
        if (
            getattr(self, "_owns_lock", False)
            and self.lock_file
            and self.lock_file.exists()
        ):
            try:
                self.lock_file.unlink()
            except (OSError, IOError):
                pass  # Ignore errors during cleanup
        self.lock_file = None

    @property
    def status(self) -> Dict[str, Union[str, dict]]:
        """Get current cluster status.

        Returns a dictionary containing the head node address, start time,
        uptime, and the most recent worker nodes status from the monitoring history.

        Returns:
            Dict containing:
                - head_address: Head node IP address
                - start_time_s: Start time as Unix timestamp
                - start_time: Formatted start time string
                - uptime: Human-readable uptime string
                - nodes: Most recent worker nodes status grouped by state
        """
        status = {
            "head_address": self.address,
            "start_time": self.start_time if self.mode != "external-cluster" else "N/A",
            "mode": self.mode,
        }

        last_status = (
            next(reversed(self.cluster_status_history.values()))
            if self.cluster_status_history
            else None
        )
        status["cluster"] = last_status["cluster"] if last_status else {}
        status["nodes"] = last_status["nodes"] if last_status else {}

        return status

    def _find_ray_executable(self) -> str:
        """
        Find the Ray executable path in the current Python environment.

        Returns:
            str: Path to the Ray executable

        Raises:
            FileNotFoundError: If Ray executable can not be found with 'which' command
        """
        try:
            ray_path = subprocess.run(
                ["which", "ray"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except FileNotFoundError:
            raise FileNotFoundError("Ray executable not found")

        self.logger.debug(f"Ray executable found at: {ray_path}")
        return str(ray_path)

    def _find_serve_executable(self) -> str:
        """
        Find the Ray Serve executable path in the current Python environment.

        Returns:
            str: Path to the Ray Serve executable

        Raises:
            FileNotFoundError: If Ray Serve executable can not be found with 'which' command
        """
        try:
            serve_path = subprocess.run(
                ["which", "serve"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except FileNotFoundError:
            raise FileNotFoundError("Ray Serve executable not found")

        self.logger.debug(f"Ray Serve executable found at: {serve_path}")
        return str(serve_path)

    def _check_slurm_available(self) -> None:
        """
        Check if SLURM is available on the system.

        Verifies that the SLURM workload manager is installed and accessible
        by attempting to run the 'sinfo' command.

        Raises:
            RuntimeError: If SLURM is not available or 'sinfo' command fails
        """
        try:
            subprocess.run(["sinfo"], capture_output=True, text=True, check=True)
        except FileNotFoundError:
            self.logger.error(
                "SLURM is not available. Please ensure you are running this on a SLURM-managed HPC system."
            )
            raise RuntimeError("SLURM is not available")

    def _check_ray_temp_dir_length(self, ray_temp_dir: Path) -> None:
        # Simulate the longest possible session directory name
        session_name = datetime.now().strftime(
            "session_%Y-%m-%d_%H-%M-%S_999999_9999999"
        )
        full_path = str(ray_temp_dir / session_name / "sockets" / "plasma_store")
        path_length = len(full_path.encode("utf-8"))

        if path_length > 107:
            self.logger.debug(f"Simulated path: {full_path}")
            self.logger.debug(f"Path length in bytes: {path_length}")
            raise ValueError(
                "Plasma store path length would exceed 107 bytes (current length: "
                f"{path_length}) with the specified Ray temp directory configuration "
                f"'{ray_temp_dir}'. Please choose a shorter path for the Ray temp directory."
            )

    def _acquire_lock(self, ray_temp_dir: Path) -> None:
        """
        Acquire an exclusive lock for the Ray temp directory using filesystem-based
        atomic operations.

        Uses atomic file creation to prevent multiple BioEngine workers from using
        the same Ray temp directory.

        Args:
            ray_temp_dir: Path to the Ray temporary directory

        Raises:
            RuntimeError: If another BioEngine worker is already using this directory
        """
        self.lock_file = ray_temp_dir / "bioengine.lock"
        process_info = (
            f"pid:{os.getpid()},uid:{uuid.uuid4().hex[:8]},started:{time.time()}"
        )

        if self.lock_file.exists():
            # Lock file exists, check if it's from a stale process
            self.logger.info(f"Lock file already exists: {self.lock_file}")

            # Attempt to read and validate the existing lock file
            try:
                with self.lock_file.open("r") as f:
                    existing_info = f.read().strip()

                # Try to extract PID from lock file content
                if existing_info.startswith("pid:"):
                    try:
                        pid_str = existing_info.split(",")[0].split(":")[1]
                        existing_pid = int(pid_str)

                        # Check if the process is still running
                        try:
                            # In container restarts, PID values (especially PID 1) can be reused.
                            # If the lock PID equals our current PID, treat the lock as stale.
                            if existing_pid == os.getpid():
                                self.logger.info(
                                    "Lock file PID matches current process PID; treating lock as stale."
                                )
                                raise ProcessLookupError()

                            os.kill(
                                existing_pid, 0
                            )  # Signal 0 checks if process exists
                            # Process is still running - cannot acquire lock
                            raise RuntimeError(
                                f"Ray session is already active in '{ray_temp_dir}' "
                                f"(PID: {existing_pid}). Please stop the existing Ray cluster "
                                "before starting a new one."
                            )
                        except ProcessLookupError:
                            # Process no longer exists, remove stale lock and retry
                            self.logger.info(
                                f"Removing stale lock file from non-existent process {existing_pid}"
                            )

                    except (ValueError, IndexError):
                        # Cannot parse PID, treat as stale lock
                        self.logger.info(
                            f"Removing malformed lock file: {existing_info}"
                        )
                else:
                    # Not standard format, treat as potentially stale
                    self.logger.info(
                        f"Removing potentially stale lock file: {existing_info}"
                    )

            except (OSError, IOError) as e:
                # Error reading lock file, assume it's corrupted
                self.logger.info(f"Removing corrupted lock file due to error: {e}")

            # Remove stale lock file before creating new one
            self.lock_file.unlink(missing_ok=True)

        # Try to create lock file atomically (exclusive creation)
        with self.lock_file.open("x") as f:
            f.write(process_info)
            f.flush()
            os.fsync(f.fileno())

        self._owns_lock = True
        self.logger.info(f"Successfully acquired lock: {self.lock_file}")

    def _set_cluster_ports(self) -> None:
        """
        Update cluster configuration with available ports.

        Checks for port availability and updates the ray_cluster_config
        dictionary with the next available ports for all Ray services.
        Also sets worker port ranges based on the client server port.
        """

        search_steps = {
            "head_node": 1,
            "node_manager": 100,
            "object_manager": 100,
            "redis_shard": 100,
            "serve": 1,
            "dashboard": 1,
            "client_server": 10000,
        }

        # Find free ports for all services; keep sockets until all ports are found
        sockets = []
        for port_name, step in search_steps.items():
            desired_port = self.ray_cluster_config["ports"][port_name]
            # Dashboard always binds to localhost, so check against 127.0.0.1
            check_ip = (
                "127.0.0.1"
                if port_name == "dashboard"
                else self.ray_cluster_config["head_node_address"]
            )
            free_port, s = acquire_free_port(
                port=desired_port,
                step=step,
                ip=check_ip,
                keep_open=True,
            )
            sockets.append(s)

            if free_port != desired_port:
                self.logger.warning(
                    f"Port {desired_port} for '{port_name}' is not available. Using {free_port} instead."
                )
            self.ray_cluster_config["ports"][port_name] = free_port

        # After finding all ports, close the sockets
        for s in sockets:
            s.close()

        self.ray_cluster_config["ports"]["min_worker"] = (
            self.ray_cluster_config["ports"]["client_server"] + 1
        )
        self.ray_cluster_config["ports"]["max_worker"] = (
            self.ray_cluster_config["ports"]["client_server"] + 9998
        )

    def _update_symlink(self, ray_temp_dir: Path) -> None:
        """
        Update the 'session_latest' symlink to use a relative path.

        Raises an error if the symlink does not exist or is not a symlink.
        """
        symlink_path = ray_temp_dir / "session_latest"
        if symlink_path.is_symlink():
            # Get the target of the symlink
            symlink_target = symlink_path.readlink()
            relative_symlink_target = Path(symlink_target.name)
            self.logger.debug(
                f"Changing symlink target from '{symlink_target}' to '{relative_symlink_target}'"
            )
            symlink_path.unlink()
            symlink_path.symlink_to(relative_symlink_target)
        else:
            self.logger.error(f"Symlink '{symlink_path}' does not exist")
            raise FileNotFoundError(f"Symlink '{symlink_path}' does not exist")

    async def _start_cluster(self) -> None:
        """Start Ray cluster head node with configured ports and resources.

        Initializes the Ray cluster head node with all configured settings,
        starts Ray Serve for model serving, and optionally starts the
        autoscaling system for SLURM environments. Handles port allocation,
        directory setup, and symlink management for containerized environments.

        Raises:
            RuntimeError: If Ray is already initialized.
            subprocess.CalledProcessError: If Ray startup command fails.
            Exception: For other initialization errors.
        """
        if self.ray_cluster_config["force_clean_up"]:
            self.logger.info("Forcing Ray cleanup...")
            try:
                await self._shutdown_head_node()
            except Exception as e:
                self.logger.error(f"Error during Ray cleanup: {e}")
        try:
            self.logger.info("Starting Ray cluster...")

            # Make sure the temporary directory exists (triggers better error message than Ray)
            ray_temp_dir = self.ray_cluster_config["ray_temp_dir"]
            await asyncio.to_thread(ray_temp_dir.mkdir, parents=True, exist_ok=True)

            # Check if a Ray cluster is already running using this temporary directory
            self._acquire_lock(ray_temp_dir)

            # Check and set cluster ports
            await asyncio.to_thread(self._set_cluster_ports)

            # Start ray as the head node with the specified parameters
            args = [
                "start",
                "--head",
                f"--num-cpus={self.ray_cluster_config['head_num_cpus']}",
                f"--num-gpus={self.ray_cluster_config['head_num_gpus']}",
                f"--node-ip-address={self.ray_cluster_config['head_node_address']}",
                f"--port={self.ray_cluster_config['ports']['head_node']}",
                f"--node-manager-port={self.ray_cluster_config['ports']['node_manager']}",
                f"--object-manager-port={self.ray_cluster_config['ports']['object_manager']}",
                f"--redis-shard-ports={self.ray_cluster_config['ports']['redis_shard']}",
                f"--ray-client-server-port={self.ray_cluster_config['ports']['client_server']}",
                f"--min-worker-port={self.ray_cluster_config['ports']['min_worker']}",
                f"--max-worker-port={self.ray_cluster_config['ports']['max_worker']}",
                "--include-dashboard=True",
                f"--dashboard-port={self.ray_cluster_config['ports']['dashboard']}",
                f"--redis-password={self.ray_cluster_config['redis_password']}",
                f"--temp-dir={ray_temp_dir}",
            ]

            # Add memory limit if specified
            if self.ray_cluster_config["head_memory_in_gb"] is not None:
                memory_limit = self.ray_cluster_config["head_memory_in_gb"] * 1024**3
                args.append(f"--memory={memory_limit}")

            # Prevent logging of Redis password in debug logs
            censored_args = [
                arg if "redis-password" not in arg else "--redis-password=****"
                for arg in args
            ]
            self.logger.debug(
                f"Ray start command: {self.ray_exec_path} {' '.join(censored_args)}"
            )

            proc = await asyncio.create_subprocess_exec(
                self.ray_exec_path,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(
                    proc.returncode, f"ray start", stderr=error_msg
                )

            self.logger.debug(
                f"Ray start command output:\n----------\n{stdout.decode()}"
            )

            # Start Ray Serve
            ray_address = f"{self.ray_cluster_config['head_node_address']}:{self.ray_cluster_config['ports']['head_node']}"
            args = [
                "start",
                "--address",
                ray_address,
                "--http-host",
                "0.0.0.0",
                "--http-port",
                str(self.ray_cluster_config["ports"]["serve"]),
            ]
            proc = await asyncio.create_subprocess_exec(
                self.serve_exec_path,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(
                    proc.returncode, f"serve start", stderr=error_msg
                )

            # Change '<ray_temp_dir>/session_latest' symlink to use relative path instead of absolute (container) path
            # (needed when starting ray in container)
            await asyncio.to_thread(self._update_symlink, ray_temp_dir)

            # If running on a HPC system, use SlurmWorkers to manage worker nodes
            if self.mode == "slurm":
                # Initialize SlurmWorkers
                self.slurm_workers = SlurmWorkers(
                    ray_cluster=self, **self.ray_cluster_config["slurm"]
                )

            self.logger.info(f"Ray cluster with Ray Serve started on '{ray_address}'")

        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Ray start command failed with error code {e.returncode}:\n{e.stderr}"
            )
            raise e
        except Exception as e:
            self.logger.error(f"Error starting Ray cluster: {e}")
            raise e

    def _set_head_node_address(self) -> None:
        """Set the head node address based on the cluster configuration."""
        head_node_address = str(self.ray_cluster_config["head_node_address"])
        if head_node_address.startswith("ray://"):
            # Choose client server port for remote head node
            port = self.ray_cluster_config["ports"]["client_server"]
        else:
            # Choose GCS server port for local head node
            port = self.ray_cluster_config["ports"]["head_node"]
        self.address = f"{head_node_address}:{port}"
        self.logger.info(f"Ray head node address: {self.address}")

    def _set_serve_http_url(self) -> None:
        """Set the Ray Serve HTTP API base URL based on the head node address and port."""
        address = self.ray_cluster_config["head_node_address"].split("://")[-1]
        self.serve_http_url = (
            f"http://{address}:{self.ray_cluster_config['ports']['serve']}"
        )
        self.logger.info(f"Ray Serve HTTP URL: {self.serve_http_url}")

    async def _connect_to_cluster(self) -> ray.client_builder.ClientContext:
        """Connect to the Ray cluster using the configured head node address.

        Establishes a connection to an existing Ray cluster using the head node
        address. This method is used both for connecting to external clusters
        and for verifying connections after starting a new cluster.

        Returns:
            ray.client_builder.ClientContext: Ray client context for the connected cluster

        Raises:
            RuntimeError: If Ray is already initialized.
            Exception: If connection to the Ray cluster fails.
        """
        try:
            # Connect to the Ray cluster.
            # For external clusters, upload the bioengine package via py_modules so
            # remote workers can import it without a pip install. In single-machine
            # and SLURM modes, the workers share the same environment/container image
            # where bioengine is already installed, so the upload is unnecessary.
            job_runtime_env = {}
            if self.mode == "external-cluster":
                job_runtime_env["py_modules"] = [os.path.dirname(bioengine.__file__)]

            context = await asyncio.to_thread(
                ray.init,
                address=self.address,
                namespace="bioengine",
                logging_format=stream_logging_format,
                runtime_env=job_runtime_env,
            )

            # Update Ray's logger formatters to use timezone-aware date format
            for handler in logging.getLogger("ray").handlers:
                if handler.formatter:
                    handler.formatter.datefmt = date_format

            # Create BioEngineProxy to access cluster state
            # TODO: Check 'exclude_head_node' setting when head cpus and gpus are set
            exclude_head_node = self.mode == "slurm"
            check_pending_resources = self.mode == "slurm"

            # In single-machine and SLURM modes, BioEngine starts the Ray
            # cluster and knows the dashboard port — pass the URL through
            # so the proxy actor does not need to look it up via private API.
            # The dashboard always binds to localhost; the proxy actor runs
            # on the head node so localhost resolves correctly there.
            if self.mode == "external-cluster":
                dashboard_url = None  # proxy actor will fall back internally
            else:
                dashboard_port = self.ray_cluster_config["ports"]["dashboard"]
                dashboard_url = f"http://127.0.0.1:{dashboard_port}"

            # Reuse a stable detached proxy actor across worker restarts
            # If it does not exist yet, create it
            proxy_actor = BioEngineProxyActor.options(
                name=self.proxy_actor_name,
                namespace="bioengine",
                lifetime="detached",
                get_if_exists=True,
            )
            self.proxy_actor_handle = proxy_actor.remote(
                exclude_head_node=exclude_head_node,
                check_pending_resources=check_pending_resources,
                dashboard_url=dashboard_url,
            )

            # Test the connection by getting the cluster state
            await self.proxy_actor_handle.get_cluster_state.remote()
            self.logger.info(
                f"Connected to Ray cluster at '{self.address}' with Serve HTTP URL {self.serve_http_url}"
            )

            return context
        except Exception as e:
            self.logger.error(f"Failed to connect to existing Ray cluster: {e}")
            raise e

    async def _shutdown_head_node(self, grace_period: int = 30) -> None:
        """Shutdown the Ray cluster head node.

        Args:
            grace_period: Grace period in seconds for graceful shutdown.
                         If negative, performs force shutdown instead.
        """
        self.logger.info(
            f"Starting graceful shutdown of Ray head node (grace period: {grace_period}s)..."
        )
        args = [self.ray_exec_path, "stop", "--grace-period", str(grace_period)]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            self.logger.error(
                f"Ray stop command failed with error code {proc.returncode}:\n{error_msg}"
            )
            raise subprocess.CalledProcessError(
                proc.returncode, " ".join(args), stderr=error_msg
            )

        output = stdout.decode()
        if re.search(r"Stopped all \d+ Ray processes\.", output):
            self.logger.info("All Ray processes stopped successfully")
        elif "Did not find any active Ray processes." in output:
            self.logger.info("Did not find any active Ray processes.")
        else:
            message = re.search(
                r"Stopped only (\d+) out of (\d+) Ray processes within the grace period (\d+) seconds\.",
                output,
            )
            if message:
                self.logger.warning(
                    f"Some Ray processes could not be stopped: {message.group(0)}"
                )
            else:
                self.logger.warning(
                    f"Unknown message during Ray shutdown:\n----------\n{output}"
                )

    async def check_connection(self) -> None:
        """Check if Ray cluster is initialized and connected.

        Raises:
            RuntimeError: If Ray is not initialized.
        """
        # Initialize Ray if RayCluster is running and Ray is not initialized
        if not self.is_ready.is_set():
            raise RuntimeError("Ray cluster is not running")

        if not ray.is_initialized():
            self.logger.warning(f"Ray client disconnected. Reconnecting...")
            await self._connect_to_cluster()

    async def reconnect(self) -> None:
        """Tear down the Ray Client session and re-run the connection sequence.

        Re-acquires every detached actor handle the worker holds:
        ``BioEngineProxyActor`` (re-attached via ``get_if_exists=True``), and
        — implicitly, on the next ``serve.*`` call — the Ray Serve controller.
        Use this when a server-side actor restart has invalidated our handles
        (see ``_is_stale_handle_error``); Ray Client does not auto-recover.

        Notes:
            ``ray.shutdown()`` only tears down our gRPC session, not the
            cluster, so this is safe in external-cluster mode. For
            single-machine/SLURM modes ``ray.shutdown()`` would stop the
            cluster itself, so reconnect is only valid when the worker is a
            Ray Client (i.e. external-cluster mode).
        """
        if self.mode != "external-cluster":
            raise RuntimeError(
                "RayCluster.reconnect() is only safe in external-cluster mode; "
                f"current mode is '{self.mode}'."
            )

        self.logger.warning("Reconnecting Ray Client to refresh stale handles...")
        try:
            await asyncio.to_thread(ray.shutdown)
        except Exception as e:
            self.logger.debug(f"ray.shutdown() during reconnect raised: {e}")
        # Drop the stale proxy actor handle before re-acquiring it; otherwise
        # callers racing against the reconnect could hold and use it.
        self.proxy_actor_handle = None
        await self._connect_to_cluster()
        self.logger.info("Ray Client reconnected; handles refreshed.")

    async def call_with_reconnect(self, fn, *args, **kwargs):
        """Run a blocking Ray API call with one automatic stale-handle recovery.

        Executes ``fn(*args, **kwargs)`` on a thread (Ray's public ``serve.*``
        functions are synchronous). If the call raises a stale-handle error
        the worker reconnects and retries once. Any other exception, or a
        second stale-handle error after reconnect, propagates to the caller.

        Args:
            fn: A blocking callable that performs a Ray operation
                (e.g. ``ray.serve.status``, ``ray.serve.run``,
                ``ray.serve.delete``, ``ray.serve.get_app_handle``).
            *args, **kwargs: Forwarded to ``fn``.

        Returns:
            Whatever ``fn`` returns.

        Raises:
            The exception ``fn`` raises if it is not a stale-handle error, or
            if the retry after reconnect also fails.
        """
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as exc:
            if not _is_stale_handle_error(exc):
                raise
            self.logger.warning(
                f"Stale Ray Client handle detected calling "
                f"{getattr(fn, '__name__', repr(fn))!r}: {exc!s}. "
                f"Reconnecting Ray Client and retrying once."
            )
            await self.reconnect()
            return await asyncio.to_thread(fn, *args, **kwargs)

    async def monitor_cluster(self) -> None:
        """Monitor cluster status and update worker nodes history."""
        try:
            # Get the current status of the cluster from the BioEngineProxy
            cluster_status = await self.proxy_actor_handle.get_cluster_state.remote()

            self.cluster_status_history[time.time()] = cluster_status

            # Limit the history size
            if len(self.cluster_status_history) > self.max_status_history_length:
                self.cluster_status_history.popitem(last=False)

            # Check if SLURM workers need to scale
            if self.slurm_workers:
                await self.slurm_workers.check_scaling()
        except Exception as e:
            self.logger.error(f"Error monitoring cluster: {e}")
            raise e

    async def start(self) -> None:
        """Start the Ray cluster based on the configured mode.

        Depending on the mode, this method will either start a new Ray cluster
        head node, connect to an existing cluster, or perform no action if already
        connected. After starting the cluster, it launches a background monitoring
        task to continuously track cluster status.

        Raises:
            RuntimeError: If the cluster is already running.
            Exception: For other startup errors.
        """
        if self.is_ready.is_set():
            raise RuntimeError("Ray cluster is already running")

        if ray.is_initialized():
            raise RuntimeError(
                "Ray is already initialized. Please stop the existing Ray instance before starting the cluster."
            )

        try:
            # If cluster is not an already running external cluster, start the Ray cluster head node
            if self.mode != "external-cluster":
                await self._start_cluster()

            self.start_time = time.time()

            # Connect a client to the Ray cluster
            self._set_head_node_address()
            self._set_serve_http_url()
            await self._connect_to_cluster()

            # Do a first cluster status check
            await self.monitor_cluster()

            self.is_ready.set()

            self.logger.info("Ray cluster started successfully.")

        except Exception as e:
            self.logger.error(f"Error in cluster startup: {e}")
            await self.stop()
            raise e

    async def stop(self) -> None:
        """Stop Ray cluster and all worker nodes.

        Performs a graceful shutdown of the Ray cluster including stopping
        all workers (if running in SLURM mode), disconnecting from the
        cluster, stopping Ray Serve, and canceling any remaining SLURM jobs.

        Raises:
            OSError: If Ray executable is not reachable.
            subprocess.CalledProcessError: If Ray stop command fails.
            Exception: For other shutdown errors.
        """
        try:
            self.is_ready.clear()
            self.start_time = None

            # Shutdown all SLURM workers if running in SLURM mode
            if self.slurm_workers:
                try:
                    await self.slurm_workers.close_all()
                except Exception as e:
                    # Log the error but do not raise, as we still want to attempt Ray shutdown
                    self.logger.error(f"Error shutting down SLURM workers: {e}")

            # Shutdown the Ray cluster head node if it is not in external-cluster mode
            if self.mode != "external-cluster":
                # Shutdown Ray Serve first while Ray client is still connected
                self.logger.info("Shutting down Ray Serve...")
                try:
                    await asyncio.to_thread(serve.shutdown)
                    self.logger.info("Ray Serve shut down successfully.")
                except Exception as e:
                    # Log the error but do not raise, as we still want to attempt Ray shutdown
                    self.logger.error(f"Error shutting down Ray Serve: {e}")

                # Disconnect from Ray cluster
                if ray.is_initialized():
                    self.logger.info("Disconnecting from Ray cluster...")
                    try:
                        await asyncio.to_thread(ray.shutdown)
                        self.logger.info("Ray cluster disconnected successfully.")
                    except Exception as e:
                        self.logger.error(f"Error disconnecting from Ray cluster: {e}")

                # Shutdown the Ray cluster head node
                try:
                    await self._shutdown_head_node(grace_period=30)
                except OSError as e:
                    if (
                        e.errno == 107
                        and os.environ.get("APPTAINER_BIND", None) is not None
                    ):
                        self.logger.warning(
                            "Ray executable is not reachable. This may be due to the container's overlay filesystem being torn down."
                        )
            else:
                # Just disconnect from Ray cluster for external clusters
                if ray.is_initialized():
                    self.logger.info("Disconnecting from Ray cluster...")
                    await asyncio.to_thread(ray.shutdown)
                    self.logger.info("Ray cluster disconnected successfully.")

        except Exception as e:
            self.logger.error(f"Error shutting down Ray cluster: {e}")
            raise e
        finally:
            # Release the lock file
            if self.lock_file and self.lock_file.exists():
                try:
                    self.lock_file.unlink()
                    self.logger.debug(f"Released lock file: {self.lock_file}")
                except (OSError, IOError) as e:
                    self.logger.warning(f"Failed to remove lock file: {e}")
            self.lock_file = None
            self.cluster_status_history.clear()
