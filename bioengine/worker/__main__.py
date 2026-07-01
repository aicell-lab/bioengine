#!/usr/bin/env python3
"""
BioEngine Worker Command-Line Interface

Enterprise-grade command-line interface for deploying and managing BioEngine workers
across diverse computational environments. Supports HPC clusters with SLURM scheduling,
single-machine deployments, and external Ray cluster connections.

This module provides comprehensive configuration management, signal handling for graceful
shutdown, and structured logging for production deployments. It serves as the primary
entry point for BioEngine worker services in both development and production environments.

Key Features:
- Multi-environment deployment support (SLURM, single-machine, external clusters)
- Automatic data server detection for scientific dataset access
- HTTP-based dataset streaming for deployed applications
- Comprehensive command-line argument parsing with validation
- Graceful shutdown handling with signal management
- Structured logging with file output and debug modes
- Production-ready error handling and cleanup procedures
- Hypha server integration with authentication management

Usage:
    python -m bioengine.worker --mode slurm --admin-users admin@institution.edu
    python -m bioengine.worker --mode single-machine --debug
    python -m bioengine.worker --mode external-cluster --server-url https://custom.hypha.io

Example Deployment:
    # SLURM HPC environment with custom configuration
    python -m bioengine.worker \\
        --mode slurm \\
        --admin-users admin@institution.edu researcher@institution.edu \\
        --workspace-dir /shared/bioengine/workspace \\  # Will auto-detect data server here
        --max-workers 20 \\
        --default-num-gpus 2 \\
        --server-url https://hypha.aicell.io
        
    # To use with a dataset server (started separately)
    # 1. Start a datasets server in one terminal:
    #    python -m bioengine.datasets --data-dir /path/to/datasets --workspace-dir /shared/bioengine/workspace
    # 2. Start the worker with the same workspace directory to auto-detect the server:
    #    python -m bioengine.worker --mode single-machine --workspace-dir /shared/bioengine/workspace

Author: BioEngine Development Team
License: MIT
"""

import argparse
import asyncio
import json
import shlex
import signal
import sys
from typing import Dict

from bioengine.worker import BioEngineWorker


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the comprehensive argument parser for BioEngine worker.

    Sets up command-line argument parsing with detailed help text, validation,
    and organized argument groups for different configuration categories.
    Provides extensive customization options for all deployment environments.

    Returns:
        Configured ArgumentParser instance with all BioEngine worker options

    Argument Groups:
        - Core Options: Basic worker configuration (mode, users, directories)
        - Hypha Options: Server connection and authentication settings
        - Ray Cluster Options: Cluster management and networking configuration
        - SLURM Job Options: HPC-specific deployment parameters
        - Ray Autoscaler Options: Autoscaling behavior and resource limits
    """
    parser = argparse.ArgumentParser(
        description="BioEngine Worker - Enterprise AI Model Deployment Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SLURM HPC deployment with autoscaling
  %(prog)s --mode slurm --max-workers 10 --admin-users admin@institution.edu

  # Single-machine development deployment  
  %(prog)s --mode single-machine --debug --workspace-dir ./workspace

  # Connect to existing Ray cluster
  %(prog)s --mode external-cluster --head-node-address 10.0.0.100

For detailed documentation, visit: https://github.com/aicell-lab/bioengine
""",
    )

    # Core configuration options
    core_group = parser.add_argument_group("Core Options", "Basic worker configuration")
    core_group.add_argument(
        "--mode",
        type=str,
        choices=["single-machine", "slurm", "external-cluster"],
        required=True,
        metavar="MODE",
        help="Deployment mode: 'single-machine' for local Ray cluster, "
        "'slurm' for HPC clusters with SLURM job scheduling, 'external-cluster' for connecting "
        "to an existing Ray cluster",
    )
    core_group.add_argument(
        "--admin-users",
        type=str,
        nargs="+",
        metavar="EMAIL",
        help="List of user emails/IDs with administrative privileges for worker management. "
        "If not specified, defaults to the authenticated user from Hypha login.",
    )
    core_group.add_argument(
        "--workspace-dir",
        type=str,
        metavar="PATH",
        help="Directory for worker workspace, temporary files, and Ray data storage. "
        "Also used to detect running data servers for dataset access. "
        "Should be accessible across worker nodes in distributed deployments.",
    )
    core_group.add_argument(
        "--ray-workspace-dir",
        type=str,
        metavar="PATH",
        help="Directory for Ray cluster workspace when connecting to an external Ray cluster. "
        "Only used in 'external-cluster' mode. This allows the remote Ray cluster to use "
        "a different workspace directory than the local machine. If not specified, uses the "
        "same directory as --workspace-dir. Not applicable for 'single-machine' or 'slurm' modes.",
    )
    core_group.add_argument(
        "--startup-applications",
        type=str,
        nargs="+",
        metavar="JSON",
        help="List of applications to deploy automatically during worker startup. "
        "Each element should be a JSON string with deployment configuration. "
        'Example: \'{"artifact_id": "my_model", "application_id": "my_app"}\'',
    )
    core_group.add_argument(
        "--monitoring-interval-seconds",
        type=int,
        metavar="SECONDS",
        help="Interval in seconds for worker status monitoring and health checks. "
        "Lower values provide faster response but increase overhead.",
    )
    core_group.add_argument(
        "--dashboard-url",
        type=str,
        metavar="URL",
        help="Base URL of the BioEngine dashboard for worker management interfaces.",
    )
    core_group.add_argument(
        "--worker-name",
        type=str,
        metavar="NAME",
        help="Display name for the Hypha service registration. "
        "Defaults to 'BioEngine Worker'. Use this to distinguish multiple workers "
        "in the same workspace.",
    )
    core_group.add_argument(
        "--log-file",
        type=str,
        metavar="PATH",
        help="Path to the log file. If set to 'off', logging will only go to console. "
        "If not specified (None), a log file will be created in '<workspace_dir>/logs'. ",
    )
    core_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging for detailed troubleshooting and development. "
        "Increases log verbosity significantly.",
    )
    core_group.add_argument(
        "--graceful-shutdown-timeout",
        type=int,
        metavar="SECONDS",
        help="Timeout in seconds for graceful shutdown operations.",
    )

    # Hypha server connection options
    hypha_group = parser.add_argument_group(
        "Hypha Options", "Server connection and authentication"
    )
    hypha_group.add_argument(
        "--server-url",
        type=str,
        metavar="URL",
        help="URL of the Hypha server for service registration and remote access. "
        "Used both by the worker itself and by deployed apps (HYPHA_SERVER_URL env "
        "var + ProxyDeployment's connection URL), so it must be reachable from every "
        "node hosting an app — including Ray worker pods.",
    )
    hypha_group.add_argument(
        "--workspace",
        type=str,
        metavar="NAME",
        help="Hypha workspace name for service isolation and organization. "
        "If not specified, uses the workspace associated with the authentication token.",
    )
    hypha_group.add_argument(
        "--token",
        type=str,
        metavar="TOKEN",
        help="Authentication token for Hypha server access. If not provided, will use "
        "the HYPHA_TOKEN environment variable or prompt for interactive login. "
        "Recommend using a long-lived token for production deployments.",
    )
    hypha_group.add_argument(
        "--client-id",
        type=str,
        metavar="ID",
        help="Unique client identifier for Hypha connection. If not specified, "
        "an identifier will be generated automatically to ensure unique registration.",
    )

    # Ray cluster management options
    ray_cluster_group = parser.add_argument_group(
        "Ray Cluster Options", "Cluster networking and resource configuration"
    )
    ray_cluster_group.add_argument(
        "--head-node-address",
        type=str,
        metavar="ADDRESS",
        help="IP address of the Ray head node. For external-cluster mode, this specifies "
        "the cluster to connect to. If not set in other modes, uses the first available "
        "system IP address.",
    )
    ray_cluster_group.add_argument(
        "--head-node-port",
        type=int,
        metavar="PORT",
        help="Port for Ray head node and GCS (Global Control Service) server. "
        "Must be accessible from all worker nodes.",
    )
    ray_cluster_group.add_argument(
        "--node-manager-port",
        type=int,
        metavar="PORT",
        help="Port for Ray node manager services. Used for inter-node communication "
        "and coordination.",
    )
    ray_cluster_group.add_argument(
        "--object-manager-port",
        type=int,
        metavar="PORT",
        help="Port for Ray object manager service. Handles distributed object storage "
        "and transfer between nodes.",
    )
    ray_cluster_group.add_argument(
        "--redis-shard-port",
        type=int,
        metavar="PORT",
        help="Port for Redis sharding in Ray's internal metadata storage. "
        "Used for cluster state management.",
    )
    ray_cluster_group.add_argument(
        "--serve-port",
        type=int,
        metavar="PORT",
        help="Port for Ray Serve HTTP endpoint serving deployed models and applications. "
        "This is where model inference requests are handled.",
    )
    ray_cluster_group.add_argument(
        "--dashboard-port",
        type=int,
        metavar="PORT",
        help="Port for Ray dashboard web interface. Provides cluster monitoring "
        "and debugging capabilities.",
    )
    ray_cluster_group.add_argument(
        "--client-server-port",
        type=int,
        metavar="PORT",
        help="Port for Ray client server connections. Used by external Ray clients "
        "to connect to the cluster.",
    )
    ray_cluster_group.add_argument(
        "--redis-password",
        type=str,
        metavar="PASSWORD",
        help="Password for Ray cluster Redis authentication. If not specified, "
        "a secure random password will be generated automatically.",
    )
    ray_cluster_group.add_argument(
        "--head-num-cpus",
        type=int,
        metavar="COUNT",
        help="Number of CPU cores allocated to the head node for task execution. "
        "Set to 0 to reserve head node for coordination only.",
    )
    ray_cluster_group.add_argument(
        "--head-num-gpus",
        type=int,
        metavar="COUNT",
        help="Number of GPU devices allocated to the head node for task execution. "
        "Typically 0 to reserve GPUs for worker nodes.",
    )
    ray_cluster_group.add_argument(
        "--head-memory-in-gb",
        type=int,
        metavar="GB",
        help="Memory allocation in GB for head node task execution. "
        "If not specified, Ray will auto-detect available memory.",
    )
    ray_cluster_group.add_argument(
        "--runtime-env-pip-cache-size-gb",
        type=int,
        metavar="GB",
        help="Size limit in GB for Ray runtime environment pip package cache. "
        "Larger cache improves environment setup time.",
    )
    ray_cluster_group.add_argument(
        "--no-ray-cleanup",
        action="store_false",
        dest="force_clean_up",
        help="Skip cleanup of previous Ray cluster processes and data. "
        "Use with caution as it may cause port conflicts or resource issues.",
    )

    # SLURM job configuration options (for HPC environments)
    slurm_job_group = parser.add_argument_group(
        "SLURM Job Options", "HPC job scheduling and worker deployment"
    )
    slurm_job_group.add_argument(
        "--image",
        type=str,
        metavar="IMAGE",
        help=f"Container image for SLURM worker jobs. Should include all required "
        f"dependencies and be accessible on compute nodes.",
    )
    slurm_job_group.add_argument(
        "--worker-workspace-dir",
        type=str,
        metavar="PATH",
        help="Workspace directory path mounted to worker containers in SLURM jobs. "
        "Must be accessible from compute nodes. Required for SLURM mode.",
    )
    slurm_job_group.add_argument(
        "--default-num-gpus",
        type=int,
        metavar="COUNT",
        help="Default number of GPU devices to request per SLURM worker job. "
        "Can be overridden per deployment.",
    )
    slurm_job_group.add_argument(
        "--default-num-cpus",
        type=int,
        metavar="COUNT",
        help="Default number of CPU cores to request per SLURM worker job. "
        "Should match typical model inference requirements.",
    )
    slurm_job_group.add_argument(
        "--default-mem-in-gb-per-cpu",
        type=int,
        metavar="GB",
        help="Default memory allocation in GB per CPU core for SLURM workers. "
        "Total memory = num_cpus * mem_per_cpu.",
    )
    slurm_job_group.add_argument(
        "--default-time-limit",
        type=str,
        metavar="TIME",
        help='Default time limit for SLURM worker jobs in "HH:MM:SS" format. '
        "Jobs will be terminated after this duration.",
    )
    slurm_job_group.add_argument(
        "--further-slurm-args",
        type=str,
        metavar="ARGS",
        help="Additional SLURM sbatch arguments for specialized cluster configurations, "
        "passed as a single quoted, shell-style string. "
        'Example: --further-slurm-args "--partition=gpu --qos=high-priority"',
    )
    slurm_job_group.add_argument(
        "--gpu-slurm-flag",
        type=str,
        metavar="TEMPLATE",
        help="Template for the GPU sbatch directive. The token '{n}' is replaced "
        "with the requested GPU count. Default: '--gpus={n}'. Use "
        "'--gres=gpu:{n}' on clusters that require gres syntax. Pass an empty "
        "string to omit the directive (e.g. when GPUs are requested via "
        "--further-slurm-args).",
    )
    slurm_job_group.add_argument(
        "--further-apptainer-args",
        type=str,
        metavar="ARGS",
        help="Additional CLI flags forwarded to 'apptainer exec' inside each "
        "SLURM worker job, passed as a single quoted, shell-style string. "
        'Example: --further-apptainer-args "--bind /proj/aicell:/proj/aicell".',
    )

    # Ray autoscaling configuration options
    ray_autoscaling_group = parser.add_argument_group(
        "Ray Autoscaler Options", "Automatic worker scaling behavior"
    )
    ray_autoscaling_group.add_argument(
        "--min-workers",
        type=int,
        metavar="COUNT",
        help="Minimum number of worker nodes to maintain in the cluster. "
        "Workers below this threshold will be started immediately.",
    )
    ray_autoscaling_group.add_argument(
        "--max-workers",
        type=int,
        metavar="COUNT",
        help="Maximum number of worker nodes allowed in the cluster. "
        "Prevents unlimited scaling and controls costs.",
    )
    ray_autoscaling_group.add_argument(
        "--scale-up-cooldown-seconds",
        type=int,
        metavar="SECONDS",
        help="Cooldown period in seconds between scaling up operations. "
        "Prevents rapid scaling oscillations.",
    )
    ray_autoscaling_group.add_argument(
        "--scale-down-check-interval-seconds",
        type=int,
        metavar="SECONDS",
        help="Interval in seconds between checks for scaling down idle workers. "
        "More frequent checks enable faster response to load changes.",
    )
    ray_autoscaling_group.add_argument(
        "--scale-down-threshold-seconds",
        type=int,
        metavar="SECONDS",
        help="Time threshold in seconds before scaling down idle worker nodes. "
        "Longer thresholds reduce churn but may waste resources.",
    )

    return parser


def get_args_by_group(parser: argparse.ArgumentParser) -> Dict[str, Dict[str, any]]:
    """
    Parse command-line arguments and organize them by argument group.

    Parses all command-line arguments and organizes them into dictionaries
    based on their argument group membership. This organization facilitates
    passing configuration to different components of the BioEngine worker.

    Arguments with None values are filtered out to allow component managers
    to use their default parameter values instead of overriding them with None.

    Args:
        parser: Configured ArgumentParser instance

    Returns:
        Dictionary mapping argument group names to their respective arguments:
            - "Core Options": Basic worker configuration
            - "Hypha Options": Server connection settings
            - "Ray Cluster Options": Cluster networking configuration
            - "SLURM Job Options": HPC job parameters
            - "Ray Autoscaler Options": Autoscaling configuration

        Only arguments with non-None values are included to preserve component
        manager defaults.

    Example:
        ```python
        parser = create_parser()
        groups = get_args_by_group(parser)
        worker_config = groups["Core Options"]
        hypha_config = groups["Hypha Options"]
        ```
    """
    args = parser.parse_args()
    args_dict = vars(args)

    group_configs = {}
    for group in parser._action_groups:
        # Skip built-in argument groups (positional arguments, options)
        if group.title in ["positional arguments", "options"]:
            continue

        group_keys = [a.dest for a in group._group_actions]
        group_configs[group.title] = {
            k: args_dict[k]
            for k in group_keys
            if k in args_dict and args_dict[k] is not None
        }

    return group_configs


def read_startup_applications(
    group_configs: Dict[str, Dict[str, any]],
) -> Dict[str, Dict[str, any]]:
    """
    Parse and validate startup application configurations from JSON strings.

    Processes startup application specifications from command-line arguments,
    parsing JSON configuration strings and validating their format. Each
    application configuration should specify deployment parameters. The parsed
    dictionaries are passed to BioEngineWorker, replacing the original JSON strings.

    Args:
        group_configs: Dictionary of grouped command-line arguments

    Returns:
        Updated group_configs with parsed startup_applications as list of dictionaries

    Raises:
        ValueError: If JSON configuration is malformed or invalid

    Example:
        Valid startup application JSON:
        ```json
        {"artifact_id": "my_artifact", "num_gpus": 1}
        ```
    """
    # Use startup_applications from Core Options group
    core_options = group_configs.get("Core Options", {})
    startup_apps_raw = core_options.get("startup_applications")

    if not startup_apps_raw:
        return group_configs

    startup_applications = []
    for json_str in startup_apps_raw:
        try:
            application_config = json.loads(json_str.strip())
            if not isinstance(application_config, dict):
                raise ValueError("Application configuration must be a JSON object")

            # Validate required fields
            if "artifact_id" not in application_config:
                raise ValueError(
                    "Application configuration must include 'artifact_id' field"
                )

            startup_applications.append(application_config)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON format for startup application: {e}. "
                "A valid format would be: "
                '\'{"artifact_id": "my_artifact", "num_gpus": 1}\'. '
                f"Received: '{json_str}'"
            )

    group_configs["Core Options"]["startup_applications"] = startup_applications
    return group_configs


def split_slurm_string_args(group_configs):
    """Split shell-style string args into List[str] for SlurmWorkers.

    ``--further-slurm-args`` and ``--further-apptainer-args`` take a single
    quoted string on the CLI (argparse rejects ``nargs="+"`` values that
    look like flags). Split here so downstream code keeps its List[str] API.
    """
    slurm_options = group_configs.get("SLURM Job Options", {})
    for key in ("further_slurm_args", "further_apptainer_args"):
        raw = slurm_options.get(key)
        if isinstance(raw, str):
            slurm_options[key] = shlex.split(raw)
    return group_configs


async def main(group_configs):
    """Main function to initialize and register BioEngine worker"""
    # Create BioEngine worker instance
    bioengine_worker = BioEngineWorker(
        **group_configs["Core Options"],
        **group_configs["Hypha Options"],
        ray_cluster_config={
            **group_configs["Ray Cluster Options"],
            **group_configs["SLURM Job Options"],
            **group_configs["Ray Autoscaler Options"],
        },
    )

    # Route k8s SIGTERM (and SIGINT) through the worker's own graceful
    # shutdown path — otherwise the process exits without running
    # BioEngineWorker._cleanup(), leaving the Hypha connection registered
    # under this pod's client_id until the server's stale-client TTL sweeps
    # it. That stranded registration blocks any future pod (rolling update,
    # crash-restart) that tries to reconnect with the same stable client_id.
    loop = asyncio.get_running_loop()

    def _request_shutdown(signame: str) -> None:
        print(f"Received {signame}, initiating graceful worker shutdown...")
        asyncio.create_task(bioengine_worker._stop(blocking=False))

    for signame in ("SIGTERM", "SIGINT"):
        try:
            loop.add_signal_handler(
                getattr(signal, signame), _request_shutdown, signame
            )
        except (NotImplementedError, RuntimeError):
            # add_signal_handler is not supported on Windows and inside
            # some embedded event loops. The KeyboardInterrupt path in
            # BioEngineWorker.start covers those cases.
            pass

    # Start the worker and wait until shutdown is triggered
    await bioengine_worker.start(blocking=True)


if __name__ == "__main__":
    """
    Entry point for BioEngine worker command-line interface.

    Parses command-line arguments, validates configuration, and starts the worker.
    """
    try:
        parser = create_parser()
        group_configs = get_args_by_group(parser)

        # Process startup applications if provided
        group_configs = read_startup_applications(group_configs)

        # Split shell-style SLURM args into List[str]
        group_configs = split_slurm_string_args(group_configs)

        # Start the main worker process
        asyncio.run(main(group_configs))

    except Exception as e:
        print(f"Failed to start BioEngine worker: {e}")
        sys.exit(1)
