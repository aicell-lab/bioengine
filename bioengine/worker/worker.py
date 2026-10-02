import asyncio
import importlib.metadata
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
import ray
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import RemoteService
from hypha_rpc.sync import login
from hypha_rpc.utils.schema import schema_method
from pydantic import Field

from bioengine import __version__
from bioengine.apps.manager import AppsManager
from bioengine.datasets import BioEngineDatasets
from bioengine.heartbeat import (
    DEFAULT_HEARTBEAT_PATH,
    STARTUP_STALE_AFTER_SECONDS,
    heartbeat_stale_after_seconds,
    write_heartbeat,
)
from bioengine.cluster.ray_cluster import RayCluster
from bioengine.utils import (
    RECONNECT_BUDGET_S,
    STARTUP_CONNECT_BUDGET_S,
    fetch_centroid_coordinates,
    fetch_geolocation,
    check_permissions,
    connect_with_retry,
    create_context,
    create_logger,
)
from bioengine.worker.code_executor import CodeExecutor

# How often the worker asks Hypha whether it still serves the worker's own
# service. hypha-rpc keeps the socket alive and re-registers on reconnect, so
# this exists only for the case the library cannot signal: Hypha dropped this
# client while the socket stayed open from our side, so echo("ping") keeps
# answering against a server that has stopped serving us.
_REGISTRATION_PROBE_INTERVAL_S = 60
# Bound on the probe itself: an unanswered round trip must not hold up the rest
# of the monitoring tick.
_REGISTRATION_PROBE_TIMEOUT_S = 10
# How long the worker's own service may stay unreachable before the check
# escalates. Hypha routinely serves nothing for a minute or two and recovers by
# itself, and reporting a worker degraded through that costs more than waiting;
# five minutes of failed re-registration is a real fault. Once past this the
# check raises every tick, so the degraded threshold (5 ticks) is reached one
# monitoring interval at a time.
_REGISTRATION_GRACE_S = 300
# The grace above resets on any successful probe, so a registration that answers
# intermittently never escalates and never logs. These bound a second, purely
# observational window that does NOT reset on success: enough failure episodes
# inside it and the worker says so once.
_REGISTRATION_FLAP_WINDOW_S = 3600
_REGISTRATION_FLAP_THRESHOLD = 5
# Bound on disconnect() while rebuilding the connection. The transport being
# closed there is the one already suspected of being wedged, so an unbounded
# close can stall the monitoring loop on exactly the socket that prompted the
# rebuild.
_DISCONNECT_TIMEOUT_S = 5.0

# Emails that stand for "nobody in particular", so a request keyed on one could
# never be granted to a person. Hypha itself reports email=None for every
# anonymous caller and flags them with is_anonymous, which is the signal to
# trust; these are the substitutes BioEngine's own code puts in its place —
# 'anonymous@example.com' from bioengine.utils.create_context, 'no-email' and
# 'anonymous-user' from the dataset proxy server.
_UNAUTHENTICATED_EMAILS = frozenset(
    {"anonymous@example.com", "no-email", "anonymous-user"}
)

# 'deny' is terminal on purpose: a denial the requester could immediately
# re-file would make the public request method a spam vector, so lifting one is
# 'clear', an admin action.
_ACCESS_REQUEST_DECISIONS = ("grant", "deny", "clear")


# What ``get_status`` tells a caller who is not a worker admin.
#
# The worker service is registered public and this is its health surface, so it
# has to keep answering without a token. The deployed Kubernetes *startup*
# probe curls it and greps ``is_ready``: gating the method fails every worker's
# startup for 18 x 30 s and then crash-loops it permanently. Only the startup
# probe — the deployed liveness probe is a local ``kill -0 1``, for the reason
# recorded on _check_service_registration below; the manifests the website's
# worker guide generates do curl ``get_status`` from both. Beyond the probes:
# the worker-list cards read ``geo_location`` and ``bioengine_version``, and the
# CLI's cluster view and the KTH gpu-cuda-watch script read ``ray_cluster``.
#
# Held back is ``admin_users``, the only field naming people. It is the same
# list PR #208 stopped permission denials from disclosing, and ``list_admin_users``
# already requires admin to return it, so publishing it here contradicted both.
#
# An allowlist rather than a denylist, for the same reason ``get_app_status``
# uses one: a field added to the status payload later must not become public by
# nobody having thought about it. Adding a name here is the decision to publish it.
#
# Top level only, deliberately: ``ray_cluster`` passes through whole, so a new
# sub-key of it — or a new ``slurm_jobs`` column — is public the day it lands.
# Nothing under it names a third party (the squeue query is scoped
# ``-u $USER -n <job_name>``, so it can only return this worker's own jobs), but
# that is where this guarantee stops. ``get_app_status`` guards its second level
# with PUBLIC_DEPLOYMENT_FIELDS because replica logs live there; nothing here
# needs the same.
PUBLIC_WORKER_STATUS_FIELDS = (
    "service_start_time",
    "service_uptime",
    "bioengine_version",
    "ray_version",
    "hypha_rpc_version",
    "worker_mode",
    "workspace",
    "client_id",
    "ray_cluster",
    "geo_location",
    "is_ready",
)


def public_worker_status(status: Dict[str, Any]) -> Dict[str, Any]:
    """The worker's status as a caller who is not a worker admin may see it."""
    return {
        field: status[field] for field in PUBLIC_WORKER_STATUS_FIELDS if field in status
    }


class BioEngineWorker:
    """
    Enterprise-grade BioEngine worker for distributed AI model deployment and execution.

    The BioEngineWorker provides a comprehensive platform for managing AI model deployments
    across diverse computational environments, from high-performance computing clusters with
    SLURM job scheduling to single-machine deployments and external Ray clusters. It serves
    as the central orchestration layer for the BioEngine ecosystem.

     Architecture Overview:
     The worker orchestrates two primary component managers, each handling specialized
     functionality while maintaining enterprise-grade security, monitoring, and lifecycle management:

     • RayCluster: Manages distributed Ray cluster lifecycle including SLURM-based autoscaling,
        resource allocation, and worker node management across HPC environments
     • AppsManager: Handles AI model deployment lifecycle through Ray Serve, including artifact
        management, deployment orchestration, and application scaling

     For datasets, the worker connects to an external data server service via HTTP:
     • Detects running data servers in the BioEngine cache directory
     • Provides data server URL to deployed applications
     • Each deployment uses BioEngineDatasets client to stream data via HTTPZarrStore
     • Each deployment receives a per-application Hypha authentication token (hypha_token) for secure dataset and API access

     Core Capabilities:
     - Multi-environment deployment support (SLURM HPC, single-machine, external clusters)
     - Enterprise-grade security with two-level permission systems (admin + resource-specific)
     - Hypha server integration for remote management and service discovery
     - Automatic Ray cluster lifecycle management with intelligent autoscaling
     - AI model deployment and serving through Ray Serve with health monitoring
     - Automatic data server detection and connection for dataset access
     - Integration with deployed applications for HTTP-based dataset streaming
     - Python code execution in distributed Ray tasks with resource allocation
     - Comprehensive monitoring, logging, and status reporting
     - Graceful shutdown and resource cleanup with signal handling

     Security Architecture:
     - Admin-level permissions for cluster and deployment management operations
     - Resource-specific authorization for dataset access and model execution
     - Context-aware permission checking with detailed audit logging
     - Secure artifact management with version control and validation
     - Isolated execution environments with resource limits and monitoring
     - Per-application Hypha authentication tokens (hypha_token) for secure dataset and API access

     Deployment Modes:
     1. **SLURM Mode**: Full HPC integration with automatic worker scheduling, resource allocation,
         and cluster autoscaling based on computational demand
     2. **Single-Machine Mode**: Local Ray cluster for development and small-scale deployments
         with configurable resource limits
     3. **External-Cluster Mode**: Connection to pre-existing Ray clusters with service registration
         and management capabilities

     Integration Points:
     - Hypha Server: Service registration, remote access, and workspace integration
     - Ray Ecosystem: Distributed computing, model serving, and resource management
     - SLURM: HPC job scheduling, resource allocation, and cluster management
     - BioEngine Datasets: HTTP-based dataset streaming service with access control (hypha_token)
     - File Systems: Artifact storage, dataset discovery, and temporary file management

     Attributes:
          admin_users (List[str]): List of user IDs/emails authorized for admin operations
          workspace_dir (Path): Directory for temporary files, Ray data, and worker state
          dashboard_url (str): URL of the BioEngine dashboard for worker management
          monitoring_interval_seconds (int): Interval for status monitoring and health checks
          log_file (Optional[str]): Path to log file for structured logging output
          heartbeat_file (Path): File rewritten after every completed monitoring pass
          graceful_shutdown_timeout (int): Timeout in seconds for graceful shutdown operations
          server_url (str): URL of the Hypha server for service registration
          workspace (str): Hypha workspace name for service isolation
          client_id (str): Unique client identifier for Hypha connection
          service_id (str): Service identifier for registration ("bioengine-worker")
          full_service_id (str): Complete service ID including workspace and user context
          ray_cluster (RayCluster): Ray cluster management component
          apps_manager (AppsManager): Application deployment management component
          data_server_url (Optional[str]): URL of the detected dataset server
          data_service_url (Optional[str]): Full URL to the dataset service endpoint
          start_time (float): Timestamp when worker was started
          is_ready (asyncio.Event): Event signaling worker initialization completion
          logger (logging.Logger): Structured logger for worker operations

    Example Usage:
        ```python
        # Initialize worker for SLURM HPC environment
        worker = BioEngineWorker(
            mode="slurm",
            admin_users=["admin@institution.edu"],
            workspace_dir=f"{os.environ['HOME']}/.bioengine",  # Will check for data server here
            server_url="https://hypha.aicell.io",
            startup_applications=[
                {"artifact_id": "<my-workspace>/<my_artifact>", "application_id": "my_custom_name"},
                {"artifact_id": "<my-workspace>/<another_artifact>", "disable_gpu": True}
            ],
            ray_cluster_config={
                "max_workers": 10,
                "default_num_gpus": 1,
                "default_num_cpus": 8
            }
        )

        # Start all services
        service_id = await worker.start()

        # Worker is now ready for model deployments with auto-detected datasets
        status = await worker.get_status()
        print(f"Data server detected: {worker.data_server_url is not None}")
        ```

    Note:
        The BioEngineWorker requires proper configuration of the deployment environment,
        including access to storage systems, network connectivity for Hypha server
        communication, and appropriate permissions for the target deployment mode.
    """

    def __init__(
        self,
        mode: Literal["single-machine", "slurm", "external-cluster"],
        admin_users: Optional[List[str]] = None,
        enable_access_requests: bool = False,
        workspace_dir: Union[str, Path] = f"{os.environ['HOME']}/.bioengine",
        ray_workspace_dir: Optional[Union[str, Path]] = None,
        startup_applications: Optional[List[dict]] = None,
        monitoring_interval_seconds: int = 10,
        # Hypha server connection configuration
        server_url: str = "https://hypha.aicell.io",
        workspace: Optional[str] = None,
        token: Optional[str] = None,
        client_id: Optional[str] = None,
        # Ray cluster configuration
        ray_cluster_config: Optional[Dict[str, Any]] = None,
        # BioEngine dashboard URL
        dashboard_url: str = "https://bioimage.io/#/bioengine",
        # Hypha service name
        worker_name: str = "BioEngine Worker",
        # Logger configuration
        log_file: Optional[Union[str, Path]] = None,
        debug: bool = False,
        # Liveness heartbeat file
        heartbeat_file: Optional[Union[str, Path]] = None,
        # Graceful shutdown timeout
        graceful_shutdown_timeout: int = 60,
    ):
        """
        Initialize BioEngine worker with enterprise-grade configuration and component managers.

        Sets up the worker with comprehensive configuration management, initializes component
        managers (RayCluster, AppsManager), checks for running data servers, configures security
        settings, and establishes logging infrastructure. Handles authentication with the
        Hypha server and prepares the worker for service registration.

        The initialization process:
        1. Validates and normalizes configuration parameters
        2. Sets up secure logging infrastructure with optional file output
        3. Performs interactive login if no token provided (for token acquisition only)
        4. Initializes RayCluster with environment-specific configuration
        5. Checks for and connects to running data server in the cache directory
        6. Prepares AppsManager with data server configuration for model-data integration
        7. Configures monitoring and health check systems

        Note: Server connection and service registration occurs later during start().

        Args:
            mode: Ray cluster deployment mode determining the operational environment:
                  - 'slurm': HPC environment with SLURM job scheduling and autoscaling
                  - 'single-machine': Local Ray cluster for development/small deployments
                  - 'external-cluster': Connect to existing Ray cluster
            admin_users: List of user IDs/emails authorized for administrative operations.
                        Auto-includes the authenticated user from Hypha connection. A '*'
                        entry is not honoured and is dropped with a warning; the users
                        named here can never lose admin permissions on a running worker
                        (to demote one, drop it from this list, restart, then call
                        remove_admin_user — the restart alone does not revoke it).
            enable_access_requests: Expose the public access-request methods so a
                        non-admin can ask to become an admin of this worker. Off by
                        default — turning it on puts a method on the public worker
                        service that an unauthenticated caller can invoke.
            workspace_dir: Directory path for temporary files, Ray data storage, and worker state.
                      Must be accessible and have sufficient space for Ray operations.
            ray_workspace_dir: Directory path for Ray cluster workspace when connecting to an external
                          Ray cluster. Only used in 'external-cluster' mode. This allows the
                          remote Ray cluster to use a different workspace directory than the local
                          machine. If not specified, uses the same directory as workspace_dir.
                          Not applicable for 'single-machine' or 'slurm' modes.
            startup_applications: List of application configuration dictionaries to deploy
                                 automatically during worker startup. Each dictionary should contain
                                 deployment parameters including 'artifact_id' and optionally
                                 resource requirements like 'num_gpus', 'num_cpus', etc.
            monitoring_interval_seconds: Interval in seconds for status monitoring, health
                                       checks, and cluster state updates.
            server_url: URL of the Hypha server for service registration and remote access.
                       Must be accessible from the deployment environment.
            workspace: Hypha workspace name for service isolation. Defaults to user's
                      workspace if not specified.
            token: Authentication token for Hypha server. Uses HYPHA_TOKEN environment
                  variable if not provided, prompts for interactive login otherwise.
            client_id: Unique client identifier for Hypha connection. Auto-generated if
                      not specified to ensure unique service registration.
            ray_cluster_config: Configuration dictionary for RayCluster component including
                              SLURM job parameters, resource limits, and autoscaling settings.
            dashboard_url: Base URL of the BioEngine dashboard for worker management and
                          monitoring interfaces.
            worker_name: Display name for the Hypha service registration. Defaults to
                        "BioEngine Worker". Use this to distinguish multiple workers in
                        the same workspace.
            log_file: File path for structured logging output. Auto-generated timestamp-based
                     filename if not specified.
            debug: Enable debug-level logging for detailed troubleshooting and development.
            heartbeat_file: File the monitoring loop rewrites after every completed
                     pass, for a purely local liveness probe to read with
                     `python -m bioengine.heartbeat <path>`. Defaults to
                     'bioengine_worker_heartbeat.json' in the system temporary
                     directory ($TMPDIR), which is node-local; the probe and the
                     writing event loop must never wait on a network filesystem.
            graceful_shutdown_timeout: Timeout in seconds for graceful shutdown operations.

        Raises:
            ValueError: If configuration parameters are invalid or incompatible
            PermissionError: If insufficient permissions for cache/data directories
            Exception: If Ray cluster initialization fails

        Note:
            The worker is not ready for use until `start()` is called, which completes
            the initialization process by starting the Ray cluster and registering with
            the Hypha server. Server connection errors will occur during start(), not init.
        """
        # Store configuration parameters
        self.admin_users = admin_users or []
        self.workspace_dir = Path(workspace_dir)
        self.ray_workspace_dir = Path(ray_workspace_dir) if ray_workspace_dir else None
        self.dashboard_url = dashboard_url.rstrip("/")
        self.monitoring_interval_seconds = monitoring_interval_seconds

        # Initialize structured logging
        if log_file == "off":
            # Disable file logging, only console output
            self.log_file = None
        elif log_file is None:
            # Create a timestamped log file in the workspace directory
            log_dir = self.workspace_dir / "logs"
            self.log_file = (
                log_dir / f"bioengine_worker_{time.strftime('%Y%m%d_%H%M%S')}.log"
            )
        else:
            self.log_file = Path(log_file)

        self.heartbeat_file = (
            Path(heartbeat_file) if heartbeat_file else DEFAULT_HEARTBEAT_PATH
        )

        self.logger = create_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if debug else logging.INFO,
            log_file=self.log_file,
        )
        self.logger.info(
            f"Initializing {self.__class__.__name__} v{__version__} with mode '{mode}'"
        )

        self._admin_users_file = self.workspace_dir / "admin_users.json"
        self._wildcard_admin_users_warned = False
        self._forbid_wildcard_admin_users("the --admin-users startup flag")
        # Captured before the persisted overlay replaces the list. Derived from
        # the post-overlay list instead, every runtime-added admin would become
        # unremovable and anyone a previous version already removed would stay
        # unprotected — so the lockout this guards would survive the fix.
        self._founding_admin_users = tuple(dict.fromkeys(self.admin_users))
        self._load_persisted_admin_users()

        self.enable_access_requests = enable_access_requests
        self._access_requests_file = self.workspace_dir / "access_requests.json"
        self._access_requests = self._load_access_requests()

        # Hypha server configuration
        self.server_url = server_url
        self.server: Optional[RemoteService] = None
        self.workspace = workspace
        self._token = token or os.environ.get("HYPHA_TOKEN")
        self._token_expires_at = 0
        self.client_id = client_id
        self.service_id = "bioengine-worker"
        self.worker_name = worker_name
        self._worker_user_email = None

        # Worker state management
        self.start_time = None
        self._last_monitoring = 0
        self._registration_probe_due_at = 0.0
        self._registration_failing = False
        self._registration_ok_at = time.time()
        # Flap detection, on a clock that deliberately never resets on success.
        self._registration_window_start = self._registration_ok_at
        self._registration_window_failures = 0

        # After this many consecutive monitoring-loop failures get_status
        # reports the worker not-ready, so dashboards and callers can see a
        # degraded worker. The loop keeps running and usually recovers; a
        # loop that has stopped running entirely is caught by the heartbeat
        # file instead, since a frozen loop never reaches this counter.
        self._monitor_consecutive_errors = 0
        self._monitor_degraded_threshold = 5

        self.is_ready = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._shutdown_event.set()
        self._monitoring_task = None
        self.graceful_shutdown_timeout = graceful_shutdown_timeout
        self.full_service_id = None
        self._admin_context = None

        # Dataset server configuration
        self.data_server: Optional[BioEngineDatasets] = None
        self.available_datasets = {}

        try:
            # Attempt interactive login if no token provided
            if not self._token:
                self.logger.info(
                    "No authentication token provided, attempting interactive login..."
                )
                print("\n" + "=" * 60)
                print("NO HYPHA TOKEN FOUND - USER LOGIN REQUIRED")
                print("-" * 60, end="\n\n")
                self._token = login(
                    {"server_url": self.server_url, "expires_in": 3600 * 24 * 365}
                )
                print("\n" + "-" * 60)
                print("Login completed successfully!")
                print("=" * 60, end="\n\n")
                self.logger.info("Interactive login completed successfully")

            # Initialize geo location with unknown placeholder; actual fetch happens in monitoring loop
            self.geo_location = {
                "region": None,
                "country_name": None,
                "country_code": None,
                "latitude": None,
                "longitude": None,
                "timezone": None,
            }

            # Configure Ray cluster with environment-specific parameters
            ray_cluster_config = ray_cluster_config or {}

            # Set core parameters with precedence for explicit values
            self._set_parameter(ray_cluster_config, "mode", mode)
            self._set_parameter(
                ray_cluster_config, "ray_temp_dir", self.workspace_dir / "ray"
            )
            self._set_parameter(ray_cluster_config, "log_file", self.log_file)
            self._set_parameter(ray_cluster_config, "debug", debug)

            # Initialize Ray cluster manager
            self.ray_cluster = RayCluster(**ray_cluster_config)

            # Determine the apps workspace directory based on mode
            # For external-cluster mode, use ray_workspace_dir if provided, otherwise use workspace_dir
            # For single-machine and slurm modes, always use workspace_dir
            if mode == "external-cluster" and self.ray_workspace_dir:
                apps_workdir = self.ray_workspace_dir / "apps"
            else:
                apps_workdir = self.workspace_dir / "apps"

            # Initialize component managers with enhanced configuration
            self.apps_manager = AppsManager(
                ray_cluster=self.ray_cluster,
                apps_workdir=apps_workdir,
                server_url=server_url,
                startup_applications=startup_applications,
                log_file=self.log_file,
                debug=debug,
            )

            self.code_executor = CodeExecutor(
                ray_cluster=self.ray_cluster,
                log_file=self.log_file,
                debug=debug,
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize BioEngineWorker: {e}")
            raise e

    def __del__(self):
        if self.start_time:
            self.logger.warning(
                "BioEngineWorker is being garbage collected with partially/completely initialized components. "
                "Consider calling stop() explicitly for proper cleanup."
            )

    def _set_parameter(
        self,
        kwargs: Dict[str, Any],
        key: str,
        value: Any,
        overwrite: bool = True,
    ) -> None:
        """
        Set parameter in configuration dictionary with optional overwrite control.

        Utility method for safely updating configuration dictionaries while respecting
        existing values when needed. Provides a consistent interface for parameter
        management across component initialization.

        Args:
            kwargs: Configuration dictionary to modify in-place
            key: Parameter key to set or update
            value: Value to assign to the parameter
            overwrite: Whether to overwrite existing values (default: True)
        """
        if overwrite:
            if key in kwargs and kwargs[key] != value:
                self.logger.warning(
                    f"Overwriting provided {key} value: {kwargs[key]!r} -> {value!r}"
                )
            kwargs[key] = value
        else:
            if key not in kwargs or kwargs[key] is None:
                kwargs[key] = value

    async def _fetch_geo_location(self) -> None:
        """Fetch geographical location and coordinates until successfully obtained.

        Called from the monitoring loop. Retries on every monitoring tick until
        country_name is populated. Coordinates are fetched via Nominatim as a
        fallback only when a provider returned country but no lat/lon.
        Failures are logged but never propagated.
        """
        # Retry as long as country_name is still unknown
        if self.geo_location.get("country_name") is None:
            try:
                geo_info = await fetch_geolocation(logger=self.logger)
                self.geo_location.update(geo_info)
            except Exception as e:
                self.logger.warning(f"Failed to fetch geo location: {e}")

        # Fallback: fetch coordinates via Nominatim if country known but lat/lon missing
        if (
            self.geo_location.get("country_name")
            and self.geo_location.get("latitude") is None
            and self.geo_location.get("longitude") is None
        ):
            try:
                coordinates = await fetch_centroid_coordinates(
                    country=self.geo_location["country_name"],
                    region=self.geo_location.get("region"),
                    logger=self.logger,
                )
                self.geo_location.update(coordinates)
            except Exception as e:
                self.logger.warning(f"Failed to fetch geo coordinates: {e}")

    def _load_persisted_admin_users(self) -> None:
        """Overlay the persisted admin users on the ``admin_users`` startup seed.

        The seed is replayed verbatim on every restart, so without this a user
        added at runtime disappears on the next pod roll and a removed one comes
        back — the silent rollback that startup flags plus out-of-band runtime
        state always produce. The overlay wins and the divergence is logged.
        """
        if not self._admin_users_file.exists():
            return

        try:
            persisted = json.loads(self._admin_users_file.read_text())
            if not isinstance(persisted, list) or not all(
                isinstance(user, str) for user in persisted
            ):
                raise ValueError("expected a list of strings")
        except Exception as e:
            self.logger.error(
                f"Ignoring unreadable admin users file '{self._admin_users_file}': {e}. "
                f"Falling back to the startup admin users: {self.admin_users}"
            )
            return

        added = [user for user in persisted if user not in self.admin_users]
        removed = [user for user in self.admin_users if user not in persisted]
        if added or removed:
            self.logger.info(
                f"Admin users from '{self._admin_users_file}' override the startup list "
                f"— added at runtime: {added or 'none'}, removed at runtime: {removed or 'none'}."
            )
        self.admin_users[:] = persisted
        self._forbid_wildcard_admin_users(f"'{self._admin_users_file}'")
        self._restore_founding_admin_users()

    def _forbid_wildcard_admin_users(self, source: str) -> None:
        """Drop every '*' entry from the worker's admin users.

        A wildcard here authorized unauthenticated callers for 'run_code',
        'deploy_app' and 'upload_app' — arbitrary execution on the Ray cluster,
        open to the internet. Removing the entry is what makes every
        check_permissions call site refuse it, rather than each of the twenty
        sites having to pass allow_wildcard=False and one of them being missed.
        """
        if "*" not in self.admin_users:
            return

        self.admin_users[:] = [user for user in self.admin_users if user != "*"]

        if self._wildcard_admin_users_warned:
            return
        self._wildcard_admin_users_warned = True
        self.logger.warning(
            f"SECURITY: '*' in this worker's admin users is no longer honoured and has "
            f"been dropped from {source}. It made every caller that can reach the Hypha "
            "server — including unauthenticated, anonymous ones — a full admin, able to "
            "run arbitrary Python on this deployment via 'run_code', 'deploy_app' or "
            "'upload_app' and to destroy it via 'stop_worker', 'stop_all_apps' or "
            "'delete_app'. The account whose token started this worker remains an admin. "
            "Name the others in --admin-users, grant them with 'add_admin_user', or set "
            "--enable-access-requests so they can ask for access."
        )

    def _restore_founding_admin_users(self) -> None:
        """Put back any startup admin the persisted store dropped.

        The users named at startup cannot be removed over RPC, so a store
        missing one was written by a worker from before that guard or edited by
        hand. Leaving it out would let the lockout outlive the fix for it: the
        overlay wins over the seed, so re-rolling with the same --admin-users
        would not bring them back.
        """
        missing = [
            user for user in self._founding_admin_users if user not in self.admin_users
        ]
        if not missing:
            return

        self.logger.warning(
            f"Restoring admin user(s) {missing} named at startup but absent from "
            f"'{self._admin_users_file}'. The users a worker is started with cannot "
            "lose admin permissions; to demote one, drop it from --admin-users, restart, "
            "and then call remove_admin_user for it."
        )
        self.admin_users[:] = missing + self.admin_users

    def _persist_admin_users(self, admin_users: List[str]) -> None:
        """Write the admin users so the next restart overlays them on the seed.

        Called before the in-memory list changes: a granted permission that only
        exists in memory would revert on the next restart without ever failing.
        """
        try:
            self._admin_users_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self._admin_users_file.with_suffix(".json.tmp")
            tmp_file.write_text(json.dumps(admin_users, indent=2))
            os.replace(tmp_file, self._admin_users_file)
        except Exception as e:
            raise RuntimeError(
                f"Failed to persist admin users to '{self._admin_users_file}': {e}. "
                "Admin users are unchanged."
            )

    def _load_access_requests(self) -> Dict[str, Dict[str, Any]]:
        """Read the persisted access requests, keyed on requester email.

        In-memory only, a pod roll would discard every pending request and the
        requester would wait forever with no signal that anything was lost.
        """
        if not self._access_requests_file.exists():
            return {}

        try:
            stored = json.loads(self._access_requests_file.read_text())
            if not isinstance(stored, dict) or not all(
                isinstance(key, str) and isinstance(value, dict)
                for key, value in stored.items()
            ):
                raise ValueError("expected an object of email -> request")
        except Exception as e:
            self.logger.error(
                f"Ignoring unreadable access requests file "
                f"'{self._access_requests_file}': {e}. Starting with no requests on "
                "record; pending requesters will have to ask again."
            )
            return {}

        return stored

    def _persist_access_requests(self, requests: Dict[str, Dict[str, Any]]) -> None:
        """Write the access requests before the in-memory copy changes.

        Same ordering as _persist_admin_users: a decision that only exists in
        memory reverts on the next restart without ever failing.
        """
        try:
            self._access_requests_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self._access_requests_file.with_suffix(".json.tmp")
            tmp_file.write_text(json.dumps(requests, indent=2))
            os.replace(tmp_file, self._access_requests_file)
        except Exception as e:
            raise RuntimeError(
                f"Failed to persist access requests to '{self._access_requests_file}': "
                f"{e}. Access requests are unchanged."
            )

    def _require_access_requests_enabled(self) -> None:
        """Refuse when the operator has not turned the request surface on.

        The methods are also left out of the registered service, so this only
        fires for an in-process caller; it is here so the feature cannot be
        reached by wiring that forgets the registration gate.
        """
        if not self.enable_access_requests:
            raise RuntimeError(
                "Access requests are disabled on this worker. Start it with "
                "--enable-access-requests to let non-admins ask for access."
            )

    def _requester_identity(self, context: Dict[str, Any]) -> tuple:
        """``(key, email)`` for the caller: the dedup key and what to grant.

        Email, not user id: Hypha's generate_token mints a fresh client id per
        token while inheriting the email, so an id-keyed request would let one
        person file unlimited requests by refreshing their token.

        The two differ by case on purpose. The key is lowercased so one account
        cannot hold several requests by varying capitalisation, but the email
        granted has to be the string Hypha will report on the caller's next call,
        because check_permissions compares it case-sensitively — store the
        lowercased form and the grant would never match the caller it was for.
        """
        if context is None or not isinstance(context, dict) or "user" not in context:
            raise PermissionError(
                "Invalid context for an access request: missing user information."
            )
        user = context["user"]
        if not isinstance(user, dict):
            raise PermissionError(
                "Invalid user information in context for an access request."
            )

        email = (user.get("email") or "").strip()
        # is_anonymous is the authoritative signal and is checked first: Hypha
        # mints a fresh random id per anonymous websocket connection, so an
        # anonymous caller can produce unlimited distinct identities just by
        # reconnecting, and one-request-per-user would bound nothing.
        if (
            user.get("is_anonymous")
            or not email
            or email.lower() in _UNAUTHENTICATED_EMAILS
            or "@" not in email
        ):
            raise PermissionError(
                "Log in before requesting admin access to this worker. An "
                "unauthenticated caller carries no email address, so the request "
                "would name no account that could be granted."
            )
        return email.lower(), email

    def _is_admin_user(self, email: str) -> bool:
        """Whether this email is already in the admin list, ignoring case."""
        lowered = email.lower()
        return any(user.lower() == lowered for user in self.admin_users)

    async def _ping_data_server(self) -> None:
        """Ping the dataset server with retries to verify connectivity."""
        # No data server configured
        if not self.data_server:
            return

        try:
            await self.data_server.ping_data_server()
        except Exception as e:
            self.logger.error(f"Error while pinging dataset server: {e}")
            self.logger.info("Clearing dataset server configuration.")
            self.data_server = None
            
            current_data_server_file = (
                self.workspace_dir / "datasets" / "bioengine_current_server"
            )
            if current_data_server_file.exists():
                try:
                    current_data_server_file.unlink()
                    self.logger.info("Removed outdated data server configuration file.")
                except Exception as unlink_e:
                    self.logger.error(f"Failed to remove data server configuration file: {unlink_e}")

    async def _discover_data_server(self) -> None:
        """
        Check for a running data server and configure connection details.

        Detects the presence of a running dataset server by checking for a server URL file
        in the BioEngine workspace directory. If found, establishes connection parameters and
        verifies server accessibility through a ping request. This enables deployed
        applications to access datasets via HTTP streaming.

        Data Server Detection Process:
        1. Checks for existence of server URL file in workspace directory
        2. Reads and validates server URL
        3. Constructs service URL with workspace information
        4. Verifies server connection with ping request

        Note:
            This method is called during initialization and periodically during monitoring
            to ensure continuous data server availability for deployed applications.
        """
        # Keep existing data server if already configured
        if self.data_server:
            return

        # Check for the presence of the current data server file
        current_data_server_file = (
            self.workspace_dir / "datasets" / "bioengine_current_server"
        )
        if not current_data_server_file.exists():
            return

        # Read the server URL from the file
        try:
            data_server_url = current_data_server_file.read_text().strip()
        except Exception as e:
            self.logger.error(f"Failed to read current data server URL: {e}")
            return

        self.data_server = BioEngineDatasets(
            data_server_url=data_server_url,
            hypha_token=None,  # no token needed for pinging and dataset listing
            logger=self.logger,
        )

        # Ping the data server to verify connectivity
        await self._ping_data_server()

        # Update AppsManager with data server URL
        self.apps_manager.app_builder.update_data_server_url(data_server_url)

    async def _refresh_datasets(self) -> None:
        """Refresh the list of available datasets from the data server."""
        if not self.data_server:
            self.available_datasets = {}
            return

        # Fetch available datasets from the data server
        try:
            updated_datasets = await self.data_server.list_datasets()
            if updated_datasets != self.available_datasets:
                self.available_datasets = updated_datasets
                self.logger.info(
                    f"Available datasets updated: {list(updated_datasets.keys())}"
                )
        except Exception as e:
            self.logger.error(f"Error fetching datasets from data server: {e}")
            self.logger.info("Clearing available datasets.")
            self.available_datasets = {}

            # Ping the data server to check connectivity; clear if unreachable
            await self._ping_data_server()

    async def _connect_to_server(
        self, retry_budget_seconds: float = RECONNECT_BUDGET_S
    ) -> None:
        """
        Establish connection to Hypha server and configure admin user permissions.

        Authenticates with the Hypha server using the configured token and workspace,
        then updates the admin users list with the authenticated user information.
        This ensures the authenticated user has administrative privileges for the worker.

        The connection process:
        1. Closes any existing connection
        2. Establishes new connection with authentication
        3. Extracts user information from the server configuration
        4. Updates admin users list with authenticated user (ID and email)
        5. Creates admin context for internal operations

        Args:
            retry_budget_seconds: How long to keep retrying a connection-level
                failure. Defaults to the reconnect budget so the monitoring
                loop's repair fits inside one pass.

        Raises:
            ConnectionError: If unable to connect to Hypha server
            AuthenticationError: If token authentication fails
            ValueError: If server configuration is invalid
        """
        if self.server:
            self.logger.debug("Closing existing Hypha server connection")
            try:
                await asyncio.wait_for(
                    self.server.disconnect(), timeout=_DISCONNECT_TIMEOUT_S
                )
            except Exception as e:
                self.logger.error(f"Error closing Hypha server connection: {e}")

        self.logger.info(f"Connecting to Hypha server at '{self.server_url}'...")
        self.server = await connect_with_retry(
            lambda: connect_to_server(
                {
                    "server_url": self.server_url,
                    "token": self._token,
                    "workspace": self.workspace,
                    "client_id": self.client_id,
                }
            ),
            description=f"Connection to Hypha server at '{self.server_url}'",
            logger=self.logger,
            total_seconds=retry_budget_seconds,
        )

        # Check if provided token has admin permission level to generate new tokens
        try:
            await self.server.generate_token()
        except Exception as e:
            if "Only admin can generate token" in str(e):
                raise ValueError("Provided token does not have admin permissions.")
            else:
                raise e

        # Update connection configuration from server response
        if self.workspace and self.workspace != self.server.config.workspace:
            raise ValueError(
                f"Workspace mismatch: {self.workspace} (local) vs {self.server.config.workspace} (server)"
            )
        self.workspace = self.server.config.workspace
        if self.client_id and self.client_id != self.server.config.client_id:
            raise ValueError(
                f"Client ID mismatch: {self.client_id} (local) vs {self.server.config.client_id} (server)"
            )
        self.client_id = self.server.config.client_id

        self.full_service_id = f"{self.workspace}/{self.client_id}:{self.service_id}"

        # Extract authenticated user information
        user_id = self.server.config.user["id"]
        user_email = self.server.config.user["email"]

        self.logger.info(
            f"User '{user_id}' ({user_email}) connected as client "
            f"'{self.client_id}' to workspace '{self.workspace}' on server '{self.server_url}'."
        )

        # Inject only the email — generate_token issues a fresh haikunated
        # `sub` per token, so adding user_id accumulates a new entry on
        # every reconnect / 3-hour refresh and never matches a human admin.
        if user_email in self.admin_users:
            self.admin_users.remove(user_email)
        self.admin_users.insert(0, user_email)
        self._worker_user_email = user_email

        # Create admin context for internal operations
        self._admin_context = create_context(user_id, user_email)

        # Pass server connection and admin users to component managers
        await self.apps_manager.complete_initialization(
            server=self.server,
            admin_users=self.admin_users,
            worker_service_id=self.full_service_id,
        )
        await self.code_executor.initialize(admin_users=self.admin_users)

        self.logger.info(
            f"Admin users for this BioEngine worker: {', '.join(self.admin_users)}"
        )

    async def _register_bioengine_worker_service(self) -> None:
        # Register service interface
        description = "Manages BioEngine Apps and Datasets"
        if self.ray_cluster.mode == "slurm":
            description += " on a HPC system with Ray Autoscaler support for dynamic resource management."
        elif self.ray_cluster.mode == "single-machine":
            description += " on a single machine Ray instance."
        else:
            description += " in a pre-existing Ray environment."

        worker_services = {
            # 🧩 Worker management
            "get_status": self.get_status,
            "stop_worker": self.stop,  # Requires admin permissions
            "check_access": self.check_access,
            "get_logs": self.get_logs,  # Requires admin permissions
            "list_admin_users": self.list_admin_users,  # Requires admin permissions
            "add_admin_user": self.add_admin_user,  # Requires admin permissions
            "remove_admin_user": self.remove_admin_user,  # Requires admin permissions
            # 📦 Dataset management
            "list_datasets": self.list_datasets,
            # 🧮 Code execution
            "run_code": self.code_executor.run_code,  # Requires admin permissions
            # 🚀 Application management
            "upload_app": self.apps_manager.upload_app,  # Admin required unless workspace+hypha_token provided
            "list_app_directories": self.apps_manager.list_app_directories,  # Requires admin permissions
            "clear_app_directory": self.apps_manager.clear_app_directory,  # Requires admin permissions
            "list_apps": self.apps_manager.list_apps,  # Requires admin permissions
            "get_app_manifest": self.apps_manager.get_app_manifest,  # Requires admin permissions
            "delete_app": self.apps_manager.delete_app,  # Requires admin permissions
            "delete_app_version": self.apps_manager.delete_app_version,  # Requires admin permissions; only 'dev' pre-releases
            "deploy_app": self.apps_manager.deploy_app,  # Requires admin permissions
            "stop_app": self.apps_manager.stop_app,  # Requires admin permissions
            "stop_all_apps": self.apps_manager.stop_all_apps,  # Requires admin permissions
            "get_app_status": self.apps_manager.get_app_status,
        }

        if self.enable_access_requests:
            # Registered only when enabled, so with the toggle off there is no
            # request surface to reach rather than one that answers with a
            # refusal. 'request_admin_access' is public by design.
            worker_services.update(
                {
                    "request_admin_access": self.request_admin_access,
                    "get_admin_access_request": self.get_admin_access_request,
                    "list_access_requests": self.list_access_requests,  # Requires admin permissions
                    "resolve_access_request": self.resolve_access_request,  # Requires admin permissions
                }
            )

        # TODO: return more informative error messages, e.g. include traceback
        service_info = await self.server.register_service(
            {
                "id": self.service_id,
                "name": self.worker_name,
                "type": "bioengine-worker",
                "description": description,
                "config": {
                    "visibility": "public",
                    "require_context": True,
                },
                **worker_services,
            }
        )

        if self.full_service_id != service_info.id:
            raise ValueError(
                f"Service ID mismatch: {self.full_service_id} (expected) vs {service_info.id} (registered)"
            )

    async def _probe_service_registration(self) -> Optional[Exception]:
        """Ask Hypha whether it still serves this worker's own service id.

        Returns the failure instead of raising it: both callers below decide
        what a failure means from where they are in the repair, not from the
        exception.
        """
        try:
            await asyncio.wait_for(
                self.server.get_service_info(self.full_service_id),
                timeout=_REGISTRATION_PROBE_TIMEOUT_S,
            )
            return None
        except Exception as exc:
            return exc

    def _absorb_or_condemn_registration(self, error: Exception, log) -> None:
        unreachable_for = time.time() - self._registration_ok_at
        if unreachable_for >= _REGISTRATION_GRACE_S:
            raise RuntimeError(
                f"Hypha service '{self.full_service_id}' has been unreachable "
                f"for {unreachable_for:.0f}s and cannot be restored: {error}"
            ) from error
        log(
            f"Failed to restore the Hypha service registration, retrying next "
            f"tick: {error}"
        )

    async def _check_service_registration(self) -> None:
        """Keep this worker's Hypha service reachable, and rebuild the client
        when it is not.

        This is the only Hypha liveness check in the monitoring loop. It
        subsumes a plain ``echo`` probe: anything that kills the socket also
        fails this one, while the reverse is not true. A freeze long enough for
        Hypha to evict this client leaves the socket open from our side, so
        ``echo`` keeps answering while
        ``<workspace>/<client-id>:bioengine-worker`` has stopped resolving for
        everyone else. Asking whether our own registration is still served is
        the only signal that separates the two.

        The repair is always a full rebuild — disconnect, reconnect, register.
        Re-registering on the client we already hold looks cheaper and cannot
        work: ``hypha_rpc`` keeps a local service registry that a server-side
        eviction never touches, so ``register_service`` on that client fails
        with *Service already exists* however the server feels about us. A
        fresh client starts with an empty registry, and is also the only thing
        that can restore an evicted *client* rather than an evicted service.

        A repair counts only once the probe answers again. Reconnecting and
        registering can both succeed locally while the service stays
        unresolvable; believing that would refill the grace clock every tick
        and make the escalation below unreachable.

        While healthy the probe costs one round trip per
        ``_REGISTRATION_PROBE_INTERVAL_S``. While failing the throttle is
        dropped and it retries every tick, so a dead registration is detected
        within one probe interval and then repaired at the monitoring interval
        rather than once a minute.

        Transient Hypha outages must not condemn the worker — Hypha can serve
        nothing for minutes and recover on its own — so failures are absorbed
        silently for ``_REGISTRATION_GRACE_S``. Past that this raises, which
        feeds the monitoring loop's degraded counter; at the degraded threshold
        ``get_status`` reports the worker not-ready.

        Nothing acts on that report today. The deployed liveness probe is a
        local PID check that never reads readiness, and ``get_status`` is
        served by the very registration that has been evicted, so in the case
        this check exists for there is no one left to ask. Escalating is
        therefore a log line and a status flag, not a restart; the restart
        needs an external consumer that does not yet exist.

        The grace is a deadline, not a budget, and it measures **continuous**
        unreachability: any successful probe resets it. A flapping connection
        therefore never condemns the worker, which is deliberate — a probe that
        answers means the service really was resolvable at that instant, and
        every other check in the monitoring loop resets on a clean tick too.
        Counting attempts instead would be strictly worse: a flap refills an
        attempt budget faster than it can be spent, so the escalation would be
        unreachable exactly when the connection is worst.

        That leaves one thing invisible, so it is reported separately: a
        registration that answers one probe in ten never condemns and never
        produces a line anyone reads. The flap window below counts failure
        *episodes* on a clock that does *not* reset on success, and only ever
        warns.
        """
        if not self.server or not self.full_service_id:
            return

        now = time.time()
        # Only throttle while healthy; a failing registration is retried on
        # every tick, which is also what feeds the degraded counter.
        if not self._registration_failing and now < self._registration_probe_due_at:
            return
        self._registration_probe_due_at = now + _REGISTRATION_PROBE_INTERVAL_S

        if now - self._registration_window_start >= _REGISTRATION_FLAP_WINDOW_S:
            if self._registration_window_failures >= _REGISTRATION_FLAP_THRESHOLD:
                self.logger.warning(
                    f"Hypha service registration for '{self.full_service_id}' "
                    f"failed {self._registration_window_failures} times in the "
                    f"last {(now - self._registration_window_start) / 60:.0f} "
                    f"minutes but recovered each time. The connection is "
                    f"flapping; this never reaches the degraded threshold "
                    f"because every recovery resets it."
                )
            self._registration_window_start = now
            self._registration_window_failures = 0

        probe_error = await self._probe_service_registration()
        if probe_error is None:
            if self._registration_failing:
                self.logger.info(
                    f"Hypha serves '{self.full_service_id}' again after "
                    f"{now - self._registration_ok_at:.0f}s."
                )
            self._registration_failing = False
            self._registration_ok_at = now
            return

        first_failure = not self._registration_failing
        self._registration_failing = True
        if first_failure:
            self._registration_window_failures += 1

        # Only the first failure of a streak is a warning. A Hypha outage is
        # retried every tick, and logging each attempt would bury the recovery.
        log = self.logger.warning if first_failure else self.logger.debug
        log(
            f"Hypha no longer serves '{self.full_service_id}' ({probe_error}). "
            "Rebuilding the connection and re-registering..."
        )
        try:
            await self._connect_to_server()
            await self._register_bioengine_worker_service()
        except Exception as repair_error:
            self._absorb_or_condemn_registration(repair_error, log)
            return

        probe_error = await self._probe_service_registration()
        if probe_error is not None:
            self._absorb_or_condemn_registration(probe_error, log)
            return

        self._registration_failing = False
        self._registration_ok_at = time.time()
        self.logger.info(
            f"Re-registered BioEngine worker service '{self.full_service_id}'."
        )

    async def _check_token_expiry(self) -> None:
        if self._token_expires_at - time.time() < 3600:
            # Renew token if it's about to expire
            self.logger.info(
                "Generating a new Hypha token with expiration time set to 3 hours."
            )
            self._token = await self.server.generate_token(
                {
                    "workspace": self.workspace,
                    "client_id": self.client_id,
                    "permission": "admin",
                    "expires_in": 3600 * 3,
                }
            )

            user_info = await self.server.parse_token(self._token)
            self._token_expires_at = user_info.expires_at

    async def _cleanup(self) -> None:
        """
        Perform comprehensive cleanup of all BioEngine worker components.

        This method handles cleanup in a robust manner, ensuring that each component
        is properly cleaned up regardless of the current state of the worker. It
        handles cases where components may not be initialized, may have failed
        during startup, or may already be in the process of shutting down.

        The cleanup process:
        1. Unregister service from Hypha server (if registered)
        2. Clean up dataset manager (if initialized)
        3. Clean up apps manager (if initialized)
        4. Stop Ray cluster (if initialized and ready)
        5. Disconnect from Hypha server (if connected)
        6. Reset worker state variables

        All cleanup operations are wrapped in try-catch blocks to ensure
        that failure in one component doesn't prevent cleanup of others.
        """
        self.logger.info("Starting cleanup of BioEngine worker...")
        cleanup_start_time = time.time()

        # Ensure the is_ready event is reset to prevent new operations
        if hasattr(self, "is_ready"):
            self.is_ready.clear()

        # Clean up apps manager. In external-cluster mode the Ray Serve
        # deployments live on the shared KubeRay cluster and are supposed
        # to survive the worker pod restart — the next worker's
        # recover_deployed_applications adopts them. Calling
        # stop_all_apps here would run serve.delete on each of them and
        # defeat that mechanism, so we skip it. In single-machine and
        # SLURM modes the head Ray process is torn down by the worker
        # itself (see RayCluster.stop), so the apps die regardless;
        # letting stop_all_apps unregister cleanly first is just hygiene.
        ray_mode = getattr(getattr(self, "ray_cluster", None), "mode", None)
        if hasattr(self, "apps_manager") and self.apps_manager:
            self.apps_manager.cancel_startup_retry()
            if ray_mode == "external-cluster":
                self.logger.info(
                    "External-cluster mode: leaving deployed apps in place on the "
                    "shared Ray cluster so the next worker can recover them."
                )
            else:
                try:
                    admin_context = getattr(self, "_admin_context", None)
                    if admin_context is None:
                        self.logger.debug(
                            "No admin context (shutdown before Hypha login completed); "
                            "skipping apps manager cleanup."
                        )
                    else:
                        # stop_all_apps is a @schema_method whose first positional
                        # parameter is timeout_seconds; the required context must
                        # be passed by name.
                        await self.apps_manager.stop_all_apps(context=admin_context)
                except Exception as e:
                    self.logger.error(f"Error cleaning up apps manager: {e}")

        # Stop Ray cluster
        if hasattr(self, "ray_cluster") and self.ray_cluster:
            try:
                # Check if Ray cluster is ready before attempting to stop
                if (
                    hasattr(self.ray_cluster, "is_ready")
                    and self.ray_cluster.is_ready.is_set()
                ):
                    await self.ray_cluster.stop()
            except Exception as e:
                self.logger.error(f"Error stopping Ray cluster: {e}")

        # Disconnect from the Hypha server
        if hasattr(self, "_server") and self.server:
            try:
                self.logger.info("Disconnecting from Hypha server...")
                await self.server.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting from Hypha server: {e}")

        # Reset worker state variables
        try:
            self.start_time = None
            self._monitoring_task = None
            if hasattr(self, "full_service_id"):
                self.full_service_id = None
            if hasattr(self, "_admin_context"):
                self._admin_context = None
        except Exception as e:
            self.logger.error(f"Error resetting worker state: {e}")

        duration = time.time() - cleanup_start_time
        self.logger.info(
            f"BioEngine worker cleanup completed in {duration:.2f} seconds."
        )

        # Signal that the worker has completed cleanup
        self._shutdown_event.set()

    def _touch_heartbeat(self, stale_after_seconds: float) -> None:
        """Write the liveness heartbeat, overwriting any previous run's."""
        try:
            write_heartbeat(self.heartbeat_file, stale_after_seconds)
        except Exception as e:
            self.logger.error(
                f"Failed to write heartbeat file {self.heartbeat_file}: {e}. "
                f"The liveness probe reads this file, so it will report this "
                f"worker dead until the path is writable."
            )

    async def _create_monitoring_task(
        self,
        backoff_initial_seconds: float = 2.0,
        backoff_max_seconds: float = 60.0,
    ) -> None:
        """Continuously monitor cluster status and update worker nodes history.

        This loop runs while the cluster is active, periodically collecting
        cluster status information and updating the history. Errors raised by
        any monitoring step (Hypha reconnect, Ray client, Serve status, apps
        manager, …) are treated as transient: the loop logs, sleeps with
        exponential backoff capped at ``backoff_max_seconds``, then retries.
        It never self-terminates the worker — only an explicit ``stop()`` /
        OS signal does. Hypha's own websocket reconnect logic recovers
        transparently once the server returns, so a multi-hour outage no
        longer brings every BioEngine worker down with it.

        Args:
            backoff_initial_seconds: Base for the exponential backoff after
                the first error (doubles each subsequent consecutive error
                until ``backoff_max_seconds``).
            backoff_max_seconds: Cap on the per-error sleep so a stuck
                worker still polls roughly once a minute. Also widens the
                heartbeat deadline, so a loop that is failing but still
                cycling is never mistaken for a frozen one.

        Raises:
            Exception: Only on unexpected errors outside the monitoring
                step (e.g. ``is_ready`` initialisation). Errors inside a
                monitoring tick are handled and retried.
        """
        try:
            # Start the registration grace clock here rather than at
            # construction: startup takes minutes, so a clock started before
            # the first probe can run is already spent when it does.
            self._registration_ok_at = time.time()
            self._registration_window_start = self._registration_ok_at

            # Signal that the worker is ready
            self.is_ready.set()

            self.logger.debug(
                "Starting monitoring task with interval "
                f"{self.monitoring_interval_seconds} seconds..."
            )
            stale_after_seconds = heartbeat_stale_after_seconds(
                self.monitoring_interval_seconds, backoff_max_seconds
            )
            self.logger.info(
                f"Writing liveness heartbeat to {self.heartbeat_file} after every "
                f"completed monitoring pass (stale after {stale_after_seconds:.0f}s)"
            )
            # Same quantity, different reader: the host census ages a worker
            # out on exactly the gap that marks it dead to the liveness probe.
            self.ray_cluster.census_stale_after_seconds = stale_after_seconds
            self._monitor_consecutive_errors = 0
            while self.is_ready.is_set():
                try:
                    # Sleep for 1 second before next iteration
                    await asyncio.sleep(1)

                    # Check if enough time has passed since the last monitoring
                    current_time = time.time()
                    if (
                        current_time - self._last_monitoring
                        < self.monitoring_interval_seconds
                    ):
                        continue  # Skip if within check interval

                    self._last_monitoring = current_time

                    # Each step runs independently so an earlier failure (e.g.
                    # a stale proxy handle in cluster monitoring) can't skip
                    # application monitoring, which fires auto-redeploy.
                    # Failures are re-raised at tick end so backoff still runs.
                    step_errors: List[tuple] = []

                    async def _step(name, coro):
                        try:
                            await coro
                        except asyncio.CancelledError:
                            raise
                        except Exception as step_exc:
                            step_errors.append((name, step_exc))
                            self.logger.warning(
                                f"Monitoring step '{name}' failed: {step_exc}",
                                exc_info=True,
                            )

                    # ===== 0. Geo location =====
                    await _step("geo_location", self._fetch_geo_location())

                    # ===== 1. Hypha server connection check =====
                    await _step(
                        "service_registration", self._check_service_registration()
                    )
                    await _step("token_expiry", self._check_token_expiry())

                    # ===== 2. Ray cluster monitoring =====
                    await _step(
                        "ray_check_connection", self.ray_cluster.check_connection()
                    )
                    await _step(
                        "cluster_monitoring", self.ray_cluster.monitor_cluster()
                    )

                    # ===== 3. Data server monitoring =====
                    await _step("data_server_ping", self._ping_data_server())
                    await _step("data_server_discover", self._discover_data_server())
                    await _step("dataset_refresh", self._refresh_datasets())

                    # ===== 4. Applications monitoring (auto-redeploy) =====
                    await _step(
                        "applications_monitoring",
                        self.apps_manager.monitor_applications(),
                    )

                    # Every step has returned control to the loop, so the loop
                    # is alive even if some of them failed. Deliberately
                    # outside the _step try/except and before the re-raise
                    # below: a step that never returns must stop the beat.
                    self._touch_heartbeat(stale_after_seconds)

                    if step_errors:
                        names = ", ".join(name for name, _ in step_errors)
                        raise RuntimeError(
                            f"{len(step_errors)} monitoring step(s) failed this "
                            f"tick: {names}"
                        )

                    # Run BioEngine Datasets monitoring
                    # await self.dataset_manager.monitor_datasets()

                    # Log recovery after a streak of failures
                    if self._monitor_consecutive_errors > 0:
                        self.logger.info(
                            f"Monitoring loop recovered after "
                            f"{self._monitor_consecutive_errors} consecutive error(s)"
                        )
                    self._monitor_consecutive_errors = 0

                except asyncio.CancelledError:
                    # Propagate so the outer handler performs clean shutdown.
                    raise
                except Exception as e:
                    self._monitor_consecutive_errors += 1
                    # Exponential backoff capped at backoff_max_seconds.
                    backoff = min(
                        backoff_max_seconds,
                        backoff_initial_seconds
                        * (2 ** (self._monitor_consecutive_errors - 1)),
                    )
                    log = (
                        self.logger.warning
                        if self._monitor_consecutive_errors <= 5
                        else self.logger.error
                    )
                    log(
                        f"Error in monitoring task "
                        f"(#{self._monitor_consecutive_errors} consecutive, "
                        f"sleeping {backoff:.0f}s before retry): {e}"
                    )
                    if (
                        self._monitor_consecutive_errors
                        == self._monitor_degraded_threshold
                    ):
                        self.logger.error(
                            f"Monitoring degraded for "
                            f"{self._monitor_consecutive_errors} consecutive ticks — "
                            f"get_status now reports the worker not-ready. The loop "
                            f"keeps retrying; nothing restarts it automatically."
                        )
                    await asyncio.sleep(backoff)

        except asyncio.CancelledError:
            self.logger.info("Monitoring task cancelled.")
        except Exception as e:
            self.logger.error(f"Unexpected error in monitoring task: {e}")
            raise
        finally:
            # Always perform comprehensive cleanup of all components
            await self._cleanup()

    async def _stop(self, blocking: bool = False) -> None:
        try:
            # Check if the worker is running
            if self._shutdown_event.is_set():
                self.logger.info("BioEngine worker is not running. Nothing to stop.")
                return

            if self._monitoring_task and not self._monitoring_task.done():
                # If monitoring task is running, cancel it and wait for graceful shutdown
                self._monitoring_task.cancel()
            else:
                # Fallback cleanup if monitoring task is not running
                asyncio.create_task(self._cleanup())

            msg = "Initiated graceful shutdown of BioEngine worker. "
            if blocking:
                self.logger.info(msg + "Waiting for graceful shutdown to complete...")
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.graceful_shutdown_timeout,
                    )
                    self.logger.info("Graceful shutdown completed successfully.")
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Graceful shutdown timed out after {self.graceful_shutdown_timeout} seconds. "
                        "Force exiting without complete cleanup."
                    )
                    # Force exit immediately with code 1 because of timeout
                    os._exit(1)
            else:
                self.logger.info(msg + "Shutdown will happen in the background.")

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.logger.info(
                "Shutdown signal received during cleanup. Force exiting without complete cleanup."
            )
            # Force exit immediately with code 0 to indicate successful shutdown
            os._exit(0)

    async def start(self, blocking: bool = True) -> str:
        """
        Start the BioEngine worker and all component services.

        Initializes the Ray cluster (or connects to existing one), establishes
        connection to the Hypha server, registers the service interface, and
        starts all component managers. Also deploys any configured startup
        deployments.

        Returns:
            str: Service ID assigned after successful registration with Hypha server.

        Raises:
            RuntimeError: If Ray is already initialized when connecting to existing cluster
            Exception: If startup of any component fails
        """
        try:
            # Reset the shutdown event to allow starting the worker
            self._shutdown_event.clear()

            # Claim the file before startup begins: a previous run's heartbeat
            # must not read as fresh, and a missing file reads as dead.
            self._touch_heartbeat(STARTUP_STALE_AFTER_SECONDS)

            # Set the start time for monitoring and uptime tracking
            self.start_time = time.time()

            # Start the Ray cluster
            await self.ray_cluster.start()

            # Connect the BioEngine worker to the Hypha server
            # Completes initialization of AppsManager and CodeExecutor
            await self._connect_to_server(STARTUP_CONNECT_BUDGET_S)

            # Check for running data server
            await self._discover_data_server()

            # Load available datasets from the data server
            await self._refresh_datasets()

            # Recover applications already running in Ray Serve before deploying startup apps
            await self.apps_manager.recover_deployed_applications()

            # Deploy startup applications
            await self.apps_manager.deploy_startup_applications()

            # Register the BioEngine worker service interface
            await self._register_bioengine_worker_service()

            # Start the monitoring task
            self._monitoring_task = asyncio.create_task(
                self._create_monitoring_task(),
                name="BioEngineWorker Monitoring Task",
            )

            # Wait for the monitoring task to signal readiness
            await self.is_ready.wait()

            self.logger.info(
                f"Manage BioEngine worker at: {self.dashboard_url}/worker?service_id={self.full_service_id}"
            )

            if blocking is True:
                # Keep the worker running until a shutdown signal is received
                self.logger.info(
                    "BioEngine worker will run until shutdown signal is received. "
                    "Press Ctrl+C for graceful shutdown. Press Ctrl+C again to force exit."
                )

                # Wait for the monitoring task to complete (which happens during shutdown)
                await self._shutdown_event.wait()

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.logger.info("Shutdown signal received.")
            await self._stop(blocking=True)
        except Exception as e:
            self.logger.error(f"Failed to start BioEngine worker: {e}")
            await self._stop(blocking=True)
            raise

        return self.full_service_id

    @schema_method
    async def list_datasets(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> None:
        """List available datasets from connected BioEngine data server."""
        # No permission check needed for listing datasets
        return self.available_datasets

    @schema_method
    async def check_access(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> bool:
        """Check if a user is in the admin users list."""
        try:
            check_permissions(
                context=context,
                authorized_users=self.admin_users,
                resource_name="accessing BioEngine Worker",
            )
            return True
        except PermissionError:
            return False

    @schema_method
    async def get_logs(
        self,
        tail: int = Field(
            100,
            description="Number of lines to retrieve from the end of the log file. Default is 100 lines.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> List[str]:
        """Retrieve the worker logs from the current log file.

        This method reads and returns the contents of the worker's log file. Only users with
        admin permissions can access the logs.

        Args:
            tail: Number of lines to retrieve from the end of the log file.
                  Set to None or 0 to retrieve the entire file.
            context: Authentication context (automatically provided by Hypha)

        Returns:
            List[str]: List of log entry lines (last tail lines)

        Raises:
            PermissionError: If the user is not authorized to access logs
            RuntimeError: If log file is not enabled or cannot be read
        """
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="accessing BioEngine Worker logs",
        )

        if not self.log_file:
            raise RuntimeError(
                "Log file is not enabled. The worker was started with log_file='off'. "
                "Cannot retrieve logs."
            )

        if not self.log_file.exists():
            raise RuntimeError(
                f"Log file does not exist: {self.log_file}. "
                "The file may have been deleted or not yet created."
            )

        try:
            with open(self.log_file, "r") as f:
                # Read all lines and return the last tail lines
                lines = f.readlines()
                return lines[-tail:]
        except Exception as e:
            raise RuntimeError(f"Failed to read log file {self.log_file}: {e}")

    @schema_method
    async def list_admin_users(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> List[str]:
        """List the users with admin permissions on this BioEngine worker.

        Requires admin permissions.

        Returns:
            List[str]: The current admin users, most recently connected first.
        """
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="listing BioEngine Worker admin users",
        )
        return list(self.admin_users)

    @schema_method
    async def add_admin_user(
        self,
        user: str = Field(
            ...,
            description="User ID or email address to grant admin permissions on this worker.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> List[str]:
        """Grant a user admin permissions on this BioEngine worker.

        Takes effect immediately for every worker method — permissions are checked
        per call — and survives a restart. Already-running applications keep the
        authorized users they were deployed with; redeploy them to widen access.

        Requires the caller to be a named admin user — being covered by a '*'
        entry is not enough.

        Args:
            user: User ID or email address to grant admin permissions to.
            context: Authentication context (automatically provided by Hypha)

        Returns:
            List[str]: The updated admin users.

        Raises:
            PermissionError: If the caller is not a named admin user.
            ValueError: If the user identifier is empty or the wildcard '*'.
        """
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="adding a BioEngine Worker admin user",
            # A caller who is only an admin through '*' must not be able to turn
            # that into a named grant the startup flag can no longer revoke.
            allow_wildcard=False,
        )

        user = user.strip()
        if not user:
            raise ValueError("Admin user identifier must not be empty.")
        if user == "*":
            # Only the operator starting the worker gets to make it world-writable.
            raise ValueError(
                "Refusing to add the wildcard '*' as an admin user. Pass "
                "--admin-users '*' at worker startup if that is intended."
            )

        if user not in self.admin_users:
            self._persist_admin_users(self.admin_users + [user])
            self.admin_users.append(user)
            self.logger.info(
                f"Added admin user '{user}' (requested by "
                f"'{context['user'].get('email') or context['user'].get('id')}'). "
                f"Admin users: {', '.join(self.admin_users)}"
            )

        return list(self.admin_users)

    @schema_method
    async def remove_admin_user(
        self,
        user: str = Field(
            ...,
            description="User ID or email address to revoke admin permissions from on this worker.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> List[str]:
        """Revoke a user's admin permissions on this BioEngine worker.

        Takes effect immediately and survives a restart. Removing a user who is
        not an admin is a no-op.

        Requires the caller to be a named admin user — being covered by a '*'
        entry is not enough.

        Args:
            user: User ID or email address to revoke admin permissions from.
            context: Authentication context (automatically provided by Hypha)

        Returns:
            List[str]: The updated admin users.

        Raises:
            PermissionError: If the caller is not a named admin user.
            ValueError: If the user is one the worker was started with, or if the
                        removal would lock the caller out, remove the last admin,
                        or drop the worker's own identity.
        """
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="removing a BioEngine Worker admin user",
            # Revoking '*' would leave the caller as the only admin, so the same
            # named-admin requirement as add_admin_user applies.
            allow_wildcard=False,
        )

        user = user.strip()
        caller = context["user"]
        if user and user in self._founding_admin_users:
            # Checked before every other guard: this is the one removal no
            # caller may make, and the persisted overlay wins over the startup
            # seed, so allowing it would be a lockout no re-roll undoes.
            raise ValueError(
                f"Refusing to remove '{user}': the worker was started with this user "
                "and a starting user cannot lose admin permissions. Demoting it takes "
                "two steps: drop it from --admin-users and restart, which stops it "
                f"being a starting user, then call remove_admin_user('{user}') again — "
                "the restart alone does not revoke it, because the persisted admin list "
                "overrides the startup flag and still carries it."
            )
        if user in self.admin_users and len(self.admin_users) == 1:
            raise ValueError(
                f"Refusing to remove '{user}': it is the last admin user and the "
                "worker would be left with none."
            )
        if user and user in (caller.get("id"), caller.get("email")):
            raise ValueError(
                f"Refusing to remove '{user}': a caller cannot revoke their own "
                "admin permissions."
            )
        if user == self._worker_user_email:
            # _connect_to_server re-inserts this on every reconnect, so removing
            # it would revert silently instead of failing here.
            raise ValueError(
                f"Refusing to remove '{user}': it is the identity this worker "
                "connects to Hypha with and would be restored on the next reconnect."
            )

        if user in self.admin_users:
            self._persist_admin_users([u for u in self.admin_users if u != user])
            self.admin_users.remove(user)
            self.logger.info(
                f"Removed admin user '{user}' (requested by "
                f"'{caller.get('email') or caller.get('id')}'). "
                f"Admin users: {', '.join(self.admin_users)}"
            )

        return list(self.admin_users)

    @schema_method
    async def request_admin_access(
        self,
        reason: str = Field(
            "",
            description="Optional note telling the worker's admins who you are and why you need access.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> Dict[str, Any]:
        """Ask the admins of this worker to grant you admin permissions.

        Deliberately public: any logged-in caller may submit, admin or not. Only
        available when the worker was started with access requests enabled.

        One request per account, keyed on the caller's email address. A second
        request is refused rather than queued or silently replacing the first, so
        a decision already taken cannot be reset by asking again — including a
        denial, which only an admin can clear.

        Args:
            reason: Free-text note shown to the admins alongside the request.
            context: Authentication context (automatically provided by Hypha)

        Returns:
            Dict: The recorded request.

        Raises:
            RuntimeError: If access requests are disabled on this worker.
            PermissionError: If the caller is not authenticated.
            ValueError: If the caller is already an admin or already has a request.
        """
        self._require_access_requests_enabled()
        key, email = self._requester_identity(context)

        if self._is_admin_user(email):
            raise ValueError(
                f"'{email}' is already an admin of this worker; there is nothing to request."
            )

        existing = self._access_requests.get(key)
        if existing:
            raise ValueError(
                f"'{email}' already has an access request on this worker with status "
                f"'{existing.get('status')}'. Only an admin can clear it."
            )

        request = {
            # The email as Hypha reports it, which is what a grant has to add to
            # the admin list; the dict key is its lowercased form.
            "email": email,
            "user_id": context["user"].get("id"),
            "status": "pending",
            "reason": str(reason or "").strip()[:500],
            "requested_at": time.time(),
            "resolved_at": None,
            "resolved_by": None,
        }
        self._persist_access_requests({**self._access_requests, key: request})
        self._access_requests[key] = request
        self.logger.info(
            f"Access request from '{email}' is pending. Grant it with "
            f"resolve_access_request(user='{email}', decision='grant')."
        )
        return dict(request)

    @schema_method
    async def get_admin_access_request(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> Optional[Dict[str, Any]]:
        """Read the state of your own access request on this worker.

        Public, and scoped to the caller: it reads the request belonging to the
        caller's email and no other. A request nobody can observe is
        indistinguishable from one that was dropped.

        Returns:
            Optional[Dict]: The caller's request, or None if they have none.
        """
        self._require_access_requests_enabled()
        key, _email = self._requester_identity(context)
        request = self._access_requests.get(key)
        return dict(request) if request else None

    @schema_method
    async def list_access_requests(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> List[Dict[str, Any]]:
        """List every access request on this worker, oldest first.

        Requires the caller to be a named admin user.
        """
        self._require_access_requests_enabled()
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="listing BioEngine Worker access requests",
            allow_wildcard=False,
        )
        return sorted(
            (dict(request) for request in self._access_requests.values()),
            key=lambda request: request.get("requested_at") or 0,
        )

    @schema_method
    async def resolve_access_request(
        self,
        user: str = Field(
            ...,
            description="Email address of the requester whose request is being decided.",
        ),
        decision: Literal["grant", "deny", "clear"] = Field(
            ...,
            description="'grant' makes the requester an admin, 'deny' refuses and blocks further requests, 'clear' deletes the record so they may ask again.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> Optional[Dict[str, Any]]:
        """Grant, deny or clear a pending access request.

        Requires the caller to be a named admin user — being covered by a '*'
        entry is not enough, for the same reason add_admin_user requires it: a
        caller who is only an admin through the wildcard must not be able to turn
        that into a named grant.

        'grant' adds the requester to the admin users, with immediate effect and
        surviving a restart. 'deny' records the refusal and leaves it in place, so
        the requester cannot re-file. 'clear' removes the record entirely, which is
        how a denial is lifted.

        Args:
            user: Email address of the requester.
            decision: 'grant', 'deny' or 'clear'.
            context: Authentication context (automatically provided by Hypha)

        Returns:
            Optional[Dict]: The updated request, or None after 'clear'.

        Raises:
            PermissionError: If the caller is not a named admin user.
            ValueError: If there is no request for that user, or the decision is unknown.
        """
        self._require_access_requests_enabled()
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="resolving a BioEngine Worker access request",
            allow_wildcard=False,
        )

        if decision not in _ACCESS_REQUEST_DECISIONS:
            # The Literal annotation only documents the schema; a direct call
            # would otherwise fall through to the 'deny' branch.
            raise ValueError(
                f"Unknown decision '{decision}'; expected one of "
                f"{', '.join(_ACCESS_REQUEST_DECISIONS)}."
            )

        # Requests are keyed on the lowercased email, so an admin may name the
        # requester in any case and still reach the record.
        key = (user or "").strip().lower()
        request = self._access_requests.get(key)
        if not request:
            raise ValueError(f"No access request on this worker for '{key}'.")

        caller = context["user"]
        resolved_by = caller.get("email") or caller.get("id")

        if decision == "clear":
            remaining = {
                stored_key: value
                for stored_key, value in self._access_requests.items()
                if stored_key != key
            }
            self._persist_access_requests(remaining)
            self._access_requests.pop(key, None)
            self.logger.info(
                f"Cleared the access request for '{key}' (by '{resolved_by}'). "
                "They may request access again."
            )
            return None

        if decision == "grant":
            # The stored email, not the key: check_permissions compares emails
            # case-sensitively, so granting the lowercased form would add an
            # entry the caller's own context never matches.
            # Reuse the guarded path rather than touching admin_users here: it is
            # what refuses the wildcard and writes the list before it takes effect.
            await self.add_admin_user(user=request["email"], context=context)

        updated = {
            **request,
            "status": "granted" if decision == "grant" else "denied",
            "resolved_at": time.time(),
            "resolved_by": resolved_by,
        }
        self._persist_access_requests({**self._access_requests, key: updated})
        self._access_requests[key] = updated
        self.logger.info(
            f"Access request for '{request['email']}' was {updated['status']} "
            f"by '{resolved_by}'."
        )
        return dict(updated)

    @schema_method
    async def get_status(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive real-time status information for the BioEngine worker and all managed components.

        This method provides a complete overview of the worker's operational state including Ray cluster health, active deployments, loaded datasets, resource utilization, and service availability. Essential for monitoring, debugging, health checks, and dashboard displays.

        SECURITY: This method is publicly accessible and does not require admin permissions, making it suitable for monitoring dashboards and health checks by any caller, including an unauthenticated one. Authentication is optional and widens the response: `admin_users` is returned only to a worker admin, who is on that list already; a "*" entry does not earn it. See PUBLIC_WORKER_STATUS_FIELDS for the fields everyone receives.

        STATUS INFORMATION CATEGORIES:

        SERVICE METADATA:
        - service_start_time: Unix timestamp when the worker was initialized
        - service_uptime: Duration in seconds since worker startup
        - workspace: Hypha workspace name where worker is registered
        - client_id: Unique client identifier for this worker instance
        - admin_users: List of user identifiers with administrative privileges (worker admins only)
        - is_ready: Boolean indicating if worker is fully operational

        RAY CLUSTER STATUS:
        - worker_mode: Deployment mode (slurm/single-machine/external-cluster)
        - ray_cluster: Complete Ray cluster state including:
          * Available and total CPU/GPU/memory resources across all nodes
          * Node health status and IP addresses
          * Cluster connectivity and operational state
          * Resource utilization metrics

        RETURN VALUE STRUCTURE:
        {
            "service_start_time": 1234567890.123,
            "service_uptime": 3600.456,
            "bioengine_version": "0.9.1",
            "ray_version": "2.55.1",
            "hypha_rpc_version": "0.21.40",
            "worker_mode": "slurm",
            "workspace": "my-workspace",
            "client_id": "client-abc123",
            "ray_cluster": {...},
            "admin_users": ["user@example.com"],
            "is_ready": true
        }

        TYPICAL USAGE:
        Health monitoring: status = await worker.get_status()
        Dashboard display: Use all returned fields for comprehensive view
        Resource planning: Focus on ray_cluster resource information

        ERROR SCENARIOS:
        Returns partial status if individual components fail to report, with error information included in the component's status section.
        """
        current_time = time.time()
        status = {
            "service_start_time": self.start_time,
            "service_uptime": current_time - self.start_time if self.start_time else 0,
            "bioengine_version": __version__,
            "ray_version": ray.__version__,
            # Read from the installed distribution, not from a requirements
            # file: those record what an image was built from, while apps now
            # inherit whatever this process actually loaded.
            "hypha_rpc_version": importlib.metadata.version("hypha-rpc"),
            "worker_mode": self.ray_cluster.mode,
            "workspace": self.workspace,
            "client_id": self.client_id,
            "ray_cluster": self.ray_cluster.status,
            "admin_users": self.admin_users,
            "geo_location": self.geo_location,
            "is_ready": (
                self.is_ready.is_set()
                and self._monitor_consecutive_errors
                < self._monitor_degraded_threshold
            ),
        }

        # SLURM jobs — refresh from squeue at read time so the dashboard
        # never observes a stale snapshot from the monitor tick (jobs that
        # die quickly otherwise fall entirely between two ticks).
        if self.ray_cluster.slurm_workers is not None:
            try:
                status["ray_cluster"]["slurm_jobs"] = (
                    await self.ray_cluster.slurm_workers.get_job_status()
                )
            except Exception as e:
                self.logger.warning(f"Failed to refresh slurm_jobs for get_status: {e}")

        try:
            check_permissions(
                context=context,
                authorized_users=self.admin_users,
                resource_name="the admin users of the BioEngine worker",
                allow_wildcard=False,
            )
        except PermissionError:
            return public_worker_status(status)

        return status

    @schema_method
    async def stop(
        self,
        blocking: bool = Field(
            False,
            description="Whether to wait for complete shutdown before returning. Set to True to ensure all resources are fully cleaned up before the method returns, or False to initiate shutdown and return immediately while cleanup continues in background. Recommended: True for production environments, False for quick shutdown.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> None:
        """
        Gracefully shutdown the BioEngine worker with comprehensive resource cleanup and service deregistration.

        This method performs an orderly shutdown of all worker components including active deployments, dataset services, Ray cluster resources, and monitoring tasks. It ensures proper cleanup to prevent resource leaks and maintains system stability during shutdown operations.

        SECURITY: Requires admin-level permissions as this operation affects the entire worker instance and all active services.

        SHUTDOWN PROCESS:
        1. Permission validation for authorized shutdown access
        2. Signal monitoring tasks to stop and wait for completion
        3. Cleanup all active applications and deployments through AppsManager
        4. Close dataset connections and stop HTTP services through DatasetsManager
        5. Shutdown Ray cluster resources (if managed by this worker instance)
        6. Deregister services from Hypha server and disconnect
        7. Reset worker state and clear readiness indicators

        BLOCKING BEHAVIOR:
        - blocking=True: Method waits for all cleanup operations to complete before returning, ensuring complete shutdown
        - blocking=False: Method initiates shutdown process and returns immediately while cleanup continues asynchronously

        TIMEOUT HANDLING:
        Shutdown operations are subject to graceful_shutdown_timeout (default 60 seconds). If cleanup exceeds this timeout, the worker will force-exit to prevent hanging processes.

        ERROR HANDLING:
        Individual component cleanup failures are logged but don't prevent shutdown of other components. Critical errors during shutdown may result in force-exit to ensure the worker doesn't remain in an inconsistent state.

        TYPICAL USAGE:
        Production shutdown: await worker.stop(blocking=True)
        Development shutdown: await worker.stop(blocking=False)
        Emergency shutdown: await worker.stop(blocking=True) with shorter timeout

        SIDE EFFECTS:
        - All active deployments will be stopped and become unavailable
        - Dataset streaming services will be terminated
        - Ray cluster will be shutdown (if managed by this worker)
        - Worker service will be deregistered from Hypha server
        - All background monitoring tasks will be cancelled
        """
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="shutdown the BioEngine worker",
        )

        await self._stop(blocking=blocking)
