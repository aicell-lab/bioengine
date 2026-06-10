import asyncio
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
from bioengine.apps import AppsManager
from bioengine.datasets import BioEngineDatasets
from bioengine.cluster.ray_cluster import RayCluster
from bioengine.utils import (
    fetch_centroid_coordinates,
    fetch_geolocation,
    check_permissions,
    create_context,
    create_logger,
)
from bioengine.worker.code_executor import CodeExecutor


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
                        Auto-includes the authenticated user from Hypha connection.
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

        self.logger = create_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if debug else logging.INFO,
            log_file=self.log_file,
        )
        self.logger.info(
            f"Initializing {self.__class__.__name__} v{__version__} with mode '{mode}'"
        )

        # Hypha server configuration
        self.server_url = server_url
        self.server: Optional[RemoteService] = None
        self.workspace = workspace
        self._token = token or os.environ.get("HYPHA_TOKEN")
        self._token_expires_at = 0
        self.client_id = client_id
        self.service_id = "bioengine-worker"
        self.worker_name = worker_name

        # Worker state management
        self.start_time = None
        self._last_monitoring = 0

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

    async def _connect_to_server(self) -> None:
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

        Raises:
            ConnectionError: If unable to connect to Hypha server
            AuthenticationError: If token authentication fails
            ValueError: If server configuration is invalid
        """
        if self.server:
            self.logger.debug("Closing existing Hypha server connection")
            try:
                await self.server.disconnect()
            except Exception as e:
                self.logger.error(f"Error closing Hypha server connection: {e}")

        self.logger.info(f"Connecting to Hypha server at '{self.server_url}'...")
        self.server = await connect_to_server(
            {
                "server_url": self.server_url,
                "token": self._token,
                "workspace": self.workspace,
                "client_id": self.client_id,
            }
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

        # Update admin users list with authenticated user (ensure at top of list)
        if user_id in self.admin_users:
            self.admin_users.remove(user_id)
        if user_email in self.admin_users:
            self.admin_users.remove(user_email)
        self.admin_users.insert(0, user_id)
        self.admin_users.insert(1, user_email)

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
            "deploy_app": self.apps_manager.deploy_app,  # Requires admin permissions
            "stop_app": self.apps_manager.stop_app,  # Requires admin permissions
            "stop_all_apps": self.apps_manager.stop_all_apps,  # Requires admin permissions
            "get_app_status": self.apps_manager.get_app_status,
        }
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

    async def _check_hypha_connection(self, reconnect: bool = True) -> None:
        try:
            await asyncio.wait_for(self.server.echo("ping"), timeout=10)
        except Exception as e:
            if reconnect:
                self.logger.warning(
                    f"Hypha server connection error. Attempting to reconnect..."
                )
                await self._connect_to_server()
                await self._register_bioengine_worker_service()
            else:
                raise RuntimeError(f"Hypha server connection error: {e}")

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

        # Clean up apps manager
        if hasattr(self, "apps_manager") and self.apps_manager:
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
                worker still polls roughly once a minute.

        Raises:
            Exception: Only on unexpected errors outside the monitoring
                step (e.g. ``is_ready`` initialisation). Errors inside a
                monitoring tick are handled and retried.
        """
        try:
            # Signal that the worker is ready
            self.is_ready.set()

            self.logger.debug(
                "Starting monitoring task with interval "
                f"{self.monitoring_interval_seconds} seconds..."
            )
            consecutive_errors = 0
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

                    # ===== 0. Geo location =====
                    await self._fetch_geo_location()

                    # ===== 1. Hypha server connection check =====
                    # Check connection to Hypha server
                    await self._check_hypha_connection()

                    # Check token expiry
                    await self._check_token_expiry()

                    # ===== 2. Ray cluster monitoring =====
                    # Check connection to Ray cluster
                    await self.ray_cluster.check_connection()

                    # Run cluster monitoring
                    await self.ray_cluster.monitor_cluster()

                    # ===== 3. Data server monitoring =====

                    # Ping the data server to verify connectivity
                    await self._ping_data_server()

                    # If not connected to a data server, attempt discovery
                    await self._discover_data_server()

                    # Refresh datasets if connected to data server
                    await self._refresh_datasets()

                    # ===== 4. Applications monitoring =====

                    # Run BioEngine Applications monitoring
                    await self.apps_manager.monitor_applications()

                    # Run BioEngine Datasets monitoring
                    # await self.dataset_manager.monitor_datasets()

                    # Log recovery after a streak of failures
                    if consecutive_errors > 0:
                        self.logger.info(
                            f"Monitoring loop recovered after "
                            f"{consecutive_errors} consecutive error(s)"
                        )
                    consecutive_errors = 0

                except asyncio.CancelledError:
                    # Propagate so the outer handler performs clean shutdown.
                    raise
                except Exception as e:
                    consecutive_errors += 1
                    # Exponential backoff capped at backoff_max_seconds.
                    backoff = min(
                        backoff_max_seconds,
                        backoff_initial_seconds * (2 ** (consecutive_errors - 1)),
                    )
                    # Escalate severity once the streak indicates a sustained
                    # outage so dashboards / alerts still notice, without
                    # ever killing the worker.
                    log = self.logger.warning if consecutive_errors <= 5 else self.logger.error
                    log(
                        f"Error in monitoring task (#{consecutive_errors} consecutive, "
                        f"sleeping {backoff:.0f}s before retry): {e}"
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

            # Set the start time for monitoring and uptime tracking
            self.start_time = time.time()

            # Start the Ray cluster
            await self.ray_cluster.start()

            # Connect the BioEngine worker to the Hypha server
            # Completes initialization of AppsManager and CodeExecutor
            await self._connect_to_server()

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

        SECURITY: This method is publicly accessible and does not require admin permissions, making it suitable for monitoring dashboards and health checks by any authenticated user.

        STATUS INFORMATION CATEGORIES:

        SERVICE METADATA:
        - service_start_time: Unix timestamp when the worker was initialized
        - service_uptime: Duration in seconds since worker startup
        - workspace: Hypha workspace name where worker is registered
        - client_id: Unique client identifier for this worker instance
        - admin_users: List of user identifiers with administrative privileges
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
            "worker_mode": self.ray_cluster.mode,
            "workspace": self.workspace,
            "client_id": self.client_id,
            "ray_cluster": self.ray_cluster.status,
            "admin_users": self.admin_users,
            "geo_location": self.geo_location,
            "is_ready": self.is_ready.is_set(),
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
