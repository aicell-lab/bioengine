import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haikunator import Haikunator
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import RemoteService
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve
from ray.serve.schema import ApplicationDetails, ServeStatus

from bioengine import __version__
from bioengine.apps.builder import AppBuilder
from bioengine.cluster.ray_cluster import RayCluster
from bioengine.utils import (
    check_permissions,
    create_application_from_files,
    create_context,
    create_logger,
    ensure_applications_collection,
    get_static_site_url,
)


class AppsManager:
    """
    Manages Ray Serve deployments using Hypha artifact manager integration.

    This class provides comprehensive management of application deployments by integrating
    with the Hypha artifact manager to deploy containerized applications as Ray Serve
    deployments. It handles the complete lifecycle from artifact creation through
    deployment management to cleanup, with robust error handling and permission control.

    The AppsManager orchestrates:
    - Artifact creation and management through Hypha
    - Ray Serve deployment lifecycle management with proper task tracking
    - Dynamic service registration and exposure
    - Resource allocation and monitoring
    - Permission-based access control
    - Graceful deployment and undeployment operations with automatic cleanup

    Key Features:
    - Artifact-based deployment system with versioning support
    - Dynamic Hypha service registration for deployed applications
    - Multi-mode deployment configurations
    - Resource-aware deployment with CPU/GPU allocation
    - Permission-based access control for deployments
    - Comprehensive error handling and logging with proper state management
    - Startup deployment automation
    - Real-time deployment status monitoring
    - Robust deployment task tracking and cleanup

    Deployment State Management:
    The class maintains deployment state in `_deployed_applications` dictionary where each
    entry contains deployment metadata, resource allocation, and an active asyncio task
    reference. Proper cleanup ensures no resource leaks or orphaned deployments.

    Attributes:
        ray_cluster (RayCluster): Ray cluster manager instance
        admin_users (List[str]): List of user emails with admin permissions
        apps_workdir (Path): Working directory for deployment artifacts
        server: Hypha server connection
        artifact_manager: Hypha artifact manager service proxy
        app_builder (AppBuilder): Application builder instance
        startup_applications (List[dict]): Deployments to start automatically
        logger: Logger instance for deployment operations
        _deployed_applications (Dict): Internal tracking of active deployments with task references
        _deployment_lock (asyncio.Lock): Ensures single deployment operation at a time

    Parameter Conventions (API):
        - application_kwargs: Dictionary of keyword arguments for each deployment class
        - application_env_vars: Dictionary of environment variables for each deployment class
        - hypha_token: Hypha authentication token for application deployments (set as env var 'HYPHA_TOKEN')
            Used for authenticating to BioEngine datasets and Hypha APIs as the logged-in user.
    """

    def __init__(
        self,
        ray_cluster: RayCluster,
        apps_workdir: Union[str, Path] = f"{os.environ['HOME']}/.bioengine/apps",
        startup_applications: Optional[List[dict]] = None,
        # Logger
        log_file: Optional[Union[str, Path]] = None,
        debug: bool = False,
    ):
        """
        Initialize the AppsManager with configuration for managing BioEngine application deployments.

        Sets up the deployment manager to handle BioEngine applications using Ray Serve and
        Hypha artifact manager integration. Configures directory paths based on the Ray
        cluster mode and initializes all necessary components.

        Directory Configuration by Mode:
        - SLURM mode: Uses fixed paths (/home/<user>/.bioengine/apps) for container compatibility
        - Single-machine mode: Uses provided paths, converted to absolute paths
        - External-cluster mode: Uses provided paths directly

        Args:
            ray_cluster: Ray cluster manager instance for compute resource management
            apps_workdir: Working directory for application artifacts and build files
            startup_applications: List of application configurations to deploy automatically
                                 when the manager initializes
            log_file: Optional path to log file for deployment operations
            debug: Enable detailed debug logging for troubleshooting

        Raises:
            ValueError: If the Ray cluster mode is not supported
            Exception: If component initialization fails

        Note:
            Admin users are set later during the initialize() call, not during construction.
        """
        # Set up logging
        self.logger = create_logger(
            name="AppsManager",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        # Initialize components
        self.ray_cluster = ray_cluster

        self.app_builder = AppBuilder(
            apps_workdir=apps_workdir,
            log_file=log_file,
            proxy_actor_name=self.ray_cluster.proxy_actor_name,
            debug=debug,
        )

        self.haikunator = Haikunator()

        # Store startup applications to deploy on initialization
        if startup_applications is None:
            startup_applications = []
        if not isinstance(startup_applications, list):
            raise ValueError(
                f"startup_applications must be a list, got {type(startup_applications)}"
            )
        self.startup_applications = startup_applications

        # Initialize state variables
        self.server = None
        self.artifact_manager = None
        self.admin_users = None
        self.worker_service_id = None
        self._deployment_lock = asyncio.Lock()
        self._deployed_applications = {}
        self.debug = debug

    def _check_initialized(self) -> None:
        """
        Verify that the manager has been properly initialized with server connections.

        Ensures both the Hypha server connection and artifact manager service are available
        before attempting any operations that require them.

        Raises:
            RuntimeError: If server connection or artifact manager is not initialized
        """
        if not self.server:
            raise RuntimeError(
                "Hypha server connection not available. Call initialize() first."
            )
        if not self.artifact_manager:
            raise RuntimeError(
                "Artifact manager not initialized. Call initialize() first."
            )

    def _get_full_artifact_id(self, artifact_id: str) -> str:
        """
        Convert a potentially short artifact ID to its full workspace-qualified form.

        Hypha artifacts can be referenced either by their full ID (workspace/artifact-name)
        or by just the artifact name if it belongs to the current workspace. This method
        ensures we always have the full form for consistent handling.

        Args:
            artifact_id: The artifact identifier to convert

        Returns:
            Full artifact ID in the format 'workspace/artifact-name'

        Raises:
            TypeError: If artifact_id is not a string

        Examples:
            - "my-app" → "workspace123/my-app"
            - "workspace123/my-app" → "workspace123/my-app" (unchanged)
        """
        if not isinstance(artifact_id, str):
            raise TypeError("Artifact ID must be a string.")

        if "/" in artifact_id:
            return artifact_id
        else:
            # If artifact_id does not contain a slash, prepend the workspace
            return f"{self.server.config.workspace}/{artifact_id}"

    async def _generate_application_id(self) -> str:
        """
        Generate a unique identifier for a new application deployment.

        Creates a human-readable, unique application ID using the Haikunator library
        which generates names like "delighted-mouse-a4f2". Ensures the generated ID
        is not already in use by checking both internal tracking and Ray Serve status.

        ID Format:
        - Two descriptive words separated by hyphens
        - 4-character hexadecimal suffix for uniqueness
        - Example: "ambitious-whale-f3c1"

        Returns:
            Unique application ID that can be used for deployment

        Note:
            This method will retry generation until a unique ID is found.
        """
        while True:
            application_id = self.haikunator.haikunate(
                delimiter="-",  # Use hyphen as delimiter
                token_length=4,  # 4-character random suffix
                token_hex=True,  # Use hexadecimal characters for the suffix
            )
            if application_id in self._deployed_applications:
                # Application ID already exists, generate a new one
                continue

            serve_status = await self.ray_cluster.call_with_reconnect(serve.status)
            if application_id not in serve_status.applications:
                # Application ID is unique and not already deployed
                break

        return application_id

    async def _check_resources(
        self, application_id: str, required_resources: Dict[str, int]
    ) -> None:
        """
        Verify that sufficient resources are available for deploying an application.

        Checks current Ray cluster resource availability against the application's
        requirements for CPU, GPU, and memory. For SLURM clusters, also considers
        the ability to scale up by adding additional worker nodes.

        When updating an existing application, the resources currently used by that
        application are accounted for (as they will be freed during the update), so
        only the net change in resources is validated.

        Resource Validation Process:
        1. Waits for Ray cluster to be ready and connected
        2. Checks each node for sufficient free resources
        3. For updates: Subtracts current app resources from node used resources
        4. For SLURM mode: Evaluates if new workers can be spawned if needed
        5. For external clusters: Issues warning but allows deployment

        Args:
            application_id: ID of the application being deployed (for logging)
            required_resources: Dictionary containing 'num_cpus', 'num_gpus', and 'memory' requirements

        Raises:
            ValueError: If insufficient resources are available and cluster cannot scale

        Note:
            External clusters are assumed to have autoscaling capabilities, so only
            warnings are issued rather than blocking deployment.
        """
        self.logger.debug(
            f"Checking resources for application '{application_id}': {required_resources}"
        )

        # Check if this is an update - if so, account for resources that will be freed
        current_resources = {"num_cpus": 0, "num_gpus": 0, "memory": 0}
        if application_id in self._deployed_applications:
            current_resources = self._deployed_applications[application_id][
                "application_resources"
            ]
            self.logger.debug(
                f"Application '{application_id}' is being updated. "
                f"Current resources: {current_resources}, New resources: {required_resources}"
            )

        # Check if the required resources are available
        insufficient_resources = True

        # Wait for Ray cluster to be ready
        await self.ray_cluster.is_ready.wait()

        for node_resource in self.ray_cluster.status["nodes"].values():
            # TODO: This currently assumes one application fits on one node.
            # For multi-node applications, replace this with aggregate placement-aware checks.
            # For updates, add back the resources that will be freed
            used_cpu = max(
                0, node_resource["used_cpu"] - current_resources["num_cpus"]
            )
            used_gpu = max(
                0, node_resource["used_gpu"] - current_resources["num_gpus"]
            )
            used_memory = max(
                0, node_resource["used_memory"] - current_resources["memory"]
            )

            free_cpu = max(0, node_resource["total_cpu"] - used_cpu)
            free_gpu = max(0, node_resource["total_gpu"] - used_gpu)
            free_memory = max(0, node_resource["total_memory"] - used_memory)

            if (
                free_cpu >= required_resources["num_cpus"]
                and free_gpu >= required_resources["num_gpus"]
                and free_memory >= required_resources["memory"]
            ):
                insufficient_resources = False

        if self.ray_cluster.mode == "slurm" and insufficient_resources:
            # Check if additional SLURM workers can be created that meet the resource requirements
            # TODO: Remove resource check when SLURM workers can adjust resources dynamically
            num_worker_jobs = await self.ray_cluster.slurm_workers.get_num_worker_jobs()
            default_num_cpus = self.ray_cluster.slurm_workers.default_num_cpus
            default_num_gpus = self.ray_cluster.slurm_workers.default_num_gpus
            default_memory_bytes = (
                self.ray_cluster.slurm_workers.default_mem_in_gb_per_cpu
                * default_num_cpus
                * 1024
                * 1024
                * 1024
            )
            if (
                num_worker_jobs < self.ray_cluster.slurm_workers.max_workers
                and default_num_cpus >= required_resources["num_cpus"]
                and default_num_gpus >= required_resources["num_gpus"]
                and default_memory_bytes >= required_resources["memory"]
            ):
                insufficient_resources = False

        if insufficient_resources:
            if self.ray_cluster.mode != "external-cluster":
                self.logger.error(
                    f"Insufficient resources for application '{application_id}'. "
                    f"Requested: {required_resources}"
                )
                raise ValueError(
                    f"Insufficient resources for application '{application_id}'. "
                    f"Requested: {required_resources}"
                )
            else:
                self.logger.warning(
                    f"Currently insufficient resources for application '{application_id}'. "
                    "Assuming Ray autoscaling is available. "
                    f"Requested resources: {required_resources}"
                )

    async def _deploy_application(
        self,
        application_id: str,
    ) -> None:
        """
        Execute the deployment of a BioEngine application and manage its lifecycle.

        This is the core deployment method that runs as a long-lived asyncio task to
        manage an application's entire lifecycle from startup to cleanup. It handles
        the actual deployment process, monitors health, and ensures proper resource
        cleanup when the deployment ends.

        Deployment Lifecycle:
        1. Reset deployment status and prepare for deployment
        2. Use the pre-built application from _deployed_applications or build if not available
        3. Deploy to Ray Serve with unique routing
        4. Keep the deployment active until cancelled
        5. Perform cleanup when task is cancelled or fails

        The method runs indefinitely to maintain the deployment until explicitly
        cancelled. All cleanup is handled in the finally block to ensure proper
        resource management regardless of how the deployment ends.

        Note:
            Application building and resource validation are now done in deploy_app()
            before this task is created, providing immediate user feedback for errors.
            The built app is stored in _deployed_applications["built_app"] to avoid duplicate builds.

        Args:
            application_id: Unique identifier for the application deployment

        Raises:
            RuntimeError: If deployment validation fails
            Exception: If deployment startup or initialization fails

        Note:
            This method is designed to run as a managed asyncio task and should
            only exit when cancelled or on unrecoverable error.
        """
        try:
            # Reset deployment status (if already deployed)
            self._deployed_applications[application_id]["is_deployed"].clear()

            artifact_id = self._deployed_applications[application_id]["artifact_id"]
            version = self._deployed_applications[application_id]["version"]

            # Use the pre-built app from _deployed_applications
            # This was built and validated in deploy_app() before this task was created
            app = self._deployed_applications[application_id]["built_app"]

            # Run the deployment in Ray Serve with unique route prefix
            await self.ray_cluster.call_with_reconnect(
                serve.run,
                target=app,
                name=application_id,
                route_prefix=f"/{application_id}",
                blocking=False,
            )

            # Track the application in the internal state
            self.logger.info(
                f"Successfully completed deployment of application '{application_id}' from "
                f"artifact '{artifact_id}', version '{version}'."
            )

            # Mark the application as deployed
            self._deployed_applications[application_id]["is_deployed"].set()

            # Keep the deployment task running until cancelled
            event = asyncio.Event()
            await event.wait()

        except asyncio.CancelledError:
            self.logger.info(
                f"Deployment task for application '{application_id}' was cancelled."
            )
        except Exception as e:
            self.logger.error(
                f"Failed to deploy application '{application_id}' with error: {e}"
            )
            try:
                serve_status = await self.ray_cluster.call_with_reconnect(serve.status)
                application = serve_status.applications.get(application_id)
                if application:
                    for deployment_name, deployment in application.deployments.items():
                        if deployment.status.value == "UNHEALTHY":
                            self.logger.error(
                                f"Ray Serve application '{application_id}' deployment "
                                f"'{deployment_name}' reported error:\n{deployment.message}",
                            )
            except Exception as status_error:
                self.logger.error(
                    f"Failed to get Ray Serve status for application '{application_id}': {status_error}"
                )
        finally:
            # Signal other processes to stop waiting for this deployment to be ready
            # Check if application still exists (might have been removed during undeployment)
            if application_id in self._deployed_applications:
                self._deployed_applications[application_id]["is_deployed"].set()
                # Signal that the deployment task is no longer active
                self._deployed_applications[application_id]["deployment_task"] = None

    async def _cancel_deployment_process(
        self,
        application_id: str,
    ) -> None:
        """
        Cancel an ongoing deployment task for an application.

        Cancels the deployment task and waits for it to finish (with timeout).
        This is used when updating an existing deployment or during undeployment.

        Args:
            application_id: The application whose deployment task should be cancelled

        Note:
            If no deployment task is active, this method does nothing.
            Uses a 60-second timeout for cancellation to complete.
        """
        deployment_task = self._deployed_applications[application_id]["deployment_task"]
        if not deployment_task:
            # No active deployment task to cancel
            return

        self.logger.info(
            f"Cancelling deployment task of application '{application_id}'..."
        )
        deployment_task.cancel()
        timeout = 60
        try:
            # Wait for the task itself to complete (handles CancelledError internally)
            await asyncio.wait_for(deployment_task, timeout=timeout)
        except asyncio.CancelledError:
            # Expected - task was cancelled successfully
            pass
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Cancellation of existing deployment task for application '{application_id}' "
                f"did not finish in time ({timeout} seconds). Proceeding anyway."
            )
        except Exception as e:
            # Unexpected error during cancellation
            self.logger.warning(
                f"Error while cancelling deployment task for '{application_id}': {e}. "
                "Proceeding anyway."
            )

    async def _undeploy_application(
        self,
        application_id: str,
        user_id: str,
    ) -> None:
        """
        Execute the complete undeployment process for an application.

        Performs the actual undeployment by:
        1. Cancelling the deployment task if active
        2. Deleting the application from Ray Serve
        3. Clearing replica history from the proxy actor
        4. Removing the application from internal tracking

        Args:
            application_id: The application to undeploy
            user_id: ID of the user initiating the undeployment (for logging)

        Note:
            This method handles errors gracefully and logs failures.
            The application is removed from _deployed_applications even if
            some cleanup steps fail.
        """
        start_time = time.time()
        self.logger.info(
            f"User '{user_id}' is starting undeployment of application '{application_id}'..."
        )
        await self._cancel_deployment_process(application_id=application_id)

        try:
            await self.ray_cluster.call_with_reconnect(
                serve.delete, application_id
            )  # Note: Doesn't throw an error if app doesn't exist
            self.logger.info(
                f"Deleted Ray Serve application '{application_id}' in {time.time() - start_time:.2f} seconds."
            )
        except Exception as delete_err:
            self.logger.error(
                f"Error deleting Ray Serve application '{application_id}' "
                f"after {time.time() - start_time:.2f} seconds: {delete_err}"
            )

        try:
            await self.ray_cluster.proxy_actor_handle.clear_application_replicas.remote(
                application_id
            )
            self.logger.info(
                f"Cleared application replica history for '{application_id}'."
            )
        except Exception as e:
            self.logger.error(
                f"Error clearing application replica history for '{application_id}': {e}"
            )

        # Remove from internal tracking after all cleanup operations complete
        self._deployed_applications.pop(application_id, None)
        self.logger.info(f"Undeployment of application '{application_id}' completed.")

    async def _get_deployment_status(
        self,
        application_id: str,
        application_status: ApplicationDetails,
        n_previous_replica: int,
        logs_tail: int,
    ) -> Dict[str, Dict[str, Any]]:
        deployments_info = {}
        for deployment_name, deployment_info in application_status.deployments.items():
            deployments_info[deployment_name] = {
                "status": deployment_info.status.value,
                "message": deployment_info.message,
                "replica_states": deployment_info.replica_states,
                "logs": None,
            }

            # Collect logs for all tracked actor IDs
            try:
                deployment_logs = await self.ray_cluster.proxy_actor_handle.get_deployment_logs.remote(
                    application_id=application_id,
                    deployment_name=deployment_name,
                    n_previous_replica=n_previous_replica,
                    tail=logs_tail,
                )
                deployments_info[deployment_name]["logs"] = deployment_logs
            except Exception as e:
                self.logger.error(
                    f"Error retrieving logs for application '{application_id}', deployment '{deployment_name}': {e}"
                )
                deployments_info[deployment_name]["logs"] = {
                    "error": f"Error retrieving logs: {str(e)}"
                }

        return deployments_info

    def _inject_admin_users(
        self, authorized_users: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Inject admin users into non-public authorized_users rules and deduplicate.

        Mirrors the logic in AppBuilder.build(). Public rules ("*" in list) are
        left unchanged. Called during recovery to update live deployments.
        """
        result = {}
        for key, rule in authorized_users.items():
            rule = list(rule)
            if "*" not in rule and self.admin_users:
                for user in self.admin_users:
                    if user not in rule:
                        rule.append(user)
            seen: set = set()
            result[key] = [u for u in rule if not (u in seen or seen.add(u))]
        if "*" not in result and self.admin_users:
            seen = set()
            result["*"] = [
                u for u in self.admin_users if not (u in seen or seen.add(u))
            ]
        return result

    def _filter_secret_env_vars(
        self, application_env_vars: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """
        Filter out secret environment variables from application_env_vars.

        Secret environment variables are those that start with an underscore (_).
        They are replaced with "*****" to hide their values in status responses.
        The underscore prefix is removed from the key names, and all environment
        variables are sorted alphabetically.

        Args:
            application_env_vars: Dictionary of environment variables for each deployment class

        Returns:
            Filtered dictionary with secret values replaced with "*****", secret keys
            without underscore prefix, and all keys sorted alphabetically
        """
        filtered_env_vars = {}
        for deployment_class, env_vars in application_env_vars.items():
            # Process environment variables: remove _ prefix from secret vars and mask values
            processed_vars = {}
            for key, value in env_vars.items():
                if key.startswith("_"):
                    # Remove the underscore prefix and mask the value
                    new_key = key[1:]  # Remove first character (_)
                    processed_vars[new_key] = "*****"
                else:
                    processed_vars[key] = value

            # Sort alphabetically by key
            filtered_env_vars[deployment_class] = dict(sorted(processed_vars.items()))
        return filtered_env_vars

    async def _get_application_service_ids(
        self, application_id: str
    ) -> List[Dict[str, Optional[str]]]:
        # Construct the service IDs for the application using the replica IDs
        replica_ids = (
            await self.ray_cluster.proxy_actor_handle.get_deployment_replicas.remote(
                application_id=application_id,
                deployment_name="ProxyDeployment",
            )
        )
        if replica_ids:
            workspace = self.server.config.workspace
            worker_client_id = self.server.config.client_id
            service_ids = [
                {
                    "websocket_service_id": f"{workspace}/{worker_client_id}-{replica_id}:{application_id}",
                    "webrtc_service_id": f"{workspace}/{worker_client_id}-{replica_id}:{application_id}-rtc",
                }
                for replica_id in replica_ids
            ]
        else:
            service_ids = [
                {
                    "websocket_service_id": None,
                    "webrtc_service_id": None,
                }
            ]
        return service_ids

    async def _get_app_status(
        self,
        application_id: str,
        serve_status: ServeStatus,
        n_previous_replica: int,
        logs_tail: int,
    ) -> Dict[str, Any]:
        """
        Application States: [NOT_STARTED, DEPLOYING, DEPLOY_FAILED, RUNNING, UNHEALTHY, DELETING]
        Deployment States: [UPDATING, HEALTHY, UNHEALTHY, UPSCALING, DOWNSCALING]
        Replica States: [STARTING, UPDATING, RECOVERING, RUNNING, STOPPING, PENDING_MIGRATION]
        """
        # Check if application is in tracked applications
        if application_id not in self._deployed_applications:
            # Application is not running, return simplified status
            return {
                "status": "NOT_RUNNING",
                "message": f"Application '{application_id}' is not currently deployed. "
                f"To deploy this application, call deploy_app(application_id='{application_id}', ...) "
                f"with the appropriate artifact_id and parameters.",
            }

        # Get application info from tracked applications
        application_info = self._deployed_applications[application_id]
        application_status = serve_status.applications.get(application_id)

        if application_status:
            status = application_status.status.value
            message = application_status.message

            deployments = await self._get_deployment_status(
                application_id=application_id,
                application_status=application_status,
                n_previous_replica=n_previous_replica,
                logs_tail=logs_tail,
            )
        else:
            if application_info["is_deployed"].is_set():
                status = "UNHEALTHY"
                message = f"Application '{application_id}' is marked as deployed but not found in Ray Serve status."
                self.logger.warning(
                    f"Application '{application_id}' for artifact '{application_info['artifact_id']}' "
                    "is marked as deployed but not found in Ray Serve status."
                )
            else:
                status = "NOT_STARTED"
                message = f"Application '{application_id}' has not been deployed yet."
            deployments = {}

        service_ids = await self._get_application_service_ids(application_id)

        # Build static site URL with runtime config params so the frontend
        # knows which Hypha server and service replica to connect to.
        base_static_url = application_info.get("static_site_url")
        static_site_url = None
        if base_static_url and service_ids:
            first = service_ids[0]
            ws_id = first.get("websocket_service_id") or ""
            rtc_id = first.get("webrtc_service_id") or ""
            server_url = self.server.config.public_base_url
            params = (
                f"server={server_url}"
                f"&ws_service_id={ws_id}"
                f"&webrtc_service_id={rtc_id}"
            )
            static_site_url = f"{base_static_url}?{params}"

        return {
            "display_name": application_info["display_name"],
            "description": application_info["description"],
            "artifact_id": application_info["artifact_id"],
            "version": application_info["version"] or "latest",
            "recovered_app": application_info["recovered_app"],
            "status": status,
            "message": message,
            "deployments": deployments,
            "application_kwargs": application_info["application_kwargs"],
            "application_env_vars": self._filter_secret_env_vars(
                application_info["application_env_vars"]
            ),
            "gpu_enabled": not application_info["disable_gpu"],
            "application_resources": application_info["application_resources"],
            "authorized_users": application_info["authorized_users"],
            "available_methods": application_info["available_methods"],
            "max_ongoing_requests": application_info["max_ongoing_requests"],
            "static_site_url": static_site_url,
            "service_ids": service_ids,
            "start_time": application_info["started_at"],
            "last_updated_at": application_info["last_updated_at"],
            "last_updated_by": application_info["last_updated_by"],
            "auto_redeploy": application_info["auto_redeploy"],
        }

    async def complete_initialization(
        self, server: RemoteService, admin_users: List[str], worker_service_id: str
    ) -> None:
        """
        Initialize the deployment manager with Hypha server connections and admin permissions.

        Sets up the complete infrastructure needed for application deployment management
        including server connections, artifact manager access, and automatic startup
        application deployment.

        Initialization Process:
        1. Establishes Hypha server connection and stores admin user list
        2. Connects to the artifact manager service for application storage
        3. Configures the AppBuilder with server connections
        4. Deploys any configured startup applications automatically
        5. Waits for all startup deployments to be ready

        Args:
            server: Active Hypha server connection instance with proper authentication
            admin_users: List of user IDs or email addresses with administrative permissions
                        for managing applications and deployments
            worker_service_id: BioEngine worker service ID

        Raises:
            Exception: If server connection fails, artifact manager is unavailable,
                      or startup application deployment fails
        """
        if not server or not isinstance(server, RemoteService):
            raise ValueError("Invalid server connection provided.")
        if not isinstance(admin_users, list) or not all(
            isinstance(user, str) for user in admin_users
        ):
            raise ValueError("Invalid admin_users list provided.")

        # Store server connection and list of admin users
        self.server = server
        self.admin_users = admin_users

        try:
            # Get artifact manager service
            self.artifact_manager = await self.server.get_service(
                "public/artifact-manager"
            )
            self.logger.info("Successfully connected to artifact manager.")
        except Exception as e:
            self.logger.error(f"Error initializing Ray Deployment Manager: {e}")
            self.server = None
            self.artifact_manager = None
            raise

        # Initialize the AppBuilder with the server and artifact manager
        self.app_builder.complete_initialization(
            server=self.server,
            artifact_manager=self.artifact_manager,
            worker_service_id=worker_service_id,
            serve_http_url=self.ray_cluster.serve_http_url,
        )

        # Ensure applications collection exists
        workspace = self.server.config.workspace
        await ensure_applications_collection(
            artifact_manager=self.artifact_manager,
            workspace=workspace,
            logger=self.logger,
        )

    async def recover_deployed_applications(self) -> None:
        """Recover app tracking state for already running Ray Serve BioEngine apps."""
        serve_status = await self.ray_cluster.call_with_reconnect(serve.status)
        recovered_count = 0

        for application_id, application in serve_status.applications.items():
            if application_id in self._deployed_applications:
                continue

            if "ProxyDeployment" not in application.deployments:
                continue

            try:
                app_handle = await self.ray_cluster.call_with_reconnect(
                    serve.get_app_handle, application_id
                )
                app_data = await asyncio.wait_for(
                    app_handle.get_app_data.remote(), timeout=10.0
                )

                if not isinstance(app_data, dict):
                    raise ValueError("get_app_data() did not return a dictionary")

                required_keys = {
                    "display_name",
                    "description",
                    "artifact_id",
                    "version",
                    "application_kwargs",
                    "application_env_vars",
                    "disable_gpu",
                    "max_ongoing_requests",
                    "application_resources",
                    "authorized_users",
                    "available_methods",
                    "started_at",
                    "last_updated_at",
                    "last_updated_by",
                    "auto_redeploy",
                    "debug",
                }
                missing_keys = sorted(required_keys - set(app_data.keys()))
                if missing_keys:
                    raise ValueError(
                        "Missing required app_data keys: "
                        f"{missing_keys}"
                    )

                is_deployed = asyncio.Event()
                is_deployed.set()

                # Inject current admin users into the recovered authorized_users
                # and update the live ProxyDeployment actor in place.
                updated_authorized_users = self._inject_admin_users(
                    app_data["authorized_users"]
                )
                await asyncio.wait_for(
                    app_handle.update_authorized_users.remote(updated_authorized_users),
                    timeout=10.0,
                )

                self._deployed_applications[application_id] = {
                    "display_name": app_data["display_name"],
                    "description": app_data["description"],
                    "artifact_id": app_data["artifact_id"],
                    "version": app_data["version"],
                    "application_kwargs": app_data["application_kwargs"],
                    "application_env_vars": app_data["application_env_vars"],
                    "hypha_token": None,
                    "disable_gpu": app_data["disable_gpu"],
                    "max_ongoing_requests": app_data["max_ongoing_requests"],
                    "application_resources": app_data["application_resources"],
                    "authorized_users": updated_authorized_users,
                    "available_methods": app_data["available_methods"],
                    "started_at": app_data["started_at"],
                    "last_updated_at": app_data["last_updated_at"],
                    "last_updated_by": app_data["last_updated_by"],
                    "built_app": None,
                    "auto_redeploy": app_data["auto_redeploy"],
                    "debug": app_data["debug"],
                    "deployment_task": None,
                    "is_deployed": is_deployed,
                    "undeployment_task": None,
                    "recovered_app": True,
                }
                recovered_count += 1
            except Exception as e:
                self.logger.warning(
                    f"Failed to recover application '{application_id}' from Ray Serve: {e}"
                )

        if recovered_count:
            self.logger.info(
                f"Recovered {recovered_count} application(s) from existing Ray Serve state."
            )
        else:
            self.logger.info("No existing Ray Serve applications needed recovery.")

    async def deploy_startup_applications(self) -> None:
        """
        Deploy any configured startup applications.
        """
        # Deploy any startup applications if provided
        if self.startup_applications:
            self.logger.info(
                f"Deploying {len(self.startup_applications)} startup application(s)..."
            )

            # Pass the worker owner's token to the startup applications (if not already set)
            startup_applications_token = await self.server.generate_token(
                {
                    "workspace": self.server.config.workspace,
                    "permission": "read_write",
                    "expires_in": 3600 * 24 * 30,  # support application for 30 days
                }
            )

            # Initialize deployment of each startup application
            application_ids = []

            # Get valid startup config keys
            deploy_app_schema = self.deploy_app.__schema__

            parameters = deploy_app_schema["parameters"]["properties"]

            valid_keys = set(parameters.keys()) - {"context"}

            for app_config in self.startup_applications:
                if not isinstance(app_config, dict):
                    raise ValueError(
                        "Each startup application config must be a dictionary."
                    )

                if "artifact_id" not in app_config:
                    raise ValueError(
                        "Each startup application config must contain an 'artifact_id'."
                    )

                # Check for invalid keys
                invalid_keys = set(app_config.keys()) - valid_keys
                if invalid_keys:
                    raise ValueError(
                        f"Invalid keys in startup application config for artifact '{app_config.get('artifact_id', 'unknown')}': "
                        f"{', '.join(sorted(invalid_keys))}. Valid keys are: {', '.join(sorted(valid_keys))}"
                    )

                if "hypha_token" not in app_config:
                    app_config["hypha_token"] = startup_applications_token

                admin_context = create_context(
                    self.admin_users[0] if len(self.admin_users) > 0 else None,
                    self.admin_users[1] if len(self.admin_users) > 1 else None,
                )

                deploy_app_kwargs = {
                    key: value
                    for key, value in app_config.items()
                    if key in valid_keys
                }
                deploy_app_kwargs["context"] = admin_context

                application_id = await self.deploy_app(**deploy_app_kwargs)
                application_ids.append(application_id)

    async def monitor_applications(self) -> None:
        """
        Monitor the health of all deployed applications and handle failures automatically.

        Continuously monitors application health by checking Ray Serve status against
        internal deployment tracking. Handles failure recovery, redeployment, and
        cleanup of consistently failing applications.

        Monitoring Actions:
        - Tracks consecutive failures for each application
        - Triggers redeployment for applications experiencing issues
        - Removes applications that fail repeatedly (>3 consecutive failures)
        - Resets failure counters for healthy applications

        Failure Handling:
        - 1-3 failures: Attempts automatic redeployment
        - >3 failures: Removes application and stops monitoring
        - Logs warnings and status changes for debugging

        This method is typically called periodically by the worker to ensure
        application health and availability.

        Raises:
            Exception: If monitoring process encounters unrecoverable errors
        """
        apps_to_redeploy = [
            app_id
            for app_id, app_info in self._deployed_applications.items()
            if app_info["auto_redeploy"]
        ]
        if not apps_to_redeploy:
            return

        try:
            # Get status of actively running deployments
            await self.ray_cluster.check_connection()
            serve_status = await self.ray_cluster.call_with_reconnect(serve.status)

            for application_id in apps_to_redeploy:
                application_info = self._deployed_applications.get(application_id)
                if not application_info:
                    # Application no longer tracked, skip monitoring
                    continue
                if not application_info["is_deployed"].is_set():
                    # Application not yet deployed, skip monitoring
                    continue

                # Get the application status from Ray Serve
                application = serve_status.applications.get(application_id)
                if not application or application.status.value in [
                    "DEPLOY_FAILED",
                    "UNHEALTHY",
                ]:
                    # Application is experiencing issues, trigger redeployment
                    self.logger.warning(
                        f"Application '{application_id}' for artifact '{application_info['artifact_id']}' "
                        f"is experiencing issues. Triggering redeployment..."
                    )

                    # Re-deploy the application
                    deployment_task = asyncio.create_task(
                        self._deploy_application(application_id=application_id),
                        name=f"Deploy_{application_id}",
                    )
                    # Overwrite the existing deployment task
                    application_info["deployment_task"] = deployment_task
        except Exception as e:
            self.logger.error(f"Error monitoring applications: {e}")
            raise e

    @schema_method
    async def upload_app(
        self,
        files: List[dict] = Field(
            ...,
            description="List of application files to upload. Each file must be a dictionary with 'name' (string), 'content' (file content), and 'type' ('text' for text files or 'base64' for binary files). Must include a 'manifest.yaml' file with application configuration.",
        ),
        workspace: Optional[str] = Field(
            None,
            description="Hypha workspace to upload the application to. When provided together with hypha_token, no admin permissions are required and the artifact is created in the specified workspace. If omitted, the worker's own workspace is used and admin access is required.",
        ),
        hypha_token: Optional[str] = Field(
            None,
            description="Hypha authentication token for the target workspace. Must be provided together with workspace. When both are given the caller does not need to be a worker admin.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> str:
        """
        Creates or updates a BioEngine application artifact in the Hypha artifact manager.

        Two usage modes:

        Admin mode (workspace and hypha_token omitted):
            Uploads to the worker's own workspace. Requires the caller to be a worker admin.
            The worker's existing artifact manager connection is reused.

        User mode (workspace and hypha_token both provided):
            Uploads to the specified workspace using the provided token. No admin permission
            is required. A temporary artifact manager connection is created and closed after
            the upload.

        Application Structure:
        - manifest.yaml: Required configuration file defining the application metadata,
          deployment settings, and entry points
        - Python files: Application code and dependencies
        - Data files: Any additional resources needed by the application

        The manifest.yaml file must contain:
        - id: Application identifier
        - type: Must be "ray-serve"
        - name: Human-readable application name
        - description: Application description
        - Additional deployment configuration

        Returns:
            The artifact ID of the created or updated application artifact

        Raises:
            ValueError: If manifest is missing, invalid, artifact ID format is incorrect,
                       or only one of workspace/hypha_token is provided
            PermissionError: If user lacks admin permissions (admin mode only)
            RuntimeError: If artifact creation fails or Hypha connection is unavailable
        """
        self._check_initialized()

        if (workspace is None) != (hypha_token is None):
            raise ValueError(
                "workspace and hypha_token must be provided together or both omitted."
            )

        user_mode = workspace is not None

        if not user_mode:
            # Admin mode: enforce admin permissions, use worker's artifact manager
            check_permissions(
                context=context,
                authorized_users=self.admin_users,
                resource_name="creating or modifying an application",
            )
            artifact_manager = self.artifact_manager
            target_workspace = self.server.config.workspace
        else:
            # User mode: connect with provided credentials, no admin check
            server_url = self.server.config.public_base_url
            remote_server = await connect_to_server(
                {"server_url": server_url, "token": hypha_token, "workspace": workspace}
            )
            artifact_manager = await remote_server.get_service("public/artifact-manager")
            target_workspace = workspace

        try:
            created_artifact_id = await create_application_from_files(
                artifact_manager=artifact_manager,
                files=files,
                workspace=target_workspace,
                user_id=context["user"]["id"],
                logger=self.logger,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create/update artifact: {e}")
        finally:
            if user_mode:
                await remote_server.disconnect()

        if not user_mode:
            # Verify the artifact is in the worker's collection (admin mode only)
            available_artifacts = await self.list_apps(context=context)
            if created_artifact_id not in available_artifacts:
                raise ValueError(
                    f"Artifact '{created_artifact_id}' could not be created or is not in the collection."
                )

        self.logger.info(
            f"Successfully created/updated application artifact '{created_artifact_id}'."
        )

        return created_artifact_id

    @schema_method
    async def list_app_directories(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> List[Dict[str, Any]]:
        """
        Lists all application working directories under the BioEngine apps workspace.

        Returns one entry per subdirectory found under the apps working directory,
        including whether that directory belongs to a currently running application.

        Returns:
            List of dictionaries, each with:
            - name: directory name (matches application_id for deployed apps)
            - path: absolute path to the directory
            - is_running: True if the directory belongs to a currently running app
            - size_bytes: total disk usage of the directory in bytes

        Raises:
            PermissionError: If the caller is not a worker admin
            RuntimeError: If the worker is not initialized
        """
        self._check_initialized()
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="listing app directories",
        )

        apps_workdir = self.app_builder.apps_workdir
        if not apps_workdir.exists():
            return []

        running = {
            app_id
            for app_id, info in self._deployed_applications.items()
            if info["is_deployed"].is_set()
        }

        result = []
        for entry in sorted(apps_workdir.iterdir()):
            if not entry.is_dir():
                continue
            size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            result.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "is_running": entry.name in running,
                    "size_bytes": size,
                }
            )
        return result

    @schema_method
    async def clear_app_directory(
        self,
        application_id: str = Field(
            ...,
            description="Application ID whose working directory should be deleted. Must match a subdirectory of the apps workspace (e.g. 'model-runner').",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> None:
        """
        Deletes an application working directory from the BioEngine apps workspace.

        The directory is identified by application_id (as returned in the 'name' field
        of list_app_directories). Raises an error if the application is currently running —
        stop it first with stop_app() before clearing its directory.

        Args:
            application_id: ID of the application whose directory should be deleted (e.g. "model-runner").

        Raises:
            PermissionError: If the caller is not a worker admin
            RuntimeError: If the worker is not initialized, the app is currently running,
                         or deletion fails
            ValueError: If the directory does not exist or application_id contains path separators
        """
        self._check_initialized()
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"clearing app directory '{application_id}'",
        )

        if "/" in application_id or "\\" in application_id or application_id in (".", ".."):
            raise ValueError(
                f"application_id must be a plain name, not a path: '{application_id}'"
            )

        target = self.app_builder.apps_workdir / application_id
        if not target.exists():
            raise ValueError(
                f"No directory found for application '{application_id}' in the apps workspace."
            )
        if not target.is_dir():
            raise ValueError(f"'{application_id}' is not a directory.")

        # Refuse if the app is currently running
        if application_id in self._deployed_applications:
            info = self._deployed_applications[application_id]
            if info["is_deployed"].is_set():
                raise RuntimeError(
                    f"Cannot clear directory for '{application_id}': application is currently running. "
                    "Stop it first with stop_app()."
                )

        import shutil
        try:
            shutil.rmtree(target)
        except Exception as e:
            raise RuntimeError(f"Failed to delete directory for '{application_id}': {e}")

        self.logger.info(f"Cleared app directory: {target}")

    @schema_method
    async def list_apps(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> Dict[str, List[str]]:
        """
        Lists all available BioEngine application artifacts stored in the Hypha artifact manager.

        This method retrieves and returns information about all BioEngine applications that have been
        uploaded to the current workspace, including their manifest data and associated files.
        Each application artifact contains the necessary code, configuration, and dependencies
        needed to deploy a BioEngine application.

        Returns:
            Dictionary mapping artifact IDs to their metadata and files. Each entry contains:
            - manifest: Application configuration and deployment specifications
            - files: List of file names included in the artifact package

        Raises:
            PermissionError: If the user lacks admin permissions to list applications
            RuntimeError: If the Hypha server connection is not initialized
        """
        self._check_initialized()

        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"listing applications",
        )

        collection_id = f"{self.server.config.workspace}/applications"
        bioengine_apps_artifacts = await self.artifact_manager.list(collection_id)
        bioengine_apps = {}
        for artifact in bioengine_apps_artifacts:
            try:
                manifest = await self.artifact_manager.read(artifact.id)
                files = await self.artifact_manager.list_files(artifact.id)
                file_names = [file.name for file in files]
                bioengine_apps[artifact.id] = {
                    "manifest": manifest,
                    "files": file_names,
                }
            except Exception as e:
                self.logger.error(f"Error reading artifact '{artifact.id}': {e}")

        return bioengine_apps

    @schema_method
    async def get_app_manifest(
        self,
        artifact_id: str = Field(
            ...,
            description="Unique identifier of the application artifact to retrieve the manifest for. Can be either the full artifact ID (workspace/artifact-name) or just the artifact name if it belongs to the current workspace.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> Dict[str, Any]:
        """
        Retrieves the manifest configuration of a specific BioEngine application artifact.

        This method fetches and returns the manifest.yaml content from the specified
        application artifact stored in the Hypha artifact manager. The manifest contains
        essential metadata and deployment specifications for the BioEngine application.

        Args:
            artifact_id: The unique identifier of the application artifact.

        Returns:
            Dictionary representing the manifest configuration of the application.

        Raises:
            ValueError: If the artifact doesn't exist or cannot be read
            PermissionError: If user lacks admin permissions to access applications
            RuntimeError: If Hypha connection is not available
        """
        self._check_initialized()

        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"accessing the artifact '{artifact_id}'",
        )

        # Get the full artifact ID
        self.logger.debug(f"Retrieving manifest for artifact '{artifact_id}'...")
        artifact_id = self._get_full_artifact_id(artifact_id)

        # Read and return the manifest
        try:
            manifest = await self.artifact_manager.read(artifact_id)
            return manifest
        except Exception as e:
            raise ValueError(
                f"Failed to read manifest for artifact '{artifact_id}': {e}"
            )

    @schema_method
    async def delete_app(
        self,
        artifact_id: str = Field(
            ...,
            description="Unique identifier of the application artifact to delete. Can be either the full artifact ID (workspace/artifact-name) or just the artifact name if it belongs to the current workspace.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> None:
        """
        Permanently deletes a BioEngine application artifact from the Hypha artifact manager.

        This method removes the specified application artifact and all its associated files
        from the artifact manager. Once deleted, the artifact cannot be recovered and any
        running deployments using this artifact should be undeployed first.

        Warning: This operation is irreversible. Ensure you have backups if needed before deletion.

        Args:
            artifact_id: The artifact to delete. Must belong to the current workspace.

        Raises:
            ValueError: If the artifact doesn't exist, doesn't belong to current workspace,
                       or cannot be deleted
            PermissionError: If user lacks admin permissions to delete applications
            RuntimeError: If Hypha connection is not available
        """
        self._check_initialized()

        # Check user permissions
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"deleting the artifact '{artifact_id}'",
        )

        # Get the full artifact ID
        self.logger.debug(f"Deleting artifact '{artifact_id}'...")
        artifact_id = self._get_full_artifact_id(artifact_id)

        # Ensure the artifact is in the correct workspace
        workspace = self.server.config.workspace
        if not artifact_id.startswith(f"{workspace}/"):
            raise ValueError(
                f"Artifact ID '{artifact_id}' does not belong to the current workspace '{workspace}'."
            )

        # Delete the artifact
        await self.artifact_manager.delete(artifact_id)

        # Verify deletion
        available_artifacts = await self.list_apps(context=context)
        if artifact_id in available_artifacts:
            raise ValueError(
                f"Artifact '{artifact_id}' could not be deleted. It still exists in the collection."
            )

        self.logger.info(f"Successfully deleted artifact '{artifact_id}'.")

    @schema_method
    async def deploy_app(
        self,
        artifact_id: str = Field(
            ...,
            description="Identifier of the application artifact to deploy. Can be either the full artifact ID (workspace/artifact-name) or just the artifact name if it belongs to the current workspace.",
        ),
        version: Optional[str] = Field(
            None,
            description="Specific version of the artifact to deploy. If not provided, deploys the latest available version of the artifact. If not specified, uses the latest version for a new application, or preserves the previously deployed version if updating an existing application (only when application_id is specified).",
        ),
        application_id: Optional[str] = Field(
            None,
            description="Unique identifier for this deployment instance. If not provided, a random unique ID will be automatically generated. To update an existing application, provide its application_id.",
        ),
        application_kwargs: Optional[Dict[str, Dict[str, Any]]] = Field(
            None,
            description="Keyword arguments to set for each deployment. Dictionary where keys are deployment class names and values are dictionaries of keyword arguments. If not specified, uses default parameters for a new application, or preserves previous values if updating an existing application (only when application_id is specified).",
            examples=[
                {"DeploymentClass": {"init_parameter": 50.0}},
                {
                    "DeploymentClass1": {"text": "Hello World!"},
                    "DeploymentClass2": {"x": 3, "y": 10},
                },
            ],
        ),
        application_env_vars: Optional[Dict[str, Dict[str, str]]] = Field(
            None,
            description="Environment variables to set for each deployment. Dictionary where keys are deployment class names and values are dictionaries of environment variables. If not specified, uses defaults for a new application, or preserves previous values if updating an existing application (only when application_id is specified).",
            examples=[
                {"DeploymentClass": {"KEY": "VALUE"}},
                {
                    "DeploymentClass1": {"TEST_VARIABLE": "1"},
                    "DeploymentClass2": {"ENV_VAR": "example"},
                },
            ],
        ),
        hypha_token: str = Field(
            None,
            description="Hypha connection token for authentication. The token will be set as environment variable 'HYPHA_TOKEN' in the application deployments. An already existing environment variable named 'HYPHA_TOKEN' will not be overwritten. The token is used to authenticate to BioEngine datasets and enables Hypha API calls as logged in user. If not specified, uses None (no token) for a new application, or preserves the previous token if updating an existing application (only when application_id is specified).",
        ),
        disable_gpu: bool = Field(
            None,
            description="Set to true to disable GPU usage for this deployment, forcing it to run on CPU only. Useful for testing or when GPU resources are limited. If not specified, uses False (GPU enabled if available) for a new application, or preserves the previous setting if updating an existing application (only when application_id is specified).",
        ),
        max_ongoing_requests: int = Field(
            None,
            description="Maximum number of concurrent requests this application instance can handle simultaneously. Higher values allow more parallelism but use more memory. If not specified, uses 10 for a new application, or preserves the previous value if updating an existing application (only when application_id is specified).",
        ),
        auto_redeploy: bool = Field(
            None,
            description="If set to true, the application will be automatically redeployed if it becomes unhealthy. If not specified, uses False for a new application, or preserves the previous setting if updating an existing application (only when application_id is specified).",
        ),
        debug: bool = Field(
            None,
            description="Set to true to enable debug logging for the whole application with all its deployments. If not specified, uses False for a new application, or preserves the previous setting if updating an existing application (only when application_id is specified).",
        ),
        ice_servers: Optional[List[Dict[str, Any]]] = Field(
            None,
            description="Custom ICE server configurations for WebRTC connections. If provided, these are used instead of fetching from the default Hypha TURN server endpoint. Each entry should have a 'urls' key and optionally 'username' and 'credential' for TURN servers. Example: [{\"urls\": \"stun:stun.example.com:19302\"}, {\"urls\": \"turn:turn.example.com:3478\", \"username\": \"user\", \"credential\": \"pass\"}]. If not specified, ICE servers are fetched automatically.",
            examples=[
                [{"urls": "stun:stun.example.com:19302"}],
                [
                    {"urls": "stun:stun.example.com:19302"},
                    {"urls": "turn:turn.example.com:3478", "username": "user", "credential": "pass"},
                ],
            ],
        ),
        authorized_users: Optional[Dict[str, List[str]]] = Field(
            None,
            description="Per-method access control for the deployed application. Keys are method names or '*' (wildcard for all methods). Values are lists of allowed user IDs or email addresses; use ['*'] for public access. Method-specific entries take precedence over the wildcard. The user calling deploy_app is always granted access to all methods. If not specified, the manifest's authorized_users field is used as a wildcard rule for all methods, or the previous value is preserved when updating an existing application.",
            examples=[
                {"*": ["*"]},
                {"*": ["user@lab.edu"], "run_inference": ["*"]},
                {"train": ["admin@lab.edu"], "infer": ["*"]},
            ],
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> str:
        """
        Deploys or updates a BioEngine application from an artifact to Ray Serve with comprehensive lifecycle management.

        This method downloads the specified application artifact, configures it according to the
        provided parameters, and creates a running deployment in the Ray Serve cluster. The
        deployment will be automatically registered as a service and made available for use.

        Deployment Process:
        1. Downloads and validates the application artifact
        2. Checks resource availability (CPU, GPU, memory)
        3. Configures the Ray Serve deployment with specified parameters
        4. Starts the deployment and monitors its health
        5. Registers the application as a callable service

        Update Process (when application_id already exists):
        1. Preserves the original creation time (started_at)
        2. Uses previous parameter values for any unspecified parameters
        3. Downloads and rebuilds the application from the artifact
        4. Updates last_updated_at and last_updated_by timestamps
        5. Redeploys with new configuration

        The deployment runs asynchronously - this method returns immediately after starting
        the deployment process. Use get_app_status() to monitor deployment progress and health.

        Resource Management:
        - Automatically scales Ray cluster if needed (SLURM mode)
        - Validates resource requirements before deployment
        - Supports both CPU and GPU deployments
        - Configurable request concurrency limits

        Returns:
            Unique application ID for the created deployment. Use this ID to monitor,
            update, or undeploy the application.

        Raises:
            ValueError: If artifact doesn't exist, deployment configuration is invalid,
                       or insufficient resources are available
            PermissionError: If user lacks admin permissions to deploy applications
            RuntimeError: If Ray cluster is unavailable or deployment initialization fails
        """
        # Only allow one deployment task creation at a time
        # This ensures unique application_id and prevents race conditions
        async with self._deployment_lock:
            self._check_initialized()

            # Get the full artifact ID
            artifact_id = self._get_full_artifact_id(artifact_id)

            # Check user permissions
            check_permissions(
                context=context,
                authorized_users=self.admin_users,
                resource_name=f"deploying an application from artifact '{artifact_id}'",
            )
            user_id = context["user"]["id"]

            # Verify Ray is initialized
            await self.ray_cluster.check_connection()

            # Generate a new application ID if not provided
            if not application_id:
                application_id = await self._generate_application_id()

            # Check if this is an update to an existing application
            is_update = application_id in self._deployed_applications

            # For updates, preserve original creation time and inherit unspecified parameters
            if is_update:
                existing_app = self._deployed_applications[application_id]
                started_at = existing_app["started_at"]
                last_updated_at = time.time()  # Update time for updates

                # Inherit previous parameters if not specified
                if version is None:
                    version = existing_app["version"]
                if application_kwargs is None:
                    application_kwargs = existing_app["application_kwargs"]
                if application_env_vars is None:
                    application_env_vars = existing_app["application_env_vars"]
                if hypha_token is None:
                    hypha_token = existing_app["hypha_token"]
                if disable_gpu is None:
                    disable_gpu = existing_app["disable_gpu"]
                if max_ongoing_requests is None:
                    max_ongoing_requests = existing_app["max_ongoing_requests"]
                if auto_redeploy is None:
                    auto_redeploy = existing_app["auto_redeploy"]
                if debug is None:
                    debug = existing_app.get("debug", False)
                if ice_servers is None:
                    ice_servers = existing_app.get("ice_servers", None)
                if authorized_users is None:
                    authorized_users = existing_app.get("authorized_users", None)
            else:
                # For new applications, set creation time and default values
                started_at = time.time()
                last_updated_at = started_at  # Same as started_at for new applications

                # Set default values for None parameters
                if application_kwargs is None:
                    application_kwargs = {}
                if application_env_vars is None:
                    application_env_vars = {}
                if disable_gpu is None:
                    disable_gpu = False
                if max_ongoing_requests is None:
                    max_ongoing_requests = 10
                if auto_redeploy is None:
                    auto_redeploy = False
                if debug is None:
                    debug = False
                # ice_servers defaults to None (fetch from URL)
                # authorized_users defaults to None (use manifest value)

            # Caller identity and admin users for injection into authorized_users by the builder
            user_email = context["user"].get("email", "")

            # Validate application_kwargs
            if not isinstance(application_kwargs, dict):
                raise ValueError(
                    "application_kwargs must be a dictionary of keyword arguments."
                )
            # Ensure all values are dictionaries
            for key, value in application_kwargs.items():
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Value for '{key}' in application_kwargs must be a dictionary."
                    )

            kwargs_str = (
                ", ".join(
                    f"{deployment_class}("
                    + ", ".join(
                        [f"{key}={value!r}" for key, value in deployment_kwargs.items()]
                    )
                    + ")"
                    for deployment_class, deployment_kwargs in application_kwargs.items()
                )
                if application_kwargs
                else "None"
            )

            # Validate application_env_vars
            if not isinstance(application_env_vars, dict):
                raise ValueError(
                    "application_env_vars must be a dictionary of environment variables."
                )
            for key, value in application_env_vars.items():
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Value for '{key}' in application_env_vars must be a dictionary."
                    )

            # Construct the environment variables string, hide secret env vars starting with underscore as *****
            env_vars_str = (
                ", ".join(
                    f"{deployment_class}: "
                    + " ".join(
                        [
                            f"{key}={value!r}"
                            for key, value in env_vars.items()
                            if not key.startswith("_")
                        ]
                    )
                    + " ".join(
                        [
                            f'{key}="*****"'
                            for key in env_vars.keys()
                            if key.startswith("_")
                        ]
                    )
                    for deployment_class, env_vars in application_env_vars.items()
                )
                if application_env_vars
                else "None"
            )

            if is_update:
                # If already deployed, cancel the existing deployment task to update deployment in a new task
                self.logger.info(
                    f"User '{user_id}' is updating existing application '{application_id}' from artifact '{artifact_id}', "
                    f"version '{version}'; kwargs: {kwargs_str}; env_vars: {env_vars_str}"
                )
                await self._cancel_deployment_process(application_id=application_id)
            else:
                # Create a new application
                self.logger.info(
                    f"User '{user_id}' is deploying new application '{application_id}' from artifact '{artifact_id}', "
                    f"version '{version}'; kwargs: {kwargs_str}; env_vars: {env_vars_str}"
                )

            # Build the application first to validate format and catch errors early
            app = await self.app_builder.build(
                application_id=application_id,
                artifact_id=artifact_id,
                version=version,
                application_kwargs=application_kwargs,
                application_env_vars=application_env_vars,
                hypha_token=hypha_token,
                disable_gpu=disable_gpu,
                max_ongoing_requests=max_ongoing_requests,
                debug=debug,
                started_at=started_at,
                last_updated_at=last_updated_at,
                last_updated_by=user_id,
                auto_redeploy=auto_redeploy,
                ice_servers=ice_servers,
                authorized_users=authorized_users,
                deploying_user=(user_id, user_email),
                admin_users=self.admin_users,
            )

            # Check resources before creating deployment task
            await self._check_resources(
                application_id=application_id,
                required_resources=app.metadata["resources"],
            )

            # Derive static site URL from the manifest's frontend_entry field.
            # If frontend_entry is set, view_config was configured during upload_app.
            static_site_url = None
            if app.metadata.get("frontend_entry"):
                static_site_url = get_static_site_url(
                    artifact_id=artifact_id,
                    server_url=self.server.config.public_base_url,
                )

            # Store deployment state with proper timestamps
            self._deployed_applications[application_id] = {
                "display_name": app.metadata["name"],
                "description": app.metadata["description"],
                "artifact_id": artifact_id,
                "version": app.metadata["version"],
                "application_kwargs": app.metadata["application_kwargs"],
                "application_env_vars": app.metadata["application_env_vars"],
                "hypha_token": hypha_token,
                "disable_gpu": disable_gpu,
                "max_ongoing_requests": max_ongoing_requests,
                "application_resources": app.metadata["resources"],
                "authorized_users": app.metadata["authorized_users"],
                "available_methods": app.metadata["available_methods"],
                "static_site_url": static_site_url,
                "started_at": started_at,  # Preserved from original deployment or set for new application
                "last_updated_at": last_updated_at,  # Same as started_at for new, current time for updates
                "last_updated_by": user_id,
                "built_app": app,
                "auto_redeploy": auto_redeploy,
                "ice_servers": ice_servers,
                "deployment_task": None,  # Control task for app deployment
                "is_deployed": asyncio.Event(),  # Track the deployment process
                "undeployment_task": None,  # Control task for app undeployment
                "recovered_app": False,
            }

            # Create and start the deployment task
            deployment_task = asyncio.create_task(
                self._deploy_application(application_id=application_id),
                name=f"Deploy_{application_id}",
            )
            self._deployed_applications[application_id][
                "deployment_task"
            ] = deployment_task

            return application_id

    @schema_method
    async def stop_app(
        self,
        application_id: str = Field(
            ...,
            description="Unique identifier of the deployed application to remove. This is the application ID that was returned when the application was deployed using deploy_app().",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> None:
        """
        Gracefully removes a deployed application from the Ray Serve cluster.

        This method stops the specified application deployment, cleans up all associated
        resources, and removes it from the internal tracking system. The application
        will no longer be available for serving requests after undeployment.

        Undeployment Process:
        1. Validates that the application is currently deployed
        2. Cancels the deployment task to initiate graceful shutdown
        3. Removes the application from Ray Serve
        4. Cleans up internal state and resource tracking
        5. Updates service registrations

        The undeployment happens asynchronously - this method returns immediately
        after initiating the shutdown process. The actual cleanup occurs in the
        background and may take a few moments to complete.

        Args:
            application_id: The deployment to remove. Must be currently deployed.

        Raises:
            RuntimeError: If the application is not currently deployed or if
                         Ray cluster connection is unavailable
            PermissionError: If user lacks admin permissions to undeploy applications

        Note:
            Once undeployed, the application ID becomes available for reuse.
            Any clients connecting to this application will receive errors.
        """
        self._check_initialized()

        # Verify Ray is initialized
        await self.ray_cluster.check_connection()

        # Check user permissions
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"undeploying the application '{application_id}'",
        )
        user_id = context["user"]["id"]

        # Check if application is currently deployed
        if application_id not in self._deployed_applications:
            raise RuntimeError(
                f"Application '{application_id}' is not currently deployed. "
                "Applications started outside of BioEngine cannot be stopped using this method."
            )

        # Start the undeployment in the background
        undeployment_task = asyncio.create_task(
            self._undeploy_application(
                application_id=application_id,
                user_id=user_id,
            )
        )
        self._deployed_applications[application_id][
            "undeployment_task"
        ] = undeployment_task

    @schema_method
    async def stop_all_apps(
        self,
        timeout_seconds: int = Field(
            180,
            description="Timeout in seconds to wait for each application to undeploy gracefully.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> None:
        """
        Removes all currently deployed applications and cleans up associated resources.

        This method performs a complete cleanup of the deployment environment by
        undeploying all currently running applications and removing them from both
        the Ray Serve cluster and internal tracking systems.

        Cleanup Process:
        1. Identifies all currently deployed applications
        2. Initiates graceful shutdown for each deployment
        3. Waits for all undeployment tasks to complete (with timeout)
        4. Cleans up any remaining resources and state
        5. Updates service registrations

        This operation is useful for:
        - Preparing for system maintenance
        - Clearing all deployments before shutdown
        - Recovering from problematic deployment states
        - Development and testing scenarios

        The cleanup process handles timeout scenarios gracefully and provides
        detailed logging of the cleanup status for each application.

        Raises:
            PermissionError: If user lacks admin permissions for cleanup operations
            RuntimeError: If Ray cluster connection is unavailable

        Note:
            This operation cannot be undone. All running applications will be stopped
            and will need to be redeployed manually if needed again.
        """
        # Check if any deployments exist
        if not self._deployed_applications:
            self.logger.info("No applications are currently deployed.")
            return

        self._check_initialized()

        # Check user permissions
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="cleaning up all deployments",
        )
        user_id = context["user"]["id"]

        self.logger.info(f"User '{user_id}' is starting cleanup of all deployments...")

        deployed_application_ids = list(self._deployed_applications.keys())

        # Cancel all deployment tasks
        for application_id in deployed_application_ids:
            await self.stop_app(application_id, context)

        async def undeploy_and_track(application_id: str) -> bool:
            try:
                deployment_info = self._deployed_applications.get(application_id)
                if deployment_info:
                    await asyncio.wait_for(
                        deployment_info["undeployment_task"], timeout=timeout_seconds
                    )
                return True
            except asyncio.TimeoutError:
                self.logger.error(
                    f"Timeout after {timeout_seconds} seconds during undeployment of application '{application_id}'."
                )
                return False
            except Exception as e:
                self.logger.error(
                    f"Error during undeployment of application '{application_id}': {e}"
                )
                return False

        # Wait for all undeployment tasks to complete
        undeployment_tasks = [
            undeploy_and_track(application_id) for application_id in deployed_application_ids
        ]
        results = await asyncio.gather(*undeployment_tasks)

        failed_attempts = len(deployed_application_ids) - sum(results)
        if failed_attempts != 0:
            self.logger.warning(
                f"Failed to clean up all deployments, {failed_attempts} remaining."
            )

    @schema_method
    async def get_app_status(
        self,
        application_ids: Optional[List[str]] = Field(
            None,
            description="List of application IDs to retrieve status for. If not provided, status for all deployed applications will be returned. If a list with only one application ID is provided, only that application's status is returned directly (not nested in a dictionary).",
        ),
        logs_tail: int = Field(
            30,
            description="Number of log lines to retrieve for each deployment replica. If set to -1, retrieves all available logs.",
        ),
        n_previous_replica: int = Field(
            0,
            description="Number of previous replicas to include in the status for each deployment. Set to -1 to retrieve all previous replicas.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive status information for deployed applications.

        Provides detailed information about application deployments, cross-referencing
        internal state with Ray Serve status to ensure accuracy. Includes deployment
        metadata, resource usage, service endpoints, and health status.

        Status Information Includes:
        - Application metadata (name, description, artifact details)
        - Deployment status and health from Ray Serve
        - Resource allocation and usage
        - Service endpoint information for client connections
        - Failure tracking and monitoring data
        - Replica states and deployment details

        For applications that are not deployed, returns a simplified status indicating
        the application is not running with instructions to deploy it.

        Consistency Validation:
        The method validates deployment state by comparing internal tracking with
        Ray Serve status, providing warnings for any inconsistencies found.

        Args:
            application_ids: Optional list of specific application IDs to query.
                           If None, returns status for all tracked applications.
                           If a list with exactly one application ID, returns only
                           that application's status directly (not wrapped in a dictionary).

        Returns:
            - If application_ids is None or contains multiple IDs: Dictionary mapping
              application IDs to their detailed status information.
            - If application_ids contains exactly one ID: The status information for
              that single application (not nested in a dictionary).
            - Empty dictionary if no applications match the criteria (when application_ids is None).

        Raises:
            RuntimeError: If Ray cluster connection is unavailable
            ValueError: If application_ids is not a list or None

        Note:
            Applications in the query list that are not deployed will return a simplified
            status with instructions for deployment.

        Examples:
            # Get all applications (returns dict of all apps)
            all_apps = await get_app_status()
            # Returns: {"app1": {...}, "app2": {...}}

            # Get multiple specific applications (returns dict)
            apps = await get_app_status(application_ids=["app1", "app2"])
            # Returns: {"app1": {...}, "app2": {...}}

            # Get single application (returns status directly)
            app = await get_app_status(application_ids=["app1"])
            # Returns: {...} (single app status, not {"app1": {...}})
        """
        # Determine which applications to check
        if application_ids is None:
            # Return status for all tracked applications
            apps_to_check = list(self._deployed_applications.keys())
            if not apps_to_check:
                return {}
        else:
            # Return status for specified applications
            if not isinstance(application_ids, list):
                raise ValueError("application_ids must be a list of strings or None")
            if len(application_ids) == 0:
                raise ValueError("application_ids list cannot be empty")
            apps_to_check = application_ids

        # Get Ray Serve status
        await self.ray_cluster.check_connection()
        serve_status = await self.ray_cluster.call_with_reconnect(serve.status)

        # Iterate over applications to check
        status_tasks = [
            self._get_app_status(
                application_id=application_id,
                serve_status=serve_status,
                n_previous_replica=n_previous_replica,
                logs_tail=logs_tail,
            )
            for application_id in apps_to_check
        ]
        status_results = await asyncio.gather(*status_tasks)

        output = {
            application_id: status
            for application_id, status in zip(apps_to_check, status_results)
        }

        return output
