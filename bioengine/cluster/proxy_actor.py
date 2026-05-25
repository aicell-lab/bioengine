import os
import re
import time
import json
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import ray
from ray.util.state import StateApiClient, get_log

from bioengine.cluster._ray_compat import (
    GcsClientOptions,
    GlobalState,
    get_dashboard_url_fallback,
)
from ray.util.state.common import (
    DEFAULT_LIMIT,
    DEFAULT_RPC_TIMEOUT,
    ListApiOptions,
    StateResource,
)
from ray.util.state.exception import RayStateApiException

logger = logging.getLogger("ray")
date_format = "%Y-%m-%d %H:%M:%S %Z"


@ray.remote(
    num_cpus=0,
    resources={"node:__internal_head__": 0.001},
    max_restarts=-1,  # Allow unlimited restarts
)
class BioEngineProxyActor:
    """
    Ray actor for monitoring cluster resources and deployment state.

    This actor provides a centralized way to query the Ray cluster for resource
    availability, node information, and deployment status. It acts as a "cluster
    dashboard" that other components can use to make informed decisions about
    resource allocation and scaling.

    Key Features:
    • Real-time cluster resource monitoring (CPU, GPU, memory)
    • Node-by-node resource breakdown with IP addresses
    • Pending resource tracking (jobs, actors, tasks)
    • Deployment replica status monitoring
    • SLURM job integration for HPC environments

    The actor runs on the head node with minimal resource requirements and
    automatically restarts if it fails.
    """

    def __init__(
        self,
        exclude_head_node: bool = False,
        check_pending_resources: bool = False,
        dashboard_url: Optional[str] = None,
    ):
        """
        Initialize the proxy actor with Ray GCS connection and monitoring options.

        Sets up connections to Ray's Global Control Service (GCS) for querying
        cluster state and configures monitoring behavior.

        Args:
            exclude_head_node: Skip head node in resource calculations (useful for worker-only metrics)
            check_pending_resources: Include pending jobs/actors/tasks in cluster state reports
            dashboard_url: Ray dashboard base URL (e.g. "http://127.0.0.1:8265"). When BioEngine
                starts the cluster (single-machine/SLURM modes), the caller knows the URL and
                passes it in. In external-cluster mode the caller passes None and the actor
                falls back to a private-API lookup via ``get_dashboard_url_fallback``.
        """
        # Get the GCS address via public API
        self.gcs_address = ray.get_runtime_context().gcs_address
        self.dashboard_url = dashboard_url

        # Create GCS client options
        gcs_options = GcsClientOptions.create(
            self.gcs_address,
            None,
            allow_cluster_id_nil=True,
            fetch_cluster_id_if_nil=False,
        )

        # Initialize global state for cluster resources
        self.global_state = GlobalState()
        self.global_state._initialize_global_state(gcs_options)

        # Initialize the state API client for querying states
        self.state_api_client = StateApiClient(self.gcs_address)
        self.exclude_head_node = exclude_head_node
        self.check_pending_resources = check_pending_resources

        # Initialize per application actor ID tracking
        # Structure: {app_id: {deployment_name: {replica_id: (actor_id, timestamp, timezone)}}}
        self.application_replicas: Dict[
            str, Dict[str, Dict[str, Tuple[str, float, str]]]
        ] = {}

        self._cached_geo_location: Optional[Dict[str, Optional[Union[str, float]]]] = None

        logger.info(f"ClusterState initialized with GCS address: {self.gcs_address}")

    def get_geo_location(self) -> Dict[str, Optional[Union[str, float]]]:
        """Return the head node's geographic location, fetched from the actor's egress IP.

        The first successful result is cached for the lifetime of the actor;
        results where every provider returned all-None are not cached, so the
        next call retries.
        """
        if self._cached_geo_location is not None:
            return self._cached_geo_location

        # Stdlib only: minimal Ray-head images (e.g. rayproject/ray:2.55.1-py311-cpu)
        # do not ship httpx, so we can't reuse bioengine.utils.geo_location here.
        def _ipwhois() -> Dict[str, Optional[Union[str, float]]]:
            with urllib.request.urlopen("https://ipwho.is/", timeout=10) as r:
                d = json.loads(r.read())
            if not d.get("success"):
                raise ValueError(d.get("message"))
            return {
                "region": d.get("region"),
                "country_name": d.get("country"),
                "country_code": d.get("country_code"),
                "latitude": d.get("latitude"),
                "longitude": d.get("longitude"),
                "timezone": (d.get("timezone") or {}).get("id"),
            }

        def _ipapi_com() -> Dict[str, Optional[Union[str, float]]]:
            with urllib.request.urlopen("http://ip-api.com/json/", timeout=10) as r:
                d = json.loads(r.read())
            if d.get("status") != "success":
                raise ValueError(d.get("message"))
            return {
                "region": d.get("regionName"),
                "country_name": d.get("country"),
                "country_code": d.get("countryCode"),
                "latitude": d.get("lat"),
                "longitude": d.get("lon"),
                "timezone": d.get("timezone"),
            }

        empty = {k: None for k in (
            "region", "country_name", "country_code", "latitude", "longitude", "timezone"
        )}
        for name, fn in (("ipwho.is", _ipwhois), ("ip-api.com", _ipapi_com)):
            try:
                geo = fn()
                logger.info(
                    f"Head-node geo via {name}: {geo['region']}, {geo['country_name']}"
                )
                self._cached_geo_location = geo
                return geo
            except Exception as e:
                logger.warning(f"Head-node geo provider '{name}' failed: {e}")
        logger.error("All head-node geo providers failed.")
        return empty

    def _get_pending_jobs(self) -> List[Dict[str, Any]]:
        """Get list of jobs waiting to start in the cluster.

        Returns:
            List of dictionaries containing job information for pending jobs.
        """
        pending_jobs = self.state_api_client.list(
            resource=StateResource.JOBS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[("status", "=", "PENDING")],
                detail=True,
                explain=False,
            ),
            raise_on_missing_output=True,
        )
        return [asdict(job) for job in pending_jobs]

    def _get_pending_actors(self) -> List[Dict[str, Any]]:
        """Get list of actors waiting to be created in the cluster.

        Returns:
            List of dictionaries containing actor information for pending actors.
        """
        pending_actors = self.state_api_client.list(
            resource=StateResource.ACTORS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[("state", "=", "PENDING_CREATION")],
                detail=True,
                explain=False,
            ),
            raise_on_missing_output=True,
        )
        return [asdict(actor) for actor in pending_actors]

    def _get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get list of tasks waiting for node assignment in the cluster.

        Returns:
            List of dictionaries containing task information for pending tasks.
            Excludes actor creation tasks to avoid duplicates with _get_pending_actors.
        """
        pending_tasks = self.state_api_client.list(
            resource=StateResource.TASKS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[
                    ("state", "=", "PENDING_NODE_ASSIGNMENT"),
                    # Avoid duplicates with `_get_pending_actors`
                    ("type", "!=", "ACTOR_CREATION_TASK"),
                ],
                detail=True,
                explain=False,
            ),
            raise_on_missing_output=True,
        )
        return [asdict(task) for task in pending_tasks]

    def _get_node_ip(self, resources: Dict[str, float]) -> Optional[str]:
        """Extract the IP address from a node's resource dictionary.

        Args:
            resources: Node resource dictionary with resource names as keys.

        Returns:
            IP address string if found (e.g., "192.168.1.1"), None otherwise.
        """
        ip_pattern = r"^node:(\d+\.\d+\.\d+\.\d+)$"
        for resource_name in resources:
            if resource_name.startswith("node:"):
                # Use regex to extract IP address from node resource name
                # (there might be multiple resources starting with "node:")
                match = re.match(ip_pattern, resource_name)
                if match:
                    return match.group(1)

    def _get_accelerator_type(self, resources: Dict[str, float]) -> Optional[str]:
        """Extract the GPU/accelerator type from a node's resource dictionary.

        Args:
            resources: Node resource dictionary with resource names as keys.

        Returns:
            Accelerator type string if found (e.g., "NVIDIA-A100"), None otherwise.
        """
        for resource_name in resources:
            if resource_name.startswith("accelerator_type:"):
                return resource_name.lstrip("accelerator_type:")

    def _get_slurm_job_id(self, resources: Dict[str, float]) -> Optional[str]:
        """Extract the SLURM job ID from a node's resource dictionary.

        Args:
            resources: Node resource dictionary with resource names as keys.

        Returns:
            SLURM job ID string if found, None otherwise.
        """
        for resource_name in resources:
            if resource_name.startswith("slurm_job_id:"):
                return resource_name.split(":")[-1]

    def _is_head_node(self, resources: Dict[str, float]) -> bool:
        """Check if a node resource map belongs to the head node."""
        return "node:__internal_head__" in resources

    def _get_dashboard_webui_url(self) -> Optional[str]:
        """Resolve the local Ray dashboard URL for node summary queries.

        Uses the URL passed in by the caller (single-machine / SLURM modes set
        this from their known dashboard port). In external-cluster mode no URL
        is known up front, so we fall back to the private-API lookup which is
        confined to ``_ray_compat.get_dashboard_url_fallback``.
        """
        webui_url = self.dashboard_url or get_dashboard_url_fallback()
        if not webui_url:
            return None
        if "://" not in webui_url:
            webui_url = f"http://{webui_url}"
        return webui_url

    def _get_per_node_gpu_memory_usage(
        self,
    ) -> Tuple[Dict[str, Dict[str, int]], bool]:
        """Fetch per-node GPU memory usage from Ray dashboard node summary.

        Returns:
            Mapping {node_id: {"total_gpu_memory": int, "used_gpu_memory": int}}
            with values in bytes, and a bool indicating if dashboard data is
            currently available.
        """
        webui_url = self._get_dashboard_webui_url()
        if not webui_url:
            return {}, False

        url = f"{webui_url}/nodes?view=summary"
        # KubeRay (and some other managed Ray distributions) put the dashboard
        # behind a Bearer-auth proxy that requires the token even from inside
        # the head pod. Ray Client auto-propagates RAY_AUTH_TOKEN to actor
        # envs, so we just forward it if present. Local Ray clusters started
        # by BioEngine itself have no auth and ignore the header.
        headers = {}
        auth_token = os.environ.get("RAY_AUTH_TOKEN")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            logger.debug(
                "Failed to fetch Ray node summary from dashboard endpoint %s",
                url,
                exc_info=True,
            )
            return {}, False

        if not payload.get("result"):
            return {}, False

        summary = payload.get("data", {}).get("summary", [])
        per_node_gpu_memory: Dict[str, Dict[str, int]] = {}
        for node in summary:
            node_id = node.get("raylet", {}).get("nodeId")
            if not node_id:
                continue

            total_gpu_memory_mb = 0
            used_gpu_memory_mb = 0
            for gpu in node.get("gpus") or []:
                total_gpu_memory_mb += int(gpu.get("memoryTotal", 0) or 0)
                used_gpu_memory_mb += int(gpu.get("memoryUsed", 0) or 0)

            per_node_gpu_memory[node_id] = {
                "total_gpu_memory": total_gpu_memory_mb * 1024 * 1024,
                "used_gpu_memory": used_gpu_memory_mb * 1024 * 1024,
            }

        return per_node_gpu_memory, True

    def get_cluster_state(self) -> Dict[str, Any]:
        """
        Get comprehensive cluster resource information including per-node breakdown.

        Returns a detailed snapshot of cluster resources showing both total capacity
        and currently used resources across all alive nodes. Useful for
        capacity planning and resource allocation decisions.

        Returns:
            Dictionary with the following structure:
            {
                "cluster": {
                    "total_cpu": float,
                    "used_cpu": float,
                    "total_gpu": float,
                    "used_gpu": float,
                    "pending_resources": {  # if check_pending_resources=True
                        "actors": List[Dict],
                        "jobs": List[Dict],
                        "tasks": List[Dict],
                        "total": int
                    }
                },
                "nodes": {
                    node_id: {
                        "node_ip": Optional[str],
                        "head": bool,
                        "total_cpu": float,
                        "used_cpu": float,
                        "total_gpu": float,
                        "used_gpu": float,
                        "total_gpu_memory": Union[int, str],  # in bytes or "NA"
                        "used_gpu_memory": Union[int, str],  # in bytes or "NA"
                        "total_memory": float,
                        "used_memory": float,
                        "total_object_store_memory": float,
                        "used_object_store_memory": float,
                        "accelerator_type": Optional[str],
                        "slurm_job_id": Optional[str]
                    }
                }
            }
        """
        total_resources_per_node = self.global_state.total_resources_per_node()
        try:
            available_resources_per_node = (
                self.global_state.available_resources_per_node()
            )
        except Exception:
            # Private-API safety net: if Ray ever removes
            # available_resources_per_node, report availability as "unknown"
            # (used = 0) so the rest of the cluster_state structure still
            # populates and callers don't crash. The cluster-level total/used
            # via ray.cluster_resources/available_resources would be the
            # public fallback, but that doesn't give per-node values — so
            # we degrade gracefully rather than guess.
            logger.warning(
                "GlobalState.available_resources_per_node() unavailable; "
                "per-node 'used' values will report 0 until a public API exists.",
                exc_info=True,
            )
            available_resources_per_node = {}
        gpu_memory_per_node, dashboard_available = self._get_per_node_gpu_memory_usage()

        cluster_state = {
            "cluster": {
                "total_cpu": 0,
                "used_cpu": 0,
                "total_gpu": 0,
                "used_gpu": 0,
            },
            "nodes": {},
        }

        # Process each node's resources
        for node_id, total_resources in total_resources_per_node.items():
            if (
                self.exclude_head_node
                and "node:__internal_head__" in total_resources.keys()
            ):
                continue

            available_resources = available_resources_per_node.get(node_id, {})

            # Add per-node resources to the cluster state
            total_cpu = total_resources.get("CPU", 0)
            available_cpu = available_resources.get("CPU", 0)
            total_gpu = total_resources.get("GPU", 0)
            available_gpu = available_resources.get("GPU", 0)
            accelerator_type = (
                "NA" if total_gpu == 0 else self._get_accelerator_type(total_resources)
            )
            total_memory = total_resources.get("memory", 0)
            available_memory = available_resources.get("memory", 0)
            total_object_store_memory = total_resources.get("object_store_memory", 0)
            available_object_store_memory = available_resources.get(
                "object_store_memory", 0
            )
            if total_gpu > 0 and not dashboard_available:
                total_gpu_memory: Union[int, str] = "NA"
                used_gpu_memory: Union[int, str] = "NA"
            else:
                gpu_memory = gpu_memory_per_node.get(
                    node_id, {"total_gpu_memory": 0, "used_gpu_memory": 0}
                )
                total_gpu_memory = gpu_memory["total_gpu_memory"]
                used_gpu_memory = gpu_memory["used_gpu_memory"]

            cluster_state["nodes"][node_id] = {
                "node_ip": self._get_node_ip(total_resources),
                "head": self._is_head_node(total_resources),
                "total_cpu": total_cpu,
                "used_cpu": max(0, total_cpu - available_cpu),
                "total_gpu": total_gpu,
                "used_gpu": max(0, total_gpu - available_gpu),
                "total_gpu_memory": total_gpu_memory,
                "used_gpu_memory": used_gpu_memory,
                "total_memory": total_memory,  # in bytes
                "used_memory": max(0, total_memory - available_memory),  # in bytes
                "total_object_store_memory": total_object_store_memory,  # in bytes
                "used_object_store_memory": max(
                    0, total_object_store_memory - available_object_store_memory
                ),  # in bytes
                "accelerator_type": accelerator_type,
                "slurm_job_id": self._get_slurm_job_id(total_resources),
            }

            # Accumulate cluster resources
            for resource_name in cluster_state["cluster"].keys():
                node_value = cluster_state["nodes"][node_id][resource_name]
                cluster_state["cluster"][resource_name] += node_value

        if self.check_pending_resources:
            # TODO: Don't count task/actor/job in runtime creation
            # Check if there are any runtime environment creations in progress
            cluster_state["cluster"]["pending_resources"] = {
                "actors": self._get_pending_actors(),
                "jobs": self._get_pending_jobs(),
                "tasks": self._get_pending_tasks(),
            }
            cluster_state["cluster"]["pending_resources"]["total"] = sum(
                len(pending_resources)
                for pending_resources in cluster_state["cluster"][
                    "pending_resources"
                ].values()
            )

        return cluster_state

    def get_deployment_replicas(
        self, application_id: str, deployment_name: str
    ) -> Dict[str, str]:
        """
        Get the list of active replica IDs for a specific Ray Serve deployment.

        Queries the Ray cluster for all ALIVE replicas of the specified deployment
        and returns their replica IDs mapped to actor IDs. Useful for monitoring
        deployment health and scaling status.

        Args:
            application_id: Name of the Ray Serve application.
            deployment_name: Name of the specific deployment within the app.

        Returns:
            Mapping of replica ID strings to their actor IDs for replicas
            currently in ALIVE state. Example: {"replica_1": "actor_abc123"}.
        """
        class_name = f"ServeReplica:{application_id}:{deployment_name}"
        deployment_actors = self.state_api_client.list(
            resource=StateResource.ACTORS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[("class_name", "=", class_name), ("state", "=", "ALIVE")],
                detail=False,
                explain=False,
            ),
            raise_on_missing_output=True,
        )
        replica_info = {
            actor.name.split("#")[-1]: actor.actor_id for actor in deployment_actors
        }
        return replica_info

    def register_serve_replica(
        self,
        application_id: str,
        deployment_name: str,
        replica_id: str,
        timezone: str,
    ) -> None:
        """
        Register a Ray Serve replica for tracking and log retrieval.

        Registers a replica's actor ID, timestamp, and timezone for later log retrieval,
        especially useful for accessing logs from replicas that have died.
        Validates that the replica exists in the Ray cluster before registration.

        Args:
            application_id: Name of the Ray Serve application.
            deployment_name: Name of the deployment within the application.
            replica_id: Unique identifier for the replica (e.g., "replica_1").
            timezone: Timezone string for the replica (e.g., "Europe/Stockholm").

        Raises:
            ValueError: If the replica is already registered, not found in the
                       cluster, or if multiple actors with the same name exist.
        """
        registration_time = time.time()
        self.application_replicas.setdefault(application_id, {})
        self.application_replicas[application_id].setdefault(deployment_name, {})

        # Check if the replica is already registered
        if replica_id in self.application_replicas[application_id][deployment_name]:
            actor_id, registration_time, _ = self.application_replicas[application_id][
                deployment_name
            ][replica_id]
            formatted_time = time.strftime(
                date_format, time.localtime(registration_time)
            )
            raise ValueError(
                f"Replica '{replica_id}' already registered for application "
                f"'{application_id}', deployment '{deployment_name}' at {formatted_time}."
            )

        # Query Ray cluster for the actor ID using the naming convention
        actor_name = f"SERVE_REPLICA::{application_id}#{deployment_name}#{replica_id}"
        actor = self.state_api_client.list(
            resource=StateResource.ACTORS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[("name", "=", actor_name)],
                detail=False,
                explain=False,
            ),
            raise_on_missing_output=True,
        )
        if len(actor) == 0:
            raise ValueError(
                f"Actor with name '{actor_name}' not found in Ray cluster."
            )
        if len(actor) > 1:
            raise ValueError(
                f"Multiple actors with name '{actor_name}' found in Ray cluster."
            )
        actor_id = actor[0].actor_id

        # Register the replica with timestamp, actor ID, and timezone
        self.application_replicas[application_id][deployment_name][replica_id] = (
            actor_id,
            registration_time,
            timezone,
        )
        logger.info(
            f"Registered replica '{replica_id}' for application '{application_id}', "
            f"deployment '{deployment_name}' with actor ID '{actor_id}' (replica timezone: {timezone})."
        )

    def clear_application_replicas(self, application_id: str) -> None:
        """
        Clear all registered replicas for a specific application.

        Removes all replica registration data for the given application,
        typically called when an application is deleted or redeployed.

        Args:
            application_id: Name of the Ray Serve application to clear.
        """
        self.application_replicas.pop(application_id, None)
        logger.info(
            f"Cleared all registered replicas for application '{application_id}'."
        )

    def get_actor_logs(
        self,
        actor_id: str,
        tail: int = 100,
        timezone: Optional[str] = None,
        creation_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get logs for a specific Ray actor by its actor ID.

        Retrieves both stdout and stderr logs from the specified actor.
        This is useful for debugging actor issues and monitoring behavior.
        Error messages are returned in place of logs if retrieval fails.

        Args:
            actor_id: Ray actor ID to retrieve logs from.
            tail: Number of lines to retrieve from the end of each log.
                  Use -1 to get the entire log. Default is 100.
            timezone: Optional timezone string for the replica.
            creation_timestamp: Optional Unix timestamp when the replica was created.

        Returns:
            Dictionary with log information. Always includes "stdout" and "stderr" keys.
            If timezone is provided, also includes "timezone" key.
            If creation_timestamp is provided, also includes "creation_timestamp" key.
            Example:
            {
                "creation_timestamp": 1234567890.123,  # if provided
                "timezone": "Europe/Stockholm",  # if provided
                "stdout": ["line1", "line2", ...],
                "stderr": ["error1", ...],
            }
        """
        logs = {}

        # Add optional metadata
        if creation_timestamp is not None:
            logs["creation_timestamp"] = creation_timestamp
        if timezone is not None:
            logs["timezone"] = timezone

        try:
            stdout = next(
                get_log(
                    address=self.gcs_address,
                    actor_id=actor_id,
                    tail=tail,
                    timeout=DEFAULT_RPC_TIMEOUT,
                    suffix="out",
                )
            )
            logs["stdout"] = stdout.splitlines()
        except RayStateApiException as e:
            logs["stdout"] = [f"Error retrieving stdout logs: {str(e)}"]

        try:
            stderr = next(
                get_log(
                    address=self.gcs_address,
                    actor_id=actor_id,
                    tail=tail,
                    timeout=DEFAULT_RPC_TIMEOUT,
                    suffix="err",
                )
            )
            logs["stderr"] = stderr.splitlines()
        except RayStateApiException as e:
            logs["stderr"] = [f"Error retrieving stderr logs: {str(e)}"]

        return logs

    def get_deployment_logs(
        self,
        application_id: str,
        deployment_name: str,
        n_previous_replica: int = 0,
        tail: int = 30,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Get logs for all replicas of a Ray Serve deployment.

        Retrieves stdout and stderr logs from all active replicas and optionally
        from recently terminated replicas. This provides a comprehensive view of
        deployment behavior across all instances.

        Args:
            application_id: Name of the Ray Serve application.
            deployment_name: Name of the deployment within the application.
            n_previous_replica: Number of most recent dead replicas to include.
                               Set to 0 to only get logs from active replicas.
                               Set to -1 to get logs from all dead replicas.
                               Default is 0.
            tail: Number of lines to retrieve from the end of each log.
                  Use -1 to get the entire log. Default is 30.

        Returns:
            Dictionary mapping replica IDs to their log dictionaries:
            {
                "replica_1": {
                    "stdout": ["line1", "line2", ...],
                    "stderr": ["error1", ...]
                },
                "replica_2": {...}
            }
            Error messages are included in stderr if log retrieval fails.
        """
        logs = {}

        # Get active replicas from Ray Serve
        active_replicas = self.get_deployment_replicas(
            application_id=application_id, deployment_name=deployment_name
        )

        # Logs of active replicas
        for replica_id, actor_id in active_replicas.items():
            try:
                # Get replica metadata for timezone and creation timestamp
                replica_info = (
                    self.application_replicas.get(application_id, {})
                    .get(deployment_name, {})
                    .get(replica_id)
                )
                replica_tz = None
                replica_creation = None
                if replica_info:
                    # Unpack: (actor_id, timestamp, timezone)
                    _, replica_creation, replica_tz = replica_info

                actor_logs = self.get_actor_logs(
                    actor_id=actor_id,
                    tail=tail,
                    timezone=replica_tz,
                    creation_timestamp=replica_creation,
                )
                logs[replica_id] = actor_logs
            except Exception as e:
                logs[replica_id] = {
                    "stdout": [],
                    "stderr": [f"Error retrieving logs: {str(e)}"],
                }

        if n_previous_replica == 0:
            return logs

        # Logs of previously registered replicas
        registered_replicas = self.application_replicas.get(application_id, {}).get(
            deployment_name, {}
        )
        dead_replica_ids = [
            replica_id
            for replica_id in registered_replicas.keys()
            if replica_id not in active_replicas
        ]
        # Add logs of the n most recent dead replicas (or all if n_previous_replica is -1)
        dead_replicas_to_fetch = (
            dead_replica_ids[::-1]
            if n_previous_replica == -1
            else dead_replica_ids[::-1][:n_previous_replica]
        )
        for replica_id in dead_replicas_to_fetch:
            actor_id, replica_creation, replica_tz = registered_replicas[replica_id]
            try:
                actor_logs = self.get_actor_logs(
                    actor_id=actor_id,
                    tail=tail,
                    timezone=replica_tz,
                    creation_timestamp=replica_creation,
                )
                logs[replica_id] = actor_logs
            except Exception as e:
                logs[replica_id] = {
                    "stdout": [],
                    "stderr": [f"Error retrieving logs: {str(e)}"],
                }

        return logs
