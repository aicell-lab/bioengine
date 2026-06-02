import asyncio
import importlib.metadata
import inspect
import logging
import os
import re
import subprocess
import tempfile
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union, get_origin

import httpx
import ray
import yaml
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import RemoteService
from hypha_rpc.utils import ObjectProxy
from ray import serve
from ray.serve.handle import DeploymentHandle

import bioengine
from bioengine.apps.proxy_deployment import ProxyDeployment
from bioengine.datasets import BioEngineDatasets
from bioengine.utils import create_logger, update_requirements, validate_manifest


class AppManifest(TypedDict, total=False):
    """
    Schema definition for BioEngine application manifest files.

    The manifest acts as a "blueprint" that describes how to deploy your application.
    It's a YAML file that tells the AppBuilder what Python code to run, who can
    access it, and how the different components work together.

    Required Fields:
    • name: Human-readable application name (shows up in UI)
    • id: Unique technical identifier for the application
    • id_emoji: Visual emoji identifier for easy recognition
    • description: What this application does (help text for users)
    • type: Deployment type (must be "ray-serve" for Ray Serve apps)
    • deployments: List of Python files to deploy (format: "file:ClassName")
    Optional Fields:
    • frontend_entry: Entry HTML file for the static frontend (e.g. "frontend/index.html").
                      When set, static site hosting is configured automatically.

    Example YAML:
    ```yaml
    name: "Image Classifier"
    id: "image-classifier-v1"
    id_emoji: "🖼️"
    description: "Classifies images using a pre-trained CNN model"
    type: "ray-serve"
    deployments: ["classifier:ImageClassifier"]
    frontend_entry: "frontend/index.html"
    ```
    """

    name: str
    id: str
    id_emoji: str
    description: str
    type: str
    deployments: List[str]
    frontend_entry: str


class AppBuilder:
    """
    Main orchestrator for building and deploying BioEngine applications.

    The AppBuilder transforms deployment artifacts (Python code + configuration) into
    fully functional distributed applications running on Ray Serve. Think of it as a
    sophisticated factory that takes your AI model code and turns it into a scalable,
    production-ready service.

    Key Responsibilities:
    • Artifact Management: Downloads and parses deployment code from remote artifacts
    • Environment Setup: Configures Python environments with proper dependencies
    • Ray Integration: Creates Ray Serve deployments with resource allocation
    • Health Monitoring: Adds initialization, testing, and health check capabilities
    • Security: Enforces authentication and authorization for deployed services
    • Communication: Sets up RTC proxy for real-time WebSocket/WebRTC connections

    Workflow Overview:
    1. Load artifact manifest to understand what needs to be deployed
    2. Download Python deployment code and execute it safely
    3. Configure Ray actor options (CPU/GPU/memory requirements)
    4. Wrap deployment classes with BioEngine-specific initialization
    5. Create composition deployments for multi-service applications
    6. Build final application with proxy for external communication

    Configuration:
        apps_workdir: Where to store downloaded artifacts and working directories
        server: Hypha RPC server connection for authentication and artifact access
        artifact_manager: Service for downloading deployment code and manifests

    Parameter Conventions (API):
        - application_kwargs: Dictionary of keyword arguments for each deployment class
        - application_env_vars: Dictionary of environment variables for each deployment class
        - hypha_token: Hypha authentication token for application deployments (set as env var 'HYPHA_TOKEN')
            Used for authenticating to BioEngine datasets and Hypha APIs as the logged-in user.
    """

    def __init__(
        self,
        apps_workdir: Union[str, Path],
        log_file: Optional[str] = None,
        proxy_actor_name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """
        Set up a new AppBuilder instance with basic configuration.

        This constructor prepares the AppBuilder but doesn't connect to external services yet.
        Call initialize() after construction to establish connections to Hypha and artifact services.

        Directory Setup:
        • apps_workdir: Creates isolated workspaces for each deployed application

        Logging Configuration:
        • debug=True: Enables verbose output for troubleshooting deployment issues
        • log_file: Redirects output to file instead of console (useful for production)

        Args:
            apps_workdir: Directory for storing downloaded code and temporary files
            log_file: Optional file path for logging output (None = console only)
            proxy_actor_name: Optional name for the Ray actor that tracks deployment replicas
            debug: Whether to enable detailed debug logging for troubleshooting

        Example:
            ```python
            builder = AppBuilder(
                apps_workdir=f"{os.environ['HOME']}/apps",
                debug=True
            )
            ```
        """
        # Set up logging
        self.logger = create_logger(
            name="AppBuilder",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        # Initialize configuration
        self.apps_workdir = Path(apps_workdir)
        self.bioengine_package_alias = "bioengine-package"
        self.server: Optional[RemoteService] = None
        self.artifact_manager: Optional[ObjectProxy] = None
        self.worker_service_id: Optional[str] = None
        self.proxy_actor_name: Optional[str] = proxy_actor_name
        self.data_server_url: Optional[str] = None

    def complete_initialization(
        self,
        server: RemoteService,
        artifact_manager: ObjectProxy,
        worker_service_id: str,
    ) -> None:
        """
        Connect the AppBuilder to external services required for operation.

        This is the "second phase" of setup that connects to live services. Must be called
        after __init__ but before building any applications. Think of this as "plugging in"
        the AppBuilder to the distributed system infrastructure.

        Service Connections:
        • server: Your authenticated connection to the Hypha workspace
        • artifact_manager: Service that stores and retrieves deployment code

        Args:
            server: Live connection to Hypha server (from connect_to_server())
            artifact_manager: Proxy to artifact management service (from server.get_service())
            worker_service_id: BioEngine worker service ID

        Example:
            ```python
            server = await connect_to_server({"server_url": url, "token": token})
            artifact_manager = await server.get_service("public/artifact-manager")

            builder.complete_initialization(
                server=server,
                artifact_manager=artifact_manager,
                worker_service_id="my-workspace/bioengine-worker",
            )
            ```
        """
        self.server = server
        self.artifact_manager = artifact_manager
        self.worker_service_id = worker_service_id

    def update_data_server_url(self, data_server_url: str) -> None:
        """
        Update the URL for the BioEngine data server used by deployments.

        This method configures the data server URL that will be injected into
        deployment environments. The data server provides access to datasets
        and storage for BioEngine applications.

        Args:
            data_server_url: Full URL of the data server (e.g., "https://data.bioengine.ai")

        Example:
            ```python
            builder.update_data_server_url("https://data.bioengine.ai")
            ```
        """
        self.data_server_url = data_server_url

    async def _load_manifest(
        self, artifact_id: str, version: Optional[str] = None
    ) -> tuple[AppManifest, Optional[str]]:
        """
        Download and parse the application manifest that describes what to deploy.

        The manifest is a YAML file that acts like a "recipe card" for your application,
        telling the AppBuilder what Python files to run, who can access the app, and
        how the different pieces fit together.

        Loading Sources:
        • Remote: Downloads from artifact manager (normal production mode)
        • Local: Reads from local filesystem (development/testing mode)

        Manifest Structure:
        The manifest must contain these required fields:
        • name: Human-readable application name
        • description: What the application does
        • deployments: List of Python files and classes to deploy
        • authorized_users: Who can access the deployed application
        • type: Must be "ray-serve" for Ray Serve deployments

        Validation is performed on the loaded manifest to ensure all required
        fields are present and have valid formats.

        Args:
            artifact_id: Artifact identifier like "my-workspace/my-app"
            version: Specific version to load (None = latest version)

        Returns:
            Tuple of (manifest, resolved_version) where resolved_version is the
            actual version string from the artifact manager (e.g. "0.0.25"), or
            None when loading from local filesystem.

        Raises:
            ValueError: Manifest not found in artifact
            Exception: Network/permission error downloading from artifact manager

        Example Manifest:
            ```yaml
            name: "Image Classifier"
            type: "ray-serve"
            deployments: ["classifier:ImageClassifier"]
            authorized_users: ["user123", "admin@example.com"]
            ```
        """
        manifest = None
        resolved_version: Optional[str] = version

        if os.environ.get("BIOENGINE_LOCAL_ARTIFACT_PATH"):
            # Try to load the file content from local path
            artifact_folder = artifact_id.split("/")[1]
            local_deployments_dir = Path(os.environ["BIOENGINE_LOCAL_ARTIFACT_PATH"])
            local_path = local_deployments_dir / artifact_folder / "manifest.yaml"
            if local_path.exists():
                with open(local_path, "r") as f:
                    manifest = yaml.safe_load(f)
            else:
                self.logger.warning(
                    f"Local manifest file not found: {local_path}. Fetching from remote artifact manager."
                )

        # If manifest was not loaded locally, fetch from remote
        if manifest is None:
            artifact = await self.artifact_manager.read(artifact_id, version=version)
            manifest = artifact.get("manifest")
            if manifest is None:
                raise ValueError(f"Manifest not found in artifact {artifact_id}.")

            # Resolve the actual version when "latest" was requested
            if version is None:
                versions = artifact.get("versions") or []
                if versions:
                    latest = max(versions, key=lambda v: v["created_at"])
                    resolved_version = latest["version"]

        # Validate the manifest structure
        validate_manifest(manifest)

        return manifest, resolved_version

    async def _update_ray_actor_options(
        self,
        deployment: serve.Deployment,
        application_id: str,
        disable_gpu: bool,
        non_secret_env_vars: Dict[str, str],
        secret_env_vars: Dict[str, str],
        artifact_id: str,
        version: str,
    ) -> serve.Deployment:
        """
        Configure the Ray runtime environment for BioEngine deployments.

        This method transforms a basic Ray deployment into a BioEngine-aware one by
        setting up the proper environment, dependencies, and directory structure.
        Think of it as "installing the BioEngine runtime" into your deployment.

        Environment Setup:
        • Creates isolated working directory for this specific application
        • Sets up temporary directories and home directory paths
        • Configures access to shared data directory
        • Injects BioEngine Python package requirements

        Hypha Integration:
        • Provides authentication token for server communication
        • Sets server URL and workspace for RPC connections
        • Enables deployments to register services and communicate

        Resource Management:
        • Optionally disables GPU allocation if not needed
        • Sets request concurrency limits for the deployment
        • Configures memory and CPU allocation

        Args:
            deployment: Base Ray deployment to enhance with BioEngine capabilities
            application_id: Unique ID for creating isolated workspace directory
            disable_gpu: Set to True to force CPU-only execution
            non_secret_env_vars: Environment variables to set directly
            secret_env_vars: Environment variables with sensitive values
            artifact_id: Artifact identifier like "my-workspace/my-app" added as env var 'HYPHA_ARTIFACT_ID'
            version: Resolved artifact version added as env var 'HYPHA_ARTIFACT_VERSION'

        Returns:
            Enhanced deployment with BioEngine runtime environment configured

        Technical Details:
        The working directory structure created is:
        - apps_workdir/application_id/ (main workspace)
        - apps_workdir/application_id/tmp/ (temporary files)
        """
        ray_actor_options = deployment.ray_actor_options.copy()

        # Disable GPU if not enabled
        if disable_gpu:
            ray_actor_options["num_gpus"] = 0

        # Update runtime environment with BioEngine requirements
        runtime_env = ray_actor_options.setdefault("runtime_env", {})
        pip_requirements = runtime_env.setdefault("pip", [])
        env_vars = runtime_env.setdefault("env_vars", {})

        # Update pip requirements
        # hypha-rpc and pydantic to serialize/de-serialize `schema_method` decorated methods
        # httpx to support streaming datasets from BioEngine data server (zarr needs to be added by the application itself)
        pip_requirements = update_requirements(
            pip_requirements,
            select=["httpx", "hypha-rpc", "pydantic"],
            extras=["worker"],
        )
        runtime_env["pip"] = pip_requirements

        # Pre-flight: if the runtime_env's resolved pydantic-core differs
        # from the BioEngine worker's, abort early with a clear error instead of
        # discovering the mismatch later when Ray Serve replicas fail to
        # unpickle the deployment definition with the cryptic
        # "'FieldInfo' object has no attribute 'exclude_if'".
        self._check_pydantic_core_compatibility(pip_requirements)

        # bioengine is available to all deployments via the job-level py_modules set
        # in ray.init() — no need to re-upload it per deployment.

        # Add user defined environment variables
        env_vars.update(non_secret_env_vars)
        # Secret env vars overwrite non-secret env vars
        hidden_secret_env_vars = {key: "*****" for key in secret_env_vars.keys()}
        env_vars.update(hidden_secret_env_vars)

        # Add BioEngine environment variables
        app_work_dir = self.apps_workdir / application_id
        env_vars["HOME"] = str(app_work_dir)

        tmp_dir = str(app_work_dir / "tmp")
        env_vars["TMPDIR"] = tmp_dir
        env_vars["TEMP"] = tmp_dir
        env_vars["TMP"] = tmp_dir

        env_vars["HYPHA_SERVER_URL"] = self.server.config.public_base_url
        env_vars["HYPHA_WORKSPACE"] = self.server.config.workspace
        env_vars["HYPHA_ARTIFACT_ID"] = artifact_id
        env_vars["HYPHA_ARTIFACT_VERSION"] = version

        env_vars["BIOENGINE_WORKER_SERVICE_ID"] = self.worker_service_id

        # Validate all environment variables
        for key, value in env_vars.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Environment variable key '{key}' must be a string, got '{type(key)}'."
                )
            if not isinstance(value, str):
                raise ValueError(
                    f"Environment variable '{key}' must be a string, got '{type(value)}'."
                )

        runtime_env["env_vars"] = env_vars

        # Update deployment options
        ray_actor_options["runtime_env"] = runtime_env
        updated_deployment = deployment.options(
            ray_actor_options=ray_actor_options,
        )
        return updated_deployment

    def _check_pydantic_core_compatibility(
        self, pip_requirements: List[str]
    ) -> None:
        """Pre-flight check: resolve the deployment's runtime_env pip list
        via ``uv pip compile`` and require the resolved ``pydantic-core``
        version to match the BioEngine worker's installed pydantic-core.

        Why this matters: Ray Serve replicas unpickle the deployment
        definition inside the runtime_env venv. The BioEngine worker
        pickles them with its own pydantic. If the venv resolves to a
        different ``pydantic-core``, cloudpickle reconstruction fails
        with errors like
        ``AttributeError: 'FieldInfo' object has no attribute 'exclude_if'``.
        Failing fast here turns that opaque downstream crash into a clear
        startup error pointing at the version conflict.

        pydantic-core is guaranteed to be installed in the BioEngine
        worker (pydantic is a hard dep, and pydantic always pulls
        pydantic-core), and ``update_requirements`` injects pydantic
        into every deployment's runtime_env pip list — so pydantic-core
        is always in the resolved set on both sides.

        Raises:
            RuntimeError: when the resolved pydantic-core version differs
                from the BioEngine worker's installed pydantic-core.
        """
        worker_pc_version = importlib.metadata.version("pydantic_core")

        if not pip_requirements:
            return

        # Write the deployment's pip list to a temp file and have uv
        # resolve it. uv is already in the worker image (preferred Ray
        # runtime_env installer from Ray 2.47+) so this adds no new
        # build dep.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".in", delete=False
        ) as f:
            f.write("\n".join(pip_requirements) + "\n")
            req_path = Path(f.name)

        try:
            result = subprocess.run(
                [
                    "uv",
                    "pip",
                    "compile",
                    "--no-header",
                    "--quiet",
                    "--no-cache",  # the worker pod may run with HOME=/nonexistent
                    str(req_path),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
        finally:
            req_path.unlink(missing_ok=True)

        match = re.search(
            r"^pydantic-core==([0-9A-Za-z.+-]+)",
            result.stdout,
            re.MULTILINE,
        )
        if not match:
            # Should not happen — update_requirements always injects pydantic
            # which pulls pydantic-core. Surface loudly if it ever does.
            raise RuntimeError(
                "uv pip compile resolved the runtime_env pip list without "
                "any pydantic-core entry. This shouldn't happen: BioEngine "
                "injects pydantic into every deployment's runtime_env, and "
                "pydantic always requires pydantic-core. Resolved output:\n"
                f"{result.stdout[:1000]}"
            )

        resolved_pc_version = match.group(1)
        if resolved_pc_version == worker_pc_version:
            self.logger.debug(
                "pydantic-core pre-flight OK: BioEngine worker and "
                "runtime_env both at %s.",
                worker_pc_version,
            )
            return

        raise RuntimeError(
            "pydantic-core version mismatch between the BioEngine worker "
            f"({worker_pc_version}) and the application's runtime_env "
            f"({resolved_pc_version}). BioEngine pickles Ray Serve "
            "deployment definitions in the worker process and Ray Serve "
            "replicas unpickle them inside the runtime_env venv, so the "
            "two sides must agree on pydantic-core. Cross-version "
            "unpickle fails with errors like \"'FieldInfo' object has "
            "no attribute 'exclude_if'\". Resolution: either pin "
            "pydantic in this application's requirements so it resolves "
            f"to pydantic-core=={worker_pc_version}, or rebuild the "
            f"BioEngine worker image with pydantic-core=={resolved_pc_version}."
        )

    def _sanitize_recovery_env_vars(
        self, application_env_vars: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """Return env vars safe to persist in proxy app_data for recovery."""
        sanitized_env_vars: Dict[str, Dict[str, str]] = {}
        for deployment_class, env_vars in application_env_vars.items():
            sanitized_env_vars[deployment_class] = {
                key: value
                for key, value in env_vars.items()
                if not key.startswith("_") and key != "HYPHA_TOKEN"
            }
        return sanitized_env_vars

    def _update_init(
        self,
        application_id: str,
        deployment: serve.Deployment,
        secret_env_vars: Dict[str, str],
        debug: bool = False,
    ) -> serve.Deployment:
        """
        Wrap the deployment's __init__ method to set up the BioEngine execution environment.

        This method intercepts the deployment class initialization to perform essential
        setup tasks before the user's __init__ code runs. It's like adding a "pre-flight
        checklist" that ensures everything is properly configured.

        Environment Preparation:
        • Creates and switches to isolated working directory
        • Sets up access to shared data directory
        • Assigns unique replica ID for logging and identification
        • Initializes deployment state tracking variables

        State Management:
        The wrapper adds these internal state variables:
        • _deployment_initialized: Tracks async initialization completion
        • _deployment_tested: Tracks deployment testing completion
        • _health_check_lock: Prevents concurrent health check execution

        Directory Structure:
        Each deployment gets its own workspace based on the application ID,
        completely isolated from other deployments for security and stability.

        Args:
            deployment: Ray deployment to enhance with BioEngine initialization

        Returns:
            Deployment with wrapped __init__ method that sets up BioEngine environment

        Note:
        This wrapping is transparent to the user's deployment code - their __init__
        method runs normally after the BioEngine setup is complete.
        """

        orig_init = getattr(deployment.func_or_class, "__init__")

        data_server_url = self.data_server_url
        proxy_actor_name = self.proxy_actor_name

        @wraps(orig_init)
        def wrapped_init(self, *args, **kwargs):
            logger = logging.getLogger("ray.serve")
            logger.setLevel(logging.DEBUG if debug else logging.INFO)

            # Register this serve replica with the BioEngineProxyActor for tracking
            proxy_actor_handle = None
            try:
                proxy_actor_handle = ray.get_actor(
                    name=proxy_actor_name, namespace="bioengine"
                )
                logger.debug(
                    f"✅ Successfully retrieved BioEngineProxyActor '{proxy_actor_name}'"
                )
            except ValueError as e:
                logger.error(
                    f"❌ BioEngineProxyActor '{proxy_actor_name}' not found: {e}"
                )
            except Exception as e:
                logger.error(
                    f"❌ Unexpected error getting BioEngineProxyActor '{proxy_actor_name}': {e}",
                    exc_info=True,
                )

            if proxy_actor_handle:
                try:
                    replica_context = serve.get_replica_context()
                    deployment_name = replica_context.deployment
                    replica_id = replica_context.replica_tag

                    # Get timezone at replica level
                    replica_timezone = time.strftime("%Z")

                    logger.debug(
                        f"Retrieved replica context: tag={replica_id}, "
                        f"deployment={replica_context.deployment}, "
                        f"app={replica_context.app_name}, "
                        f"timezone={replica_timezone}"
                    )

                    proxy_actor_handle.register_serve_replica.remote(
                        application_id=application_id,
                        deployment_name=deployment_name,
                        replica_id=replica_id,
                        timezone=replica_timezone,
                    )
                    logger.info(
                        f"✅ Registered replica '{replica_id}' with BioEngineProxyActor."
                    )
                except ValueError as e:
                    logger.error(f"❌ Invalid replica registration parameters: {e}")
                except Exception as e:
                    logger.error(
                        f"❌ Unable to register replica with BioEngineProxyActor: {e}",
                        exc_info=True,
                    )

            # Ensure the current working directory is set to the application working directory
            workdir = Path.home().resolve()  # Home directory is set to the apps_workdir
            os.environ["HOME"] = str(
                workdir
            )  # Update the HOME environment variable with resolved path
            workdir.mkdir(parents=True, exist_ok=True)
            os.chdir(workdir)
            logger.info(f"📁 Working directory: {workdir}/")

            # Initialize deployment states
            self._bioengine_replica_initialized = False
            self._bioengine_replica_test_failed = False
            self._bioengine_test_task = None  # Background task for test_deployment

            # Create a health check lock to synchronize health checks
            self._bioengine_health_check_lock = asyncio.Lock()

            # Update secret environment variable with real value (previously set to "*****")
            for env_var, value in secret_env_vars.items():
                os.environ[env_var] = value

            # Initialize BioEngine datasets
            self.bioengine_datasets = BioEngineDatasets(
                data_server_url=data_server_url,
                hypha_token=os.getenv("HYPHA_TOKEN"),
                logger=logger,
            )

            # Call the original __init__ method
            orig_init(self, *args, **kwargs)

        setattr(deployment.func_or_class, "__init__", wrapped_init)
        return deployment

    def _update_async_init(self, deployment: serve.Deployment) -> serve.Deployment:
        """
        Wrap the deployment's async_init method to handle initialization with proper tracking.

        Many deployments need to do expensive setup work after basic initialization -
        loading models, connecting to databases, downloading files, etc. This wrapper
        ensures that work is properly tracked and timed for monitoring purposes.

        Flexibility Support:
        • Handles deployments with no async_init method (creates default no-op)
        • Supports both async and sync async_init implementations
        • Automatically converts sync methods to async using thread execution

        Progress Tracking:
        • Logs initialization start/completion with timing information
        • Sets _deployment_initialized flag when complete
        • Provides detailed error reporting if initialization fails

        Monitoring Integration:
        Uses the replica_id for clear logging so you can track which specific
        deployment instances are initializing in a multi-replica deployment.

        Args:
            deployment: Ray deployment to enhance with async initialization tracking

        Returns:
            Deployment with wrapped async_init method that provides progress tracking

        Example User Code:
            ```python
            class MyDeployment:
                async def async_init(self):
                    # This will be wrapped and tracked automatically
                    self.model = await load_large_model()
                    logger.info("Model loaded successfully")
            ```
        """
        orig_async_init = getattr(
            deployment.func_or_class, "async_init", lambda self: None
        )

        @wraps(orig_async_init)
        async def wrapped_async_init(self):
            logger = logging.getLogger("ray.serve")
            logger.info(f"⚡ Starting async initialization...")

            start_time = time.time()
            try:
                # Check if the original async_init method is async
                if inspect.iscoroutinefunction(orig_async_init):
                    await orig_async_init(self)
                else:
                    await asyncio.to_thread(orig_async_init, self)

                elapsed_time = time.time() - start_time
                self._bioengine_replica_initialized = True
                logger.info(
                    f"✅ Async initialization completed successfully in {elapsed_time:.2f}s"
                )

            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(
                    f"❌ Async initialization failed after {elapsed_time:.2f}s: {e}"
                )
                raise

        setattr(deployment.func_or_class, "async_init", wrapped_async_init)
        return deployment

    def _update_test_deployment(self, deployment: serve.Deployment) -> serve.Deployment:
        """
        Wrap the deployment's test_deployment method to validate functionality before going live.

        This wrapper ensures that deployments can prove they're working correctly before
        they start serving real requests. Think of it as a "smoke test" that runs
        automatically during deployment startup to catch problems early.

        Testing Philosophy:
        • Tests run after initialization but before the deployment accepts traffic
        • Failed tests prevent the deployment from becoming "healthy"
        • Tests should be lightweight but validate core functionality

        Implementation Support:
        • Creates default no-op test if deployment doesn't define one
        • Supports both async and sync test_deployment methods
        • Converts sync tests to async using thread execution for consistency

        State Management:
        • Sets _deployment_tested flag when tests pass
        • Provides detailed timing and error reporting
        • Tracks test status for health check validation

        Args:
            deployment: Ray deployment to enhance with test execution tracking

        Returns:
            Deployment with wrapped test_deployment method that validates functionality

        Example User Code:
            ```python
            class MyDeployment:
                def test_deployment(self):
                    # This will be wrapped and tracked automatically
                    result = self.predict_sample_input()
                    assert result is not None, "Model prediction failed"
                    logger.info("Deployment test passed!")
            ```
        """
        orig_test_deployment = getattr(
            deployment.func_or_class, "test_deployment", lambda self: None
        )

        @wraps(orig_test_deployment)
        async def wrapped_test_deployment(self):
            logger = logging.getLogger("ray.serve")
            logger.info(f"🧪 Starting deployment test...")

            start_time = time.time()
            try:
                # Check if the original test_deployment method is async
                if inspect.iscoroutinefunction(orig_test_deployment):
                    await orig_test_deployment(self)
                else:
                    await asyncio.to_thread(orig_test_deployment, self)

                # Mark the deployment as tested
                elapsed_time = time.time() - start_time
                self._bioengine_replica_test_failed = False
                logger.info(
                    f"✅ Deployment test completed successfully in {elapsed_time:.2f}s"
                )

            except Exception as e:
                elapsed_time = time.time() - start_time
                self._bioengine_replica_test_failed = True
                logger.error(
                    f"❌ Deployment test failed after {elapsed_time:.2f}s: {e}"
                )
                raise RuntimeError(f"Deployment test failed: {e}")

        setattr(deployment.func_or_class, "test_deployment", wrapped_test_deployment)
        return deployment

    def _update_health_check(self, deployment: serve.Deployment) -> serve.Deployment:
        """
        Add a comprehensive health check system to control deployment readiness.

        This wrapper creates a sophisticated health check that ensures deployments
        don't start accepting requests until they're fully ready. It's like having
        a "traffic light" that stays red until everything is properly set up.

        Health Check Orchestration:
        • Runs async_init if not already completed
        • Executes test_deployment to validate functionality
        • Calls original health check method if it exists
        • Uses a lock to prevent concurrent health check execution

        Ray Serve Integration:
        Ray Serve calls check_health() repeatedly during deployment startup.
        The deployment remains in "DEPLOYING" state until health checks pass,
        ensuring users only see fully functional deployments.

        Concurrency Safety:
        Uses an async lock to ensure only one health check runs at a time,
        preventing race conditions during the initialization phase.

        Args:
            deployment: Ray deployment to enhance with comprehensive health checking

        Returns:
            Deployment with integrated health check that validates readiness

        Note:
        The health check ensures both async_init and test_deployment complete
        successfully before marking the deployment as healthy and ready for traffic.
        """
        orig_health_check = getattr(
            deployment.func_or_class, "check_health", lambda self: None
        )

        @wraps(orig_health_check)
        async def check_health(self):
            logger = logging.getLogger("ray.serve")

            async with self._bioengine_health_check_lock:
                # Ensure async initialization has completed
                if not self._bioengine_replica_initialized:
                    await self.async_init()

                # Launch test_deployment in the background if not already started
                if self._bioengine_test_task is None:
                    logger.info("🚀 Launching deployment test in the background...")
                    # Run test_deployment in thread pool executor to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    self._bioengine_test_task = loop.run_in_executor(
                        None, lambda: asyncio.run(self.test_deployment())
                    )

                # Check if the background test task failed
                if self._bioengine_replica_test_failed:
                    raise RuntimeError(
                        "Deployment test failed - deployment is unhealthy"
                    )

                try:
                    # Ensure data server can be reached
                    await self.bioengine_datasets.ping_data_server()

                    # Check if the original health check method is async
                    if inspect.iscoroutinefunction(orig_health_check):
                        await orig_health_check(self)
                    else:
                        await asyncio.to_thread(orig_health_check, self)

                except Exception as e:
                    logger.error(f"❌ Health check failed: {e}")
                    raise

        # Add the updated health check method to the deployment class
        setattr(deployment.func_or_class, "check_health", check_health)
        return deployment

    def _get_init_param_info(
        self, deployment: serve.Deployment
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a deployment class to understand what parameters it expects.

        This method uses Python's introspection capabilities to examine the __init__
        method signature and extract information about what parameters the deployment
        class needs for initialization. It's like reading the "ingredient list" before
        cooking a recipe.

        Parameter Analysis:
        • Extracts parameter names, types (if annotated), and default values
        • Identifies which parameters are required vs optional
        • Handles special parameter types like *args and **kwargs
        • Excludes 'self' parameter (not relevant for external callers)

        Type Information:
        • Captures type hints if provided (e.g., str, int, DeploymentHandle)
        • Records default values for optional parameters
        • Identifies parameters that accept variable arguments

        Special Handling:
        • DeploymentHandle parameters are flagged for composition support
        • *args and **kwargs are tracked but not included in main parameter list
        • Unknown types are recorded as None rather than failing

        Args:
            deployment: Ray deployment to analyze for parameter requirements

        Returns:
            Dictionary with parameter info structure:
            {
                "param_name": {"type": param_type, "default": default_value},
                "__has_var_positional__": {"type": None, "default": True/False},
                "__has_var_keyword__": {"type": None, "default": True/False}
            }

        Example:
            For a class with `__init__(self, model_path: str, batch_size: int = 32)`:
            Returns: {
                "model_path": {"type": str, "default": None},
                "batch_size": {"type": int, "default": 32}
            }
        """
        sig = inspect.signature(deployment.func_or_class.__init__)
        params = {}
        has_var_positional = False
        has_var_keyword = False

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            # Skip *args and **kwargs parameters as they have special handling
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                has_var_positional = True
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                has_var_keyword = True
                continue

            param_type = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else None
            )

            # Use a sentinel value to distinguish "no default" from "default is None"
            if param.default is inspect.Parameter.empty:
                has_default = False
                default_value = None
            else:
                has_default = True
                default_value = param.default

            params[name] = {
                "type": param_type,
                "default": default_value,
                "has_default": has_default,
            }

        # Store information about *args and **kwargs for validation
        params["__has_var_positional__"] = {"type": None, "default": has_var_positional}
        params["__has_var_keyword__"] = {"type": None, "default": has_var_keyword}
        return params

    def _is_instance_of_type(self, value: Any, expected_type: Any) -> bool:
        """
        Check if a value is an instance of the expected type, handling subscripted generics.

        This method works around the limitation that subscripted generics like typing.List[str]
        cannot be used directly with isinstance(). It extracts the origin type (e.g., list from
        List[str]) and performs the type check against that.

        Args:
            value: The value to check
            expected_type: The expected type, which may be a subscripted generic

        Returns:
            True if the value is an instance of the expected type, False otherwise
        """
        try:
            # First try the direct isinstance check (works for regular types)
            return isinstance(value, expected_type)
        except TypeError:
            # Handle subscripted generics like typing.List[str], typing.Dict[str, int], etc.
            # Use get_origin to extract the base type (e.g., list from List[str])
            origin = get_origin(expected_type)
            if origin is not None:
                return isinstance(value, origin)
            # If we can't handle it, just skip the type check
            return True

    def _check_params(
        self, init_params: Dict[str, Dict[str, Any]], kwargs: Dict[str, Any]
    ) -> None:
        """
        Validate that user-provided parameters match what the deployment expects.

        This method acts like a "compatibility checker" that ensures the parameters
        you want to pass to a deployment class are actually accepted by that class.
        It prevents runtime errors by catching parameter mismatches early.

        Validation Rules:
        • All provided parameters must be expected by the __init__ method
        • Required parameters (no default value) must be provided
        • Parameter types must match if type hints are available
        • DeploymentHandle parameters are handled specially for composition

        Flexibility Support:
        • Classes with **kwargs accept any additional parameters
        • Classes without **kwargs reject unexpected parameters
        • *args parameters are not supported (would complicate composition)

        Error Prevention:
        • Catches typos in parameter names before deployment
        • Validates type compatibility where possible
        • Ensures required parameters aren't missing

        Args:
            init_params: Parameter structure from _get_init_param_info()
            kwargs: Dictionary of parameters to validate

        Raises:
            ValueError: Parameter name not expected, required parameter missing,
                       type mismatch, or *args detected (not supported)

        Example Validation:
            For a deployment expecting (model_path: str, batch_size: int = 32):
            ✓ {"model_path": "/path/to/model"} - Valid (uses default batch_size)
            ✓ {"model_path": "/path", "batch_size": 64} - Valid
            ✗ {"model_file": "/path"} - Invalid (typo in parameter name)
            ✗ {"batch_size": 64} - Invalid (missing required model_path)
        """
        # Extract special flags for **kwargs handling (*args is not supported)
        has_var_keyword = init_params.get("__has_var_keyword__", {}).get(
            "default", False
        )

        # Remove special flags from init_params for normal processing
        filtered_init_params = {
            k: v for k, v in init_params.items() if not k.startswith("__has_var_")
        }

        # Check if all provided parameters are expected
        for key in kwargs:
            if key not in filtered_init_params:
                if not has_var_keyword:
                    raise ValueError(
                        f"Unexpected parameter '{key}' provided. "
                        f"Expected one of {list(filtered_init_params.keys())}."
                    )
                # If **kwargs is present, allow any additional parameters
            else:
                expected_type = filtered_init_params[key]["type"]
                if expected_type and not self._is_instance_of_type(
                    kwargs[key], expected_type
                ):
                    raise ValueError(
                        f"Parameter '{key}' must be of type {expected_type.__name__}, "
                        f"but got {type(kwargs[key]).__name__}."
                    )

        # Check if all required parameters are provided
        for key, param_info in filtered_init_params.items():
            if param_info["type"] == DeploymentHandle:
                # DeploymentHandle parameters are handled separately
                continue

            # A parameter is required if it has no default value
            if not param_info["has_default"] and key not in kwargs:
                param_type = param_info["type"]
                type_name = param_type.__name__ if param_type else "unknown"
                raise ValueError(
                    f"Missing required parameter '{key}' of type {type_name}."
                )

    async def _load_deployment(
        self,
        application_id: str,
        artifact_id: str,
        version: Optional[str],
        python_file: str,
        class_name: str,
        disable_gpu: bool,
        env_vars: Dict[str, str],
        debug: bool = False,
    ) -> serve.Deployment:
        """
        Download and transform Python code into a Ray Serve deployment.

        This method performs the "magic" of converting stored Python code into a
        running deployment. It downloads the code, executes it safely, and wraps
        it with all the BioEngine functionality needed for production deployment.

        Code Loading Process:
        • Downloads Python file from artifact storage (or loads from local path)
        • Executes code in controlled environment to extract deployment class
        • Validates that the specified class exists and is properly defined
        • Handles both remote artifacts and local development files

        BioEngine Enhancement:
        After loading the base class, it enhances it with:
        • Resource allocation (CPU/GPU/memory configuration)
        • Environment setup (working directories, data access)
        • Lifecycle management (__init__, async_init, test_deployment, health_check)
        • Authentication and workspace isolation

        Security Considerations:
        • Code execution happens in a restricted globals environment
        • Each deployment gets isolated working directory
        • Authentication tokens are properly scoped

        Args:
            application_id: Unique ID for creating isolated workspace
            artifact_id: Where to find the deployment code (e.g., "workspace/app-name")
            version: Specific version to load (None = latest)
            python_file: Which file to load (format: "filename.py")
            class_name: Which class to load (format: "ClassName")
            disable_gpu: Force CPU-only execution regardless of class defaults
            env_vars: Environment variables to set for the deployment, secret env vars start with "_"

        Returns:
            Fully configured Ray Serve deployment ready for use in applications

        Raises:
            FileNotFoundError: Local file not found in development mode
            ValueError: Invalid import_path format or class not found in code
            RuntimeError: Code execution failed or deployment configuration failed
            Exception: Network error downloading code or permission issues

        Example:
            Loading a model deployment:
            import_path="model_server:ImageClassifier" loads the ImageClassifier
            class from model_server.py file in the artifact.
        """
        code_content = None
        local_path = None

        if os.environ.get("BIOENGINE_LOCAL_ARTIFACT_PATH"):
            # Try to load the file content from local path
            artifact_folder = artifact_id.split("/")[1]
            local_deployments_dir = Path(os.environ["BIOENGINE_LOCAL_ARTIFACT_PATH"])
            local_path = local_deployments_dir / artifact_folder / python_file
            if local_path.exists():
                self.logger.debug(
                    f"Loading deployment code from local path: {python_file} in folder {artifact_folder}/"
                )
                with open(local_path, "r") as f:
                    code_content = f.read()
            else:
                self.logger.warning(
                    f"Local deployment file not found: {local_path}. Fetching from remote artifact manager."
                )

        # If code was not loaded locally, fetch from remote
        if code_content is None:
            try:
                # Get download URL for the file
                self.logger.debug(
                    f"Downloading deployment code from artifact: {artifact_id}, file: {python_file}"
                )
                download_url = await self.artifact_manager.get_file(
                    artifact_id=artifact_id,
                    version=version,
                    file_path=python_file,
                )

                # Download the file content with timeout (30s for all operations)
                download_timeout = httpx.Timeout(30.0)
                async with httpx.AsyncClient(timeout=download_timeout) as client:
                    response = await client.get(download_url)

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to download deployment code from {download_url}: "
                        f"HTTP {response.status_code} - {response.text}"
                    )

                code_content = response.text
            except Exception as e:
                self.logger.error(
                    f"Error downloading deployment code from {artifact_id}: {e}"
                )
                raise e

        # Split environment variables into secret and non-secret
        # Secret env vars start with "_" and are hidden in the logs and ray actor config
        # Remove prepended "_" from secret env var names
        non_secret_env_vars = {
            k: v for k, v in env_vars.items() if not k.startswith("_")
        }
        secret_env_vars = {k[1:]: v for k, v in env_vars.items() if k.startswith("_")}

        # Create a restricted globals dictionary for sandboxed execution - pass some deployment options
        try:
            # Execute the code in a sandboxed environment
            safe_globals = non_secret_env_vars | secret_env_vars  # merged all env vars
            exec(code_content, safe_globals)
            if class_name not in safe_globals:
                raise ValueError(f"{class_name} not found in {artifact_id}")
            deployment = safe_globals[class_name]
            if not deployment:
                raise RuntimeError(f"Error loading {class_name} from {artifact_id}")

            # Update environment variables and requirements
            deployment = await self._update_ray_actor_options(
                deployment=deployment,
                application_id=application_id,
                disable_gpu=disable_gpu,
                non_secret_env_vars=non_secret_env_vars,
                secret_env_vars=secret_env_vars,
                artifact_id=artifact_id,
                version=version,
            )

            # Update the deployment class methods
            deployment = self._update_init(application_id, deployment, secret_env_vars, debug)
            deployment = self._update_async_init(deployment)
            deployment = self._update_test_deployment(deployment)
            deployment = self._update_health_check(deployment)

            if local_path:
                self.logger.info(
                    f"Successfully loaded and configured deployment class '{class_name}' from local path '{local_path}/'"
                )
            else:
                self.logger.info(
                    f"Successfully loaded and configured deployment class '{class_name}' from artifact '{artifact_id}'"
                )
            return deployment
        except Exception as e:
            self.logger.error(
                f"Error creating deployment class from {artifact_id}: {e}"
            )
            raise e

    def _calculate_required_resources(
        self, deployments: List[serve.Deployment]
    ) -> Dict[str, Union[int, float]]:
        """
        Calculate total resource requirements across all deployment components.

        Before deploying an application, it's helpful to know what computational
        resources will be needed. This method sums up the CPU, GPU, and memory
        requirements from all deployment classes to give you the total "bill."

        Resource Aggregation:
        • CPU cores: Sums num_cpus from all deployments
        • GPU devices: Sums num_gpus from all deployments
        • Memory: Sums memory requirements from all deployments

        Use Cases:
        • Capacity planning: Ensure cluster has enough resources
        • Cost estimation: Understand resource costs before deployment
        • Scheduling: Help Ray Serve schedule deployments efficiently
        • Monitoring: Track actual vs expected resource usage

        Args:
            deployments: List of configured Ray deployments to analyze

        Returns:
            Resource summary with keys "num_cpus", "num_gpus", "memory"
            containing the total requirements across all deployments

        Example:
            If you have 2 deployments:
            - Deployment A: 2 CPUs, 1 GPU, 4GB memory
            - Deployment B: 1 CPU, 0 GPU, 2GB memory

            Result: {"num_cpus": 3, "num_gpus": 1, "memory": 6442450944}
        """
        required_resources = {
            "num_cpus": 0,
            "num_gpus": 0,
            "memory": 0,
        }
        for deployment in deployments:
            ray_actor_options = deployment.ray_actor_options
            required_resources["num_cpus"] += ray_actor_options.get("num_cpus", 0)
            required_resources["num_gpus"] += ray_actor_options.get("num_gpus", 0)
            required_resources["memory"] += ray_actor_options.get("memory", 0)

        return required_resources

    async def build(
        self,
        application_id: str,
        artifact_id: str,
        version: str,
        application_kwargs: Dict[str, Dict[str, Any]],
        application_env_vars: Dict[str, Dict[str, Any]],
        hypha_token: Optional[str],
        disable_gpu: bool,
        max_ongoing_requests: int,
        debug: bool,
        started_at: Optional[float] = None,
        last_updated_at: Optional[float] = None,
        last_updated_by: Optional[str] = None,
        auto_redeploy: bool = False,
        ice_servers: Optional[List[Dict[str, Any]]] = None,
        authorized_users: Optional[Dict[str, List[str]]] = None,
        deploying_user: Optional[tuple] = None,
        admin_users: Optional[List[str]] = None,
    ) -> serve.Application:
        """
        Transform a deployment artifact into a fully functional BioEngine application.

        This is the main "assembly line" method that takes your stored Python code
        and configuration, then builds it into a complete, production-ready application
        running on Ray Serve. Think of it as a sophisticated build system that handles
        all the complexity of distributed deployment.

        Complete Build Process:
        1. Download and parse the application manifest (the "recipe")
        2. Load all Python deployment classes from the artifact
        3. Configure each deployment with proper resources and environment
        4. Set up deployment composition for multi-service applications
        5. Create RTC proxy for WebSocket/WebRTC communication
        6. Validate all parameters and dependencies
        7. Package everything into a deployable Ray Serve application

        Resource Management:
        • Calculates total CPU/GPU/memory requirements
        • Configures isolated working directories for each deployment
        • Sets up shared data directory access
        • Handles GPU allocation and CPU-only fallbacks

        Security & Authentication:
        • Enforces user authorization rules from manifest
        • Provides secure token-based authentication
        • Isolates deployments in separate workspaces

        Communication Setup:
        • Creates RTC proxy for real-time WebSocket connections
        • Exposes schema methods for remote procedure calls
        • Integrates with Hypha server for service registration

        Args:
            application_id: Unique identifier for this deployment instance
            artifact_id: Location of deployment code (format: "workspace/app-name")
            version: Specific artifact version to deploy
            application_kwargs: Initialization parameters for each deployment class
            application_env_vars: Environment variables for each deployment class
            disable_gpu: Force CPU-only execution regardless of deployment defaults
            max_ongoing_requests: Request concurrency limit for the entire application
            debug: Sets logging level to DEBUG in all deployments

        Returns:
            Complete Ray Serve application ready for deployment with metadata including:
            - Available RPC methods exposed by the application
            - Resource requirements for capacity planning
            - Authorization rules and user access controls
            - Service health and status information

        Raises:
            ValueError: Invalid application_id, artifact_id format, or deployment config
            FileNotFoundError: Artifact files not found (in local development mode)
            Exception: Manifest parsing, code loading, or configuration errors

        Example:
            ```python
            app = await builder.build(
                application_id="my-classifier-v1",
                artifact_id="my-workspace/image-classifier",
                version="1.2.0",
                application_kwargs={
                    "ImageClassifier": {"model_path": "/data/models/resnet50.pt"}
                },
                application_env_vars={
                    "ImageClassifier": {"BATCH_SIZE": "32"}
                },
                disable_gpu=False,
                max_ongoing_requests=10
            )
            ```
        """
        self.logger.info(
            f"Building application '{application_id}' from artifact '{artifact_id}' (version: {version})"
        )

        try:
            # Load the artifact manifest and resolve the actual version
            manifest, version = await self._load_manifest(artifact_id, version)
            self.logger.info(
                f"Resolved application '{application_id}' artifact '{artifact_id}' to version '{version}'"
            )

            # Load all deployments defined in the manifest
            deployment_import_paths = manifest["deployments"]
            if (
                not isinstance(deployment_import_paths, list)
                or not deployment_import_paths
            ):
                raise ValueError(
                    f"Invalid deployments format in artifact {artifact_id}. "
                    "Expected a non-empty list of deployment import paths."
                )

            deployments = []
            for import_path in deployment_import_paths:
                try:
                    filename, class_name = import_path.split(":")
                    python_file = f"{filename}.py"  # Add .py extension
                except ValueError:
                    raise ValueError(
                        f"Invalid import path format: {import_path}. "
                        "Expected format is 'filename:ClassName' (without .py extension)."
                    )

                # Get or create deployment environment variables dictionary
                if class_name not in application_env_vars:
                    application_env_vars[class_name] = {}
                deployment_env_vars = application_env_vars[class_name]

                # Add user provided Hypha token as secret environment variable
                if hypha_token is not None:
                    deployment_env_vars["_HYPHA_TOKEN"] = hypha_token

                deployment = await self._load_deployment(
                    application_id=application_id,
                    artifact_id=artifact_id,
                    version=version,
                    python_file=python_file,
                    class_name=class_name,
                    disable_gpu=disable_gpu,
                    env_vars=deployment_env_vars,
                    debug=debug,
                )
                deployments.append(deployment)

                # Extract complete runtime environment variables from the deployment
                application_env_vars[class_name] = deployment.ray_actor_options[
                    "runtime_env"
                ]["env_vars"]

            # Calculate the total number of required resources
            proxy_deployment = ProxyDeployment
            required_resources = self._calculate_required_resources(
                deployments + [proxy_deployment]
            )

            # Get all schema_methods from the entry deployment class
            entry_deployment = deployments[0]
            class_name = entry_deployment.func_or_class.__name__
            method_schemas = []
            for method_name in dir(entry_deployment.func_or_class):
                method = getattr(entry_deployment.func_or_class, method_name)
                if callable(method) and hasattr(method, "__schema__"):
                    method_schemas.append(method.__schema__)

            if not method_schemas:
                raise ValueError(
                    f"No schema methods found in the entry deployment class: {class_name}."
                )

            # Get kwargs for the entry deployment
            if class_name not in application_kwargs:
                application_kwargs[class_name] = {}
            entry_deployment_kwargs = application_kwargs[class_name].copy()
            entry_init_params = self._get_init_param_info(entry_deployment)
            self._check_params(entry_init_params, entry_deployment_kwargs)

            # If multiple deployment classes are found, create a composition deployment
            if len(deployments) > 1:
                self.logger.debug(
                    f"Creating a composition deployment with {len(deployments)} classes."
                )

                # Add the composition deployment class(es) to the entry deployment kwargs
                # Use the parameter name from import_path (first part before ':') as the kwarg name
                for import_path, deployment in zip(
                    deployment_import_paths[1:], deployments[1:]
                ):
                    parameter_name, class_name = import_path.split(":")

                    # Check that the parameter exists and is of type DeploymentHandle
                    if parameter_name not in entry_init_params:
                        raise ValueError(
                            f"Parameter '{parameter_name}' not found in entry deployment "
                            f"'{entry_deployment.func_or_class.__name__}' init method. "
                            f"Available parameters: {list(entry_init_params.keys())}"
                        )

                    param_info = entry_init_params[parameter_name]
                    if param_info["type"] != DeploymentHandle:
                        raise ValueError(
                            f"Parameter '{parameter_name}' in entry deployment "
                            f"'{entry_deployment.func_or_class.__name__}' must be of type "
                            f"DeploymentHandle, but got {param_info['type']}"
                        )

                    init_params = self._get_init_param_info(deployment)
                    deployment_kwargs = application_kwargs.get(class_name, {})
                    self._check_params(init_params, deployment_kwargs)
                    entry_deployment_kwargs[parameter_name] = deployment.bind(
                        **deployment_kwargs
                    )

            # Create the entry deployment handle
            entry_deployment_handle = entry_deployment.bind(**entry_deployment_kwargs)

            # Generate a token to register the application service
            proxy_service_token = await self.server.generate_token(
                {
                    "workspace": self.server.config.workspace,
                    "permission": "read_write",
                    "expires_in": 3600 * 24 * 30,  # support application for 30 days
                }
            )

            # Resolve authorized_users: deploy-time override > manifest.
            # Handle both dict (new format) and list (legacy recovered state).
            if authorized_users is not None:
                if isinstance(authorized_users, dict):
                    effective_authorized_users = authorized_users
                else:
                    effective_authorized_users = {"*": authorized_users}
            else:
                manifest_users = manifest.get("authorized_users", ["*"])
                if isinstance(manifest_users, dict):
                    effective_authorized_users = manifest_users
                else:
                    effective_authorized_users = {"*": manifest_users}

            # Collect deploying user identifiers — always injected into all method rules.
            deploying_user_entries = []
            if deploying_user:
                dep_id, dep_email = deploying_user
                deploying_user_entries = [v for v in [dep_id, dep_email] if v]

            # Inject users and deduplicate each rule list.
            # Admin users are only injected when the rule is not already public (i.e. "*" not in rule),
            # since public access already covers everyone.
            for key in list(effective_authorized_users):
                rule = list(effective_authorized_users[key])
                is_public = "*" in rule
                if not is_public:
                    to_add = list(deploying_user_entries)
                    if admin_users:
                        to_add.extend(admin_users)
                    for user in to_add:
                        if user not in rule:
                            rule.append(user)
                # Deduplicate while preserving order
                seen = set()
                effective_authorized_users[key] = [
                    u for u in rule if not (u in seen or seen.add(u))
                ]

            # If there is no wildcard key, ensure deploying user + admins can still call all methods.
            if "*" not in effective_authorized_users:
                fallback = list(deploying_user_entries)
                if admin_users:
                    fallback.extend(admin_users)
                seen = set()
                effective_authorized_users["*"] = [
                    u for u in fallback if not (u in seen or seen.add(u))
                ]

            # Create the application
            sanitized_env_vars = self._sanitize_recovery_env_vars(application_env_vars)
            app_data = {
                "display_name": manifest["name"],
                "description": manifest["description"],
                "artifact_id": artifact_id,
                "version": version,
                "application_kwargs": application_kwargs,
                "application_env_vars": sanitized_env_vars,
                "disable_gpu": disable_gpu,
                "max_ongoing_requests": max_ongoing_requests,
                "application_resources": required_resources,
                "authorized_users": effective_authorized_users,
                "available_methods": [
                    method_schema["name"] for method_schema in method_schemas
                ],
                "started_at": started_at if started_at is not None else time.time(),
                "last_updated_at": (
                    last_updated_at if last_updated_at is not None else time.time()
                ),
                "last_updated_by": (
                    last_updated_by
                    if last_updated_by is not None
                    else self.server.config.user["id"]
                ),
                "auto_redeploy": auto_redeploy,
                "debug": debug,
            }

            app = proxy_deployment.bind(
                application_id=application_id,
                application_name=manifest["name"],
                application_description=manifest["description"],
                app_data=app_data,
                entry_deployment_handle=entry_deployment_handle,
                method_schemas=method_schemas,
                max_ongoing_requests=max_ongoing_requests,
                server_url=self.server.config.public_base_url,
                workspace=self.server.config.workspace,
                worker_client_id=self.server.config.client_id,
                proxy_service_token=proxy_service_token,
                authorized_users=effective_authorized_users,
                proxy_actor_name=self.proxy_actor_name,
                debug=debug,
                ice_servers=ice_servers,
            )

            # Create application metadata
            app.metadata = {
                "name": manifest["name"],
                "description": manifest["description"],
                "version": version,
                "resources": required_resources,
                "authorized_users": effective_authorized_users,
                "available_methods": [
                    method_schema["name"] for method_schema in method_schemas
                ],
                "application_kwargs": application_kwargs,
                "application_env_vars": application_env_vars,
                "frontend_entry": manifest.get("frontend_entry"),
            }

            self.logger.info(
                f"Successfully built application '{application_id}' with "
                f"available methods: {app.metadata['available_methods']}"
            )

        except Exception as e:
            self.logger.error(f"Error building application '{application_id}': {e}")
            raise e

        return app


if __name__ == "__main__":
    from hypha_rpc import connect_to_server

    server_url = "https://hypha.aicell.io"
    token = os.environ["HYPHA_TOKEN"]

    base_dir = Path(__file__).parent.parent.parent
    os.environ["BIOENGINE_LOCAL_ARTIFACT_PATH"] = str(base_dir / "tests")

    apps_workdir = Path.home() / ".bioengine" / "apps"

    async def test_app_builder():
        server = await connect_to_server({"server_url": server_url, "token": token})
        artifact_manager = await server.get_service("public/artifact-manager")

        app_builder = AppBuilder(
            apps_workdir=apps_workdir,
            log_file=None,
            debug=True,
            proxy_actor_name="TEST_PROXY_ACTOR",
        )
        app_builder.complete_initialization(
            server, artifact_manager, worker_service_id="test-worker"
        )

        app = await app_builder.build(
            application_id="test-application-1234",
            artifact_id="test-workspace/composition-app",
            version=None,
            deployment_options={
                "num_cpus": 1,
                "num_gpus": 0,
                "memory": 1024 * 1024 * 1024,  # 1 GB
            },
            application_kwargs={
                "CompositionDeployment": {"demo_input": "Hello World!"},
                "Deployment2": {"start_number": 10},
            },
        )

    asyncio.run(test_app_builder())
