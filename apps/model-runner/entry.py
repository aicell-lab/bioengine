"""@bioengine.app EntryApp — the public RPC surface of the model-runner.

The app provides bioimage.io model search, RDF/documentation retrieval,
RDF validation, end-to-end model testing, and inference. It delegates the
GPU-bound work (``predict`` and the heavy ``bioimageio.core.test_model``
call) to :class:`runtime.RuntimeApp` via the v0.6 type-hint composition.

Heavier helper modules:

* :mod:`model_cache.cache` — LRU cache of downloaded bioimage.io model
  packages, with on-disk markers for cross-replica coordination
* :mod:`model_cache.package` — per-use lock context manager around a
  cached package; the entry uses it via ``async with package: …`` to
  keep models from being evicted mid-inference
"""


import asyncio
import json
import logging
import os
import random
import shutil
import time
import traceback
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import bioengine
import httpx
import numpy as np
import yaml
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_method
from pydantic import Field

from bioengine import __version__

from model_cache import BioimageioPackage, ModelCache
from runtime import RuntimeApp


def _arbitrary_types_method(fn):
    """Marker variant of ``@bioengine.method`` that allows non-pydantic
    types (``np.ndarray``) in the signature. Used on :meth:`EntryApp.infer`.
    The framework picks up the schema via ``__schema__`` and the kind
    marker exactly as it does for ``@bioengine.method``; this helper just
    plumbs ``arbitrary_types_allowed=True`` through to ``schema_method``.
    """
    wrapped = schema_method(fn, arbitrary_types_allowed=True)
    wrapped._bioengine_kind = "method"
    return wrapped


logger = logging.getLogger("ray.serve")
logger.setLevel("INFO")

SUPPORTED_FILES_TYPES = Literal[".npy", ".png", ".tiff", ".tif", ".jpeg", ".jpg"]


@bioengine.app(
    num_cpus=1,
    num_gpus=0,
    memory_mb=4 * 1024,
    pip=[
        "aiofiles>=23.0.0",
        "bioimageio.core==0.10.0",
        "imageio>=2.37.0",
        "numpy==1.26.4",
        "tqdm>=4.64.0",
    ],
    env_vars={
        # Shared NFS path so EntryApp and RuntimeApp see the same model cache
        # across pods. Override per-cluster via the manifest if the deployment
        # has a different writable shared path.
        "MODEL_CACHE_DIR": "/home/bioengine/staging/model-cache",
    },
    max_ongoing_requests=10,
    max_queued_requests=30,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 1,
        "target_num_ongoing_requests_per_replica": 1,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=300.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class EntryApp:
    """
    Ray Serve deployment for bioimage.io model operations.

    Handles model downloading, caching, validation, testing, and inference
    with cross-replica coordination and atomic filesystem operations.

    Concurrency Design:
    - Uses atomic filesystem operations (mkdir, rename) for replica coordination
    - Download markers prevent duplicate downloads across replicas
    - Access tracking (.last_access files) prevents eviction of active models
    - LRU eviction with retry logic handles cache space management
    - Context managers ensure proper access time tracking during model usage
    - Graceful error handling for filesystem race conditions and I/O errors
    """

    def __init__(
        self,
        runtime: RuntimeApp,
        cache_size_in_gb: float = 50.0,
    ) -> None:
        self.runtime = runtime

        # Set Hypha server and workspace
        self.server_url = "https://hypha.aicell.io"
        self._hypha_token = os.getenv("HYPHA_TOKEN")
        if not self._hypha_token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set")

        # Get replica identifier for logging
        try:
            from ray import serve as _serve
            self.replica_id = _serve.get_replica_context().replica_tag
        except Exception:
            self.replica_id = "unknown"

        # Set up model cache
        self.model_cache = ModelCache(
            cache_size_in_gb=cache_size_in_gb,
            replica_id=self.replica_id,
        )

        logger.info(
            f"🚀 {self.__class__.__name__} initialized with models directory: "
            f"{self.model_cache.cache_dir} (cache_size={self.model_cache.cache_size_bytes / (1024*1024*1024):.3f} GB)"
        )

    # === BioEngine App Method - will be called when the deployment is started ===

    @bioengine.async_init
    async def _async_init(self) -> None:
        self.hypha_client = await connect_to_server(
            {
                "server_url": self.server_url,
                "token": self._hypha_token,
            }
        )
        self.artifact_manager = await self.hypha_client.get_service(
            "public/artifact-manager"
        )
        self.s3_controller = await self.hypha_client.get_service("public/s3-storage")
        logger.info(f"Connected to Hypha Server at {self.server_url}")

        # Decide once at startup whether this deployment's HYPHA_TOKEN can
        # write back to the upstream bioimage-io/* model artifacts (used by
        # the publish-test-report path in test()). Reads of public artifacts
        # work for any token, so the rest of the service is unaffected.
        self._can_publish_to_bioimage_io = self._check_bioimage_io_write_access()
        if self._can_publish_to_bioimage_io:
            logger.info(
                "✅ HYPHA_TOKEN has write access to bioimage-io workspace; "
                "test reports can be published to source artifacts."
            )
        else:
            logger.warning(
                "⚠️ HYPHA_TOKEN does not have write access to bioimage-io workspace; "
                "calls to test(publish_test_report=True) will raise a PermissionError "
                "(test(publish_test_report=False) still works)."
            )

    def _check_bioimage_io_write_access(self) -> bool:
        """Return True if the deployment's HYPHA_TOKEN can write to the bioimage-io workspace.

        Inspects ``self.hypha_client.config.user["scope"]["workspaces"]`` —
        Hypha exposes per-workspace permission letters there
        (``r`` read, ``rw`` write, ``rw+`` write+create, ``a`` admin).
        Anything that includes ``w`` or equals ``a`` is sufficient for the
        edit/put_file/commit flow used in publish_test_report.
        """
        try:
            scope = self.hypha_client.config.user.get("scope", {}) or {}
            workspaces = scope.get("workspaces", {}) or {}
            perm = workspaces.get("bioimage-io", "")
            return bool(perm) and ("w" in perm or "a" in perm)
        except Exception as e:
            # Conservative fallback: assume no write access. Better to skip
            # publishing than to crash an inference deployment.
            logger.warning(
                f"Could not introspect HYPHA_TOKEN scope for bioimage-io write access: {e}. "
                f"Treating as read-only."
            )
            return False

    # === Ray Serve Health Check Method - will be called periodically to check the health of the deployment ===

    async def _check_runtime_available(self) -> None:
        """Ping the runtime deployment and raise immediately if it is not responding.

        Called at the top of every GPU method so callers get a fast, clear error
        instead of a 30 s+ Ray timeout when the GPU runtime is not running.
        """
        try:
            await asyncio.wait_for(
                self.runtime.ping(),
                timeout=2.0,
            )
        except Exception as e:
            raise RuntimeError(
                "GPU runtime deployment is not available. "
                "Inference, test, and validate are unavailable until the runtime starts."
            ) from e

    @bioengine.health_check
    async def _health_check(self) -> None:
        # Test connection to the Hypha server only — runtime availability is
        # checked per-call in GPU methods so partial registration is preserved
        # (service stays registered for CPU-only methods when GPU is down).
        await self.hypha_client.echo("ping")

    # === Internal Helper Methods ===

    def _get_bioimageio_versions(self) -> Dict[str, str]:
        """
        Return identities the test cache invalidates on.

        Includes installed versions of the bioimageio packages and the
        model-runner artifact's own version — so the existing env-mismatch
        check in ``test()`` re-runs when the runner itself is upgraded.
        """
        from importlib.metadata import PackageNotFoundError, version

        package_names = ["bioimageio.core", "bioimageio.spec"]
        versions: Dict[str, str] = {}

        for package_name in package_names:
            try:
                versions[package_name] = version(package_name)
            except PackageNotFoundError:
                versions[package_name] = "not-installed"

        versions["bioimage-io/model-runner"] = os.environ.get(
            "HYPHA_ARTIFACT_VERSION", "unknown"
        )

        logger.debug(f"📦 Bioimage.io package versions: {versions}")
        return versions

    def _stamp_runtime_versions_in_test_env(self, test_report: dict) -> dict:
        """Upsert runtime-identity rows in ``test_report['env']``.

        The test cache invalidation in ``test()`` reuses this env list to
        detect when a stored report was produced under different runtime
        versions; downstream consumers (e.g. bioimage.io CI) also read it
        from the published report to drive their own cache keys.
        """
        rows = [
            ("bioengine", __version__),
            (
                "bioimage-io/model-runner",
                os.environ.get("HYPHA_ARTIFACT_VERSION", "unknown"),
            ),
        ]

        env = test_report.get("env")
        if not isinstance(env, list):
            env = []
        else:
            env = list(env)

        for name, version_value in rows:
            row_index: Optional[int] = None
            for i, row in enumerate(env):
                if (
                    isinstance(row, (list, tuple))
                    and len(row) >= 1
                    and str(row[0]) == name
                ):
                    row_index = i
                    break

            new_row = [name, version_value, "", ""]
            if row_index is None:
                env.append(new_row)
            else:
                existing_row = env[row_index]
                if isinstance(existing_row, tuple):
                    existing_row = list(existing_row)
                while len(existing_row) < 4:
                    existing_row.append("")
                existing_row[1] = version_value
                env[row_index] = existing_row

        test_report["env"] = env
        return test_report

    async def _get_download_url(self, file_path: str) -> str:
        # Temporary S3 file path — resolve to a presigned download URL
        try:
            download_url = await self.s3_controller.get_file(
                file_path=file_path, use_proxy=True
            )
            return download_url
        except Exception as e:
            raise RuntimeError(
                f"Failed to get download URL for temporary file '{file_path}': {e}"
            ) from e

    async def _load_image_from_source(self, source: str) -> np.ndarray:
        """
        Load an image from a URL or a temporary S3 file path into a numpy array.

        Accepts either:
        - A direct HTTP/HTTPS URL (fetched as-is), or
        - A temporary file path returned by ``get_upload_url`` (resolved to a
          presigned S3 download URL via BioEngine S3 storage).

        The file content is decoded based on the file extension.

        Args:
            source: Direct URL (``http://…`` / ``https://…``) or temporary
                    file path returned by ``get_upload_url``

        Returns:
            np.ndarray: NumPy array containing the image data

        Raises:
            FileNotFoundError: If the remote resource does not exist or has expired
            ValueError: If the file extension is not supported
        """
        # Check file extension for supported formats
        ext = Path(
            source.split("?")[0]
        ).suffix.lower()  # strip query string for URL sources
        if ext not in SUPPORTED_FILES_TYPES.__args__:
            raise ValueError(
                f"Unsupported file extension '{ext}' in source '{source}'. "
                f"Supported extensions: {SUPPORTED_FILES_TYPES.__args__}"
            )

        logger.info(f"📥 Loading image from source '{source}'...")

        if source.startswith(("http://", "https://")):
            # Direct URL — fetch without S3 indirection
            download_url = source
        else:
            download_url = await self._get_download_url(source)

        # Download file content
        response = await self.model_cache._get_url_with_retry(download_url, params=None)

        if response.status_code == 404:
            raise FileNotFoundError(f"Source '{source}' does not exist or has expired.")
        try:
            response.raise_for_status()
        except Exception as e:
            raise FileNotFoundError(f"Failed to download source '{source}': {e}") from e

        # Parse and load based on file extension
        try:
            buffer = BytesIO(response.content)
            if ext == ".npy":
                array = await asyncio.to_thread(np.load, buffer)
            else:
                import imageio.v3 as iio

                array = await asyncio.to_thread(iio.imread, buffer)
        except Exception as e:
            raise ValueError(
                f"Failed to parse image from source '{source}': {e}"
            ) from e

        logger.info(
            f"✅ Loaded image from '{source}': shape={array.shape}, dtype={array.dtype}"
        )
        return array

    async def _save_array_to_temp_file(self, array: np.ndarray) -> str:
        """
        Save a NumPy array to a temporary ``.npy`` file in S3 and return a presigned download URL.

        The array is serialised with ``numpy.save`` and uploaded to BioEngine S3 storage using a
        presigned upload URL obtained from ``get_upload_url``. The file is given a 1-hour TTL.

        Args:
            array: NumPy array to save

        Returns:
            str: Presigned download URL for the uploaded ``.npy`` file (valid for 1 hour)

        Raises:
            RuntimeError: If saving the array to a temporary file fails
        """
        try:
            upload_info = await self.get_upload_url(file_type=".npy")
            logger.info(
                f"💾 Saving array (shape: {array.shape}, dtype: {array.dtype}) "
                f"to temporary file '{upload_info['file_path']}'..."
            )
            buffer = BytesIO()
            np.save(buffer, array)
            await self.model_cache.client.put(
                upload_info["upload_url"], data=buffer.getvalue()
            )
            logger.info(
                f"✅ Array saved to temporary file '{upload_info['file_path']}'"
            )

            download_url = await self._get_download_url(upload_info["file_path"])
        except Exception as e:
            raise RuntimeError(f"Failed to save array to temporary file: {e}") from e

        return download_url

    # === Exposed BioEngine App Methods - all methods decorated with @bioengine.method will be exposed as API endpoints ===
    # Note: Parameter type hints and docstrings will be used to generate the API documentation.

    @bioengine.method
    async def get_version(self) -> Dict[str, str]:
        """Return the artifact identity this replica was deployed from.

        Callers (e.g. bioimage.io CI cache invalidation) use this to detect
        when a stored test report was produced under a different runner
        version and should be re-tested.
        """
        return {
            "artifact_id": os.environ.get("HYPHA_ARTIFACT_ID", "unknown"),
            "version": os.environ.get("HYPHA_ARTIFACT_VERSION", "unknown"),
            "bioengine_version": __version__,
        }

    @bioengine.method
    async def search_models(
        self,
        keywords: Optional[List[str]] = Field(
            None,
            description="List of keywords to filter models by (e.g., ['cell', 'nuclei', 'segmentation']",
        ),
        limit: Optional[int] = Field(
            10, description="Maximum number of models to return in the search results"
        ),
        ignore_checks: Optional[bool] = Field(
            False,
            description="Whether to ignore bioengine inference checks and return all models (True) or only models that passed checks (False)",
        ),
    ) -> List[Dict[str, str]]:
        """
        Search for models in the bioimage.io collection.

        Returns a list of model identifiers with their descriptions that match the search query.
        """
        logger.info(f"🔍 Searching models with keywords={keywords}, limit={limit}")
        collection_id = "bioimage-io/bioimage.io"

        try:
            results = await self.artifact_manager.list(
                parent_id=collection_id,
                filters={"type": "model"},
                keywords=keywords,
                limit=limit,
                stage=False,
            )

            if not ignore_checks:
                collection = await self.artifact_manager.read(collection_id)
                bioengine_inference_results = collection["manifest"][
                    "bioengine_inference"
                ]
                runnable_models = {
                    model_id
                    for model_id, result in bioengine_inference_results.items()
                    if result.get("status") == "passed"
                }

            models = []
            for artifact in results:
                manifest = artifact["manifest"]
                if not ignore_checks and artifact["alias"] not in runnable_models:
                    continue
                models.append(
                    {
                        "model_id": artifact["alias"],
                        "description": manifest.get("description", ""),
                    }
                )

            logger.info(f"✅ Found {len(models)} models matching query.")
            return models

        except Exception as e:
            error_msg = f"Failed to search models: {e}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    @bioengine.method
    async def get_model_rdf(
        self,
        model_id: str = Field(
            ...,
            description="Unique identifier of the bioimage.io model (e.g., 'ambitious-ant')",
        ),
        stage: Optional[bool] = Field(
            False,
            description="Whether to get RDF from the staged version of the model (True) or the committed version (False)",
        ),
    ) -> Dict[str, Union[str, int, float, List, Dict]]:
        """
        Retrieve the Resource Description Framework (RDF) metadata for a bioimage.io model.

        Returns:
            Dictionary containing the complete RDF metadata structure with nested
            configuration for inputs, outputs, preprocessing, postprocessing, and model weights

        Raises:
            ValueError: If model_id is invalid or model not found
            RuntimeError: If download fails
        """
        logger.info(f"📋 Downloading RDF for model '{model_id}' (stage={stage}).")

        rdf_url = f"{self.server_url}/bioimage-io/artifacts/{model_id}/files/rdf.yaml"
        response = await self.model_cache._get_url_with_retry(
            rdf_url, params={"stage": str(stage).lower()}
        )

        if response.status_code == 404 and stage:
            # If staged version doesn't exist, try with stage=false
            logger.warning(
                f"⚠️ Staged RDF not found for model '{model_id}', trying committed version..."
            )
            response = await self.model_cache._get_url_with_retry(
                rdf_url, params={"stage": "false"}
            )

        try:
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to download RDF from {rdf_url}") from e

        model_rdf = await asyncio.to_thread(yaml.safe_load, response.text)

        logger.info(f"✅ Successfully downloaded RDF for model '{model_id}'.")
        return model_rdf

    @bioengine.method
    async def get_model_documentation(
        self,
        model_id: str = Field(
            ...,
            description="Unique identifier of the bioimage.io model (e.g., 'ambitious-ant')",
        ),
        stage: Optional[bool] = Field(
            False,
            description="Whether to fetch documentation from the staged version (True) or committed version (False)",
        ),
    ) -> Optional[str]:
        """
        Retrieve the documentation text for a bioimage.io model.

        Reads the 'documentation' field from the model RDF. If the field is set,
        downloads the referenced file from the artifact and returns its content.

        Returns:
            The documentation file content as a string, or None if the
            'documentation' field is absent/None or the file does not exist.
        """
        rdf = await self.get_model_rdf(model_id=model_id, stage=stage)

        doc_path = rdf.get("documentation")
        if not doc_path:
            logger.info(f"📄 No documentation field in RDF for model '{model_id}'.")
            return None

        doc_url = f"{self.server_url}/bioimage-io/artifacts/{model_id}/files/{doc_path}"
        logger.info(f"📄 Downloading documentation '{doc_path}' for model '{model_id}'.")

        response = await self.model_cache._get_url_with_retry(
            doc_url, params={"stage": str(stage).lower()}
        )

        if response.status_code == 404:
            logger.info(
                f"📄 Documentation file '{doc_path}' not found in artifact for model '{model_id}'."
            )
            return None

        try:
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"⚠️ Failed to download documentation for model '{model_id}': {e}")
            return None

        logger.info(f"✅ Successfully downloaded documentation for model '{model_id}'.")
        return response.text

    @bioengine.method
    async def validate(
        self,
        rdf_dict: Dict[str, Union[str, int, float, List, Dict]] = Field(
            ..., description="Complete RDF dictionary structure to validate"
        ),
        known_files: Optional[Dict[str, str]] = Field(
            None,
            description="Mapping of relative file paths to their content hashes for validating file references within the RDF",
        ),
    ) -> Dict[str, Union[bool, str]]:
        """
        Validate a model Resource Description Framework (RDF) against bioimage.io specifications.

        Returns:
            Validation result containing:
            - success: Boolean indicating overall validation status
            - details: Detailed validation report with specific issues or confirmation

        Note:
            This method performs format validation only (perform_io_checks=False).
            File existence is not verified unless known_files mapping is provided.
        """
        from bioimageio.spec import ValidationContext, validate_format

        logger.info(
            f"🔬 Validating RDF (known_files: {len(known_files or {})} files)..."
        )

        ctx = ValidationContext(perform_io_checks=False, known_files=known_files or {})
        summary = await asyncio.to_thread(validate_format, rdf_dict, context=ctx)

        result = {
            "success": summary.status == "valid-format",
            "details": summary.format(),
        }

        logger.info(f"✅ RDF validation {'passed' if result['success'] else 'failed'}.")
        return result

    @bioengine.method
    async def test(
        self,
        model_id: str = Field(
            ..., description="Unique identifier of the bioimage.io model to test"
        ),
        stage: Optional[bool] = Field(
            False,
            description="Whether to get the staged version of the model (True) or the committed version (False)",
        ),
        additional_requirements: Optional[List[str]] = Field(
            None,
            description='Extra Python packages to install in the test environment (e.g., ["scipy>=1.7.0", "scikit-image"])',
        ),
        skip_cache: Optional[bool] = Field(
            False,
            description="Force a complete model package re-download and bypass cached test results before testing",
        ),
        publish_test_report: Optional[bool] = Field(
            False,
            description="Automatically publish the test report to the model artifact after testing",
        ),
    ) -> Dict[str, Union[str, bool, List, Dict]]:
        """
        Execute comprehensive model testing using the `bioimageio.core.test_model` test suite.

        Caching behavior:
        - Cached test reports are locally stored at ``<model_package>/.test_cache.json``.
        - Cached results are reused only when ``skip_cache=False`` AND the model
            package has not changed (same ``latest_remote_modified``) AND the cached
            ``test_report['env']`` versions for ``bioimageio.core`` and
            ``bioimageio.spec`` match the currently installed versions.
        - ``skip_cache=True`` forces a complete model package re-download,
            bypasses cached test results, and runs a fresh test.

        Additional requirements:
        - ``additional_requirements`` are persisted in the cache metadata for
            observability but are NOT part of automatic cache invalidation.
            If you change them, use ``skip_cache=True`` to force re-testing.

        Publishing behavior:
        - If ``publish_test_report=True``, a compact ``test_summary`` entry is
            written to the artifact manifest, ``test_report.json`` is uploaded,
            and the artifact is committed.
        - If the artifact had an open staging version before publishing, staging is
            re-opened after commit.
        """
        import aiofiles

        await self._check_runtime_available()
        # Fail fast before running any compute if the caller wants to publish
        # but this deployment's HYPHA_TOKEN has no write access to the
        # bioimage-io workspace. Better than burning GPU time and surprising
        # the caller with a permission error from artifact_manager.edit() at
        # the end.
        if publish_test_report and not self._can_publish_to_bioimage_io:
            raise PermissionError(
                f"Cannot publish test report for '{model_id}': "
                f"this deployment's HYPHA_TOKEN has no write access to the "
                f"bioimage-io workspace. Either deploy with a HYPHA_TOKEN that "
                f"has bioimage-io write permission, or call test() with "
                f"publish_test_report=False."
            )
        logger.info(
            f"🧪 Testing model '{model_id}' (stage={stage}, skip_cache={skip_cache}, publish_test_report={publish_test_report})."
        )

        # Get model package with access tracking
        package = await self.model_cache.get_model_package(
            model_id=model_id,
            stage=stage,
            allow_unpublished=True,
            skip_cache=skip_cache,
        )

        # Use context manager to track access and prevent eviction during test
        async with package:
            logger.info(f"📍 Model source for '{model_id}': {package.source}")
            test_report: Optional[dict] = None
            tested_at: Optional[float] = None
            should_run_test = True
            should_cache_report = True

            # Check for cached test results
            test_report_path = package.package_path / ".test_cache.json"
            current_versions = self._get_bioimageio_versions()

            if not skip_cache and await asyncio.to_thread(test_report_path.exists):
                try:
                    # Load cached test results
                    async with aiofiles.open(test_report_path, "r") as f:
                        content = await f.read()
                        cached_data = await asyncio.to_thread(json.loads, content)

                    # Check if model files have changed since last test
                    cached_remote_modified = cached_data["latest_remote_modified"]
                    model_unchanged = (
                        package.latest_remote_modified == cached_remote_modified
                    )

                    # Validate cached environment versions against currently installed packages.
                    cached_test_report = cached_data["test_report"]
                    cached_env = cached_test_report.get("env", [])
                    cached_env_versions: Dict[str, str] = {}

                    for row in cached_env:
                        if isinstance(row, (list, tuple)) and len(row) >= 2:
                            pkg_name = str(row[0])
                            pkg_version = str(row[1])
                            if pkg_name in current_versions:
                                cached_env_versions[pkg_name] = pkg_version

                    env_versions_match = True
                    for pkg_name, installed_version in current_versions.items():
                        cached_version = cached_env_versions.get(pkg_name)
                        if cached_version != installed_version:
                            env_versions_match = False
                            logger.info(
                                f"🔄 Cached test env mismatch for '{model_id}' on {pkg_name}: "
                                f"cached={cached_version}, installed={installed_version}. "
                                f"Re-running tests."
                            )

                    if model_unchanged and env_versions_match:
                        # Model hasn't changed, return cached results
                        logger.info(
                            f"💾 Model '{model_id}' unchanged since last test, using cached results."
                        )
                        test_report = cached_data["test_report"]
                        tested_at = test_report["tested_at"]
                        should_run_test = False
                        should_cache_report = False
                    else:
                        if model_unchanged and not env_versions_match:
                            logger.info(
                                f"🔄 Model '{model_id}' unchanged but test environment changed, re-running tests."
                            )
                        elif not model_unchanged:
                            logger.info(
                                f"🔄 Model '{model_id}' has been updated, re-running tests "
                                f"(cached: {cached_remote_modified}, current: {package.latest_remote_modified})"
                            )
                except (json.JSONDecodeError, KeyError, OSError, IOError) as e:
                    logger.warning(
                        f"⚠️ Failed to load cached test results for '{model_id}': {e}. Running fresh test."
                    )

            # Run the test unless we already accepted a cached result
            if should_run_test:
                tested_at = time.time()
                try:
                    test_report = await self.runtime.test(
                        rdf_path=package.source,
                        additional_requirements=additional_requirements,
                    )
                    logger.info(f"✅ Model test completed for '{model_id}'.")
                except Exception as e:
                    error_traceback = traceback.format_exc()
                    logger.warning(
                        f"⚠️ Model test failed for '{model_id}': {str(e)}. Generating fallback report."
                    )
                    should_cache_report = False

                    # Load RDF from package for fallback report
                    try:
                        async with aiofiles.open(package.source, "r") as f:
                            rdf_content = await f.read()
                            rdf = await asyncio.to_thread(yaml.safe_load, rdf_content)
                            artifact_type = rdf.get("type")
                    except Exception as rdf_error:
                        logger.error(
                            f"⚠️ Failed to load RDF for fallback report: {rdf_error}"
                        )
                        artifact_type = None

                    # Generate fallback test report
                    #! WARNING: Ensure that bioimageio.core and bioimageio.spec versions are the same in the EntryDeployment and RuntimeDeployment environments
                    test_report = {
                        "name": "bioimageio format validation",
                        "source_name": package.source,
                        "id": model_id,
                        "type": artifact_type,
                        "format_version": current_versions.get(
                            "bioimageio.spec", "unknown"
                        ),
                        "status": "failed",
                        "details": [{"errors": [{"msg": error_traceback}]}],
                        "env": [
                            [
                                "bioimageio.core",
                                current_versions.get("bioimageio.core", "unknown"),
                                "",
                                "",
                            ],
                            [
                                "bioimageio.spec",
                                current_versions.get("bioimageio.spec", "unknown"),
                                "",
                                "",
                            ],
                        ],
                        "saved_conda_list": "",
                    }

                    logger.warning(
                        f"⚠️ Generated fallback test report for '{model_id}' due to test error."
                    )

                test_report = self._stamp_runtime_versions_in_test_env(test_report)

                # Add tested_at timestamp to the test report so the report is self-contained.
                test_report["tested_at"] = tested_at

            # Save test results only for fresh, successful calculations.
            if should_cache_report:
                try:
                    cache_data = {
                        "test_report": test_report,
                        "latest_remote_modified": package.latest_remote_modified,
                        "additional_requirements": additional_requirements,
                    }
                    async with aiofiles.open(test_report_path, "w") as f:
                        await f.write(json.dumps(cache_data, indent=2))
                    logger.info(f"💾 Test report cached for model '{model_id}'")
                except (OSError, IOError) as e:
                    logger.warning(
                        f"⚠️ Failed to cache test report for '{model_id}': {e}"
                    )

            # Publish test report to artifact (caller already validated as
            # having bioimage-io write access at the top of this method).
            if publish_test_report:
                artifact_id = f"bioimage-io/{model_id}"
                report_file_name = "test_report.json"
                should_publish_report = True

                try:
                    download_url = await self.artifact_manager.get_file(
                        artifact_id=artifact_id,
                        file_path=report_file_name,
                    )
                    async with httpx.AsyncClient(timeout=30) as client:
                        response = await client.get(download_url)
                        response.raise_for_status()

                    remote_test_report = await asyncio.to_thread(
                        json.loads, response.text
                    )
                    remote_tested_at = float(remote_test_report.get("tested_at", 0.0))
                    local_tested_at = test_report["tested_at"]
                    should_publish_report = remote_tested_at != local_tested_at
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to load remote test report for '{artifact_id}' before publishing: {e}"
                    )
                    should_publish_report = True

                if not should_publish_report:
                    logger.info(
                        f"ℹ️ Existing test report for '{artifact_id}' is up to date; skipping publish."
                    )
                    return test_report

                # Check current staging state.
                artifact = await self.artifact_manager.read(artifact_id)
                is_staged = artifact.get("staging") is not None

                # Create a compact test report for the artifact manifest (excluding details and env to save space) and merge it with existing manifest data.
                test_report_summary = {
                    "status": test_report["status"],
                    "tested_at": test_report["tested_at"],
                    "env": test_report["env"],
                }

                # 'test_reports' and 'test_report' are legacy manifest keys.
                updated_manifest = dict(artifact["manifest"])
                updated_manifest.pop("test_reports", None)
                updated_manifest.pop("test_report", None)
                updated_manifest.pop("score", None)
                updated_manifest["test_summary"] = test_report_summary

                # Edit the artifact and stage it for review.
                artifact = await self.artifact_manager.edit(
                    artifact_id=artifact.id,
                    manifest=updated_manifest,
                    stage=True,
                )

                upload_url = await self.artifact_manager.put_file(
                    artifact.id, file_path=report_file_name
                )

                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.put(
                        upload_url, data=json.dumps(test_report)
                    )
                    response.raise_for_status()

                # 'test_reports.json' is a legacy file name
                try:
                    existing_files = await self.artifact_manager.list_files(artifact.id)
                    if any(file.name == "test_reports.json" for file in existing_files):
                        await self.artifact_manager.remove_file(
                            artifact.id, file_path="test_reports.json"
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to remove legacy test report file for '{artifact_id}': {e}"
                    )

                # Commit the artifact.
                await self.artifact_manager.commit(artifact_id=artifact.id)

                # If it was staged before this update, put it back into stage mode.
                if is_staged:
                    await self.artifact_manager.edit(
                        artifact_id=artifact.id,
                        stage=True,
                    )

                logger.info(
                    f"📤 Published test report for model '{model_id}' to artifact '{artifact_id}'."
                )

        return test_report

    @bioengine.method
    async def get_upload_url(
        self,
        file_type: SUPPORTED_FILES_TYPES = Field(
            ...,
            description='File type for the upload. Supported types: ".npy" (NumPy array), ".png" (PNG image), ".tiff"/".tif" (TIFF image), ",jpeg"/".jpg" (JPEG image)',
        ),
    ) -> Dict[str, str]:
        """
        Request a presigned upload URL for uploading an input image to temporary storage.

        Creates a unique temporary file in BioEngine S3 storage with a 1-hour TTL.
        Upload the file to the returned URL via an HTTP PUT request, then pass the
        returned ``file_path`` as the ``inputs`` parameter of the ``infer`` endpoint.

        Returns:
            Dictionary containing:
            - upload_url: Presigned URL for uploading the file via HTTP PUT
            - file_path: Unique temporary file path to reference the uploaded file

        Example::
            import httpx, imageio.v3 as iio, io

            result = await model_runner_service.get_upload_url(file_type=".png")
            buf = io.BytesIO()
            iio.imwrite(buf, image, extension=".png")
            async with httpx.AsyncClient() as client:
                await client.put(result["upload_url"], content=buf.getvalue())
            output = await model_runner_service.infer(model_id="...", inputs=result["file_path"])
        """
        unique_id = str(uuid.uuid4())
        file_path = f"temp/{unique_id}{file_type}"

        logger.info(f"📤 Requesting presigned upload URL for '{file_path}'...")

        try:
            upload_url = await self.s3_controller.put_file(
                file_path=file_path,
                ttl=3600,  # 1-hour TTL
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get upload URL for temporary file '{file_path}': {e}"
            ) from e

        logger.info(f"✅ Presigned upload URL generated for '{file_path}'.")
        return {"upload_url": upload_url, "file_path": file_path}

    @_arbitrary_types_method
    async def infer(
        self,
        model_id: str = Field(
            ..., description="Unique identifier of the published bioimage.io model"
        ),
        inputs: Union[np.ndarray, Dict[str, Union[np.ndarray, str]], str] = Field(
            ...,
            description="Input data as numpy array, dictionary mapping input names to arrays/strings, or a single string. "
            "Accepted string formats: a direct HTTP/HTTPS URL (fetched as-is) or a temporary file path returned by "
            "``get_upload_url`` (resolved via S3 storage). "
            "Must match the model's input specification for shape and data type. "
            "For single-input models, provide a np.ndarray or a string. "
            "For multi-input models, provide a dict with input names as keys; each value may be a np.ndarray or a string.",
        ),
        weights_format: Optional[str] = Field(
            None,
            description='Preferred model weights format ("pytorch_state_dict", "torchscript", "onnx", "tensorflow_saved_model"). If None, automatically selects best available.',
        ),
        device: Optional[Literal["cuda", "cpu"]] = Field(
            None,
            description='Target computation device. "cuda" for GPU acceleration, "cpu" for CPU-only. If None, automatically selects based on availability and model compatibility.',
        ),
        default_blocksize_parameter: Optional[int] = Field(
            None,
            description="Override default tiling block size for memory management. Larger values use more memory but may be faster. Only applicable for models supporting tiled inference.",
        ),
        sample_id: Optional[str] = Field(
            "sample",
            description="Identifier for this inference request, used for logging and debugging",
        ),
        skip_cache: Optional[bool] = Field(
            False, description="Force re-download of model package before inference"
        ),
        return_download_url: Optional[bool] = Field(
            False,
            description="If True, each array in the output will be saved to a temporary .npy file in S3 and the output value will be a presigned download URL (str) instead of the raw np.ndarray. The URL is valid for 1 hour.",
        ),
    ) -> Dict[str, Union[np.ndarray, str]]:
        """
        Execute inference on a bioimage.io model with provided input data.

        Performs end-to-end inference including:
        - Automatic input preprocessing according to model specification
        - Model execution with optimized framework backend
        - Output postprocessing and format standardization
        - Memory-efficient processing for large inputs using tiling if supported

        Returns:
            Dictionary mapping output names to inference results. By default each value is a
            ``np.ndarray`` whose shape and data type match the model's output specification
            (e.g. ``{"output": result_array}``). When ``return_download_url=True``, each value
            is instead a presigned S3 download URL (``str``) pointing to the result serialised
            as a ``.npy`` file; the URL is valid for 1 hour.

        Raises:
            ValueError: If model_id is a URL (only model IDs allowed) or inputs don't match specification
            FileNotFoundError: If a URL or temporary file path is provided but the resource does not exist or has expired
            RuntimeError: If model loading, preprocessing, inference, or postprocessing fails

        Note:
            Only published models from the bioimage.io model zoo are supported for inference.
            This method delegates to the model_inference deployment for optimized execution.
            String inputs are resolved via ``_load_image_from_source``: direct HTTP/HTTPS URLs are
            fetched as-is; all other strings are treated as temporary S3 file paths and resolved
            through BioEngine S3 storage. To upload large inputs, first call ``get_upload_url``
            to obtain a presigned URL, upload the file, then pass the returned ``file_path`` as ``inputs``.
        """
        from ray.exceptions import RayTaskError

        await self._check_runtime_available()
        logger.info(f"🤖 Running inference for model '{model_id}'...")

        # Resolve any URL or temporary file path strings to numpy arrays
        if isinstance(inputs, str):
            inputs = await self._load_image_from_source(inputs)
        elif isinstance(inputs, dict):
            resolved: Dict[str, np.ndarray] = {}
            for key, value in inputs.items():
                if isinstance(value, str):
                    array = await self._load_image_from_source(value)
                    resolved[key] = array
                else:
                    resolved[key] = value
            inputs = resolved

        try:
            # Get model package with access tracking
            package = await self.model_cache.get_model_package(
                model_id=model_id,
                stage=False,
                allow_unpublished=False,
                skip_cache=skip_cache,
            )

            # Use context manager to track access and prevent eviction during inference
            async with package:
                logger.info(
                    f"📍 Model source for '{model_id}': {package.source} "
                    f"(latest_remote_modified: {package.latest_remote_modified})"
                )

                result = await self.runtime.predict(
                    rdf_path=package.source,
                    inputs=inputs,
                    weights_format=weights_format,
                    device=device,
                    default_blocksize_parameter=default_blocksize_parameter,
                    sample_id=sample_id,
                    latest_remote_modified=package.latest_remote_modified,
                )
        except RayTaskError as e:
            error_msg = f"Failed to run inference for model '{model_id}': {e}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        if return_download_url:
            new_result = {}
            for key, value in result.items():
                new_result[key] = await self._save_array_to_temp_file(value)
            result = new_result

        logger.info(f"✅ Inference completed for model '{model_id}'.")
        return result


