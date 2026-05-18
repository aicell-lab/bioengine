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

import httpx
import numpy as np
import yaml
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve
from ray.exceptions import RayTaskError
from ray.serve.handle import DeploymentHandle

from bioengine import __version__

logger = logging.getLogger("ray.serve")
logger.setLevel("INFO")

SUPPORTED_FILES_TYPES = Literal[".npy", ".png", ".tiff", ".tif", ".jpeg", ".jpg"]


class BioimageioPackage:
    """Wrapper for cached bioimage.io model package with access tracking."""

    def __init__(
        self,
        package_path: Path,
        latest_remote_modified: float,
        replica_id: str,
    ) -> None:
        self.package_path = package_path
        self.source = str(self.package_path / "rdf.yaml")
        self.latest_remote_modified = latest_remote_modified
        self.replica_id = replica_id
        self._lock_file: Optional[Path] = None

    async def __aenter__(self):
        """Create a per-use lock file so the model is not evicted while in use."""
        token = int(time.time() * 1_000_000)
        self._lock_file = self.package_path / f".in_use_{self.replica_id}_{token}"
        try:
            await asyncio.to_thread(self._lock_file.write_text, str(token))
        except (OSError, IOError) as e:
            logger.warning(f"⚠️ Failed to create in-use lock file: {e}")
            self._lock_file = None
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Remove the per-use lock file and update the last-access timestamp."""
        if self._lock_file is not None:
            try:
                await asyncio.to_thread(self._lock_file.unlink, True)  # missing_ok=True
            except (OSError, IOError) as e:
                logger.warning(f"⚠️ Failed to remove in-use lock file: {e}")
        access_file = self.package_path / ".last_access"
        try:
            await asyncio.to_thread(access_file.write_text, str(time.time()))
        except (OSError, IOError) as e:
            logger.warning(f"⚠️ Failed to update access time on exit: {e}")


class ModelCache:
    def __init__(
        self,
        cache_size_in_gb: float,
        replica_id: str,
    ):
        self.cache_dir = Path().resolve() / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_size_bytes = int(
            cache_size_in_gb * 1024 * 1024 * 1024
        )  # Convert GB to bytes
        self.replica_id = replica_id

        self.per_file_download_timeout = 180.0
        download_timeout = httpx.Timeout(self.per_file_download_timeout)
        self.client = httpx.AsyncClient(timeout=download_timeout, follow_redirects=True)

        self.timeout_threshold = self.per_file_download_timeout + 60.0  # 60s buffer

        num_existing_models = len(list(self.cache_dir.glob("**/rdf.yaml")))
        logger.info(
            f"🔄 Found {num_existing_models} existing models in cache at "
            f"{self.cache_dir}/. {'Starting model validation in the background.' if num_existing_models > 0 else ''}"
        )
        asyncio.create_task(self._scan_cache_dir())

    async def _remove_package(self, package_path: Path) -> None:
        """Safely remove package directory using atomic operations across replicas."""
        if not await asyncio.to_thread(package_path.exists):
            return

        try:
            # Use atomic rename for safe removal across replicas
            temp_dir = (
                package_path.parent
                / f".removing_{package_path.name}_{int(time.time() * 1000000)}"
            )
            await asyncio.to_thread(package_path.rename, temp_dir)
            logger.info(f"🔄 Atomically moved model for removal: '{package_path.name}'")

            # Remove the temporary directory
            await asyncio.to_thread(shutil.rmtree, temp_dir)
            logger.info(f"🗑️ Successfully removed cached model: '{package_path.name}'")

        except FileNotFoundError:
            # Another replica already removed it
            logger.info(
                f"🔍 Model '{package_path.name}' already removed by another replica"
            )
        except OSError as e:
            # Package might be in use, log but don't fail
            logger.warning(
                f"⚠️ Could not remove cached model '{package_path.name}': {e}"
            )
        except Exception as e:
            logger.error(
                f"❌ Unexpected error removing model '{package_path.name}': {e}"
            )

    async def _scan_cache_dir(self) -> None:
        """Scan the cache directory and validate existing models."""
        try:
            all_dirs = await asyncio.to_thread(lambda: list(self.cache_dir.iterdir()))
            local_dirs = []
            for d in all_dirs:
                if await asyncio.to_thread(d.is_dir) and not d.name.startswith("."):
                    local_dirs.append(d)
        except (OSError, IOError) as e:
            logger.warning(f"⚠️ Error reading cache directory: {e}")
            return

        # Check for any stale temporary directories
        all_temp_dirs = await asyncio.to_thread(
            lambda: list(self.cache_dir.glob(".temp_*"))
        )
        temp_dirs = []
        for d in all_temp_dirs:
            if await asyncio.to_thread(d.is_dir) and d.name != ".temp_":
                temp_dirs.append(d)
        for temp_dir in temp_dirs:
            # Remove stale temporary directories if timeout exceeded
            try:
                stat_result = await asyncio.to_thread(temp_dir.stat)
                if time.time() - stat_result.st_ctime > self.timeout_threshold:
                    await asyncio.to_thread(shutil.rmtree, temp_dir)
                    logger.info(f"🧹 Cleaned up stale temporary directory: {temp_dir}")
            except (OSError, IOError) as e:
                logger.warning(
                    f"⚠️ Failed to clean up stale temporary directory {temp_dir}: {e}"
                )

    async def _get_url_with_retry(
        self, url: str, params: Dict[str, str]
    ) -> httpx.Response:
        """
        Helper method to fetch a URL with retries.

        Implements a simple retry mechanism for HTTP GET requests to handle
        transient network issues. Retries the request up to 3 times with
        exponential backoff.

        Args:
            url: The URL to fetch
        Returns:
            The HTTP response object
        Raises:
            httpx.HTTPError: If all retry attempts fail
        """
        max_attempts = 4
        backoff = 0.2  # backoff: 0.2s, 0.4s, 0.8s
        backoff_multiplier = 2.0

        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                return response
            except Exception as e:
                # Don't retry on 4xx client errors (except 429 Too Many Requests)
                if isinstance(e, httpx.HTTPStatusError):
                    if (
                        400 <= e.response.status_code < 500
                        and e.response.status_code != 429
                    ):
                        return response

                if attempt < max_attempts:
                    # Sleep with exponential backoff before retrying
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for URL {url}, "
                        f"params: {params}, error: {e}. Retrying in {backoff:.1f}s..."
                    )
                    await asyncio.sleep(backoff)
                    backoff *= backoff_multiplier
                else:
                    # If we get here, all retries failed due to errors (network, transport, etc.)
                    logger.error(
                        f"Failed to fetch URL '{url}' after {max_attempts} attempts: {e}"
                    )
                    if isinstance(e, httpx.HTTPStatusError):
                        return response
                    else:
                        raise e

    async def _check_model_published_status(self, model_id: str, stage: bool) -> None:
        """
        Check if a model is published by looking at its manifest status.

        Behavior:
        - Retries with exponential backoff on transient errors.
        - Falls back from stage=true to stage=false on 404.
        - Only raises 'not published' if status == 'request-review'.
        - Raises a RuntimeError if status could not be determined after retries.
        """
        artifact_url = f"https://hypha.aicell.io/bioimage-io/artifacts/{model_id}"

        response = await self._get_url_with_retry(
            url=artifact_url, params={"stage": str(stage).lower()}
        )

        if response.status_code == 404 and stage:
            logger.warning(
                f"⚠️ Staged version not found for model '{model_id}', trying committed version..."
            )
            response = await self._get_url_with_retry(
                url=artifact_url, params={"stage": "false"}
            )

        try:
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Failed to download manifest from {artifact_url}"
            ) from e

        artifact = await asyncio.to_thread(yaml.safe_load, response.text)
        status = artifact["manifest"].get("status")

        # Explicit logic:
        # - Only treat as not published if status == "request-review"
        # - Any other status (including None) is considered published/acceptable
        if status == "request-review":
            raise ValueError(
                f"Model '{model_id}' is not published (status='request-review'). "
                f"Only published models are allowed."
            )

    async def _wait_for_download_completion(
        self, package_dir: Path, max_wait_time: int = 300
    ) -> bool:
        """Wait for another replica to finish downloading. Returns True if successful."""
        import aiofiles

        start_time = time.time()
        downloading_marker = (
            package_dir.parent / f".downloading_{package_dir.name}.lock"
        )

        logger.info(f"⏳ Waiting for download completion: {package_dir.name}")

        check_interval = 2.0
        while time.time() - start_time < max_wait_time:
            try:
                # Check if download is complete (package exists and no downloading marker)
                if await asyncio.to_thread(
                    package_dir.exists
                ) and not await asyncio.to_thread(downloading_marker.exists):
                    logger.info(
                        f"✅ Download of model '{package_dir.name}' completed by another replica."
                    )
                    return True

                # Check if download failed (no package and no downloading marker)
                if not await asyncio.to_thread(
                    package_dir.exists
                ) and not await asyncio.to_thread(downloading_marker.exists):
                    logger.warning(
                        f"⚠️ Download of model '{package_dir.name}' appears to have failed on another replica."
                    )
                    return False

                # Check if download has timed out
                if await asyncio.to_thread(downloading_marker.exists):
                    try:
                        async with aiofiles.open(downloading_marker, "r") as f:
                            lock_data = await asyncio.to_thread(
                                json.loads, await f.read()
                            )

                        download_start_time = lock_data.get("start_time", 0)
                        elapsed_time = time.time() - download_start_time

                        if elapsed_time > self.timeout_threshold:

                            logger.warning(
                                f"🕒 Download by replica '{lock_data.get('replica_id', 'unknown')}' has timed out ({elapsed_time:.1f}s > {self.timeout_threshold:.1f}s)"
                            )

                            # Remove stale downloading directory
                            temp_download_dir = (
                                self.cache_dir
                                / f".temp_{package_dir.name}_{int(download_start_time * 1000000)}"
                            )
                            if await asyncio.to_thread(temp_download_dir.exists):
                                try:
                                    await asyncio.to_thread(
                                        shutil.rmtree, temp_download_dir
                                    )
                                    logger.info(
                                        f"🧹 Cleaned up stale download directory: {temp_download_dir}"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"⚠️ Failed to clean up stale download directory: {e}"
                                    )

                            return False

                    except (json.JSONDecodeError, KeyError, OSError, IOError):
                        # Corrupted or unreadable lock file, treat as timed out
                        logger.warning(
                            f"⚠️ Corrupted lock file detected, treating as timed out"
                        )
                        return False

            except (OSError, IOError) as e:
                # Handle filesystem errors gracefully
                logger.warning(f"⚠️ Filesystem error while waiting: {e}")

            await asyncio.sleep(check_interval)

        # Timeout reached
        logger.warning(
            f"⏰ Timeout waiting for '{package_dir.name}' download completion."
        )
        return False

    async def _get_cached_models_info(self) -> List[Dict[str, Union[str, float, bool]]]:
        """Get information about all cached models including access times and locks."""
        import aiofiles

        models_info = []

        try:
            items = await asyncio.to_thread(lambda: list(self.cache_dir.iterdir()))
        except (OSError, IOError) as e:
            logger.warning(f"⚠️ Error reading models directory: {e}")
            return models_info

        for item in items:
            try:
                if not await asyncio.to_thread(item.is_dir) or item.name.startswith(
                    "."
                ):
                    continue

                access_file = item / ".last_access"
                meta_file = item / ".file_metadata.json"
                downloading_marker = self.cache_dir / f".downloading_{item.name}.lock"

                # Check if currently downloading
                is_downloading = await asyncio.to_thread(downloading_marker.exists)

                # Get last access time
                last_access = 0
                if await asyncio.to_thread(access_file.exists):
                    try:
                        access_content = await asyncio.to_thread(access_file.read_text)
                        last_access = float(access_content.strip())
                    except (ValueError, FileNotFoundError, OSError, IOError):
                        last_access = 0

                # A model is locked if any non-stale per-use lock file exists
                # (created by BioimageioPackage.__aenter__, removed by __aexit__).
                # Lock files older than 10 minutes are treated as stale (e.g. from
                # a crashed replica) and ignored to prevent indefinite eviction block.
                in_use_lock_max_age_s = 600  # 10 minutes
                in_use_files = await asyncio.to_thread(
                    lambda: list(item.glob(".in_use_*"))
                )
                is_locked = False
                for lock_file in in_use_files:
                    try:
                        # Filename format: .in_use_{replica_id}_{token_microseconds}
                        token_us = int(lock_file.name.rsplit("_", 1)[-1])
                        age_s = time.time() - token_us / 1_000_000
                        if age_s < in_use_lock_max_age_s:
                            is_locked = True
                            break
                        else:
                            logger.warning(
                                f"⚠️ Ignoring stale in-use lock '{lock_file.name}' "
                                f"({age_s:.0f}s old > {in_use_lock_max_age_s}s limit)"
                            )
                    except (ValueError, IndexError):
                        # Unrecognised filename format – ignore
                        logger.warning(
                            f"⚠️ Ignoring in-use lock file with unrecognised format: '{lock_file.name}'"
                        )
                        continue

                # Get download time from file metadata
                download_time = 0
                if await asyncio.to_thread(meta_file.exists):
                    try:
                        async with aiofiles.open(meta_file, "r") as f:
                            content = await f.read()
                            metadata = await asyncio.to_thread(json.loads, content)
                        # Get the newest timestamp from all files
                        timestamps = [
                            float(ts)
                            for ts in metadata.values()
                            if isinstance(ts, (int, float, str))
                        ]
                        download_time = max(timestamps) if timestamps else 0
                    except (
                        ValueError,
                        FileNotFoundError,
                        OSError,
                        IOError,
                        json.JSONDecodeError,
                    ):
                        download_time = 0

                # Calculate model size in bytes
                model_size_bytes = await self._calculate_model_size(item)

                models_info.append(
                    {
                        "model_id": item.name,
                        "path": item,
                        "last_access": last_access,
                        "download_time": download_time,
                        "size_bytes": model_size_bytes,
                        "is_locked": is_locked,
                        "is_downloading": is_downloading,
                    }
                )
            except (OSError, IOError) as e:
                # Skip problematic directories but continue processing others
                logger.warning(f"⚠️ Error processing cache directory {item}: {e}")
                continue

        return models_info

    async def _calculate_model_size(self, model_dir: Path) -> int:
        """Calculate the total size of a model directory in bytes."""
        total_size = 0
        try:
            all_items = await asyncio.to_thread(lambda: list(model_dir.rglob("*")))
            for item in all_items:
                if await asyncio.to_thread(item.is_file):
                    try:
                        stat_result = await asyncio.to_thread(item.stat)
                        total_size += stat_result.st_size
                    except (OSError, IOError):
                        # Skip files that can't be accessed
                        continue
        except (OSError, IOError):
            # Return 0 if directory can't be accessed
            pass
        return total_size

    async def _ensure_cache_space(
        self,
        model_id: str,
        model_size_bytes: int,
        max_retries: int = 10,
        retry_delay: float = 5.0,
    ) -> None:
        """Ensure there's space in cache for a new model, evicting old ones if necessary."""
        logger.info(
            f"🔍 Checking cache space for new model: '{model_id}' ({model_size_bytes / (1024*1024):.1f} MB)"
        )

        for attempt in range(max_retries):
            # Add small random delay to reduce contention between replicas
            if attempt > 0:
                delay = retry_delay + random.uniform(0, 2)
                await asyncio.sleep(delay)

            models_info = await self._get_cached_models_info()

            # Calculate current cache size in bytes.
            current_size_bytes = sum(model["size_bytes"] for model in models_info)

            logger.info(
                f"📊 Current cache usage: {current_size_bytes / (1024*1024*1024):.3f} GB / {self.cache_size_bytes / (1024*1024*1024):.3f} GB"
            )

            if current_size_bytes + model_size_bytes <= self.cache_size_bytes:
                logger.info(f"✅ Cache space available for model '{model_id}'")
                return

            # Need to evict models - sort by last access time (oldest first)
            evictable_models = [
                model
                for model in models_info
                if not model["is_locked"] and not model["is_downloading"]
            ]

            if not evictable_models:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ No evictable models found, retrying...")
                    continue
                else:
                    logger.warning(f"⚠️ Could not evict any models, proceeding anyway")
                    return

            # Sort by last access time (oldest first)
            evictable_models.sort(key=lambda x: x["last_access"])

            # Evict models until we have enough space
            space_needed = (
                current_size_bytes + model_size_bytes
            ) - self.cache_size_bytes

            for oldest_model in evictable_models:
                if space_needed <= 0:
                    break

                logger.info(
                    f"🗑️ Evicting model: {oldest_model['model_id']} ({oldest_model['size_bytes'] / (1024*1024):.1f} MB, last accessed: {oldest_model['last_access']})"
                )

                try:
                    await self._remove_package(oldest_model["path"])
                    logger.info(
                        f"✅ Successfully evicted model: {oldest_model['model_id']} ({oldest_model['size_bytes'] / (1024*1024):.1f} MB)"
                    )
                    space_needed -= oldest_model["size_bytes"]
                except Exception as e:
                    logger.error(
                        f"❌ Failed to evict model '{oldest_model['model_id']}': {e}"
                    )

            # Check if we've freed enough space
            if space_needed <= 0:
                logger.info(
                    f"✅ Successfully freed enough cache space for model '{model_id}'"
                )
                return
            elif attempt < max_retries - 1:
                logger.warning(
                    f"⚠️ Still need {space_needed / (1024*1024):.1f} MB more space, retrying..."
                )
                await asyncio.sleep(retry_delay)
                continue
            else:
                logger.warning(f"⚠️ Could not free enough space, proceeding anyway")
                return

    async def _fetch_file_list(self, model_id: str, stage: bool = False) -> List[dict]:
        """Fetch the list of files for a model from the bioimage.io artifacts API."""
        files_url = f"https://hypha.aicell.io/bioimage-io/artifacts/{model_id}/files/"
        response = await self._get_url_with_retry(
            url=files_url, params={"stage": str(stage).lower()}
        )

        if response.status_code == 404 and stage:
            # If staged version doesn't exist, try with stage=false
            logger.warning(
                f"⚠️ Staged file list not found for model '{model_id}', trying committed version..."
            )
            response = await self._get_url_with_retry(
                url=files_url, params={"stage": "false"}
            )

        try:
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch file list for model '{model_id}'"
            ) from e

        return response.json()

    async def _calculate_remote_model_size(self, file_list: List[dict]) -> int:
        """Calculate the total size of a model from its file list."""
        total_size = 0
        for file_info in file_list:
            if file_info.get("type") == "file" and "size" in file_info:
                total_size += file_info["size"]
        return total_size

    async def _download_file(
        self,
        model_id: str,
        model_dir: Path,
        file_meta: dict,
        stage: bool = False,
    ):
        """Download a single file for a model."""
        import aiofiles

        file_url = f"https://hypha.aicell.io/bioimage-io/artifacts/{model_id}/files/{file_meta['name']}"
        file_path = model_dir / file_meta["name"]

        # Create parent directories if needed
        await asyncio.to_thread(file_path.parent.mkdir, parents=True, exist_ok=True)

        response = await self._get_url_with_retry(
            url=file_url, params={"stage": str(stage).lower()}
        )

        if response.status_code == 404 and stage:
            # If staged version doesn't exist, try with stage=false
            response = await self._get_url_with_retry(
                url=file_url, params={"stage": "false"}
            )

        try:
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Failed to download file '{file_meta['name']}' for model '{model_id}'"
            ) from e

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(response.content)

        return file_meta["name"], file_meta["last_modified"]

    async def _download_model_files(
        self,
        model_id: str,
        model_dir: Path,
        stage: bool = False,
        check_newer_files: bool = True,
        file_list: Optional[List[dict]] = None,
    ):
        """Download all files for a model using concurrent downloads."""
        import aiofiles

        await asyncio.to_thread(model_dir.mkdir, parents=True, exist_ok=True)

        meta_path = model_dir / ".file_metadata.json"
        old_meta = {}
        if check_newer_files and await asyncio.to_thread(meta_path.exists):
            async with aiofiles.open(meta_path, "r") as f:
                content = await f.read()
                old_meta = await asyncio.to_thread(json.loads, content)

        # Use provided file list or fetch it if not provided
        if file_list is None:
            file_list = await self._fetch_file_list(model_id, stage=stage)
        remote_files = {
            f["name"]: f
            for f in file_list
            if f["type"] == "file" and f["name"] != "test_report.json"
        }

        # Get local files (excluding metadata files)
        all_files = await asyncio.to_thread(lambda: list(model_dir.glob("*")))
        local_files = set()
        for f in all_files:
            if await asyncio.to_thread(f.is_file) and not f.name.startswith("."):
                local_files.add(f.name)

        # Determine files to delete
        remote_file_names = set(remote_files.keys())
        files_to_delete = local_files - remote_file_names
        for fname in files_to_delete:
            (model_dir / fname).unlink()

        # Determine files to download
        files_to_download = []
        for name, meta in remote_files.items():
            if not check_newer_files:
                files_to_download.append(meta)
            elif name not in old_meta or meta["last_modified"] > old_meta[name]:
                files_to_download.append(meta)

        tasks = [
            self._download_file(model_id, model_dir, f, stage=stage)
            for f in files_to_download
        ]
        results = await asyncio.gather(*tasks)

        # Update metadata
        # Keep metadata only for currently tracked remote files.
        new_meta = {name: old_meta[name] for name in remote_files if name in old_meta}
        for name, ts in results:
            new_meta[name] = ts

        async with aiofiles.open(meta_path, "w") as f:
            await f.write(json.dumps(new_meta, indent=2))

        return {
            "downloaded": [name for name, _ in results],
            "deleted": list(files_to_delete),
            "skipped": list(remote_file_names - {name for name, _ in results}),
        }

    async def _create_package(self, model_id: str, stage: bool) -> None:
        """
        Create or update a model package in the cache directory.

        Downloads all files of a model artifact from the bioimage.io workspace.
        If files already exist, they are updated only if newer versions are available.
        Uses atomic operations to prevent conflicts between replicas.
        """
        import aiofiles

        package_dir = self.cache_dir / model_id
        downloading_marker = self.cache_dir / f".downloading_{model_id}.lock"

        # Fetch file list once at the beginning
        file_list = None
        try:
            file_list = await self._fetch_file_list(model_id, stage=stage)
        except Exception as e:
            logger.error(f"❌ Failed to fetch file list for model '{model_id}': {e}")
            raise RuntimeError(f"Failed to fetch file list for model {model_id}: {e}")

        # Calculate model size from file list
        model_size_bytes = await self._calculate_remote_model_size(file_list)
        logger.info(
            f"📊 Model '{model_id}' size: {model_size_bytes / (1024*1024):.1f} MB"
        )

        # Check if model already exists
        if await asyncio.to_thread(package_dir.exists):
            logger.info(
                f"💾 Model '{model_id}' already exists, checking for updates..."
            )

            # Check if files need updating by comparing with remote file list
            try:
                remote_files = {
                    f["name"]: f
                    for f in file_list
                    if f["type"] == "file" and f["name"] != "test_report.json"
                }

                # Get local file metadata
                meta_path = package_dir / ".file_metadata.json"
                local_meta = {}
                if await asyncio.to_thread(meta_path.exists):
                    async with aiofiles.open(meta_path, "r") as f:
                        content = await f.read()
                        local_meta = await asyncio.to_thread(json.loads, content)

                # Check for files that need updating
                files_need_update = False
                for name, remote_file in remote_files.items():
                    if (
                        name not in local_meta
                        or remote_file["last_modified"] > local_meta[name]
                    ):
                        files_need_update = True
                        logger.info(
                            f"📄 File '{name}' needs update (remote: {remote_file['last_modified']}, local: {local_meta.get(name, 'missing')})"
                        )
                        break

                # Check for files that no longer exist remotely
                all_local_files = await asyncio.to_thread(
                    lambda: list(package_dir.glob("*"))
                )
                local_files = set()
                for f in all_local_files:
                    if await asyncio.to_thread(f.is_file) and not f.name.startswith(
                        "."
                    ):
                        local_files.add(f.name)
                remote_file_names = set(remote_files.keys())
                files_to_delete = local_files - remote_file_names
                if files_to_delete:
                    files_need_update = True
                    logger.info(f"🗑️ Files to delete: {list(files_to_delete)}")

                if not files_need_update:
                    logger.info(f"✅ Model '{model_id}' is up to date")
                    # Update access time and return
                    access_file = package_dir / ".last_access"
                    try:
                        await asyncio.to_thread(
                            access_file.write_text, str(time.time())
                        )
                    except (OSError, IOError) as e:
                        logger.warning(
                            f"⚠️ Failed to update access time for existing model: {e}"
                        )
                    return

                logger.info(
                    f"🔄 Model '{model_id}' needs updates, proceeding with download..."
                )

            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to check for updates: {e}. Proceeding with download..."
                )
                # Continue with download if update check fails

        # Try to claim the download by creating a downloading marker atomically
        try:
            lock_data = {
                "replica_id": self.replica_id,
                "start_time": time.time(),
                "model_id": model_id,
                "stage": stage,
            }
            async with aiofiles.open(downloading_marker, "x") as f:
                await f.write(json.dumps(lock_data, indent=2))
            logger.info(f"🔒 Claimed download for model '{model_id}'.")
        except FileExistsError:
            # Another replica is downloading, wait for completion
            logger.info(f"⏳ Another replica is downloading '{model_id}', waiting...")
            if await self._wait_for_download_completion(package_dir):
                # Update access time
                access_file = package_dir / ".last_access"
                try:
                    await asyncio.to_thread(access_file.write_text, str(time.time()))
                except (OSError, IOError) as e:
                    logger.warning(f"⚠️ Failed to update access time after waiting: {e}")
                return
            else:
                # Download failed or timed out, try to claim it ourselves

                # Remove stale marker
                downloading_marker.unlink()

                logger.info(
                    f"🧹 Cleaned up stale download marker for model '{model_id}'."
                )

                # Retry claiming
                try:
                    # Update lock data with new start time
                    lock_data["start_time"] = time.time()
                    async with aiofiles.open(downloading_marker, "x") as f:
                        await f.write(json.dumps(lock_data, indent=2))
                    logger.info(
                        f"🔄 Claimed download after timeout for model '{model_id}'."
                    )
                except FileExistsError:
                    raise RuntimeError(
                        f"Failed to claim download for model {model_id} after timeout"
                    )

        try:
            # Ensure cache space AFTER claiming download (so download marker counts towards limit)
            await self._ensure_cache_space(model_id, model_size_bytes)

            # Create temporary download directory
            temp_download_dir = (
                self.cache_dir
                / f".temp_{model_id}_{int(lock_data['start_time'] * 1000000)}"
            )

            await asyncio.to_thread(temp_download_dir.mkdir)
            logger.info(
                f"📁 Starting download of model '{model_id}' to temporary directory."
            )

            # If updating an existing package, copy existing files to temp directory first
            if await asyncio.to_thread(package_dir.exists):
                logger.info(
                    f"📋 Copying existing files to temporary directory for update..."
                )
                try:
                    # Copy all existing files except access tracking files
                    for item in await asyncio.to_thread(
                        lambda: list(package_dir.iterdir())
                    ):
                        if await asyncio.to_thread(item.is_file) and item.name not in [
                            ".last_access"
                        ]:
                            dest_file = temp_download_dir / item.name
                            await asyncio.to_thread(
                                dest_file.parent.mkdir, parents=True, exist_ok=True
                            )
                            await asyncio.to_thread(shutil.copy2, item, dest_file)
                        elif await asyncio.to_thread(item.is_dir):
                            dest_dir = temp_download_dir / item.name
                            await asyncio.to_thread(
                                shutil.copytree, item, dest_dir, dirs_exist_ok=True
                            )
                    logger.info(f"✅ Copied existing files to temporary directory")
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to copy existing files: {e}. Starting fresh download..."
                    )

            download_start = time.time()

            # Use concurrent file download for all models, passing the pre-fetched file list
            download_result = await self._download_model_files(
                model_id=model_id,
                model_dir=temp_download_dir,
                stage=stage,
                check_newer_files=True,  # Always check for newer files when updating
                file_list=file_list,  # Pass the file list we fetched at the beginning
            )

            logger.info(
                f"📦 Downloaded {len(download_result['downloaded'])} files for model '{model_id}'."
            )

            download_duration = time.time() - download_start
            logger.info(
                f"⚡ Download completed in {download_duration:.2f}s for model '{model_id}'."
            )

            # Atomically move to final location (handle existing directory)
            if await asyncio.to_thread(package_dir.exists):
                # For updates: move existing dir away, move temp dir to final location, then remove old dir
                backup_dir = (
                    self.cache_dir / f".backup_{model_id}_{int(time.time() * 1000000)}"
                )
                await asyncio.to_thread(package_dir.rename, backup_dir)
                await asyncio.to_thread(temp_download_dir.rename, package_dir)
                await asyncio.to_thread(shutil.rmtree, backup_dir)
                logger.info(
                    f"🔄 Atomically updated model '{model_id}' in final location."
                )
            else:
                # For new models: simple rename
                temp_download_dir.rename(package_dir)
                logger.info(
                    f"🔄 Atomically moved model '{model_id}' to final location."
                )

            # Create last access file (file metadata is already created by self._download_model_files)
            current_time = time.time()
            access_file = package_dir / ".last_access"

            try:
                await asyncio.to_thread(access_file.write_text, str(current_time))
            except (OSError, IOError) as e:
                logger.warning(f"⚠️ Failed to create access file for new model: {e}")

        except Exception as e:
            logger.error(f"❌ Failed to download model '{model_id}': {e}")
            # Clean up temporary directory
            if await asyncio.to_thread(temp_download_dir.exists):
                try:
                    await asyncio.to_thread(shutil.rmtree, temp_download_dir)
                except Exception as cleanup_error:
                    logger.warning(
                        f"⚠️ Failed to cleanup temp directory: {cleanup_error}"
                    )
            raise RuntimeError(f"Failed to download model {model_id}: {e}")
        finally:
            # Remove downloading marker
            try:
                downloading_marker.unlink()
                logger.info(f"🔓 Released download claim for model '{model_id}'.")
            except (FileNotFoundError, OSError) as e:
                logger.warning(f"⚠️ Failed to remove downloading marker: {e}")

        logger.info(f"🎉 Successfully completed download of model '{model_id}'.")

    async def _get_latest_remote_modified_time(self, package_path: Path) -> float:
        """Get the latest tracked remote last-modified time from .file_metadata.json."""
        import aiofiles

        meta_path = package_path / ".file_metadata.json"
        if await asyncio.to_thread(meta_path.exists):
            async with aiofiles.open(meta_path, "r") as f:
                content = await f.read()
                metadata = await asyncio.to_thread(json.loads, content)
                return max(metadata.values(), default=0.0)
        return 0.0

    async def get_model_package(
        self,
        model_id: str,
        stage: bool,
        allow_unpublished: bool,
        skip_cache: bool,
    ) -> "BioimageioPackage":
        """Get a cached model package or download it if not available."""

        # Check if model is published
        if not allow_unpublished:
            await self._check_model_published_status(model_id, stage=stage)

        # Force a complete re-download if skip_cache is True
        package_path = self.cache_dir / model_id
        if await asyncio.to_thread(package_path.exists) and skip_cache:
            await self._remove_package(package_path)

        # Create or update the local package
        await self._create_package(model_id, stage=stage)

        # Get latest tracked remote last-modified time from .file_metadata.json
        latest_remote_modified = await self._get_latest_remote_modified_time(
            package_path
        )

        return BioimageioPackage(
            package_path=package_path,
            latest_remote_modified=latest_remote_modified,
            replica_id=self.replica_id,
        )


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
        "memory": 4 * 1024 * 1024 * 1024,  # 4GB RAM limit
        "runtime_env": {
            "pip": [
                "aiofiles>=23.0.0",
                "bioimageio.core==0.10.0",
                "imageio>=2.37.0",
                "numpy==1.26.4",
                "tqdm>=4.64.0",
            ],
        },
    },
    max_ongoing_requests=10,  # Should be smaller than the number of cached models to prevent race conditions between requests
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
class EntryDeployment:
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
        runtime_deployment: DeploymentHandle,
        cache_size_in_gb: float = 50.0,
    ) -> None:
        self.runtime_deployment = runtime_deployment

        # Set Hypha server and workspace
        self.server_url = "https://hypha.aicell.io"
        self._hypha_token = os.getenv("HYPHA_TOKEN")
        if not self._hypha_token:
            raise RuntimeError("HYPHA_TOKEN environment variable is not set")

        # Get replica identifier for logging
        try:
            self.replica_id = serve.get_replica_context().replica_tag
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

    async def async_init(self) -> None:
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
                "test reports will be published to source artifacts."
            )
        else:
            logger.warning(
                "⚠️ HYPHA_TOKEN does not have write access to bioimage-io workspace; "
                "publish_test_report=True will be silently downgraded to a no-op. "
                "The test() result is still returned to the caller."
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
                self.runtime_deployment.check_health.remote(),
                timeout=2.0,
            )
        except Exception as e:
            raise RuntimeError(
                "GPU runtime deployment is not available. "
                "Inference, test, and validate are unavailable until the runtime starts."
            ) from e

    async def check_health(self) -> None:
        # Test connection to the Hypha server only — runtime availability is
        # checked per-call in GPU methods so partial registration is preserved
        # (service stays registered for CPU-only methods when GPU is down).
        await self.hypha_client.echo("ping")

    # === Internal Helper Methods ===

    def _get_bioimageio_versions(self) -> Dict[str, str]:
        """
        Return installed versions of bioimageio packages used by this deployment.

        Returns:
            Dictionary containing package versions for:
            - bioimageio.core
            - bioimageio.spec
        """
        from importlib.metadata import PackageNotFoundError, version

        package_names = ["bioimageio.core", "bioimageio.spec"]
        versions: Dict[str, str] = {}

        for package_name in package_names:
            try:
                versions[package_name] = version(package_name)
            except PackageNotFoundError:
                versions[package_name] = "not-installed"

        logger.debug(f"📦 Bioimage.io package versions: {versions}")
        return versions

    def _ensure_bioengine_in_test_env(self, test_report: dict) -> dict:
        """Ensure test_report['env'] contains the current BioEngine version row."""
        env = test_report.get("env")
        if not isinstance(env, list):
            env = []
        else:
            env = list(env)

        bioengine_row_index: Optional[int] = None
        for i, row in enumerate(env):
            if (
                isinstance(row, (list, tuple))
                and len(row) >= 1
                and str(row[0]) == "bioengine"
            ):
                bioengine_row_index = i
                break

        bioengine_row = ["bioengine", __version__, "", ""]
        if bioengine_row_index is None:
            env.append(bioengine_row)
        else:
            existing_row = env[bioengine_row_index]
            if isinstance(existing_row, tuple):
                existing_row = list(existing_row)
            while len(existing_row) < 4:
                existing_row.append("")
            existing_row[1] = __version__
            env[bioengine_row_index] = existing_row

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

    # === Exposed BioEngine App Methods - all methods decorated with @schema_method will be exposed as API endpoints ===
    # Note: Parameter type hints and docstrings will be used to generate the API documentation.

    @schema_method
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

    @schema_method
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

    @schema_method
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

    @schema_method
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

    @schema_method
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
                    test_report = await self.runtime_deployment.test.remote(
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

                # Add BioEngine version to the test report environment
                test_report = self._ensure_bioengine_in_test_env(test_report)

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

            # Publish test report to artifact
            if publish_test_report and not self._can_publish_to_bioimage_io:
                logger.info(
                    f"ℹ️ Skipping publish_test_report for '{model_id}': "
                    f"deployment's HYPHA_TOKEN has no write access to the bioimage-io workspace. "
                    f"Returning the test report to the caller without publishing it back to the source artifact."
                )
            elif publish_test_report:
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

    @schema_method
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

    @schema_method(arbitrary_types_allowed=True)
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

                result = await self.runtime_deployment.predict.remote(
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


if __name__ == "__main__":
    import imageio.v3 as iio

    async def run_deployment_test():

        class MockMethod:
            def __init__(self, name: str):
                self.name = name

            async def remote(self, *args, **kwargs):
                logger.info(
                    f"🎭 Mocked method '{self.name}' called with args={args}, kwargs={kwargs}"
                )
                if self.name == "test":
                    # Return a mock test result
                    return {
                        "status": "passed",
                        "details": "All tests passed successfully.",
                    }
                elif self.name == "predict":
                    # Return a mock prediction result
                    return {"output": np.zeros((1, 1, 64, 64), dtype=np.float32)}
                else:
                    return {"mock_result": True}

        class MockDeploymentHandle:
            def __getattr__(self, name):
                return MockMethod(name)

        # Set up the environment variables like in the real deployment
        deployment_workdir = Path.home() / ".bioengine" / "apps" / "model-runner"
        deployment_workdir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(deployment_workdir / "tmp")
        os.environ["HOME"] = str(deployment_workdir)
        os.chdir(deployment_workdir)

        # Create a mock replica context for standalone testing
        class MockReplicaContext:
            replica_tag = "test-replica-001"

        # Mock serve.get_replica_context for testing
        original_get_replica_context = serve.get_replica_context
        serve.get_replica_context = lambda: MockReplicaContext()

        try:
            model_runner = EntryDeployment.func_or_class(
                runtime_deployment=MockDeploymentHandle(),
                cache_size_in_gb=0.230,  # 230 MB cache for testing
            )

            await model_runner.async_init()
            await model_runner.check_health()

            # await model_runner.test(model_id="ambitious-ant", publish_test_report=True)

            # Search for segmentation models in the bioimage.io collection
            search_results = await model_runner.search_models(
                keywords=["cell", "nuclei", "segmentation"], limit=5
            )
            logger.info(f"Search results: {search_results}")

            # Test all methods of the deployment (ambitious-ant ~18 MB)
            model_id = "ambitious-ant"

            logger.info(f"🔍 Test 1/5: Getting model RDF...")
            t0 = time.time()
            model_rdf = await model_runner.get_model_rdf(model_id=model_id, stage=False)
            logger.info(f"✅ RDF retrieval successful ({time.time()-t0:.2f}s)")

            logger.info(f"🔬 Test 2/5: Validating RDF...")
            t0 = time.time()
            validation_result = await model_runner.validate(rdf_dict=model_rdf)
            logger.info(
                f"✅ Validation {'passed' if validation_result['success'] else 'failed'} ({time.time()-t0:.2f}s)"
            )

            logger.info(f"🧩 Test 3/5: Testing model...")
            t0 = time.time()
            _ = await model_runner.test(model_id=model_id, stage=False)
            logger.info(f"✅ Model test completed ({time.time()-t0:.2f}s)")

            logger.info(f"🔄 Test 4/5: Testing with cache skip...")
            t0 = time.time()
            _ = await model_runner.test(model_id=model_id, stage=False, skip_cache=True)
            logger.info(f"✅ Skip cache test completed ({time.time()-t0:.2f}s)")

            logger.info(f"🤖 Test 5/5: Running inference...")
            local_package = await model_runner.model_cache.get_model_package(
                model_id=model_id, stage=False, allow_unpublished=False, skip_cache=False
            )
            async with local_package:
                test_input_source = model_rdf["test_inputs"][0]
                test_image_path = local_package.package_path / test_input_source
                test_image = np.load(test_image_path).astype("float32")
            t0 = time.time()
            _ = await model_runner.infer(model_id=model_id, inputs=test_image)
            logger.info(f"✅ Inference completed ({time.time()-t0:.2f}s)")

            # Simulate newer remote files for testing updates
            file_metadata_path = (
                model_runner.model_cache.cache_dir
                / "ambitious-ant"
                / ".file_metadata.json"
            )
            with open(file_metadata_path, "r") as f:
                metadata = json.load(f)
                metadata["rdf.yaml"] -= 1000  # Make it older

            with open(file_metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            await model_runner.test(model_id="ambitious-ant")

            # Test result caching of "test" method
            await model_runner.test(model_id="ambitious-ant")

            # Load a test image from the model package to upload
            image_path = (
                model_runner.model_cache.cache_dir / "ambitious-ant" / "test-input.npy"
            )
            array = np.load(image_path)

            # Test image upload to temporary storage
            upload_info_npy = await model_runner.get_upload_url(file_type=".npy")

            buffer = BytesIO()
            np.save(buffer, array)
            # Bash equivalent (upload .npy from filesystem file; works for .npy, .png, .tiff, .tif, .jpeg, .jpg):
            # curl -fsSL -X PUT --data-binary @/path/to/file.npy "<upload_url>"
            async with httpx.AsyncClient() as client:
                await client.put(upload_info_npy["upload_url"], data=buffer.getvalue())

            result = await model_runner.infer(
                model_id="ambitious-ant",
                inputs=upload_info_npy["file_path"],
                return_download_url=True,
            )
            logger.info(f"Inference result: {result}")

            # Bash equivalent (download .npy result from URL):
            # curl -fsSL --compressed --output output.npy "<download_url>"
            async with httpx.AsyncClient() as client:
                response = await client.get(result["output"])
                response.raise_for_status()

            buffer = BytesIO(response.content)
            array_result = np.load(buffer)
            logger.info(
                f"Result array shape: {array_result.shape}, dtype: {array_result.dtype}"
            )

            # Test image upload as PNG to temporary storage
            upload_info_png = await model_runner.get_upload_url(file_type=".png")

            buffer = BytesIO()
            array_png = array[0, 0, :, :]
            array_png = (array_png - array_png.min()) / (
                array_png.max() - array_png.min() + 1e-8
            )  # Normalize to [0, 1]
            array_png = (array_png * 255).clip(0, 255).astype(np.uint8)
            iio.imwrite(buffer, array_png, extension=".png")
            async with httpx.AsyncClient() as client:
                await client.put(
                    upload_info_png["upload_url"], content=buffer.getvalue()
                )

            result = await model_runner.infer(
                model_id="ambitious-ant", inputs=upload_info_png["file_path"]
            )
            logger.info(f"Inference result: {result}")

            # This should exceed the cache size limit of 230 MB and trigger eviction of the older model
            await model_runner.test(model_id="charismatic-whale")  # ~224 MB

        finally:
            # Restore original function
            serve.get_replica_context = original_get_replica_context

    asyncio.run(run_deployment_test())
