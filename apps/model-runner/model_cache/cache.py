"""LRU model cache for bioimage.io model packages.

The ``ModelCache`` downloads model artifacts from the BioImage.IO Hypha
workspace, stores them under a per-worker ``./models`` directory, and
coordinates across replicas with on-disk markers so two replicas don't
download the same model twice. It also implements LRU eviction so the
cache stays under a configured size budget.

Public surface used by ``EntryApp``:

* ``ModelCache(cache_size_in_gb, replica_id)`` — constructor
* ``await cache.get_model_package(model_id, stage, allow_unpublished, skip_cache)``
  — returns a :class:`BioimageioPackage` ready to use in an ``async with``
* ``cache.client`` — the shared ``httpx.AsyncClient`` for downloads
* ``cache._get_url_with_retry(url, params)`` — exposed because the entry
  uses the same retry/backoff for RDF and documentation fetches

Everything else on the class is private orchestration the entry code
doesn't talk to directly.
"""


import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx

from .package import BioimageioPackage

logger = logging.getLogger("ray.serve")


class ModelCache:
    def __init__(
        self,
        cache_size_in_gb: float,
        replica_id: str,
    ):
        # Place the model cache under ``$MODEL_CACHE_DIR`` if the operator
        # configured one (e.g. an NVMe scratch path), otherwise under
        # ``/tmp/bioengine/model-runner-cache``. We deliberately *don't*
        # use ``$TMPDIR`` — the BioEngine framework sets it per-app to
        # ``<apps_workdir>/<app_id>/tmp``, which on shared-NFS workers
        # may live on a mount the replica's UID cannot write to. ``/tmp``
        # is always writable inside the actor process namespace.
        cache_base = (
            os.environ.get("MODEL_CACHE_DIR")
            or "/tmp/bioengine/model-runner-cache"
        )
        self.cache_dir = Path(cache_base).expanduser().resolve() / "models"
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
