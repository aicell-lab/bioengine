from __future__ import annotations

import base64
import logging
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx
import yaml

if TYPE_CHECKING:
    from hypha_rpc.rpc import ObjectProxy

from .logger import create_logger


def create_file_list_from_directory(
    directory_path: Union[str, Path],
    _artifact_id_suffix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert a local directory to a list of file dictionaries for artifact creation.

    This utility function reads all files from a directory and converts them to the
    format expected by create_application_from_files. It automatically handles text
    and binary files, and can modify the manifest ID with a suffix for testing.

    Args:
        directory_path: Path to the directory containing application files
        _artifact_id_suffix: Optional suffix to append to the manifest ID
                           (useful for creating unique test artifacts)

    Returns:
        List of file dictionaries with keys:
        - 'name': relative file path (str)
        - 'content': file content (str for text, base64 string for binary)
        - 'type': 'text' for text files, 'base64' for binary files

    Raises:
        ValueError: If directory doesn't exist or is not a directory
        RuntimeError: If file reading fails

    Example:
        files = create_file_list_from_directory(
            directory_path="/path/to/my-app",
            _artifact_id_suffix="test-123"
        )
        # Returns list of file dictionaries
        # The manifest.yaml ID will be modified if suffix is provided
    """
    directory_path = Path(directory_path)
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    files = []

    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            # Get relative path from the directory
            relative_path = file_path.relative_to(directory_path)

            # Read file content
            try:
                # Try to read as text first
                content = file_path.read_text(encoding="utf-8")
                file_type = "text"
            except UnicodeDecodeError:
                # If it fails, read as binary and encode as base64
                content = base64.b64encode(file_path.read_bytes()).decode("ascii")
                file_type = "base64"
            except Exception as e:
                raise RuntimeError(f"Failed to read file {relative_path}: {e}")

            # Update manifest files with suffix if specified
            if str(relative_path) == "manifest.yaml" and _artifact_id_suffix:
                try:
                    manifest = yaml.safe_load(content)
                    if not isinstance(manifest, dict):
                        raise ValueError(
                            f"Invalid manifest structure: expected dict, got {type(manifest)}"
                        )
                    if "id" not in manifest:
                        raise ValueError("Manifest missing required 'id' field")

                    # Append suffix to ID
                    hyphen_suffix = _artifact_id_suffix.replace("_", "-")
                    original_id = manifest["id"]
                    manifest["id"] = f"{original_id}-{hyphen_suffix}"
                    content = yaml.dump(manifest)
                except yaml.YAMLError as e:
                    raise ValueError(f"Failed to parse manifest YAML: {e}")
                except Exception as e:
                    raise ValueError(f"Failed to update manifest: {e}")

            files.append(
                {"name": str(relative_path), "content": content, "type": file_type}
            )

    return files


async def ensure_applications_collection(
    artifact_manager: ObjectProxy,
    workspace: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Ensure the 'applications' collection exists in the Hypha artifact manager.

    Creates the applications collection if it doesn't exist, providing organized
    storage for BioEngine application artifacts. Ensures public read access.

    Args:
        artifact_manager: Hypha artifact manager service instance
        workspace: Hypha workspace identifier
        logger: Optional logger instance for debugging

    Returns:
        The collection ID in format 'workspace/applications'
    """
    if logger is None:
        logger = create_logger("ArtifactUtils")

    collection_id = f"{workspace}/applications"

    try:
        collection = await artifact_manager.read(collection_id)
        logger.info(f"Collection '{collection_id}' already exists")

        # Check and fix permissions if needed
        # We want permissions to be public read: {"*": "r"}
        collection_config = collection.config or {}
        permissions = collection_config.get("permissions", {})
        if permissions.get("*") != "r":
            logger.info("Updating collection permissions for public access")
            permissions["*"] = "r"
            collection_config["permissions"] = permissions
            await artifact_manager.edit(
                artifact_id=collection_id, config=collection_config
            )

    except Exception as collection_error:
        expected_error = (
            f"KeyError: \"Artifact with ID '{collection_id}' does not exist."
        )
        if expected_error in str(collection_error).strip():
            collection_manifest = {
                "name": "Applications",
                "description": f"A collection of applications for workspace {workspace}",
            }

            try:
                collection = await artifact_manager.create(
                    type="collection",
                    alias="applications",
                    manifest=collection_manifest,
                    config={
                        "permissions": {"*": "r"}
                    },  # Set public permission on create
                )
                logger.info(f"Applications collection created with ID: {collection.id}")
            except Exception as e:
                raise RuntimeError(f"Failed to create applications collection: {e}")
        else:
            raise RuntimeError(
                f"Failed to check applications collection '{collection_id}': {collection_error}"
            )

    return collection_id


async def load_manifest_from_files(
    files: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract and parse the manifest from a list of files.

    This function only loads and parses the manifest YAML file from the files list.
    It does NOT perform validation - use validate_manifest() for that.

    Args:
        files: List of file dictionaries

    Returns:
        Parsed manifest dictionary

    Raises:
        ValueError: If manifest file is missing or YAML is invalid
    """
    # Find manifest file
    manifest_file = None
    for file in files:
        if file["name"].lower() == "manifest.yaml":
            manifest_file = file
            break

    if not manifest_file:
        raise ValueError(
            "No manifest file found in files list. Expected 'manifest.yaml'"
        )

    # Load the manifest content
    manifest_content = manifest_file["content"]
    if manifest_file["type"] == "base64":
        # Remove `data:...` prefix if present and decode base64
        if manifest_content.startswith("data:"):
            manifest_content = manifest_content.split(",")[1]
        manifest_content = base64.b64decode(manifest_content).decode("utf-8")

    try:
        manifest = yaml.safe_load(manifest_content)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in manifest file: {e}")

    return manifest


def validate_manifest(manifest: Dict[str, Any]) -> None:
    """
    Validate that a manifest contains all required fields and has correct format.

    Validates manifest structure for Ray Serve application deployments.
    This function can be called from both artifact creation (via load_manifest_from_files)
    and deployment time (via AppBuilder._load_manifest).

    Args:
        manifest: The manifest dictionary to validate

    Raises:
        ValueError: If manifest is missing required fields or has invalid format

    Example:
        manifest = yaml.safe_load(manifest_content)
        validate_manifest(manifest)  # Raises ValueError if invalid
    """
    # Validate required fields
    required_fields = [
        "name",
        "id",
        "id_emoji",
        "description",
        "type",
        "deployments",
    ]
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Manifest is missing required field: '{field}'")

    # Validate type
    if manifest["type"] != "ray-serve":
        raise ValueError(
            f"Invalid manifest type: '{manifest['type']}'. Expected 'ray-serve'."
        )

    # Validate deployments format
    deployments = manifest["deployments"]
    if not isinstance(deployments, list) or len(deployments) == 0:
        raise ValueError(
            "Invalid deployments format in manifest. "
            "Expected a non-empty list of deployment descriptions in the format 'python_file:class_name'."
        )



def validate_artifact_id(
    manifest: Dict[str, Any],
    workspace: str,
    artifact_id: Optional[str] = None,
) -> str:
    """
    Validate and normalize the artifact ID.

    Args:
        deployment_manifest: The deployment manifest
        workspace: Hypha workspace identifier
        artifact_id: Optional artifact ID for updates

    Returns:
        Full normalized artifact ID

    Raises:
        ValueError: If artifact ID format is incorrect
    """
    if artifact_id is None:
        if "id" not in manifest:
            raise ValueError(
                "No 'artifact_id' provided and no 'id' field found in manifest"
            )
        alias = manifest["id"]

        # Validate alias format
        invalid = any(
            [
                not alias.islower(),
                "_" in alias,
                "/" in alias,
                not alias.replace("-", "_").isidentifier(),
            ]
        )
        if invalid:
            raise ValueError(
                f"Invalid artifact alias: '{alias}'. Please use lowercase letters, numbers, and hyphens only."
            )

        full_artifact_id = f"{workspace}/{alias}"
    else:
        # Use provided artifact_id, ensure it's fully qualified
        if "/" not in artifact_id:
            full_artifact_id = f"{workspace}/{artifact_id}"
        else:
            full_artifact_id = artifact_id

        # Ensure artifact belongs to workspace
        if not full_artifact_id.startswith(f"{workspace}/"):
            raise ValueError(
                f"Artifact ID '{full_artifact_id}' does not belong to workspace '{workspace}'"
            )

    return full_artifact_id


async def stage_artifact(
    artifact_manager: ObjectProxy,
    workspace: str,
    manifest: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    permissions: Optional[Dict[str, Any]] = None,
) -> tuple:
    """
    Put an artifact into stage mode. Creates a new artifact if it doesn't exist.

    Versioning behaviour:
    - New artifact or version tag not yet present → branches a new version snapshot
      (``edit(version='new')``); the caller should commit with the version tag.
    - Version tag already exists → stages on top of the current head without
      branching (``edit(version=None)``); the caller should commit *without* a tag
      so the existing named version is updated in place and other versions are
      left untouched.

    Args:
        artifact_manager: Hypha artifact manager service instance
        workspace: Hypha workspace identifier
        manifest: The updated artifact manifest
        logger: Optional logger instance for debugging
        permissions: Optional permissions to set on the artifact

    Returns:
        Tuple of (artifact, version_is_new) where version_is_new is True when the
        manifest version did not previously exist and should be tagged on commit.

    Raises:
        RuntimeError: If artifact creation fails
    """
    if logger is None:
        logger = create_logger("ArtifactUtils")

    # Get full artifact ID
    artifact_id = validate_artifact_id(manifest, workspace)

    # Define collection_id early so it's available for artifact creation
    collection_id = f"{workspace}/applications"

    # Build view_config from frontend_entry if present
    view_config = None
    frontend_entry = manifest.get("frontend_entry")
    if frontend_entry:
        entry_path = PurePosixPath(frontend_entry)
        parent = str(entry_path.parent)
        view_config = {
            "branch": "main",
            "root_directory": "" if parent == "." else parent,
            "headers": {},
            "index": entry_path.name,
        }

    # Check if artifact exists and handle collection placement
    artifact = None
    version_is_new = True  # new artifact always gets a version tag on commit
    manifest_version = manifest.get("version")

    try:
        # Check if artifact already exists
        existing_artifact = await artifact_manager.read(artifact_id)

        # Check if artifact is in the correct collection
        current_parent_id = getattr(existing_artifact, "parent_id", None)
        if current_parent_id != collection_id:
            logger.info(
                f"Artifact '{artifact_id}' exists but is in wrong collection "
                f"(current: {current_parent_id}, expected: {collection_id}). "
                "Deleting and recreating..."
            )
            await artifact_manager.delete(artifact_id=artifact_id)
            existing_artifact = None

        if existing_artifact:
            # Edit existing artifact — preserve existing config, then apply overrides
            artifact_config = dict(existing_artifact.config or {})
            if permissions:
                artifact_config["permissions"] = permissions
            if view_config is not None:
                artifact_config["view_config"] = view_config

            # Decide how to stage based on whether the manifest version is new:
            #
            # - New version tag: edit(version='new') branches a fresh snapshot so
            #   all previous versions stay intact; caller commits with the tag.
            # - Same version as the LATEST: edit(version=None) stages on the
            #   current head; caller commits without a tag — updates in place.
            # - Same version as an OLDER (non-latest) tag: not supported by Hypha
            #   (edit always targets the latest head), so raise a clear error.
            existing_versions = [
                v["version"] for v in (existing_artifact.versions or [])
            ]
            version_is_new = bool(
                manifest_version and manifest_version not in existing_versions
            )

            if not version_is_new and existing_versions:
                latest_version = existing_versions[-1]
                if manifest_version and manifest_version != latest_version:
                    raise ValueError(
                        f"Cannot re-save artifact '{artifact_id}' at version "
                        f"'{manifest_version}': a newer version '{latest_version}' "
                        f"already exists. Bump the version in manifest.yaml to save "
                        f"changes (existing versions: {existing_versions})."
                    )

            edit_kwargs = {
                "artifact_id": artifact_id,
                "manifest": manifest,
                "type": "application",
                "stage": True,
                "version": "new" if version_is_new else None,
            }
            if artifact_config:
                edit_kwargs["config"] = artifact_config

            artifact = await artifact_manager.edit(**edit_kwargs)
            if version_is_new:
                logger.info(
                    f"Editing existing artifact '{artifact_id}' "
                    f"(new version: {manifest_version})"
                )
            else:
                logger.info(
                    f"Editing existing artifact '{artifact_id}' "
                    f"(updating existing version: {manifest_version or 'latest'})"
                )

    except Exception as e:
        expected_error = f"KeyError: \"Artifact with ID '{artifact_id}' does not exist."
        if expected_error not in str(e).strip():
            raise e

    # Create new artifact if it doesn't exist or was deleted
    if artifact is None:
        try:
            initial_config = {}
            if permissions:
                initial_config["permissions"] = permissions
            if view_config is not None:
                initial_config["view_config"] = view_config

            create_kwargs = {
                "parent_id": collection_id,
                "type": "application",
                "alias": artifact_id.split("/")[1],
                "manifest": manifest,
                "stage": True,
            }
            if initial_config:
                create_kwargs["config"] = initial_config

            artifact = await artifact_manager.create(**create_kwargs)
            logger.info(f"Created new artifact '{artifact.id}'")
        except Exception as e:
            raise RuntimeError(f"Failed to create artifact '{artifact_id}': {e}")

    return artifact, version_is_new


async def upload_file_to_artifact(
    artifact_manager: ObjectProxy,
    artifact_id: str,
    file_name: str,
    file_content: Union[str, bytes],
    file_type: str = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Upload a file to an artifact.

    Args:
        artifact_manager: Hypha artifact manager service instance
        artifact_id: The artifact ID
        file_name: The file name/path
        file_content: The file content (str for text, str for base64, or raw bytes)
        file_type: The file type ('text', 'base64', or None for raw binary)
        logger: Optional logger instance for debugging

    Raises:
        ValueError: If file type is unsupported
        RuntimeError: If file upload fails
    """
    if logger is None:
        logger = create_logger("ArtifactUtils")

    logger.info(f"Uploading file '{file_name}' to artifact...")

    # Get upload URL
    try:
        upload_url = await artifact_manager.put_file(artifact_id, file_path=file_name)
    except Exception as e:
        raise RuntimeError(f"Failed to get upload URL for '{file_name}': {e}")

    # Prepare content for upload
    if isinstance(file_content, bytes):
        # Direct binary content
        upload_data = file_content
        content_type = "binary"
    elif file_type == "text" or (file_type is None and isinstance(file_content, str)):
        # Text content
        upload_data = file_content
        content_type = "text"
    elif file_type == "base64":
        # Decode base64 content for binary files
        if isinstance(file_content, str):
            if file_content.startswith("data:"):
                file_content = file_content.split(",")[1]
            upload_data = base64.b64decode(file_content)
            content_type = "binary"
        else:
            raise ValueError("Base64 content must be a string")
    else:
        raise ValueError(
            f"Unsupported file type '{file_type}'. Expected 'text', 'base64', or None for raw binary"
        )

    # Upload the file with timeout
    upload_timeout = httpx.Timeout(30.0)
    try:
        async with httpx.AsyncClient(timeout=upload_timeout) as client:
            if content_type == "text":
                response = await client.put(upload_url, data=upload_data)
            else:
                response = await client.put(upload_url, content=upload_data)
            response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to upload file '{file_name}': {e}")


async def remove_file_from_artifact(
    artifact_manager: ObjectProxy,
    artifact_id: str,
    file_name: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Remove a file from an artifact.

    Args:
        artifact_manager: Hypha artifact manager service instance
        artifact_id: The artifact ID
        file_name: The file name/path to remove
        logger: Optional logger instance for debugging

    Raises:
        RuntimeError: If file removal fails
    """
    if logger is None:
        logger = create_logger("ArtifactUtils")

    logger.info(f"Removing file '{file_name}' from artifact...")

    try:
        await artifact_manager.remove_file(artifact_id, file_path=file_name)
    except Exception as e:
        raise RuntimeError(f"Failed to remove file '{file_name}': {e}")


async def commit_artifact(
    artifact_manager: ObjectProxy,
    artifact_id: str,
    version: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Commit an artifact.

    Args:
        artifact_manager: Hypha artifact manager service instance
        artifact_id: The artifact ID
        version: Optional version tag (e.g. "1.0.0"). If None, Hypha assigns "v0" for the first commit, then increments (v1, v2, …) for subsequent unversioned commits.
        logger: Optional logger instance for debugging

    Raises:
        RuntimeError: If artifact commit fails
    """
    if logger is None:
        logger = create_logger("ArtifactUtils")

    try:
        kwargs = {"artifact_id": artifact_id}
        if version is not None:
            kwargs["version"] = version
        await artifact_manager.commit(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to commit artifact '{artifact_id}': {e}")

    logger.info(f"Successfully committed artifact '{artifact_id}' (version={version or 'latest'})")


def get_static_site_url(artifact_id: str, server_url: str) -> str:
    """
    Return the static site URL for a BioEngine artifact.

    The URL follows the Hypha convention::

        {server_url}/{workspace}/view/{alias}/

    Args:
        artifact_id: Fully-qualified artifact ID (``workspace/alias``).
        server_url: Public base URL of the Hypha server.

    Returns:
        The public static site URL.
    """
    workspace, alias = artifact_id.split("/", 1)
    return f"{server_url}/{workspace}/view/{alias}/"



async def create_application_from_files(
    artifact_manager: ObjectProxy,
    files: List[Dict[str, Any]],
    workspace: str,
    user_id: str,
    logger: Optional[logging.Logger] = None,
    permissions: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create or update a Hypha artifact from a list of files.

    This utility function handles the common pattern of creating artifacts in Hypha
    from a list of files, including manifest validation, collection management,
    file uploads, and artifact commitment.

    When updating an existing artifact, this function will:
    1. Validate the manifest (must have type='ray-serve')
    2. Add 'created_by' field to the manifest with the provided user_id
    3. Upload all files from the provided files list
    4. Only if all uploads succeed, remove any files that existed in the artifact
       before but are not in the new files list

    This ensures the artifact contains exactly the files specified in the files list,
    with no orphaned files from previous versions. The removal of old files only
    happens after all new files are successfully uploaded to prevent data loss.

    Args:
        artifact_manager: Hypha artifact manager service instance
        files: List of file dictionaries with keys:
               - 'name': relative file path (str)
               - 'content': file content (str or bytes)
               - 'type': 'text' for text files, 'base64' for binary files
        workspace: Hypha workspace identifier
        user_id: User ID to set as the 'created_by' field in the manifest
        logger: Optional logger instance for debugging
        permissions: Optional permissions to set on the artifact

    Returns:
        The artifact ID of the created or updated artifact

    Raises:
        ValueError: If manifest is missing, invalid, type is not 'ray-serve',
                   or artifact ID format is incorrect
        RuntimeError: If artifact creation, file upload, or commit fails

    Example:
        files = [
            {'name': 'manifest.yaml', 'content': manifest_yaml, 'type': 'text'},
            {'name': 'app.py', 'content': app_code, 'type': 'text'},
            {'name': 'data.bin', 'content': base64_data, 'type': 'base64'}
        ]
        artifact_id = await create_application_from_files(
            artifact_manager=am,
            files=files,
            workspace="my-workspace",
            user_id="user@example.com",
        )
    """
    if logger is None:
        logger = create_logger("ArtifactUtils")

    # Ensure applications collection exists
    await ensure_applications_collection(
        artifact_manager=artifact_manager,
        workspace=workspace,
        logger=logger,
    )

    # Extract and validate manifest
    application_manifest = await load_manifest_from_files(files)
    validate_manifest(application_manifest)

    # Add created_by field to the manifest
    application_manifest["created_by"] = user_id

    # Edit or create artifact, put in stage mode.
    # version_is_new tells us whether to tag the commit with the manifest version.
    artifact, version_is_new = await stage_artifact(
        artifact_manager=artifact_manager,
        workspace=workspace,
        manifest=application_manifest,
        logger=logger,
        permissions=permissions,
    )

    # Get existing files if artifact already exists (to remove old files later)
    existing_files = set()
    try:
        files_list = await artifact_manager.list_files(artifact.id)
        existing_files = {file.name for file in files_list}
        logger.info(
            f"Found {len(existing_files)} existing files in artifact '{artifact.id}'"
        )
    except Exception as e:
        # Artifact might be new, so no existing files
        logger.debug(f"Could not list existing files (artifact may be new): {e}")

    # Upload all files - track success
    new_files = set()
    upload_errors = []

    for file in files:
        file_name = file["name"]
        file_content = file["content"]
        file_type = file["type"]

        try:
            await upload_file_to_artifact(
                artifact_manager=artifact_manager,
                artifact_id=artifact.id,
                file_name=file_name,
                file_content=file_content,
                file_type=file_type,
                logger=logger,
            )
            new_files.add(file_name)
        except Exception as e:
            upload_errors.append((file_name, str(e)))
            logger.error(f"Failed to upload file '{file_name}': {e}")

    # If any uploads failed, raise an error before removing old files
    if upload_errors:
        error_details = "; ".join([f"{name}: {err}" for name, err in upload_errors])
        raise RuntimeError(
            f"Failed to upload {len(upload_errors)} file(s): {error_details}"
        )

    # Only remove old files if all new files were successfully uploaded
    files_to_remove = existing_files - new_files
    if files_to_remove:
        logger.info(
            f"Removing {len(files_to_remove)} old files that are no longer present"
        )
        for file_name in files_to_remove:
            try:
                await remove_file_from_artifact(
                    artifact_manager=artifact_manager,
                    artifact_id=artifact.id,
                    file_name=file_name,
                    logger=logger,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to remove old file '{file_name}': {e}. Continuing anyway..."
                )

    # Commit the artifact.
    # Only tag with the manifest version when it is genuinely new; if the version
    # tag already existed we commit without a tag (updates the artifact in place
    # without creating a duplicate version entry).
    commit_version = application_manifest.get("version") if version_is_new else None
    await commit_artifact(
        artifact_manager=artifact_manager,
        artifact_id=artifact.id,
        version=commit_version,
        logger=logger,
    )

    return artifact.id
