"""
End-to-end tests for BioEngine Worker AppsManager component.

This module tests the AppsManager functionality through the Hypha service API,
including application deployment, undeployment, startup applications, WebSocket services,
peer connections, artifact management, and cleanup operations.
"""

import asyncio
import base64
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import yaml
from hypha_rpc import get_rtc_service
from hypha_rpc.rpc import ObjectProxy, RemoteService

from bioengine.utils import create_file_list_from_directory


def top_level_name(file: Dict) -> str:
    """Return the first path segment of an uploaded file's relative name."""
    return file["name"].split("/")[0]


async def resolve_service(lookup, timeout: int = 60):
    """Retry a Hypha service lookup until the registration shows up.

    `get_app_status` derives `service_ids` from the worker's client id and only
    gates on a live ProxyDeployment replica, so an id is reported before that
    replica has finished registering it with Hypha.
    """
    start_time = time.time()
    while True:
        try:
            return await lookup()
        except Exception:
            if time.time() - start_time >= timeout:
                raise
            await asyncio.sleep(2)


def bump_manifest_version(files: List[Dict], new_version: str) -> List[Dict]:
    """Return a copy of ``files`` with the manifest's ``version`` replaced.

    ``upload_app`` rejects a version that is not strictly greater than every
    existing one, so re-uploading an artifact requires a fresh version.
    """
    bumped = []
    for file in files:
        if file["name"] == "manifest.yaml":
            manifest = yaml.safe_load(file["content"])
            manifest["version"] = new_version
            file = {**file, "content": yaml.safe_dump(manifest)}
        bumped.append(file)
    return bumped


@pytest.mark.end_to_end
@pytest.mark.asyncio
async def test_create_and_delete_artifacts(
    bioengine_worker_service: ObjectProxy,
    bioengine_apps_dir: Path,
    test_id: str,
    hypha_workspace: str,
    hypha_user_id: str,
):
    """
    Test creating and deleting artifacts from both demo-app and composition-demo applications.

    Steps:
    - Create artifacts from both applications
    - Wait for artifact creation completion
    - Verify artifacts appear in available artifacts list
    - Check all files in artifact directories
    - Delete both artifacts using delete_artifact API
    - Verify artifacts are removed from available list
    - Confirm storage cleanup and accessibility removal
    """
    # Define paths to test applications
    demo_app_path = bioengine_apps_dir / "demo-app"
    composition_app_path = bioengine_apps_dir / "composition-demo"

    # Verify the test directories exist
    assert demo_app_path.exists(), f"Demo app directory not found: {demo_app_path}"
    assert (
        composition_app_path.exists()
    ), f"Composition app directory not found: {composition_app_path}"

    # Create file lists from both directories
    demo_app_files = create_file_list_from_directory(
        directory_path=demo_app_path, _artifact_id_suffix=test_id
    )

    composition_app_files = create_file_list_from_directory(
        directory_path=composition_app_path, _artifact_id_suffix=test_id
    )

    # Verify we have files to upload
    assert len(demo_app_files) > 0, "Demo app directory should contain files"
    assert (
        len(composition_app_files) > 0
    ), "Composition app directory should contain files"

    # Verify artifact has manifest files and extract artifact IDs
    demo_manifest_file = next(
        (f for f in demo_app_files if f["name"] == "manifest.yaml"), None
    )
    composition_manifest_file = next(
        (f for f in composition_app_files if f["name"] == "manifest.yaml"), None
    )

    # Ensure manifest files are present
    assert demo_manifest_file, "Demo app manifest not found"
    assert composition_manifest_file, "Composition app manifest not found"

    # Ensure manifest files are valid YAML and extract artifact aliases
    try:
        demo_manifest = yaml.safe_load(demo_manifest_file["content"])
        demo_artifact_alias = demo_manifest["id"]
        demo_artifact_id = f"{hypha_workspace}/{demo_artifact_alias}"
    except yaml.YAMLError as e:
        pytest.fail(f"Invalid YAML in demo app manifest: {e}")
    except KeyError:
        pytest.fail("Demo app manifest missing 'id' field")

    try:
        composition_manifest = yaml.safe_load(composition_manifest_file["content"])
        composition_artifact_alias = composition_manifest["id"]
        composition_artifact_id = f"{hypha_workspace}/{composition_artifact_alias}"
    except yaml.YAMLError as e:
        pytest.fail(f"Invalid YAML in composition app manifest: {e}")
    except KeyError:
        pytest.fail("Composition app manifest missing 'id' field")

    # Verify the artifact IDs do not already exist
    existing_artifacts = await bioengine_worker_service.list_apps()
    assert (
        demo_artifact_id not in existing_artifacts
    ), f"Demo artifact ID {demo_artifact_id} already exists"
    assert (
        composition_artifact_id not in existing_artifacts
    ), f"Composition artifact ID {composition_artifact_id} already exists"

    test_completed = False
    try:
        # Create demo-app artifact
        created_demo_artifact_id = await bioengine_worker_service.upload_app(
            files=demo_app_files
        )

        # Verify artifact creation returned the correct ID
        assert created_demo_artifact_id, "Demo artifact creation should return an ID"
        assert isinstance(
            created_demo_artifact_id, str
        ), "Artifact ID should be a string"
        assert (
            created_demo_artifact_id == demo_artifact_id
        ), "Created demo artifact ID should match manifest ID and workspace"

        # Create composition-demo artifact
        created_composition_artifact_id = (
            await bioengine_worker_service.upload_app(
                files=composition_app_files
            )
        )

        # Verify artifact creation returned the correct ID
        assert (
            created_composition_artifact_id
        ), "Composition artifact creation should return an ID"
        assert isinstance(
            created_composition_artifact_id, str
        ), "Artifact ID should be a string"
        assert (
            created_composition_artifact_id == composition_artifact_id
        ), "Created composition artifact ID should match manifest ID and workspace"

        # Verify artifacts exist
        available_artifacts = await bioengine_worker_service.list_apps()
        assert (
            demo_artifact_id in available_artifacts
        ), "Demo artifact should be listed in available artifacts"
        assert (
            composition_artifact_id in available_artifacts
        ), "Composition artifact should be listed in available artifacts"

        # Update both artifacts. `upload_app` only accepts a version strictly
        # greater than every existing one, so the update has to bump it.
        updated_version = "9999.0.1"
        demo_manifest["version"] = updated_version
        composition_manifest["version"] = updated_version

        updated_demo_artifact_id = await bioengine_worker_service.upload_app(
            files=bump_manifest_version(demo_app_files, updated_version)
        )
        assert (
            updated_demo_artifact_id == demo_artifact_id
        ), "Updated demo artifact ID should match original"

        updated_composition_artifact_id = (
            await bioengine_worker_service.upload_app(
                files=bump_manifest_version(composition_app_files, updated_version)
            )
        )
        assert (
            updated_composition_artifact_id == composition_artifact_id
        ), "Updated composition artifact ID should match original"

        # Verify updated artifacts still exist
        available_artifacts = await bioengine_worker_service.list_apps()
        assert (
            demo_artifact_id in available_artifacts
        ), "Demo artifact should be listed in available artifacts"
        assert (
            composition_artifact_id in available_artifacts
        ), "Composition artifact should be listed in available artifacts"

        # Verify all files in demo-app artifact. `list_apps` reports a
        # non-recursive listing, so a nested upload shows up as its top-level
        # directory entry rather than its full relative path.
        assert all(
            top_level_name(f) in available_artifacts[demo_artifact_id]["files"]
            for f in demo_app_files
        ), "All demo app files should be listed in artifact files"
        received_manifest = available_artifacts[demo_artifact_id]["manifest"].toDict()
        created_by = received_manifest["manifest"].pop("created_by")
        assert (
            received_manifest["manifest"] == demo_manifest
        ), "Demo app manifest should match expected manifest"
        assert (
            created_by == hypha_user_id
        ), "Created by user ID should match the test user ID"
        assert (
            received_manifest["parent_id"] == f"{hypha_workspace}/applications"
        ), "Demo app manifest should be in applications collection"

        # Verify all files in composition-demo artifact
        assert all(
            top_level_name(f) in available_artifacts[composition_artifact_id]["files"]
            for f in composition_app_files
        ), "All composition app files should be listed in artifact files"
        received_manifest = available_artifacts[composition_artifact_id][
            "manifest"
        ].toDict()
        created_by = received_manifest["manifest"].pop("created_by")
        assert (
            received_manifest["manifest"] == composition_manifest
        ), "Composition app manifest should match expected manifest"
        assert (
            created_by == hypha_user_id
        ), "Created by user ID should match the test user ID"
        assert (
            received_manifest["parent_id"] == f"{hypha_workspace}/applications"
        ), "Composition app manifest should be in applications collection"

        # Delete both artifacts
        await bioengine_worker_service.delete_app(artifact_id=demo_artifact_id)

        await bioengine_worker_service.delete_app(
            artifact_id=composition_artifact_id
        )

        # Verify artifacts no longer exist
        available_artifacts = await bioengine_worker_service.list_apps()
        assert (
            demo_artifact_id not in available_artifacts
        ), "Demo artifact should be removed from available artifacts"
        assert (
            composition_artifact_id not in available_artifacts
        ), "Composition artifact should be removed from available artifacts"

        test_completed = True

    finally:
        # Cleanup: Ensure artifacts are deleted even if test fails
        cleanup_errors = []
        if demo_artifact_id and not test_completed:
            try:
                await bioengine_worker_service.delete_app(
                    artifact_id=demo_artifact_id
                )
            except Exception as e:
                cleanup_errors.append(
                    f"Failed to cleanup demo artifact {demo_artifact_id}: {e}"
                )

        if composition_artifact_id and not test_completed:
            try:
                await bioengine_worker_service.delete_app(
                    artifact_id=composition_artifact_id
                )
            except Exception as e:
                cleanup_errors.append(
                    f"Failed to cleanup composition artifact {composition_artifact_id}: {e}"
                )

        # Log cleanup errors but don't fail the test if cleanup fails
        if cleanup_errors:
            for error in cleanup_errors:
                warnings.warn(error)


@pytest.mark.end_to_end
@pytest.mark.asyncio
async def test_startup_application(
    bioengine_worker_service: ObjectProxy,
    startup_applications: List[Dict],
    application_check_timeout: int,
):
    """
    Test that all startup applications are properly deployed.

    This test validates the AppsManager status reporting and startup application deployment:
    1. Retrieves application status using get_app_status
    2. Validates application status structure and content
    3. Checks that startup applications are properly deployed and healthy
    4. Verifies application metadata, resource allocation, and service registration
    5. Ensures deployment configuration matches expected startup specifications

    Expected Application Status Structure:
    - Dict[application_id, application_info] where each application_info contains:
      - display_name, description, artifact_id, version
      - start_time, status (RUNNING/HEALTHY/etc), message
      - deployments: Dict of deployment status and replica states
      - resource allocation: application_resources, application_kwargs, gpu_enabled
      - service_ids: WebSocket and WebRTC service endpoints
      - access control: authorized_users, last_updated_by
      - available_methods: List of exposed application methods
    """
    # Ensure at least one startup application is configured
    assert (
        startup_applications and len(startup_applications) > 0
    ), "No startup applications configured for this test. Please define at least one in the fixture."

    # `worker.start()` returns once the startup deployment is submitted, while
    # its replicas are still building their pip runtime_env, so wait for them.
    expected_app_count = len(startup_applications)
    start_time = time.time()
    while time.time() - start_time < application_check_timeout:
        apps_status = await bioengine_worker_service.get_app_status()
        running = [
            app for app in apps_status.values() if app["status"] == "RUNNING"
        ]
        if len(running) >= expected_app_count:
            break
        await asyncio.sleep(2)

    # Get application status (returns all deployed applications when application_ids is None)
    apps_status = await bioengine_worker_service.get_app_status()

    # Validate apps_status is properly structured
    assert isinstance(apps_status, dict), "Application status should be a dictionary"

    # Assert that applications are deployed based on startup configuration
    assert (
        len(apps_status) > 0
    ), f"Expected {expected_app_count} startup applications to be deployed, but found {len(apps_status)} applications"

    # Validate each deployed application's status structure
    for application_id, app_info in apps_status.items():
        assert isinstance(
            application_id, str
        ), f"Application ID '{application_id}' should be a string"
        assert isinstance(
            app_info, dict
        ), f"Application info for '{application_id}' should be a dictionary"

        # Required application metadata fields
        required_fields = [
            "display_name",
            "description",
            "artifact_id",
            "version",
            "status",
            "message",
            "deployments",
            "application_kwargs",
            "gpu_enabled",
            "application_resources",
            "authorized_users",
            "available_methods",
            "service_ids",
            "last_updated_by",
        ]

        for field in required_fields:
            assert (
                field in app_info
            ), f"Application '{application_id}' should contain '{field}' field"

        # Validate field types and values
        assert isinstance(
            app_info["display_name"], str
        ), f"display_name should be a string for '{application_id}'"
        assert isinstance(
            app_info["description"], str
        ), f"description should be a string for '{application_id}'"
        assert isinstance(
            app_info["artifact_id"], str
        ), f"artifact_id should be a string for '{application_id}'"
        assert isinstance(
            app_info["version"], str
        ), f"version should be a string for '{application_id}'"
        assert isinstance(
            app_info["status"], str
        ), f"status should be a string for '{application_id}'"
        assert isinstance(
            app_info["message"], str
        ), f"message should be a string for '{application_id}'"
        assert isinstance(
            app_info["deployments"], dict
        ), f"deployments should be a dictionary for '{application_id}'"
        assert isinstance(
            app_info["application_kwargs"], dict
        ), f"application_kwargs should be a dictionary for '{application_id}'"
        assert isinstance(
            app_info["gpu_enabled"], bool
        ), f"gpu_enabled should be a boolean for '{application_id}'"
        assert isinstance(
            app_info["application_resources"], dict
        ), f"application_resources should be a dictionary for '{application_id}'"
        assert isinstance(
            app_info["authorized_users"], dict
        ), f"authorized_users should be a dictionary for '{application_id}'"
        assert isinstance(
            app_info["available_methods"], list
        ), f"available_methods should be a list for '{application_id}'"
        assert isinstance(
            app_info["service_ids"], dict
        ), f"service_ids should be a dictionary for '{application_id}'"

        # Validate application status is in expected states
        valid_statuses = [
            "NOT_STARTED",
            "DEPLOYING",
            "DEPLOY_FAILED",
            "RUNNING",
            "UNHEALTHY",
            "DELETING",
        ]
        assert (
            app_info["status"] in valid_statuses
        ), f"Application status '{app_info['status']}' should be one of {valid_statuses}"

        # Validate start_time if present (can be None for failed deployments)
        if "start_time" in app_info and app_info["start_time"] is not None:
            assert isinstance(
                app_info["start_time"], (int, float)
            ), f"start_time should be a number for '{application_id}'"
            assert (
                app_info["start_time"] > 0
            ), f"start_time should be positive for '{application_id}'"

        # For healthy applications, check deployment details
        if app_info["status"] == "RUNNING":
            # Should have deployments
            assert (
                len(app_info["deployments"]) > 0
            ), f"Running application '{application_id}' should have active deployments"

            # Validate each deployment
            for deployment_name, deployment_info in app_info["deployments"].items():
                assert isinstance(
                    deployment_name, str
                ), f"Deployment name should be a string in '{application_id}'"
                assert isinstance(
                    deployment_info, dict
                ), f"Deployment info should be a dictionary in '{application_id}'"

                # Required deployment fields
                deployment_fields = ["status", "message", "replica_states"]
                for field in deployment_fields:
                    assert (
                        field in deployment_info
                    ), f"Deployment '{deployment_name}' should contain '{field}' field"

                # Validate deployment status
                valid_deployment_statuses = [
                    "UPDATING",
                    "HEALTHY",
                    "UNHEALTHY",
                    "UPSCALING",
                    "DOWNSCALING",
                ]
                assert (
                    deployment_info["status"] in valid_deployment_statuses
                ), f"Deployment status '{deployment_info['status']}' should be one of {valid_deployment_statuses}"

                # Validate replica states
                assert isinstance(
                    deployment_info["replica_states"], dict
                ), f"replica_states should be a dictionary for deployment '{deployment_name}'"

            # Should have service IDs for running applications
            service_ids = app_info["service_ids"]
            assert isinstance(
                service_ids, dict
            ), f"service_ids should be a dictionary for '{application_id}'"
            for key in ("websocket_service_id", "webrtc_service_id"):
                assert (
                    service_ids.get(key)
                ), f"Running application '{application_id}' should have a {key}"

        # Validate resource allocation structure
        if app_info["application_resources"]:
            assert isinstance(
                app_info["application_resources"], dict
            ), f"application_resources should be a dictionary for '{application_id}'"
            # Common resource fields (may vary based on deployment)
            for resource_key, resource_value in app_info[
                "application_resources"
            ].items():
                assert isinstance(
                    resource_key, str
                ), f"Resource key should be a string in '{application_id}'"
                assert isinstance(
                    resource_value, (int, float, str)
                ), f"Resource value should be numeric or string in '{application_id}'"

        # Validate deployment kwargs structure
        if app_info["application_kwargs"]:
            assert isinstance(
                app_info["application_kwargs"], dict
            ), f"application_kwargs should be a dictionary for '{application_id}'"

        # Validate gpu_enabled field
        assert isinstance(
            app_info["gpu_enabled"], bool
        ), f"gpu_enabled should be a boolean for '{application_id}'"

        # Validate authorized users: a per-method map of method name (or '*')
        # to the list of user IDs/emails allowed to call it.
        assert (
            len(app_info["authorized_users"]) > 0
        ), f"Application '{application_id}' should have authorized users"
        for method, users in app_info["authorized_users"].items():
            assert isinstance(
                method, str
            ), f"Authorized method should be a string in '{application_id}'"
            assert isinstance(
                users, list
            ), f"Authorized users for '{method}' should be a list in '{application_id}'"
            for user in users:
                assert isinstance(
                    user, str
                ), f"Authorized user should be a string in '{application_id}'"

        # Validate available methods
        for method in app_info["available_methods"]:
            assert isinstance(
                method, str
            ), f"Available method should be a string in '{application_id}'"

    # Log summary of application status for debugging
    app_count = len(apps_status)
    running_count = sum(1 for app in apps_status.values() if app["status"] == "RUNNING")
    healthy_count = sum(
        1
        for app in apps_status.values()
        if app["status"] == "RUNNING"
        and any(dep["status"] == "HEALTHY" for dep in app["deployments"].values())
    )

    print(
        f"AppsManager Status Summary: {app_count} total applications, "
        f"{running_count} running, {healthy_count} healthy"
    )
    print(f"Expected startup applications: {expected_app_count}")

    # Validate startup applications deployment - ALL must be running and healthy

    # Assert that exactly the expected number of applications are deployed
    assert (
        app_count == expected_app_count
    ), f"Expected exactly {expected_app_count} startup applications, but found {app_count} total applications"

    # ALL startup applications must be running
    assert (
        running_count == expected_app_count
    ), f"Expected all {expected_app_count} startup applications to be running, but found only {running_count} running applications"

    # ALL startup applications must be healthy
    assert (
        healthy_count == expected_app_count
    ), f"Expected all {expected_app_count} startup applications to be healthy, but found only {healthy_count} healthy applications"

    # Validate that each configured startup application exists and is in perfect state
    startup_apps_validated = 0
    for startup_app in startup_applications:
        artifact_id = startup_app.get("artifact_id")
        if artifact_id:
            # Check if any deployed app has this artifact_id
            matching_apps = [
                app
                for app in apps_status.values()
                if app.get("artifact_id").endswith(artifact_id)
            ]
            assert (
                len(matching_apps) == 1
            ), f"Expected exactly one deployment for startup application with artifact_id '{artifact_id}', but found {len(matching_apps)}"

            app_info = matching_apps[0]

            # Check that the app is running
            assert (
                app_info.get("status") == "RUNNING"
            ), f"Startup application with artifact_id '{artifact_id}' is not running (status: {app_info.get('status')})"

            # Check that all deployments are healthy
            assert (
                len(app_info["deployments"]) > 0
            ), f"Startup application with artifact_id '{artifact_id}' has no deployments"

            for deployment_name, deployment_info in app_info["deployments"].items():
                assert (
                    deployment_info["status"] == "HEALTHY"
                ), f"Deployment '{deployment_name}' of startup application '{artifact_id}' is not healthy (status: {deployment_info['status']})"

            # Check that service IDs are properly configured
            service_ids = app_info["service_ids"]
            assert (
                service_ids.get("websocket_service_id") is not None
            ), f"Startup application with artifact_id '{artifact_id}' has no websocket_service_id configured"

            # Check that the application has available methods
            assert (
                len(app_info["available_methods"]) > 0
            ), f"Startup application with artifact_id '{artifact_id}' has no available methods"

            # Check that start_time is set (indicating successful deployment)
            assert (
                app_info.get("start_time") is not None
            ), f"Startup application with artifact_id '{artifact_id}' has no start_time set"
            assert isinstance(
                app_info["start_time"], (int, float)
            ), f"Startup application with artifact_id '{artifact_id}' has invalid start_time type"
            assert (
                app_info["start_time"] > 0
            ), f"Startup application with artifact_id '{artifact_id}' has invalid start_time value"

            startup_apps_validated += 1

    # Ensure we validated all expected startup applications
    assert (
        startup_apps_validated == expected_app_count
    ), f"Expected to validate {expected_app_count} startup applications, but only validated {startup_apps_validated}"


@pytest.mark.end_to_end
@pytest.mark.asyncio
async def test_deploy_app_locally(
    monkeypatch: pytest.MonkeyPatch,
    bioengine_apps_dir: Path,
    hypha_workspace: str,
    test_id: str,
    bioengine_worker_service: ObjectProxy,
):
    """
    Test deploying the 'demo-app' and 'composition-demo' applications from local artifact path.
    """
    # Local artifact resolution is by directory name under this root
    monkeypatch.setenv("BIOENGINE_LOCAL_ARTIFACT_PATH", str(bioengine_apps_dir))
    assert os.getenv("BIOENGINE_LOCAL_ARTIFACT_PATH") == str(bioengine_apps_dir)

    # Note: the demo app is already deployed by startup_applications, but the deployment below will use a different application ID

    demo_artifact_id = f"{hypha_workspace}/demo-app"
    demo_app_config = {
        "artifact_id": demo_artifact_id,
        "application_kwargs": {"DemoDeployment": {"test_param": "local_value"}},
        "disable_gpu": True,
    }  # Test random application ID generation and deployment kwargs

    composition_artifact_id = f"{hypha_workspace}/composition-demo"
    hyphen_test_id = test_id.replace("_", "-")
    composition_app_config = {
        "artifact_id": composition_artifact_id,
        "application_id": f"composition-demo-{hyphen_test_id}",
        "disable_gpu": True,
    }  # Provide custom application id

    app_configs = [demo_app_config, composition_app_config]
    deployed_app_ids = []

    try:
        for app_config in app_configs:
            # Deploy the application
            application_id = await bioengine_worker_service.deploy_app(
                **app_config
            )
            deployed_app_ids.append(application_id)
            print(f"Deployed application: {application_id}")

        # Wait for both applications to finish deploying
        # Generous: each replica builds a pip runtime_env before it turns HEALTHY.
        timeout = 300
        poll_interval = 2  # Check every 2 seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            bioengine_apps = await bioengine_worker_service.get_app_status(
                application_ids=deployed_app_ids
            )

            # Check if all apps are no longer in DEPLOYING state
            all_deployed = True
            for app_id in deployed_app_ids:
                if app_id in bioengine_apps:
                    app_status = bioengine_apps[app_id].get("status", "")
                    if app_status in ["NOT_STARTED", "DEPLOYING"]:
                        all_deployed = False
                        break
                else:
                    all_deployed = False
                    break

            if all_deployed:
                break

            await asyncio.sleep(poll_interval)
        else:
            raise TimeoutError(
                f"Applications did not finish deploying within {timeout} seconds"
            )

        # Check that both apps are healthy and have running replicas
        bioengine_apps = await bioengine_worker_service.get_app_status(
            application_ids=deployed_app_ids
        )

        for app_id in deployed_app_ids:
            assert app_id in bioengine_apps, f"Application {app_id} not found in status"

            app_info = bioengine_apps[app_id]
            assert (
                app_info["status"] == "RUNNING"
            ), f"Application {app_id} is not running: {app_info['status']}"

            # Check deployments are healthy
            assert (
                len(app_info["deployments"]) > 0
            ), f"Application {app_id} should have active deployments"

            for deployment_name, deployment_info in app_info["deployments"].items():
                assert (
                    deployment_info["status"] == "HEALTHY"
                ), f"Deployment {deployment_name} of app {app_id} is not healthy: {deployment_info['status']}"

                # Check replica states
                replica_states = deployment_info["replica_states"]
                assert (
                    len(replica_states) > 0
                ), f"Deployment {deployment_name} of app {app_id} should have replicas"

                # Ensure at least one replica is running
                running_replicas = replica_states.get("RUNNING", 0)
                assert (
                    running_replicas > 0
                ), f"Deployment {deployment_name} of app {app_id} should have at least one running replica"

            print(f"Application {app_id} is healthy with running replicas")

    finally:
        # Cleanup: Ensure applications are undeployed (even if test fails)
        for app_id in deployed_app_ids:
            try:
                await bioengine_worker_service.stop_app(
                    application_id=app_id
                )
                print(f"Undeployed application: {app_id}")
            except Exception as e:
                warnings.warn(f"Failed to undeploy application {app_id}: {e}")


@pytest.mark.end_to_end
@pytest.mark.asyncio
async def test_deploy_app_from_artifact(
    monkeypatch: pytest.MonkeyPatch,
    bioengine_apps_dir: Path,
    test_id: str,
    hypha_workspace: str,
    bioengine_worker_service: ObjectProxy,
):
    """
    Test deploying the 'demo-app' and 'composition-demo' applications from remote artifact.

    Note: The demo app is already deployed by startup_applications, deploying again will update the app
    """
    # Ensure BIOENGINE_LOCAL_ARTIFACT_PATH is not set to avoid local deployment
    monkeypatch.delenv("BIOENGINE_LOCAL_ARTIFACT_PATH", raising=False)
    assert os.getenv("BIOENGINE_LOCAL_ARTIFACT_PATH") is None

    hyphen_test_id = test_id.replace("_", "-")

    demo_app_path = bioengine_apps_dir / "demo-app"
    demo_artifact_id = f"{hypha_workspace}/demo-app-{hyphen_test_id}"
    demo_app_config = {
        "artifact_id": demo_artifact_id,
        "application_kwargs": {"DemoDeployment": {"test_param": "custom_value"}},
        "disable_gpu": True,
    }  # Test random application ID generation and deployment kwargs

    composition_app_path = bioengine_apps_dir / "composition-demo"
    # Alias follows the `id` in composition-demo/manifest.yaml plus the suffix
    # `create_file_list_from_directory` appends.
    composition_artifact_id = (
        f"{hypha_workspace}/bioengine-composition-demo-{hyphen_test_id}"
    )
    composition_app_config = {
        "artifact_id": composition_artifact_id,
        "application_id": f"composition-demo-{hyphen_test_id}",
        "disable_gpu": True,
    }  # Provide custom application id

    app_paths = [demo_app_path, composition_app_path]
    app_configs = [demo_app_config, composition_app_config]
    artifact_ids = [demo_artifact_id, composition_artifact_id]
    deployed_app_ids = []

    # Verify the test directories exist
    assert demo_app_path.exists(), f"Demo app directory not found: {demo_app_path}"
    assert (
        composition_app_path.exists()
    ), f"Composition app directory not found: {composition_app_path}"

    try:
        # Create artifacts first
        for app_path, artifact_id in zip(app_paths, artifact_ids):
            # Create file list from directory
            files = create_file_list_from_directory(
                directory_path=app_path, _artifact_id_suffix=test_id
            )

            # Extract artifact alias from manifest to verify it matches expected
            manifest_file = next(
                (f for f in files if f["name"] == "manifest.yaml"), None
            )
            assert manifest_file, f"Manifest not found in {app_path}"

            manifest = yaml.safe_load(manifest_file["content"])
            artifact_alias = manifest["id"]
            created_artifact_id = f"{hypha_workspace}/{artifact_alias}"

            assert (
                created_artifact_id == artifact_id
            ), f"Artifact ID mismatch: expected {artifact_id}, got {created_artifact_id}"

            # Create the artifact
            result_artifact_id = await bioengine_worker_service.upload_app(
                files=files
            )
            assert (
                result_artifact_id == artifact_id
            ), f"Created artifact ID should match expected: {artifact_id}"
            print(f"Created artifact: {artifact_id}")

        # Verify artifacts exist
        available_artifacts = await bioengine_worker_service.list_apps()
        for artifact_id in artifact_ids:
            assert (
                artifact_id in available_artifacts
            ), f"Artifact {artifact_id} should be listed in available artifacts"

        # Deploy applications from artifacts
        for app_config in app_configs:
            application_id = await bioengine_worker_service.deploy_app(
                **app_config
            )
            deployed_app_ids.append(application_id)
            print(f"Deployed application: {application_id}")

        # Wait for both applications to finish deploying
        # Generous: each replica builds a pip runtime_env before it turns HEALTHY.
        timeout = 300
        poll_interval = 2  # Check every 2 seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            bioengine_apps = await bioengine_worker_service.get_app_status(
                application_ids=deployed_app_ids
            )

            # Check if all apps are no longer in DEPLOYING state
            all_deployed = True
            for app_id in deployed_app_ids:
                if app_id in bioengine_apps:
                    app_status = bioengine_apps[app_id].get("status", "")
                    if app_status in ["NOT_STARTED", "DEPLOYING"]:
                        all_deployed = False
                        break
                else:
                    all_deployed = False
                    break

            if all_deployed:
                break

            await asyncio.sleep(poll_interval)
        else:
            raise TimeoutError(
                f"Applications did not finish deploying within {timeout} seconds"
            )

        # Check that both apps are healthy and have running replicas
        bioengine_apps = await bioengine_worker_service.get_app_status(
            application_ids=deployed_app_ids
        )

        for app_id in deployed_app_ids:
            assert app_id in bioengine_apps, f"Application {app_id} not found in status"

            app_info = bioengine_apps[app_id]
            assert (
                app_info["status"] == "RUNNING"
            ), f"Application {app_id} is not running: {app_info['status']}"

            # Check deployments are healthy
            assert (
                len(app_info["deployments"]) > 0
            ), f"Application {app_id} should have active deployments"

            for deployment_name, deployment_info in app_info["deployments"].items():
                assert (
                    deployment_info["status"] == "HEALTHY"
                ), f"Deployment {deployment_name} of app {app_id} is not healthy: {deployment_info['status']}"

                # Check replica states
                replica_states = deployment_info["replica_states"]
                assert (
                    len(replica_states) > 0
                ), f"Deployment {deployment_name} of app {app_id} should have replicas"

                # Ensure at least one replica is running
                running_replicas = replica_states.get("RUNNING", 0)
                assert (
                    running_replicas > 0
                ), f"Deployment {deployment_name} of app {app_id} should have at least one running replica"

            print(f"Application {app_id} is healthy with running replicas")

    finally:
        # Cleanup: Delete all created artifacts (even if test fails)
        for artifact_id in artifact_ids:
            try:
                await bioengine_worker_service.delete_app(
                    artifact_id=artifact_id
                )
                print(f"Deleted artifact: {artifact_id}")
            except Exception as e:
                warnings.warn(f"Failed to delete artifact {artifact_id}: {e}")

        # Cleanup: Ensure applications are undeployed (even if test fails)
        for app_id in deployed_app_ids:
            try:
                await bioengine_worker_service.stop_app(
                    application_id=app_id
                )
                print(f"Undeployed application: {app_id}")
            except Exception as e:
                warnings.warn(f"Failed to undeploy application {app_id}: {e}")


@pytest.mark.end_to_end
@pytest.mark.asyncio
async def test_call_demo_app_functions(
    monkeypatch: pytest.MonkeyPatch,
    bioengine_apps_dir: Path,
    hypha_workspace: str,
    bioengine_worker_service: ObjectProxy,
    hypha_client: RemoteService,
):
    """
    Test calling functions of the deployed demo application.

    Exposed methods of `DemoDeployment`:
    - `ping`
    - `ascii_art`
    - `list_datasets`
    - `reverse_text`
    - `set_fail_health_check`
    """
    # Local artifact resolution is by directory name under this root
    monkeypatch.setenv("BIOENGINE_LOCAL_ARTIFACT_PATH", str(bioengine_apps_dir))
    assert os.getenv("BIOENGINE_LOCAL_ARTIFACT_PATH") == str(bioengine_apps_dir)

    # Deploy the demo-app with apps_manager.deploy_app from local path
    demo_artifact_id = f"{hypha_workspace}/demo-app"

    app_id = await bioengine_worker_service.deploy_app(
        artifact_id=demo_artifact_id, disable_gpu=True
    )

    try:
        # Wait for deployment to complete
        # Generous: each replica builds a pip runtime_env before it turns HEALTHY.
        timeout = 300
        poll_interval = 2  # Check every 2 seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            app_status_result = await bioengine_worker_service.get_app_status(
                application_ids=[app_id]
            )

            if app_id in app_status_result:
                app_status = app_status_result[app_id]
                service_ids = app_status.get("service_ids") or {}
                if app_status["status"] == "RUNNING" and service_ids.get(
                    "websocket_service_id"
                ):
                    break

            await asyncio.sleep(poll_interval)
        else:
            pytest.fail("Demo app deployment timed out")

        # Get the service ID from the application status
        app_status_result = await bioengine_worker_service.get_app_status(
            application_ids=[app_id]
        )
        app_status = app_status_result[app_id]
        service_ids = app_status["service_ids"]

        websocket_service_id = service_ids["websocket_service_id"]
        webrtc_service_id = service_ids["webrtc_service_id"]

        websocket_service = await resolve_service(
            lambda: hypha_client.get_service(websocket_service_id)
        )
        assert (
            websocket_service
        ), f"Could not connect to WebSocket service {websocket_service_id}"

        # Call the application functions using the WebSocket service
        # Test ping method
        ping_result = await asyncio.wait_for(websocket_service.ping(), timeout=10)
        assert ping_result is not None, "Ping should return a result"
        assert isinstance(ping_result, dict), "Ping result should be a dictionary"
        assert (
            ping_result["status"] == "ok"
        ), f"Expected status 'ok', got {ping_result.get('status')}"
        assert "message" in ping_result, "Ping result should contain 'message'"
        assert "timestamp" in ping_result, "Ping result should contain 'timestamp'"
        assert "uptime" in ping_result, "Ping result should contain 'uptime'"

        # Test ascii_art method
        ascii_result = await asyncio.wait_for(websocket_service.ascii_art(), timeout=10)
        assert ascii_result is not None, "ASCII art should return a result"
        assert isinstance(ascii_result, list), "ASCII art result should be a list"
        assert len(ascii_result) > 0, "ASCII art should not be empty"
        assert all(
            isinstance(line, str) for line in ascii_result
        ), "All ASCII lines should be strings"

        # Get the peer connection
        peer_connection = await resolve_service(
            lambda: get_rtc_service(hypha_client, webrtc_service_id)
        )
        assert (
            peer_connection
        ), f"Could not connect to WebRTC service {webrtc_service_id}"

        try:
            # Get the service using the peer connection instead of hypha_client
            peer_service = await peer_connection.get_service(app_id)
            assert peer_service, "Could not get peer service from WebRTC"

            # Call the application functions using the peer connection service
            # Test ping method through WebRTC
            rtc_ping_result = await asyncio.wait_for(
                peer_service.ping(context=hypha_client.config), timeout=10
            )
            assert rtc_ping_result is not None, "WebRTC ping should return a result"
            assert isinstance(
                rtc_ping_result, dict
            ), "WebRTC ping result should be a dictionary"
            assert (
                rtc_ping_result["status"] == "ok"
            ), f"Expected status 'ok', got {rtc_ping_result.get('status')}"

            # Test ascii_art method through WebRTC
            rtc_ascii_result = await asyncio.wait_for(
                peer_service.ascii_art(context=hypha_client.config), timeout=10
            )
            assert (
                rtc_ascii_result is not None
            ), "WebRTC ASCII art should return a result"
            assert isinstance(
                rtc_ascii_result, list
            ), "WebRTC ASCII art result should be a list"
            assert len(rtc_ascii_result) > 0, "WebRTC ASCII art should not be empty"

            # Results should be the same through both channels
            assert (
                rtc_ping_result["status"] == ping_result["status"]
            ), "Ping results should match"
            assert rtc_ascii_result == ascii_result, "ASCII art results should match"

        finally:
            # Clean up WebRTC connection
            await peer_connection.disconnect()

    finally:
        # Cleanup: Ensure applications are undeployed (even if test fails)
        try:
            await bioengine_worker_service.stop_app(application_id=app_id)
            print(f"Undeployed application: {app_id}")
        except Exception as e:
            warnings.warn(f"Failed to undeploy application {app_id}: {e}")


@pytest.mark.end_to_end
@pytest.mark.asyncio
async def test_call_composition_app_functions(
    monkeypatch: pytest.MonkeyPatch,
    bioengine_apps_dir: Path,
    hypha_workspace: str,
    bioengine_worker_service: ObjectProxy,
    hypha_client: RemoteService,
):
    """
    Test calling functions of the deployed composition application.

    Only `EntryDeployment` is exposed as a service; it fans out to
    RuntimeA/B/C through deployment handles. Exposed methods:
    - `status`
    - `process_text`
    - `analyze_numbers`
    - `time_operations`
    - `run_all`
    """
    # Local artifact resolution is by directory name under this root
    monkeypatch.setenv("BIOENGINE_LOCAL_ARTIFACT_PATH", str(bioengine_apps_dir))
    assert os.getenv("BIOENGINE_LOCAL_ARTIFACT_PATH") == str(bioengine_apps_dir)

    # Deploy the composition-demo with apps_manager.deploy_app from local path
    # EntryDeployment takes only deployment handles, so no application_kwargs.
    composition_app_config = {
        "artifact_id": f"{hypha_workspace}/composition-demo",
        "disable_gpu": True,
    }

    app_id = await bioengine_worker_service.deploy_app(**composition_app_config)

    try:
        # Wait for deployment to complete
        # Generous: each replica builds a pip runtime_env before it turns HEALTHY.
        timeout = 300
        poll_interval = 2  # Check every 2 seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            app_status_result = await bioengine_worker_service.get_app_status(
                application_ids=[app_id]
            )

            if app_id in app_status_result:
                app_status = app_status_result[app_id]
                service_ids = app_status.get("service_ids") or {}
                if app_status["status"] == "RUNNING" and service_ids.get(
                    "websocket_service_id"
                ):
                    break

            await asyncio.sleep(poll_interval)
        else:
            pytest.fail("Composition app deployment timed out")

        # Get the service ID from the application status
        app_status_result = await bioengine_worker_service.get_app_status(
            application_ids=[app_id]
        )
        app_status = app_status_result[app_id]
        service_ids = app_status["service_ids"]

        websocket_service_id = service_ids["websocket_service_id"]
        webrtc_service_id = service_ids["webrtc_service_id"]

        websocket_service = await resolve_service(
            lambda: hypha_client.get_service(websocket_service_id)
        )
        assert (
            websocket_service
        ), f"Could not connect to WebSocket service {websocket_service_id}"

        # Call the application functions using the WebSocket service
        # Test status method — fans out to all three runtimes
        status_result = await asyncio.wait_for(websocket_service.status(), timeout=30)
        assert isinstance(status_result, dict), "Status result should be a dictionary"
        assert "entry_uptime" in status_result, "Status should report the entry uptime"
        for runtime_key, runtime_name in (
            ("runtime_a", "runtime_a"),
            ("runtime_b", "runtime_b"),
            ("runtime_c", "runtime_c"),
        ):
            runtime_status = status_result[runtime_key]
            assert isinstance(
                runtime_status, dict
            ), f"{runtime_key} status should be a dictionary"
            assert (
                runtime_status["name"] == runtime_name
            ), f"Expected name '{runtime_name}', got {runtime_status.get('name')}"
            assert runtime_status["status"] == "ok", f"{runtime_key} should be ok"
        # RuntimeB's `pip=` runtime_env is what makes numpy importable there
        assert status_result["runtime_b"][
            "numpy_version"
        ], "RuntimeB should report its numpy version"

        # Test process_text method — routed through RuntimeA
        text_result = await asyncio.wait_for(
            websocket_service.process_text(text="hello bioengine"), timeout=30
        )
        assert isinstance(text_result, dict), "process_text should return a dictionary"
        assert text_result["word_count"] == 2, f"Expected 2 words, got {text_result}"
        assert text_result["words"] == ["hello", "bioengine"], text_result
        assert text_result["upper"] == "HELLO BIOENGINE", text_result
        assert text_result["reversed"] == "enigneoib olleh", text_result

        # Test analyze_numbers method — routed through RuntimeB (numpy)
        numbers_result = await asyncio.wait_for(
            websocket_service.analyze_numbers(values=[1, 2, 3, 4, 5]), timeout=30
        )
        assert isinstance(
            numbers_result, dict
        ), "analyze_numbers should return a dictionary"
        assert numbers_result["mean"] == 3.0, numbers_result
        assert numbers_result["sum"] == 15.0, numbers_result
        assert numbers_result["min"] == 1.0 and numbers_result["max"] == 5.0
        assert numbers_result["count"] == 5, numbers_result

        # Test time_operations method — routed through RuntimeC
        time_result = await asyncio.wait_for(
            websocket_service.time_operations(count=3), timeout=30
        )
        assert isinstance(
            time_result, dict
        ), "time_operations should return a dictionary"
        assert time_result["count"] == 3, time_result
        assert len(time_result["timestamps"]) == 3, time_result

        # Test run_all method — fans out to all three runtimes in parallel
        run_all_result = await asyncio.wait_for(
            websocket_service.run_all(text="hello bioengine", values=[2, 4], count=2),
            timeout=30,
        )
        assert isinstance(run_all_result, dict), "run_all should return a dictionary"
        assert run_all_result["text_result"]["word_count"] == 2, run_all_result
        assert run_all_result["data_result"]["mean"] == 3.0, run_all_result
        assert len(run_all_result["time_result"]["timestamps"]) == 2, run_all_result

        # Get the peer connection
        peer_connection = await resolve_service(
            lambda: get_rtc_service(hypha_client, webrtc_service_id)
        )
        assert (
            peer_connection
        ), f"Could not connect to WebRTC service {webrtc_service_id}"

        try:
            # Get the service using the peer connection instead of hypha_client
            peer_service = await peer_connection.get_service(app_id)
            assert peer_service, "Could not get peer service from WebRTC"

            # Call the application functions using the peer connection service
            # Test status method through WebRTC
            rtc_status_result = await asyncio.wait_for(
                peer_service.status(context=hypha_client.config), timeout=30
            )
            assert isinstance(
                rtc_status_result, dict
            ), "WebRTC status result should be a dictionary"
            for runtime_key in ("runtime_a", "runtime_b", "runtime_c"):
                assert (
                    rtc_status_result[runtime_key]["status"] == "ok"
                ), f"WebRTC {runtime_key} should be ok"

            # Test process_text method through WebRTC
            rtc_text_result = await asyncio.wait_for(
                peer_service.process_text(
                    text="hello bioengine", context=hypha_client.config
                ),
                timeout=30,
            )
            assert isinstance(
                rtc_text_result, dict
            ), "WebRTC process_text result should be a dictionary"

            # Deterministic results should be the same through both channels
            assert rtc_text_result == text_result, "process_text results should match"

        finally:
            # Clean up WebRTC connection
            await peer_connection.disconnect()

    finally:
        # Cleanup: Ensure applications are undeployed (even if test fails)
        try:
            await bioengine_worker_service.stop_app(application_id=app_id)
            print(f"Undeployed application: {app_id}")
        except Exception as e:
            warnings.warn(f"Failed to undeploy application {app_id}: {e}")


# TODO: test proxy deployment autoscaling and load balancing
