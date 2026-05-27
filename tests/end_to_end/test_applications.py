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


@pytest.mark.end_to_end
@pytest.mark.asyncio
async def test_create_and_delete_artifacts(
    bioengine_worker_service: ObjectProxy,
    tests_dir: Path,
    bioengine_apps_dir: Path,
    test_id: str,
    hypha_workspace: str,
    hypha_user_id: str,
):
    """
    Test creating and deleting artifacts from both demo-app and composition-app applications.

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
    composition_app_path = bioengine_apps_dir / "demo-app"

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

        # Create composition-app artifact
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

        # Update demo-app artifact
        updated_demo_artifact_id = await bioengine_worker_service.upload_app(
            files=demo_app_files
        )
        assert (
            updated_demo_artifact_id == demo_artifact_id
        ), "Updated demo artifact ID should match original"

        # Update composition-app artifact
        updated_composition_artifact_id = (
            await bioengine_worker_service.upload_app(
                files=composition_app_files
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

        # Verify all files in demo-app artifact
        assert all(
            f["name"] in available_artifacts[demo_artifact_id]["files"]
            for f in demo_app_files
        ), "All demo app files should be listed in artifact files"
        received_manifest = available_artifacts[demo_artifact_id]["manifest"].toDict()
        created_by = received_manifest["manifest"].pop("created_by")
        assert (
            received_manifest["manifest"] == demo_manifest
        ), "Demo app manifest should match expected manifest"
        assert (
            created_by == hypha_user_id,
            "Created by user ID should match the test user ID",
        )
        assert (
            received_manifest["parent_id"] == f"{hypha_workspace}/applications"
        ), "Demo app manifest should be in applications collection"

        # Verify all files in composition-app artifact
        assert all(
            f["name"] in available_artifacts[composition_artifact_id]["files"]
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
            created_by == hypha_user_id,
            "Created by user ID should match the test user ID",
        )
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

    # Get application status (returns all deployed applications when application_ids is None)
    apps_status = await bioengine_worker_service.get_app_status()

    # Validate apps_status is properly structured
    assert isinstance(apps_status, dict), "Application status should be a dictionary"

    # Assert that applications are deployed based on startup configuration
    expected_app_count = len(startup_applications)
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
            app_info["authorized_users"], list
        ), f"authorized_users should be a list for '{application_id}'"
        assert isinstance(
            app_info["available_methods"], list
        ), f"available_methods should be a list for '{application_id}'"
        assert isinstance(
            app_info["service_ids"], list
        ), f"service_ids should be a list for '{application_id}'"

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

        # Validate authorized users
        assert (
            len(app_info["authorized_users"]) > 0
        ), f"Application '{application_id}' should have authorized users"
        for user in app_info["authorized_users"]:
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
    tests_dir: Path,
    hypha_workspace: str,
    test_id: str,
    bioengine_worker_service: ObjectProxy,
):
    """
    Test deploying the 'demo-app' and 'composition-app' applications from local artifact path.
    """
    # Set environment variables for startup application deployment from local path
    monkeypatch.setenv("BIOENGINE_LOCAL_ARTIFACT_PATH", str(tests_dir))
    assert os.getenv("BIOENGINE_LOCAL_ARTIFACT_PATH") == str(tests_dir)

    # iterate over demo_app and composition_app directories
    # Note: the demo app is already deployed by startup_applications, but the deployment below will use a different application ID

    demo_artifact_id = f"{hypha_workspace}/demo-app"
    demo_app_config = {
        "artifact_id": demo_artifact_id,
        "disable_gpu": True,
    }  # Test random application ID generation

    composition_artifact_id = f"{hypha_workspace}/composition-app"
    hyphen_test_id = test_id.replace("_", "-")
    composition_app_config = {
        "artifact_id": composition_artifact_id,
        "application_id": f"composition-app-{hyphen_test_id}",
        "application_kwargs": {
            "CompositionDeployment": {"demo_input": "Hello World!"},
            "Deployment2": {"start_number": 10},
        },
        "disable_gpu": True,
    }  # Provide custom application id and deployment kwargs

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
        timeout = 30  # 30 seconds timeout
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
    tests_dir: Path,
    bioengine_apps_dir: Path,
    test_id: str,
    hypha_workspace: str,
    bioengine_worker_service: ObjectProxy,
):
    """
    Test deploying the 'demo-app' and 'composition-app' applications from remote artifact.

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
        "disable_gpu": True,
    }  # Test random application ID generation

    composition_app_path = bioengine_apps_dir / "demo-app"
    composition_artifact_id = f"{hypha_workspace}/composition-app-{hyphen_test_id}"
    composition_app_config = {
        "artifact_id": composition_artifact_id,
        "application_id": f"composition-app-{hyphen_test_id}",
        "application_kwargs": {
            "CompositionDeployment": {"demo_input": "Hello World!"},
            "Deployment2": {"start_number": 10},
        },
        "disable_gpu": True,
    }  # Provide custom application id and deployment kwargs

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
        timeout = 30  # 30 seconds timeout
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
    tests_dir: Path,
    hypha_workspace: str,
    bioengine_worker_service: ObjectProxy,
    hypha_client: RemoteService,
):
    """
    Test calling functions of the deployed demo application.

    Exposed methods:
    - `ping`
    - `ascii_art`
    """
    # Set environment variables for startup application deployment from local path
    monkeypatch.setenv("BIOENGINE_LOCAL_ARTIFACT_PATH", str(tests_dir))
    assert os.getenv("BIOENGINE_LOCAL_ARTIFACT_PATH") == str(tests_dir)

    # Deploy the demo-app with apps_manager.deploy_app from local path
    demo_artifact_id = f"{hypha_workspace}/demo-app"

    app_id = await bioengine_worker_service.deploy_app(
        artifact_id=demo_artifact_id, disable_gpu=True
    )

    try:
        # Wait for deployment to complete
        timeout = 30  # 30 seconds timeout
        poll_interval = 2  # Check every 2 seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            app_status_result = await bioengine_worker_service.get_app_status(
                application_ids=[app_id]
            )

            if app_id in app_status_result:
                app_status = app_status_result[app_id]
                if (
                    app_status["status"] == "RUNNING"
                    and len(app_status.get("service_ids", [])) > 0
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

        websocket_service = await hypha_client.get_service(websocket_service_id)
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
        peer_connection = await get_rtc_service(hypha_client, webrtc_service_id)
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
    tests_dir: Path,
    hypha_workspace: str,
    bioengine_worker_service: ObjectProxy,
    hypha_client: RemoteService,
):
    """
    Test calling functions of the deployed composition application.

    Exposed methods:
    - `ping`
    - `calculate_result`
    """
    # Set environment variables for startup application deployment from local path
    monkeypatch.setenv("BIOENGINE_LOCAL_ARTIFACT_PATH", str(tests_dir))
    assert os.getenv("BIOENGINE_LOCAL_ARTIFACT_PATH") == str(tests_dir)

    # Deploy the composition-app with apps_manager.deploy_app from local path
    composition_app_config = {
        "artifact_id": f"{hypha_workspace}/composition_app",
        "application_kwargs": {
            "CompositionDeployment": {"demo_input": "Test Hello World!"},
            "Deployment2": {"start_number": 100},
        },
        "disable_gpu": True,
    }

    app_id = await bioengine_worker_service.deploy_app(**composition_app_config)

    try:
        # Wait for deployment to complete
        timeout = 30  # 30 seconds timeout
        poll_interval = 2  # Check every 2 seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            app_status_result = await bioengine_worker_service.get_app_status(
                application_ids=[app_id]
            )

            if app_id in app_status_result:
                app_status = app_status_result[app_id]
                if (
                    app_status["status"] == "RUNNING"
                    and len(app_status.get("service_ids", [])) > 0
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

        websocket_service = await hypha_client.get_service(websocket_service_id)
        assert (
            websocket_service
        ), f"Could not connect to WebSocket service {websocket_service_id}"

        # Call the application functions using the WebSocket service
        # Test ping method
        ping_result = await asyncio.wait_for(websocket_service.ping(), timeout=10)
        assert ping_result is not None, "Ping should return a result"
        assert isinstance(ping_result, str), "Ping result should be a string"
        assert ping_result == "pong", f"Expected 'pong', got {ping_result}"

        # Test calculate_result method
        test_number = 42
        calc_result = await asyncio.wait_for(
            websocket_service.calculate_result(number=test_number), timeout=15
        )
        assert calc_result is not None, "Calculate result should return a result"
        assert isinstance(calc_result, str), "Calculate result should be a string"
        assert "Uptime:" in calc_result, "Result should contain uptime information"
        assert "Result:" in calc_result, "Result should contain calculation result"
        assert "Demo string:" in calc_result, "Result should contain demo string"
        assert (
            "Test Hello World!" in calc_result
        ), "Result should contain the demo input"
        # The result should be start_number (100) + test_number (42) = 142
        assert (
            "142" in calc_result
        ), f"Result should contain 142 (100 + 42), got: {calc_result}"

        # Get the peer connection
        peer_connection = await get_rtc_service(hypha_client, webrtc_service_id)
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
                rtc_ping_result, str
            ), "WebRTC ping result should be a string"
            assert rtc_ping_result == "pong", f"Expected 'pong', got {rtc_ping_result}"

            # Test calculate_result method through WebRTC
            rtc_calc_result = await asyncio.wait_for(
                peer_service.calculate_result(
                    number=test_number, context=hypha_client.config
                ),
                timeout=15,
            )
            assert (
                rtc_calc_result is not None
            ), "WebRTC calculate result should return a result"
            assert isinstance(
                rtc_calc_result, str
            ), "WebRTC calculate result should be a string"
            assert (
                "Uptime:" in rtc_calc_result
            ), "WebRTC result should contain uptime information"
            assert (
                "Result:" in rtc_calc_result
            ), "WebRTC result should contain calculation result"
            assert (
                "Demo string:" in rtc_calc_result
            ), "WebRTC result should contain demo string"
            assert (
                "Test Hello World!" in rtc_calc_result
            ), "WebRTC result should contain the demo input"
            assert (
                "142" in rtc_calc_result
            ), f"WebRTC result should contain 142 (100 + 42), got: {rtc_calc_result}"

            # Results should be the same through both channels
            assert rtc_ping_result == ping_result, "Ping results should match"
            # Note: Uptime may differ slightly between calls, so we check key components
            assert "Test Hello World!" in rtc_calc_result, "Demo string should match"
            assert "142" in rtc_calc_result, "Calculation result should match"

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
