"""
End-to-end test configuration for BioEngine Worker.

Provides specialized fixtures for comprehensive testing including worker
initialization, application deployment, and service interaction patterns.

Supports both local worker instances and connecting to running workers
via USE_RUNNING_WORKER environment variable.
"""

import os
from typing import AsyncGenerator, Dict, List

import pytest
import pytest_asyncio
from anyio import Path
from hypha_rpc.rpc import ObjectProxy, RemoteService

from bioengine.worker import BioEngineWorker


# Test application configurations for automatic startup deployment
@pytest.fixture(scope="session")
def startup_applications() -> List[Dict]:
    """Return startup applications for BioEngine Worker tests."""
    return [
        {
            "artifact_id": "demo-app",
            "application_id": "demo-app",
            "disable_gpu": True,
        },
    ]


@pytest.fixture(scope="session")
def monitoring_interval_seconds() -> int:
    """Return 10-second monitoring interval for responsive test feedback."""
    return 10


@pytest.fixture(scope="session")
def dashboard_url() -> str:
    """Return BioEngine dashboard URL for worker integration."""
    return "https://bioimage.io/#/bioengine"


@pytest.fixture(scope="session")
def graceful_shutdown_timeout() -> int:
    """Return 60-second timeout for graceful worker shutdown."""
    return 60


@pytest.fixture(scope="session")
def application_check_timeout() -> int:
    """Return 30-second timeout for application readiness checks."""
    return 30


@pytest.fixture(scope="session")
def num_cpus(worker_mode: str) -> int:
    """Return CPU cores based on worker mode."""
    return 4 if worker_mode != "external-cluster" else 0


@pytest.fixture(scope="session")
def num_gpus() -> int:
    """Return 0 GPUs for testing (no GPU required)."""
    return 0


@pytest.fixture(scope="session")
def memory_in_gb() -> int:
    """Return 6 GB memory for testing."""
    return 6


@pytest_asyncio.fixture(scope="function")
async def bioengine_worker_service_id(
    monkeypatch: pytest.MonkeyPatch,
    worker_mode: str,
    workspace_dir: Path,
    startup_applications: List[Dict],
    monitoring_interval_seconds: int,
    server_url: str,
    hypha_token: str,
    test_id: str,
    num_cpus: int,
    num_gpus: int,
    memory_in_gb: int,
    head_node_address: str,
    head_node_port: int,
    dashboard_url: str,
    graceful_shutdown_timeout: int,
    bioengine_apps_dir: Path,
) -> AsyncGenerator[str, None]:
    """
    Create BioEngine worker instance and return service ID.

    Initializes worker with startup applications and manages lifecycle.
    Automatically starts worker and cleans up after test completion.
    """

    # Local artifact resolution is by directory name under this root, so the
    # reference apps in `apps/` are addressable as `<workspace>/<dir-name>`.
    monkeypatch.setenv("BIOENGINE_LOCAL_ARTIFACT_PATH", str(bioengine_apps_dir))
    assert os.getenv("BIOENGINE_LOCAL_ARTIFACT_PATH") == str(bioengine_apps_dir)

    # Initialize the BioEngine worker with startup applications
    bioengine_worker = BioEngineWorker(
        mode=worker_mode,
        admin_users=None,
        workspace_dir=workspace_dir,
        startup_applications=startup_applications,
        monitoring_interval_seconds=monitoring_interval_seconds,
        server_url=server_url,
        workspace=None,
        token=hypha_token,
        client_id=f"bioengine_test_worker_{test_id}",
        ray_cluster_config={
            "head_node_address": head_node_address,
            "head_node_port": head_node_port,
            "head_num_cpus": num_cpus,
            "head_num_gpus": num_gpus,
            "head_memory_in_gb": memory_in_gb,
        },
        dashboard_url=dashboard_url,
        log_file="off",
        debug=True,
        graceful_shutdown_timeout=graceful_shutdown_timeout,
    )

    try:
        # Start the worker
        await bioengine_worker.start(blocking=False)

        # Return the worker service for use in tests
        yield bioengine_worker.full_service_id

    finally:
        # Cleanup after all tests are done
        await bioengine_worker._stop(blocking=True)


@pytest_asyncio.fixture(scope="function")
async def bioengine_worker_service(
    hypha_client: RemoteService, bioengine_worker_service_id: str
) -> ObjectProxy:
    """Get BioEngine worker service from created worker instance."""
    # Get the BioEngine worker service
    bioengine_worker_service = await hypha_client.get_service(
        bioengine_worker_service_id
    )

    # Return the worker service for use in tests
    return bioengine_worker_service


@pytest.fixture(scope="function")
def bioengine_worker_workspace(bioengine_worker_service_id: str) -> str:
    """Extract workspace from worker service ID."""
    return bioengine_worker_service_id.split("/")[0]


@pytest.fixture(scope="function")
def bioengine_worker_client_id(bioengine_worker_service_id: str) -> str:
    """Extract client ID from worker service ID."""
    return bioengine_worker_service_id.split("/")[1].split(":")[0]
