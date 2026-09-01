"""
Global pytest configuration for BioEngine Worker tests.

Provides common fixtures for environment setup, Hypha authentication,
and test isolation across the entire test suite.

Requires HYPHA_TOKEN in environment and bioengine-worker conda environment.
"""

import asyncio
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import ObjectProxy, RemoteService

from bioengine.cluster.ray_cluster import RayCluster

# Load environment variables from .env file
load_dotenv()


@pytest.fixture(
    scope="session",
    params=[
        "external-cluster",
        "single-machine",
        "slurm",
    ],
)
def worker_mode(request) -> str:
    """
    Provide worker mode for tests based on environment configuration.

    Modes:
    - 'external-cluster': Connect to existing Ray cluster
    - 'single-machine': Start local Ray cluster
    - 'slurm': Use Slurm for job scheduling (if available)
    """
    if os.getenv("BIOENGINE_TEST_SINGLE_CLUSTER", "0") == "1":
        if request.param in ["single-machine", "slurm"]:
            pytest.skip(
                "Single cluster mode enabled. Skipping test for worker mode: "
                f"{request.param}"
            )

    if request.param == "slurm":
        try:
            subprocess.run(["sinfo"], capture_output=True, text=True, check=True)
        except FileNotFoundError:
            pytest.skip("Slurm not available. Skipping test for worker mode: slurm")

    return request.param


@pytest.fixture(scope="session")
def workspace_folder() -> Path:
    """
    Return project root directory and set BIOENGINE_LOCAL_ARTIFACT_PATH.

    Configures environment for local test artifact discovery.
    """
    folder = Path(__file__).resolve().parent.parent
    return folder


@pytest.fixture(scope="session")
def tests_dir(workspace_folder: Path) -> Path:
    """Return test directory."""
    return workspace_folder / "tests"


@pytest.fixture(scope="session")
def bioengine_apps_dir(workspace_folder: Path) -> Path:
    """Return bioengine_apps directory."""
    return workspace_folder / "apps"


@pytest.fixture(scope="session", autouse=True)
def validate_environment(workspace_folder) -> str:
    requirements_file = workspace_folder / "requirements-worker.txt"
    with open(requirements_file, "r") as file:
        for req in file:
            req = req.strip()
            if not req or req.startswith("#"):
                continue

            # Use regex to match package names (also ray[serve] and similar)
            match = re.match(r"^\s*([a-zA-Z0-9_\-\.]+)", req)
            if match:
                package_name = match.group(1)
                try:
                    # Query the interpreter running the tests, not whichever
                    # pip happens to be first on PATH.
                    subprocess.run(
                        [sys.executable, "-m", "pip", "show", package_name],
                        check=True,
                        capture_output=True,
                    )
                except subprocess.CalledProcessError:
                    pytest.exit(
                        f"Required package '{package_name}' not installed. "
                        "Please install it before running tests."
                    )
            else:
                pytest.exit(f"Invalid requirement format: {req}")


@pytest.fixture(scope="session")
def data_dir(workspace_folder: Path) -> Path:
    """Create and return BioEngine Worker data directory."""
    data_dir = workspace_folder / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture(scope="function")
def test_id() -> str:
    """Generate unique timestamp-based session ID for test isolation."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


@pytest.fixture(scope="function")
def workspace_dir() -> Generator[Path, None, None]:
    """Create temporary workspace directory with automatic cleanup."""
    with tempfile.TemporaryDirectory(prefix="bioengine_test_") as temp_workspace_dir:
        workspace_dir = Path(temp_workspace_dir)
        yield workspace_dir

    # Cleanup is handled automatically by TemporaryDirectory context manager


@pytest.fixture(scope="session")
def ray_address(worker_mode: str) -> Generator[str, None, None]:
    """Start a Ray cluster a BioEngine Worker can connect to."""
    if worker_mode == "external-cluster":
        with tempfile.TemporaryDirectory(prefix=f"bioengine_test_ray_") as temp_dir:

            # Use RayCluster to start a local Ray cluster
            ray_cluster = RayCluster(
                mode="single-machine",
                head_num_cpus=6,
                head_num_gpus=0,
                head_memory_in_gb=12,
                ray_temp_dir=temp_dir,
                debug=True,
            )
            asyncio.run(ray_cluster._start_cluster())
            ray_cluster._set_head_node_address()
            ray_cluster.is_ready.set()

            yield ray_cluster.address

            # Stop the Ray cluster after tests complete
            asyncio.run(ray_cluster.stop())
    else:
        # For single-machine or slurm modes, return no address
        yield None


@pytest.fixture(scope="session")
def head_node_address(ray_address: str) -> str:
    """Return head node address based on worker mode."""
    if ray_address is None:
        return None

    # Extract address from Ray address format "address:port"
    address, _ = ray_address.split(":")
    return address


@pytest.fixture(scope="session")
def head_node_port(ray_address: str) -> int:
    """Return head node port based on worker mode."""
    if ray_address is None:
        return 6379  # Default Ray port

    # Extract port from Ray address format "address:port"
    _, port = ray_address.split(":")
    return int(port)


@pytest.fixture(scope="session")
def server_url() -> str:
    """Return Hypha server URL for test connections."""
    return "https://hypha.aicell.io"


@pytest.fixture(scope="session")
def hypha_token() -> str:
    """
    Get Hypha authentication token from HYPHA_TOKEN environment variable.

    Skips test if token not found. Required for Hypha server access.
    """
    token = os.environ.get("HYPHA_TOKEN")
    if not token:
        pytest.exit(
            "HYPHA_TOKEN environment variable not set. "
            "Please ensure .env file contains HYPHA_TOKEN"
        )
    return token


@pytest_asyncio.fixture(scope="function")
async def hypha_client(
    hypha_token: str, test_id: str
) -> AsyncGenerator[RemoteService, None]:
    """
    Create Hypha RPC client with unique ID for each test function.

    Automatically connects and disconnects for proper test isolation.
    """
    client = await connect_to_server(
        {
            "server_url": "https://hypha.aicell.io",
            "token": hypha_token,
            "client_id": f"bioengine_test_client_{test_id}",
        }
    )
    yield client

    await client.disconnect()


@pytest.fixture(scope="function")
def hypha_workspace(hypha_client: RemoteService) -> str:
    """Extract workspace ID from connected Hypha client."""
    return hypha_client.config.workspace


@pytest.fixture(scope="function")
def hypha_client_id(hypha_client: RemoteService) -> str:
    """Extract unique client ID from connected Hypha client."""
    return hypha_client.config.client_id


@pytest.fixture(scope="function")
def hypha_user_id(hypha_client: RemoteService) -> str:
    """Extract unique user ID from connected Hypha client."""
    return hypha_client.config.user["id"]


@pytest_asyncio.fixture(scope="function")
async def artifact_manager(hypha_client: RemoteService) -> ObjectProxy:
    """
    Get artifact manager service for the Hypha workspace.
    """
    artifact_manager_service = await hypha_client.get_service("public/artifact-manager")
    return artifact_manager_service


# Configure asyncio for pytest
pytest_plugins = ("pytest_asyncio",)
