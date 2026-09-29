"""
Global pytest configuration for BioEngine Worker tests.

Provides common fixtures for environment setup, Hypha authentication,
and test isolation across the entire test suite.

Requires HYPHA_TOKEN in environment and bioengine-worker conda environment.
"""

import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator, List, Optional, Tuple

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import ObjectProxy, RemoteService

from bioengine.cluster.ray_cluster import RayCluster

# Load environment variables from .env file
load_dotenv()

# Every Hypha connection in the suite is hard-coded to this deployment; no
# environment variable redirects it.
HYPHA_SERVER_URL = "https://hypha.aicell.io"

# Requesting one of these fixtures means the test authenticates against the
# real deployment above. Anything else that reaches a cluster has to say so
# with @pytest.mark.live.
LIVE_FIXTURES = frozenset({"hypha_client", "hypha_token", "model_runner"})

# Credentials the suite picks up on its own, in the order a reader should
# worry about them. `load_dotenv()` above supplies these from a repo-root
# .env, so whether they resolve depends on which checkout pytest was run from.
LIVE_TOKEN_VARS = ("HYPHA_TOKEN", "BIOIMAGE_IO_TOKEN")

_START_TIME = pytest.StashKey[float]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help=(
            "Run tests that deploy to and call a real Hypha cluster. Without "
            "this flag they are deselected even when a token is available."
        ),
    )


def _token_workspace(token: str) -> str:
    """Workspace a Hypha JWT is scoped to, or '' if it is not a readable JWT."""
    try:
        segment = token.split(".")[1]
        claims = json.loads(
            base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
        )
        scope = str(claims.get("scope", ""))
    except Exception:
        # A token we cannot parse must not abort the session; the caller
        # reports the credential as present with an unknown workspace.
        return ""
    for part in scope.split():
        if part.startswith("wid:"):
            return part[len("wid:") :]
    return ""


def resolved_live_targets(
    environ: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """(variable, workspace) for every cluster credential visible to this run."""
    environ = os.environ if environ is None else environ
    return [
        (var, _token_workspace(environ[var]) or "<unreadable token>")
        for var in LIVE_TOKEN_VARS
        if environ.get(var)
    ]


def live_exposure_banner(
    targets: List[Tuple[str, str]], live_count: int, enabled: bool
) -> str:
    """The text printed before the first test when a cluster credential resolves.

    The count is stated whether or not --live was given: the flag only stops
    the accidental case, and once someone has opted in deliberately the count
    in the log is the only thing that makes an after-the-fact audit possible.
    """
    lines = [
        "=" * 22 + " live cluster credentials resolved " + "=" * 22,
        f"{live_count} collected test(s) can reach the live cluster at "
        f"{HYPHA_SERVER_URL}, and a credential for it resolved:",
    ]
    lines += [f"  {var} -> workspace {workspace}" for var, workspace in targets]
    lines.append(
        f"THEY WILL RUN: --live was given, so {live_count} test(s) will act on "
        "that workspace."
        if enabled
        else f"Deselected {live_count} test(s); pass --live to run them."
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def _emit(config: pytest.Config, text: str) -> None:
    """Write where the controller will actually see it.

    Under pytest-xdist -- which `addopts` turns on with `--numprocesses=1` --
    collection runs inside a worker whose terminalreporter output is dropped
    and whose deselection stats never reach the summary. The worker's stderr
    is inherited by the controller, so that is the channel that survives.
    """
    worker = getattr(config, "workerinput", None)
    if worker is not None:
        if worker.get("workerid", "gw0") == "gw0":
            sys.stderr.write(text + "\n")
            sys.stderr.flush()
        return

    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        print(text)
    else:
        reporter.write_line(text)


def _is_live(item: pytest.Item) -> bool:
    if any(item.iter_markers("live")):
        return True
    return bool(LIVE_FIXTURES.intersection(getattr(item, "fixturenames", ())))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    # Attaching the marker is what makes `-m live` agree with `--live`: most
    # live tests are detected from their fixture closure, so without this they
    # are gated but unnamed and `-m live` returns a wrong subset. It has to
    # happen before pytest's own mark filtering -- which a conftest hookimpl
    # already does under pluggy's reverse-registration order, so tryfirst is a
    # guarantee against a plugin registering an earlier modifyitems, not a
    # load-bearing fix. Switching it to trylast does break the invariant.
    live, offline = [], []
    for item in items:
        if _is_live(item):
            item.add_marker(pytest.mark.live)
            live.append(item)
        else:
            offline.append(item)

    enabled = config.getoption("--live")
    if live and not enabled:
        items[:] = offline
        config.hook.pytest_deselected(items=live)

    targets = resolved_live_targets()
    if targets:
        _emit(config, live_exposure_banner(targets, len(live), enabled))


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_START_TIME] = time.time()


def pytest_terminal_summary(
    terminalreporter, exitstatus: int, config: pytest.Config
) -> None:
    """Restate the exposure next to the wall clock.

    Runtime is the cheapest signal that live tests ran -- the same command
    takes ~30s offline and several minutes against a cluster -- so an audit
    needs the credential and the duration in one place.
    """
    targets = resolved_live_targets()
    if not targets:
        return
    elapsed = time.time() - config.stash.get(_START_TIME, time.time())
    workspaces = ", ".join(workspace for _, workspace in targets)
    terminalreporter.write_line(
        f"live cluster exposure: credentials for {workspaces} on "
        f"{HYPHA_SERVER_URL} were in scope for this {elapsed:.1f}s run "
        f"({'--live GIVEN' if config.getoption('--live') else '--live not given'}).",
        red=True,
    )


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


@pytest.fixture(scope="session", autouse=True)
def no_leaked_working_directory() -> Generator[None, None, None]:
    """Fail the session if a test left the process in another directory.

    An unrestored ``os.chdir`` is silent until some later test opens a
    relative path or spawns a child that inherits the cwd, and the failure is
    then charged to that later test -- which passes again when run alone.

    A test that needs to run elsewhere should use ``monkeypatch.chdir``;
    anything else lands here as a teardown error charged to whichever test
    happened to run last.
    """
    entry = os.getcwd()
    yield
    try:
        final = os.getcwd()
    except OSError as e:
        # A deleted cwd is the same defect; without this it surfaces as a bare
        # FileNotFoundError from the fixture and names nothing.
        final = f"a directory that no longer exists ({e})"
    if final != entry:
        pytest.fail(
            f"a test left the process in {final}; the session started in "
            f"{entry}. Use monkeypatch.chdir, or restore it where it changes."
        )


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
    return HYPHA_SERVER_URL


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


_UNAWAITED_DESTRUCTOR = "coroutine 'ProxyDeployment.__del__' was never awaited"
_unawaited_destructor_nodeids: list = []


def pytest_warning_recorded(warning_message, nodeid, **_):
    """Record proxies left for the collector to finalise.

    Ray Serve awaits ``ProxyDeployment.__del__``; CPython's collector calls it,
    gets a coroutine and discards it. The warning is charged to whichever test
    was running when the collector tripped, not to the one that built the
    object, so it has to be caught session-wide rather than per test.
    """
    if _UNAWAITED_DESTRUCTOR in str(warning_message.message):
        _unawaited_destructor_nodeids.append(nodeid)


def pytest_sessionfinish(session) -> None:
    if not _unawaited_destructor_nodeids:
        return
    # Only claim a passing run: INTERRUPTED and the error statuses say more.
    if session.exitstatus == 0:
        session.exitstatus = 1
    print(
        f"\n{len(_unawaited_destructor_nodeids)} un-awaited ProxyDeployment "
        f"destructor(s). Build test proxies from tests.apps._proxy_double."
        f"ProxyDouble. Charged to: "
        f"{', '.join(sorted(set(_unawaited_destructor_nodeids)))}"
    )


# Configure asyncio for pytest
pytest_plugins = ("pytest_asyncio",)
