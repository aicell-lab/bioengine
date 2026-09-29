"""
bioengine worker — start, stop and follow a BioEngine worker container.

Wraps the container invocation from docs/deployment-guide.md. Worker arguments
are forwarded verbatim to ``python -m bioengine.worker`` inside the image, so
this module never has to know what they are.

Examples:
  bioengine worker start -- --mode single-machine --head-num-cpus 4
  bioengine worker start --dry-run -- --mode single-machine
  bioengine worker logs -f
  bioengine worker stop
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import click

from bioengine import __version__
from bioengine.cli.utils import error_exit

DEFAULT_IMAGE_REPO = "ghcr.io/aicell-lab/bioengine-worker"
DEFAULT_CONTAINER_NAME = "bioengine-worker"
DEFAULT_WORKSPACE_DIR = Path.home() / ".bioengine"
# The path the image mounts the workspace at — see docs/deployment-guide.md.
CONTAINER_WORKSPACE_DIR = "/.bioengine"
DEFAULT_SHM_SIZE = "8g"

# Only the GPU flag differs between docker and podman (deployment-guide.md).
_GPU_FLAGS = {
    "docker": ["--gpus=all"],
    "podman": ["--device", "nvidia.com/gpu=all"],
    "apptainer": ["--nv"],
}

_RUNTIME_PREFERENCE = ("docker", "podman", "apptainer")


def _passthrough_env(token: Optional[str], server_url: Optional[str]) -> dict:
    """The variables the container needs, keyed by name.

    Built from the resolved option values, not from ``os.environ`` — ``--token``
    only reaches the environment later, in ``_subprocess_env``.
    """
    resolved = (("HYPHA_TOKEN", token), ("BIOENGINE_SERVER_URL", server_url))
    return {name: value for name, value in resolved if value}


def _detect_runtime() -> Optional[str]:
    for runtime in _RUNTIME_PREFERENCE:
        if shutil.which(runtime):
            return runtime
    return None


def _has_gpu() -> bool:
    """Whether to pass a GPU flag by default.

    Passing ``--gpus=all`` on a host without the NVIDIA container toolkit makes
    the runtime refuse to start, so it cannot simply be on by default.
    """
    return shutil.which("nvidia-smi") is not None


def build_command(
    runtime: str,
    image: str,
    worker_args: Tuple[str, ...],
    workspace_dir: Path,
    container_name: str,
    shm_size: str,
    gpus: bool,
    detach: bool,
    tty: bool,
    token: Optional[str],
    server_url: Optional[str],
) -> List[str]:
    """Build the container invocation. Secrets travel in the environment, never argv."""
    entrypoint = ["python", "-m", "bioengine.worker", *worker_args]

    if runtime == "native":
        return entrypoint

    if runtime == "apptainer":
        command = ["apptainer", "exec"]
        if gpus:
            command += _GPU_FLAGS["apptainer"]
        command += ["--bind", f"{workspace_dir}:{CONTAINER_WORKSPACE_DIR}"]
        return command + [f"docker://{image}", *entrypoint]

    command = [runtime, "run", "--rm"]
    command += ["--detach"] if detach else ["-it" if tty else "-i"]
    command += ["--name", container_name]
    command += ["--user", f"{os.getuid()}:{os.getgid()}"]
    command += ["--shm-size", shm_size]
    if gpus:
        command += _GPU_FLAGS[runtime]
    command += ["-v", f"{workspace_dir}:{CONTAINER_WORKSPACE_DIR}"]
    for name in _passthrough_env(token, server_url):
        command += ["-e", name]
    return command + [image, *entrypoint]


_SECRET_OPTION_HINTS = ("token", "password", "secret")


def _is_secret_option(arg: str) -> bool:
    return arg.startswith("-") and any(hint in arg.lower() for hint in _SECRET_OPTION_HINTS)


def redact_secrets(command: List[str]) -> List[str]:
    """Mask credential-bearing option values so the command can be printed."""
    redacted = []
    mask_next = False
    for arg in command:
        if mask_next:
            redacted.append("<redacted>")
            mask_next = False
            continue
        name, separator, _ = arg.partition("=")
        if separator and _is_secret_option(name):
            redacted.append(f"{name}=<redacted>")
            continue
        if _is_secret_option(arg):
            mask_next = True
        redacted.append(arg)
    return redacted


def _container_exists(runtime: str, container_name: str) -> bool:
    result = subprocess.run(
        [runtime, "ps", "--all", "--quiet", "--filter", f"name=^{container_name}$"],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _subprocess_env(runtime: str, token: Optional[str], server_url: Optional[str]) -> dict:
    passthrough = _passthrough_env(token, server_url)
    env = dict(os.environ)
    env.update(passthrough)
    if runtime == "apptainer":
        # Apptainer only forwards host variables it is told about explicitly.
        env.update({f"APPTAINERENV_{name}": value for name, value in passthrough.items()})
    return env


def _resolve_runtime(runtime: str, require_available: bool = True) -> str:
    """Resolve 'auto'. ``require_available`` is off for --dry-run, whose whole
    point is producing a command to run somewhere else."""
    if runtime != "auto":
        if require_available and runtime != "native" and not shutil.which(runtime):
            error_exit(
                f"Container runtime '{runtime}' is not on PATH.",
                "Install it, or pass --runtime native to run the worker in this environment.",
            )
        return runtime

    detected = _detect_runtime()
    if not detected:
        if not require_available:
            return _RUNTIME_PREFERENCE[0]
        error_exit(
            "No container runtime found (looked for docker, podman, apptainer).",
            "Install one, or pass --runtime native to run the worker in this environment.",
        )
    return detected


@click.group("worker")
def worker_group():
    """Start, stop and follow a BioEngine worker container."""


@worker_group.command(
    "start",
    context_settings={"ignore_unknown_options": True},
)
@click.argument("worker_args", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--runtime",
    type=click.Choice(["auto", "docker", "podman", "apptainer", "native"]),
    default="auto",
    help="Container runtime. 'auto' picks the first of docker, podman, apptainer on PATH. "
    "'native' runs the worker in this environment instead (requires the 'worker' extra).",
)
@click.option(
    "--image",
    default=None,
    metavar="IMAGE",
    help=f"Worker image (default: {DEFAULT_IMAGE_REPO}:<installed bioengine version>).",
)
@click.option(
    "--workspace-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_WORKSPACE_DIR,
    show_default=True,
    help="Host directory mounted as the worker workspace.",
)
@click.option(
    "--name",
    "container_name",
    default=DEFAULT_CONTAINER_NAME,
    show_default=True,
    help="Container name, used by 'bioengine worker stop' and 'logs'.",
)
@click.option(
    "--shm-size", default=DEFAULT_SHM_SIZE, show_default=True, help="Shared memory size."
)
@click.option(
    "--gpus/--no-gpus",
    default=None,
    help="Request GPUs. Defaults to on when nvidia-smi is present.",
)
@click.option("--detach", "-d", is_flag=True, help="Run the container in the background.")
@click.option(
    "--token",
    envvar=["HYPHA_TOKEN", "BIOENGINE_TOKEN"],
    default=None,
    metavar="TOKEN",
    help="Hypha auth token (or HYPHA_TOKEN env var). Passed via the environment, not the command line.",
)
@click.option("--server-url", envvar="BIOENGINE_SERVER_URL", default=None, hidden=True)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the command instead of running it. Credential values are masked.",
)
def worker_start(
    worker_args,
    runtime,
    image,
    workspace_dir,
    container_name,
    shm_size,
    gpus,
    detach,
    token,
    server_url,
    dry_run,
):
    """
    Start a BioEngine worker.

    Everything after ``--`` is forwarded verbatim to ``python -m bioengine.worker``
    inside the container, so every worker option is available without this command
    knowing about it. Run ``bioengine worker start -- --help`` to see them.

    \b
    Examples:
      bioengine worker start -- --mode single-machine --head-num-cpus 4
      bioengine worker start -d --no-gpus -- --mode single-machine
      bioengine worker start --dry-run -- --mode single-machine
    """
    runtime = _resolve_runtime(runtime, require_available=not dry_run)

    if gpus is None:
        gpus = _has_gpu()

    # Pinned to the installed version so the CLI and the worker it starts cannot
    # silently diverge.
    image = image or f"{DEFAULT_IMAGE_REPO}:{__version__}"

    workspace_dir = workspace_dir.expanduser()
    if runtime != "native" and not dry_run:
        workspace_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(
        runtime=runtime,
        image=image,
        worker_args=worker_args,
        workspace_dir=workspace_dir,
        container_name=container_name,
        shm_size=shm_size,
        gpus=gpus,
        detach=detach,
        tty=sys.stdin.isatty(),
        token=token,
        server_url=server_url,
    )

    if dry_run:
        click.echo(" ".join(redact_secrets(command)))
        return

    if runtime in ("docker", "podman") and _container_exists(runtime, container_name):
        error_exit(
            f"A container named '{container_name}' already exists.",
            f"Stop it with 'bioengine worker stop --name {container_name}', "
            f"or start this one under a different --name.",
        )

    env = _subprocess_env(runtime, token, server_url)
    try:
        raise SystemExit(subprocess.call(command, env=env))
    except FileNotFoundError:
        error_exit(f"Failed to execute '{command[0]}': not found on PATH.")


@worker_group.command("stop")
@click.option(
    "--runtime",
    type=click.Choice(["auto", "docker", "podman"]),
    default="auto",
    help="Container runtime. 'auto' picks the first of docker, podman on PATH.",
)
@click.option(
    "--name",
    "container_name",
    default=DEFAULT_CONTAINER_NAME,
    show_default=True,
    help="Container name.",
)
def worker_stop(runtime, container_name):
    """Stop a running BioEngine worker container."""
    runtime = _resolve_runtime(runtime)
    if runtime not in ("docker", "podman"):
        error_exit(
            f"'{runtime}' has no named containers to stop.",
            "Stop the worker process directly.",
        )
    raise SystemExit(subprocess.call([runtime, "stop", container_name]))


@worker_group.command("logs")
@click.option(
    "--runtime",
    type=click.Choice(["auto", "docker", "podman"]),
    default="auto",
    help="Container runtime. 'auto' picks the first of docker, podman on PATH.",
)
@click.option(
    "--name",
    "container_name",
    default=DEFAULT_CONTAINER_NAME,
    show_default=True,
    help="Container name.",
)
@click.option("--follow", "-f", is_flag=True, help="Follow log output.")
@click.option("--tail", default=None, metavar="N", help="Show only the last N lines.")
def worker_logs(runtime, container_name, follow, tail):
    """Show the logs of a running BioEngine worker container."""
    runtime = _resolve_runtime(runtime)
    if runtime not in ("docker", "podman"):
        error_exit(
            f"'{runtime}' has no named containers to read logs from.",
            "Read the worker's own log file under the workspace directory instead.",
        )
    command = [runtime, "logs"]
    if follow:
        command.append("-f")
    if tail:
        command += ["--tail", str(tail)]
    raise SystemExit(subprocess.call(command + [container_name]))
