"""
Contract for ``bioengine worker`` — the container launcher.

The command's whole job is turning options into an argv list, so these tests
pin that argv: the shape documented in docs/deployment-guide.md for each
runtime, worker arguments forwarded verbatim, and the auth token never
appearing on a command line where `ps` would show it.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from bioengine import __version__
from bioengine.cli.worker import (
    CONTAINER_WORKSPACE_DIR,
    DEFAULT_IMAGE_REPO,
    _has_gpu,
    _subprocess_env,
    build_command,
    redact_secrets,
    worker_group,
)

WORKSPACE = Path("/home/someone/.bioengine")
IMAGE = f"{DEFAULT_IMAGE_REPO}:{__version__}"
WORKER_ARGS = ("--mode", "single-machine", "--head-num-cpus", "4")
# Not a credential — a literal that must never survive into printed output.
FAKE_TOKEN = "FAKE-TOKEN-NOT-A-CREDENTIAL"
FAKE_REDIS_PASSWORD = "FAKE-REDIS-PASSWORD"


def _build(runtime, **overrides):
    kwargs = dict(
        runtime=runtime,
        image=IMAGE,
        worker_args=WORKER_ARGS,
        workspace_dir=WORKSPACE,
        container_name="bioengine-worker",
        shm_size="8g",
        gpus=False,
        detach=False,
        tty=True,
        token=None,
        server_url=None,
    )
    kwargs.update(overrides)
    return build_command(**kwargs)


def _run(args, env=None):
    """Invoke the CLI with the credential variables cleared unless a test sets them."""
    base = {"HYPHA_TOKEN": "", "BIOENGINE_SERVER_URL": "", "BIOENGINE_TOKEN": ""}
    base.update(env or {})
    return CliRunner().invoke(worker_group, args, env=base)


# ── The documented invocation, per runtime ────────────────────────────────────


def test_the_docker_command_matches_the_deployment_guide():
    command = _build("docker", gpus=True)
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--user" in command and f"{os.getuid()}:{os.getgid()}" in command
    assert command[command.index("--shm-size") + 1] == "8g"
    assert "--gpus=all" in command
    assert f"{WORKSPACE}:{CONTAINER_WORKSPACE_DIR}" in command
    assert command[-len(WORKER_ARGS) - 4 :] == [
        IMAGE,
        "python",
        "-m",
        "bioengine.worker",
        *WORKER_ARGS,
    ]


def test_podman_uses_its_own_gpu_flag():
    command = _build("podman", gpus=True)
    assert command[:2] == ["podman", "run"]
    assert "--gpus=all" not in command
    assert ["--device", "nvidia.com/gpu=all"] == command[
        command.index("--device") : command.index("--device") + 2
    ]


def test_apptainer_binds_instead_of_mounting():
    command = _build("apptainer", gpus=True)
    assert command[:2] == ["apptainer", "exec"]
    assert "--nv" in command
    assert command[command.index("--bind") + 1] == f"{WORKSPACE}:{CONTAINER_WORKSPACE_DIR}"
    assert f"docker://{IMAGE}" in command
    # No container to name, detach or size — those flags belong to docker/podman.
    for flag in ("--name", "--detach", "--shm-size", "--user"):
        assert flag not in command


def test_native_runs_the_worker_without_a_container():
    command = _build("native", gpus=True)
    assert command == ["python", "-m", "bioengine.worker", *WORKER_ARGS]


def test_the_gpu_flag_is_omitted_when_gpus_are_off():
    for runtime in ("docker", "podman", "apptainer"):
        command = _build(runtime, gpus=False)
        assert "--gpus=all" not in command
        assert "--device" not in command
        assert "--nv" not in command


def test_detaching_replaces_the_interactive_flags():
    assert "-it" in _build("docker")
    detached = _build("docker", detach=True)
    assert "--detach" in detached and "-it" not in detached


def test_no_tty_is_requested_when_stdin_is_not_one():
    command = _build("docker", tty=False)
    assert "-it" not in command
    assert "-i" in command


def test_a_non_interactive_invocation_does_not_ask_for_a_tty():
    """CliRunner's stdin is not a terminal, the same as CI, nohup or systemd."""
    result = _run(["start", "--runtime", "docker", "--dry-run", "--", "--mode", "single-machine"])
    assert result.exit_code == 0, result.output
    tokens = result.output.split()
    assert "-it" not in tokens
    assert "-i" in tokens


# ── The token must never reach argv ───────────────────────────────────────────


def test_the_token_is_named_not_valued_in_the_container_command():
    for runtime in ("docker", "podman"):
        command = _build(runtime, token=FAKE_TOKEN)
        assert FAKE_TOKEN not in command
        assert command[command.index("-e") + 1] == "HYPHA_TOKEN"


def test_the_token_never_appears_in_any_runtimes_command():
    for runtime in ("docker", "podman", "apptainer", "native"):
        assert FAKE_TOKEN not in " ".join(_build(runtime, token=FAKE_TOKEN))


def test_the_token_is_passed_through_the_environment():
    env = _subprocess_env("docker", token=FAKE_TOKEN, server_url=None)
    assert env["HYPHA_TOKEN"] == FAKE_TOKEN


def test_apptainer_needs_the_prefixed_variable_to_forward_anything():
    env = _subprocess_env("apptainer", token=FAKE_TOKEN, server_url=None)
    assert env["APPTAINERENV_HYPHA_TOKEN"] == FAKE_TOKEN


def test_an_unset_variable_is_not_forwarded():
    assert "HYPHA_TOKEN" not in _build("docker")
    assert "APPTAINERENV_HYPHA_TOKEN" not in _subprocess_env("apptainer", None, None)


def test_a_dry_run_does_not_print_a_secret_worker_argument():
    """``python -m bioengine.worker`` takes --token and --redis-password; the
    printed command must not carry their values into scrollback or CI logs."""
    result = _run(
        [
            "start",
            "--runtime",
            "docker",
            "--dry-run",
            "--",
            "--mode",
            "single-machine",
            "--token",
            FAKE_TOKEN,
            "--redis-password",
            FAKE_REDIS_PASSWORD,
        ]
    )
    assert result.exit_code == 0, result.output
    assert FAKE_TOKEN not in result.output
    assert FAKE_REDIS_PASSWORD not in result.output
    assert "--token <redacted>" in result.output
    assert "--redis-password <redacted>" in result.output


def test_an_inline_secret_argument_is_redacted_too():
    assert redact_secrets(["docker", f"--token={FAKE_TOKEN}"]) == ["docker", "--token=<redacted>"]


def test_redaction_keeps_the_named_environment_passthrough_readable():
    assert redact_secrets(["-e", "HYPHA_TOKEN"]) == ["-e", "HYPHA_TOKEN"]


def test_a_dry_run_still_prints_a_runnable_command():
    result = _run(["start", "--runtime", "docker", "--dry-run", "--", "--mode", "single-machine"])
    assert result.output.startswith("docker run --rm")
    assert IMAGE in result.output


# ── Worker arguments are forwarded, not interpreted ───────────────────────────


def test_worker_arguments_are_forwarded_verbatim():
    args = ("--mode", "slurm", "--admin-users", "a@x.org,b@y.org", "--debug")
    assert _build("native", worker_args=args)[3:] == list(args)


def test_an_option_the_cli_also_defines_still_reaches_the_worker(monkeypatch):
    """``--workspace-dir`` after ``--`` configures the worker, not the container.

    Asserted against the argv the runtime was called with rather than against
    ``result.output``: click 8.2 dropped ``mix_stderr``, so ``output`` is stdout
    and stderr merged, and anything else the process writes to stderr inside the
    invoke window would land in it.
    """
    started = _fake_runtime(monkeypatch, "")
    result = _run(["start", "--runtime", "native", "--", "--workspace-dir", "/data/ws"])
    assert result.exit_code == 0, result.output
    assert started[0][0] == [
        "python",
        "-m",
        "bioengine.worker",
        "--workspace-dir",
        "/data/ws",
    ]


def test_no_worker_arguments_still_starts_the_worker_module():
    assert _build("native", worker_args=())[-1] == "bioengine.worker"


# ── The CLI surface ───────────────────────────────────────────────────────────


def test_the_image_is_pinned_to_the_installed_version():
    result = _run(["start", "--runtime", "docker", "--dry-run", "--", "--mode", "single-machine"])
    assert result.exit_code == 0, result.output
    assert f"{DEFAULT_IMAGE_REPO}:{__version__}" in result.output


def test_a_dry_run_neither_creates_the_workspace_nor_needs_the_runtime(tmp_path):
    workspace = tmp_path / "never-created"
    result = _run(
        [
            "start",
            "--runtime",
            "podman",
            "--workspace-dir",
            str(workspace),
            "--dry-run",
            "--",
            "--mode",
            "single-machine",
        ]
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith("podman run")
    assert not workspace.exists()


def test_a_missing_runtime_is_refused_when_actually_starting(monkeypatch):
    monkeypatch.setattr("bioengine.cli.worker.shutil.which", lambda _: None)
    result = _run(["start", "--runtime", "podman", "--", "--mode", "single-machine"])
    assert result.exit_code == 1
    assert "not on PATH" in result.output


def test_stop_and_logs_refuse_runtimes_without_named_containers(monkeypatch):
    """On an apptainer-only host there is no container name to act on."""
    monkeypatch.setattr(
        "bioengine.cli.worker.shutil.which", lambda name: name if name == "apptainer" else None
    )
    for command in ("stop", "logs"):
        result = _run([command])
        assert result.exit_code == 1
        assert "no named containers" in result.output


@pytest.mark.parametrize("command", ["start", "stop", "logs"])
def test_every_subcommand_is_reachable(command):
    result = _run([command, "--help"])
    assert result.exit_code == 0
    assert "bioengine-worker" in result.output or "worker" in result.output


# ── A second worker must not silently collide with the first ──────────────────


def _fake_runtime(monkeypatch, existing_container: str):
    monkeypatch.setattr("bioengine.cli.worker.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        "bioengine.cli.worker.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=existing_container),
    )
    started = []
    monkeypatch.setattr(
        "bioengine.cli.worker.subprocess.call",
        lambda command, **kwargs: started.append((command, kwargs["env"])) or 0,
    )
    return started


def test_starting_a_second_worker_under_the_same_name_is_refused(monkeypatch, tmp_path):
    started = _fake_runtime(monkeypatch, "9f1c2b3d4e5f\n")
    result = _run(
        [
            "start",
            "--runtime",
            "docker",
            "--workspace-dir",
            str(tmp_path / "ws"),
            "--",
            "--mode",
            "single-machine",
        ]
    )
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert "bioengine worker stop --name bioengine-worker" in result.output
    assert started == []


def test_a_free_container_name_starts_the_worker(monkeypatch, tmp_path):
    started = _fake_runtime(monkeypatch, "")
    result = _run(
        [
            "start",
            "--runtime",
            "docker",
            "--workspace-dir",
            str(tmp_path / "ws"),
            "--",
            "--mode",
            "single-machine",
        ]
    )
    assert result.exit_code == 0, result.output
    assert started and started[0][0][:3] == ["docker", "run", "--rm"]


# ── The token reaches the container, however it was supplied ──────────────────
#
# Asserting on build_command or _subprocess_env alone passes on either side of
# the seam between them: build_command decides which variables the container is
# told about, _subprocess_env supplies their values. These tests start the
# worker for real — only the subprocess call is faked — with no credential in
# the environment, and read what the container would actually receive.


def _container_environment(runtime, command, env):
    """What the worker process inside the container will see.

    docker and podman forward a variable named by ``-e`` from the launcher's own
    environment; apptainer forwards the ``APPTAINERENV_``-prefixed copy instead.
    """
    if runtime == "apptainer":
        prefix = "APPTAINERENV_"
        return {
            name[len(prefix) :]: value for name, value in env.items() if name.startswith(prefix)
        }
    return {name: env.get(name) for flag, name in zip(command, command[1:]) if flag == "-e"}


def _start(monkeypatch, tmp_path, runtime, extra_args=(), env=None):
    """Start a worker with the runtime faked out; return its argv and environment."""
    started = _fake_runtime(monkeypatch, "")
    result = _run(
        [
            "start",
            "--runtime",
            runtime,
            "--workspace-dir",
            str(tmp_path / "ws"),
            *extra_args,
            "--",
            "--mode",
            "single-machine",
        ],
        env=env,
    )
    assert result.exit_code == 0, result.output
    assert len(started) == 1
    return started[0]


@pytest.mark.parametrize("runtime", ["docker", "podman", "apptainer"])
def test_the_token_option_reaches_the_container(monkeypatch, tmp_path, runtime):
    command, env = _start(monkeypatch, tmp_path, runtime, ["--token", FAKE_TOKEN])
    assert _container_environment(runtime, command, env)["HYPHA_TOKEN"] == FAKE_TOKEN
    assert FAKE_TOKEN not in command


@pytest.mark.parametrize("runtime", ["docker", "podman", "apptainer"])
@pytest.mark.parametrize("source", ["HYPHA_TOKEN", "BIOENGINE_TOKEN"])
def test_either_token_environment_variable_reaches_the_container(
    monkeypatch, tmp_path, runtime, source
):
    command, env = _start(monkeypatch, tmp_path, runtime, env={source: FAKE_TOKEN})
    assert _container_environment(runtime, command, env)["HYPHA_TOKEN"] == FAKE_TOKEN


def test_the_server_url_reaches_the_container_too(monkeypatch, tmp_path):
    command, env = _start(
        monkeypatch, tmp_path, "docker", ["--server-url", "https://hypha.example.org"]
    )
    container = _container_environment("docker", command, env)
    assert container["BIOENGINE_SERVER_URL"] == "https://hypha.example.org"


def test_no_token_means_no_passthrough(monkeypatch, tmp_path):
    command, env = _start(monkeypatch, tmp_path, "docker")
    assert "HYPHA_TOKEN" not in _container_environment("docker", command, env)


def test_the_token_is_masked_in_a_dry_run_and_delivered_in_a_real_one(monkeypatch, tmp_path):
    printed = _run(
        [
            "start",
            "--runtime",
            "docker",
            "--token",
            FAKE_TOKEN,
            "--dry-run",
            "--",
            "--mode",
            "single-machine",
        ]
    )
    assert printed.exit_code == 0, printed.output
    assert FAKE_TOKEN not in printed.output
    assert "-e HYPHA_TOKEN" in printed.output

    command, env = _start(monkeypatch, tmp_path, "docker", ["--token", FAKE_TOKEN])
    assert _container_environment("docker", command, env)["HYPHA_TOKEN"] == FAKE_TOKEN


# ── GPUs are requested only when the host can serve them ──────────────────────


def _fake_nvidia_smi(monkeypatch, present: bool):
    monkeypatch.setattr(
        "bioengine.cli.worker.shutil.which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" and present else None,
    )


@pytest.mark.parametrize("nvidia_smi_present", [True, False])
def test_nvidia_smi_decides_whether_gpus_are_requested_by_default(monkeypatch, nvidia_smi_present):
    """Neither --gpus nor --no-gpus is given, so the default consults _has_gpu()."""
    _fake_nvidia_smi(monkeypatch, nvidia_smi_present)
    assert _has_gpu() is nvidia_smi_present

    result = _run(["start", "--runtime", "docker", "--dry-run", "--", "--mode", "single-machine"])
    assert result.exit_code == 0, result.output
    assert ("--gpus=all" in result.output) is nvidia_smi_present


@pytest.mark.parametrize("nvidia_smi_present", [True, False])
def test_an_explicit_gpu_choice_overrides_the_host(monkeypatch, nvidia_smi_present):
    _fake_nvidia_smi(monkeypatch, nvidia_smi_present)
    for flag, expected in (("--gpus", True), ("--no-gpus", False)):
        result = _run(
            ["start", "--runtime", "docker", flag, "--dry-run", "--", "--mode", "single-machine"]
        )
        assert result.exit_code == 0, result.output
        assert ("--gpus=all" in result.output) is expected


# --- singularity, apptainer's predecessor ------------------------------------
#
# The deployment guide has always said "Apptainer / Singularity", but
# --runtime singularity was rejected by the Choice. The CLI surface this
# launcher uses is identical; only the env-forwarding prefix differs.

def test_singularity_is_an_accepted_runtime():
    from bioengine.cli.worker import _RUNTIME_PREFERENCE

    assert "singularity" in _RUNTIME_PREFERENCE


def test_singularity_takes_the_apptainer_command_shape():
    singularity = _build("singularity", gpus=True)
    apptainer = _build("apptainer", gpus=True)
    assert singularity[0] == "singularity"
    assert apptainer[0] == "apptainer"
    # Identical apart from the binary name — if they ever diverge, that is a
    # decision someone should have to make explicitly.
    assert singularity[1:] == apptainer[1:]


def test_singularity_binds_rather_than_mounting():
    command = _build("singularity", gpus=False)
    assert command[:2] == ["singularity", "exec"]
    assert "--bind" in command
    assert "-v" not in command
    assert f"docker://{IMAGE}" in command


def test_singularity_gets_the_nv_flag_only_with_gpus():
    # Position, not mere presence. _GPU_FLAGS carries "--nv" for singularity
    # regardless of which branch builds the command, so `"--nv" in command`
    # also passes when singularity wrongly takes the docker `run` shape — the
    # test would be green while naming a property that had been broken.
    assert _build("singularity", gpus=True)[:3] == ["singularity", "exec", "--nv"]
    assert _build("singularity", gpus=False)[:2] == ["singularity", "exec"]
    assert "--nv" not in _build("singularity", gpus=False)


def test_each_sif_runtime_uses_its_own_env_prefix():
    # Apptainer honours APPTAINERENV_; singularity honours SINGULARITYENV_.
    # Forwarding under the wrong prefix is silent: the variable simply never
    # arrives, and the worker starts with no token.
    apptainer_env = _subprocess_env("apptainer", FAKE_TOKEN, None)
    singularity_env = _subprocess_env("singularity", FAKE_TOKEN, None)
    assert apptainer_env["APPTAINERENV_HYPHA_TOKEN"] == FAKE_TOKEN
    assert singularity_env["SINGULARITYENV_HYPHA_TOKEN"] == FAKE_TOKEN
    assert "SINGULARITYENV_HYPHA_TOKEN" not in apptainer_env
    assert "APPTAINERENV_HYPHA_TOKEN" not in singularity_env


def test_docker_and_podman_get_no_sif_prefixes():
    for runtime in ("docker", "podman"):
        env = _subprocess_env(runtime, FAKE_TOKEN, None)
        assert env["HYPHA_TOKEN"] == FAKE_TOKEN
        assert not [k for k in env if k.startswith(("APPTAINERENV_", "SINGULARITYENV_"))]
