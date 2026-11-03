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
    DEFAULT_MEMORY_FRACTION,
    _has_gpu,
    _subprocess_env,
    build_command,
    redact_secrets,
    resolve_memory,
    worker_group,
)
from bioengine.utils import read_meminfo

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
        memory=None,
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


# --- NVIDIA device nodes in the OCI spec -------------------------------------
#
# `--gpus=all` records only a DeviceRequest; the nvidia-container-runtime hook
# patches the device cgroup out of band, so the OCI spec never names the nodes.
# A cgroup re-apply rebuilds the allowlist from that spec and the container
# silently loses the GPU. These pin the nodes into the command instead — and
# pin the set to exactly what the hook injects, no wider.

import stat as _stat

# Two GPUs, the control nodes, and the MIG caps the hook does NOT inject.
_CHAR_NODES = {
    "/dev/nvidia0",
    "/dev/nvidia1",
    "/dev/nvidiactl",
    "/dev/nvidia-modeset",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-caps/nvidia-cap1",
    "/dev/nvidia-caps/nvidia-cap2",
}
_DIR_NODES = {"/dev/nvidia-caps"}


@pytest.fixture
def fake_nvidia_host(monkeypatch):
    """A host with two GPUs, the control nodes, and a caps directory."""
    from bioengine.cli import worker as worker_module

    everything = sorted(_CHAR_NODES | _DIR_NODES)

    def fake_glob(pattern):
        import fnmatch

        return [p for p in everything if fnmatch.fnmatch(p, pattern)]

    # Delegate anything that is not one of our fakes to the real os.stat, and
    # keep the real signature. Patching os.stat is process-wide: a fake that
    # swallows every path (or drops follow_symlinks) breaks pytest's own
    # teardown, which calls Path.exists() long after the test has finished.
    real_stat = os.stat

    def fake_stat(path, *args, **kwargs):
        name = str(path)
        if name in _CHAR_NODES:
            return os.stat_result((_stat.S_IFCHR, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        if name in _DIR_NODES:
            return os.stat_result((_stat.S_IFDIR, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(worker_module.glob, "glob", fake_glob)
    monkeypatch.setattr(worker_module.os, "stat", fake_stat)


def _devices(command):
    return [a[len("--device=") :] for a in command if a.startswith("--device=")]


def test_docker_names_the_nodes_the_hook_injects(fake_nvidia_host):
    assert _devices(_build("docker", gpus=True)) == [
        "/dev/nvidia0",
        "/dev/nvidia1",
        "/dev/nvidiactl",
        "/dev/nvidia-modeset",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
    ]


def test_the_mig_capability_nodes_are_not_granted(fake_nvidia_host):
    # Measured: `--gpus=all` alone does NOT grant /dev/nvidia-caps, so naming
    # them protects nothing a cgroup rebuild could drop. It would hand out the
    # MIG-configuration capability that nvidia-container-toolkit deliberately
    # gates behind NVIDIA_MIG_CONFIG_DEVICES. The fix must be no wider than the
    # bug.
    granted = _devices(_build("docker", gpus=True))
    assert not [node for node in granted if "nvidia-caps" in node]


def test_the_caps_directory_is_not_passed_as_a_device(fake_nvidia_host):
    # Belt and braces: the directory is outside the glob set now, and the
    # character-device check would reject it even if a glob reached it.
    assert "/dev/nvidia-caps" not in _devices(_build("docker", gpus=True))


def test_a_node_that_is_not_a_character_device_is_skipped(monkeypatch):
    # Handing `docker run` a non-device makes it hard-fail with "not a device
    # node", so the filter has to be a real check rather than decoration. The
    # narrowed globs mean no realistic host hits this, which is exactly why it
    # needs a test: without one, deleting the filter changes no verdict.
    from bioengine.cli import worker as worker_module

    real_stat = os.stat

    def fake_stat(path, *args, **kwargs):
        if str(path) == "/dev/nvidia0":
            return os.stat_result((_stat.S_IFREG, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        if str(path) == "/dev/nvidiactl":
            return os.stat_result((_stat.S_IFCHR, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        return real_stat(path, *args, **kwargs)

    # Pin a single pattern of our own rather than keying on the production
    # tuple, so this test stays about the filter and does not go red merely
    # because the glob set was retuned.
    monkeypatch.setattr(worker_module, "_NVIDIA_DEV_GLOBS", ("/fake/nvidia*",))
    monkeypatch.setattr(
        worker_module.glob, "glob", lambda pattern: ["/dev/nvidia0", "/dev/nvidiactl"]
    )
    monkeypatch.setattr(worker_module.os, "stat", fake_stat)

    assert _devices(_build("docker", gpus=True)) == ["/dev/nvidiactl"]


def test_the_device_flags_do_not_replace_the_gpu_flag(fake_nvidia_host):
    # Measured: a container given the nodes WITHOUT --gpus fails at
    # "libcuda.so.1: cannot open shared object file" — the driver libraries
    # come from the hook, which only runs for --gpus. The two are a pair.
    command = _build("docker", gpus=True)
    assert "--gpus=all" in command
    assert _devices(command)


def test_no_device_flags_when_gpus_are_off(fake_nvidia_host):
    assert not _devices(_build("docker", gpus=False))


def test_a_host_with_no_nvidia_nodes_adds_nothing(monkeypatch):
    from bioengine.cli import worker as worker_module

    monkeypatch.setattr(worker_module.glob, "glob", lambda pattern: [])
    command = _build("docker", gpus=True)
    assert not _devices(command)
    assert "--gpus=all" in command


def test_podman_and_apptainer_are_untouched(fake_nvidia_host):
    # podman's CDI reference already names the devices in the spec — measured:
    # `nvidia-ctk cdi generate` emits explicit deviceNodes and its only hook is
    # update-ldcache, a library-path edit rather than a cgroup patch. apptainer
    # does not use the device cgroup this way. Widening either is scope creep.
    assert not _devices(_build("podman", gpus=True))
    assert not _devices(_build("apptainer", gpus=True))


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


def test_every_detected_runtime_is_also_an_accepted_choice():
    """The --runtime choices and the detection preference must agree.

    They drifted once already: the guide and the SLURM prerequisites offered
    Singularity while the Choice refused it. Adding a runtime to
    _RUNTIME_PREFERENCE alone makes `auto` detect and run something the flag
    cannot name, while the help text advertises it. Reproduced with a fake
    entry and the suite stayed green, because nothing compared the two lists.
    """
    from bioengine.cli.worker import _RUNTIME_PREFERENCE

    choice = next(
        p.type for p in worker_group.commands["start"].params if p.name == "runtime"
    )
    accepted = set(choice.choices)
    assert set(_RUNTIME_PREFERENCE) <= accepted, (
        "runtimes `auto` can detect but --runtime refuses: "
        f"{sorted(set(_RUNTIME_PREFERENCE) - accepted)}"
    )
    assert {"auto", "native"} <= accepted


# ── Container memory limit (svamp #0125) ─────────────────────────────────────
#
# Ray sizes its memory monitor from `min(cgroup limit, host total)`. With no
# cgroup limit it reads the host's total, so an unlimited worker only starts
# shedding its own tasks at 95% of the WHOLE MACHINE — by which point the
# kernel's OOM killer has usually acted first, and it picks Ray's actors
# because Ray sets oom_score_adj=1000 on them. A limit moves that budget onto
# the container, where it can act early and locally.


def test_the_memory_limit_reaches_the_runtime():
    command = _build("docker", memory="32g")
    assert command[command.index("--memory") + 1] == "32g"


def test_podman_takes_the_memory_limit_too():
    command = _build("podman", memory="32g")
    assert command[command.index("--memory") + 1] == "32g"


def test_no_memory_limit_means_no_flag():
    assert "--memory" not in _build("docker", memory=None)


def test_docker_gets_a_limit_by_default():
    """The default is 'auto', so a worker started with no thought about memory
    still gets a cgroup limit — which is the whole point. Without one Ray sizes
    its monitor from host total and only sheds tasks at 95% of the machine."""
    result = _run(["start", "--runtime", "docker", "--dry-run", "--", "--mode", "single-machine"])
    assert result.exit_code == 0, result.output
    assert "--memory" in result.output


def test_auto_resolves_to_a_share_of_host_memory():
    total = read_meminfo()["MemTotal"]
    expected = f"{int(total * DEFAULT_MEMORY_FRACTION) // 1024**3}g"
    assert resolve_memory("auto") == expected
    assert resolve_memory("AUTO") == expected, "the sentinel is case-insensitive"


@pytest.mark.parametrize(
    "value, expected",
    [("32g", "32g"), ("  32g  ", "32g"), ("none", None), ("0", None), ("", None)],
)
def test_memory_values_resolve(value, expected):
    assert resolve_memory(value) == expected


def test_an_explicit_limit_is_refused_rather_than_dropped_for_sif_runtimes():
    """Silently ignoring it is the exact failure this option exists to prevent.

    An operator who passes --memory and gets no limit believes the worker is
    capped when it is not — strictly worse than being told no, because they
    stop looking.
    """
    for runtime in ("apptainer", "singularity", "native"):
        result = _run(
            ["start", "--runtime", runtime, "--memory", "32g", "--dry-run",
             "--", "--mode", "single-machine"]
        )
        assert result.exit_code != 0, f"{runtime} accepted --memory silently"
        assert "--memory is not supported" in result.output


def test_the_default_does_not_refuse_those_runtimes():
    """The regression a default introduces, and the reason this test exists.

    Once --memory defaults to 'auto' the option is always set, so a refusal
    keyed on "was it given" fails every apptainer and native run on a flag its
    operator never typed. Only an EXPLICIT limit may be refused.
    """
    for runtime in ("apptainer", "singularity", "native"):
        result = _run(
            ["start", "--runtime", runtime, "--dry-run", "--", "--mode", "single-machine"]
        )
        assert result.exit_code == 0, f"{runtime} failed on the default: {result.output}"
        assert "--memory" not in result.output


def test_explicitly_disabling_it_is_not_refused_either():
    """'none' is the operator agreeing there is no limit, not asking for one."""
    result = _run(
        ["start", "--runtime", "apptainer", "--memory", "none", "--dry-run",
         "--", "--mode", "single-machine"]
    )
    assert result.exit_code == 0, result.output
    assert "--memory" not in result.output
