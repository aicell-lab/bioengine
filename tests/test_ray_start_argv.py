"""No secret may reach the argv of the Ray head or of anything Ray spawns from it.

``/proc/<pid>/cmdline`` is mode 0444, so every argument BioEngine hands to
``ray start`` is readable by every uid on the host for the lifetime of the
process — and Ray copies the password it is given into the raylet's Python
worker command template, so it lands in every worker's argv too.
"""

import re
import tempfile

import pytest

from bioengine.cluster.ray_cluster import RayCluster
from bioengine.worker.__main__ import create_parser

# The shape os.urandom(16).hex() used to produce.
HEX32 = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")


class _FakeProcess:
    returncode = 0

    async def communicate(self):
        return b"", b""


@pytest.fixture
async def head_start_argv(monkeypatch):
    """Argv of every subprocess a real ``_start_cluster`` run would exec."""
    import asyncio

    calls = []

    async def fake_exec(program, *args, **kwargs):
        calls.append((program, list(args)))
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(RayCluster, "_update_symlink", lambda self, path: None)

    # Ray's plasma store socket path must stay under 107 bytes, which pytest's
    # tmp_path already exceeds on its own.
    with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
        cluster = RayCluster(
            mode="single-machine",
            head_node_address="127.0.0.1",
            head_num_cpus=1,
            ray_temp_dir=temp_dir,
            force_clean_up=False,
        )
        await cluster._start_cluster()

        yield cluster, calls


def test_ray_start_argv_carries_no_secret(head_start_argv):
    _, calls = head_start_argv

    ray_start = next(args for _, args in calls if args[:2] == ["start", "--head"])
    joined = " ".join(ray_start)

    # Positive control: we really captured the head command, not an empty list.
    assert "--num-cpus=1" in ray_start
    assert any(arg.startswith("--temp-dir=") for arg in ray_start)

    assert not [arg for arg in ray_start if "password" in arg.lower()]
    assert not HEX32.findall(joined)


def test_no_subprocess_argv_carries_a_secret(head_start_argv):
    """Ray Serve is started too; neither command may carry a credential."""
    _, calls = head_start_argv

    assert len(calls) >= 2
    for program, args in calls:
        joined = " ".join([program] + args)
        assert "password" not in joined.lower()
        assert not HEX32.findall(joined)


def test_cluster_config_holds_no_credential(head_start_argv):
    cluster, _ = head_start_argv

    assert not [key for key in cluster.ray_cluster_config if "password" in key]
    assert not HEX32.findall(str(cluster.ray_cluster_config))


def test_ray_cluster_rejects_a_redis_password():
    with pytest.raises(TypeError):
        RayCluster(mode="external-cluster", redis_password="dummy-not-a-real-secret")


def test_worker_cli_offers_no_redis_password():
    """A flag on the worker's own command line has the same 0444 exposure."""
    parser = create_parser()

    # Positive control: the same invocation without the flag must parse.
    parser.parse_args(["--mode", "single-machine"])

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--mode", "single-machine", "--redis-password", "dummy-not-a-real-secret"]
        )


def test_ray_defaults_to_no_redis_password():
    """The upstream guarantee the fix rests on: omitting the flag means no password.

    Ray gates both injection sites (``gcs_server`` argv and the raylet's Python
    worker command template) on the password being truthy, and its default is
    the empty string. If a future Ray ships a non-empty default, the head starts
    publishing a credential again without this repository changing.
    """
    from ray._private import ray_constants
    from ray.scripts.scripts import start as ray_start_command

    assert ray_constants.REDIS_DEFAULT_PASSWORD == ""

    option = next(p for p in ray_start_command.params if p.name == "redis_password")
    assert option.default == ray_constants.REDIS_DEFAULT_PASSWORD
