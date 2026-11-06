"""``run_code`` refuses the ``resources`` remote_option at the boundary.

Ray accepts the key and then fails the task with an error that names neither it
nor anything in the caller's payload, because a nested value arrives from Hypha
RPC as an ``ObjectProxy`` rather than a dict. Measured both ways: on a direct
connection Ray's ``submit_task`` raises ``TypeError: Expected dict, got
ObjectProxy``, and over a Ray client the options are re-pickled in the head pod,
where the proxy cannot be unpickled — ``ModuleNotFoundError: No module named
'hypha_rpc'``. The second escaped as a task error naming a package the payload
never imports; the first escaped the ``except ValueError`` here entirely and
reached the caller as a raw RPC exception.

Everything else stays a pass-through to ``.options()``: ``scheduling_strategy``
is undocumented and works, and nothing below may start refusing it.
"""

import pytest

from bioengine.utils import create_context
from bioengine.worker.code_executor import CodeExecutor

ADMIN = create_context("admin-id", "admin@example.org")

CODE = "def analyze():\n    return 'ok'\n"


class _Logger:
    def __init__(self):
        self.records = []

    def _record(self, message):
        self.records.append(str(message))

    info = warning = error = debug = _record


class _RayCluster:
    mode = "single-machine"


class _FakeTask:
    """Stands in for the ``@ray.remote`` ``ray_task``, recording submissions."""

    def __init__(self):
        self.options_seen = []

    def options(self, **kwargs):
        self.options_seen.append(kwargs)
        return self

    def remote(self, func, args, kwargs):
        async def _result():
            return {"result": func(*args, **kwargs)}

        return _result()


def _executor(monkeypatch):
    executor = CodeExecutor.__new__(CodeExecutor)
    executor.logger = _Logger()
    executor.ray_cluster = _RayCluster()
    executor.admin_users = ["admin@example.org"]
    task = _FakeTask()
    monkeypatch.setattr("bioengine.worker.code_executor.ray_task", task)
    return executor, task


@pytest.mark.parametrize(
    "resources",
    [
        {"node:172.25.1.180": 0.001},
        {"custom_zzz": 1},
        {},
    ],
    ids=["node-affinity", "unknown-custom-resource", "empty-dict"],
)
async def test_resources_is_refused_whatever_it_asks_for(monkeypatch, resources):
    executor, task = _executor(monkeypatch)

    result = await executor.run_code(
        code=CODE,
        function_name="analyze",
        remote_options={"resources": resources},
        context=ADMIN,
    )

    assert result["error"].startswith(
        "Unsupported remote_options key 'resources'"
    ), result
    assert "ModuleNotFoundError" not in result["error"]
    assert task.options_seen == [], "the task must not be submitted at all"


async def test_the_refusal_survives_other_keys_in_the_same_dict(monkeypatch):
    executor, task = _executor(monkeypatch)

    result = await executor.run_code(
        code=CODE,
        function_name="analyze",
        remote_options={"num_cpus": 0, "resources": {"node:172.25.1.180": 0.001}},
        context=ADMIN,
    )

    assert result["error"].startswith(
        "Unsupported remote_options key 'resources'"
    ), result
    assert task.options_seen == []


@pytest.mark.parametrize(
    "remote_options",
    [
        {},
        {"num_cpus": 1},
        {"num_gpus": 0},
        {"memory": 100_000_000},
        {"runtime_env": {"pip": ["numpy"]}},
        {"num_cpus": 1, "scheduling_strategy": "SPREAD"},
    ],
    ids=[
        "bare",
        "num_cpus",
        "num_gpus",
        "memory",
        "runtime_env",
        "scheduling_strategy",
    ],
)
async def test_every_other_option_still_reaches_ray(monkeypatch, remote_options):
    executor, task = _executor(monkeypatch)

    result = await executor.run_code(
        code=CODE,
        function_name="analyze",
        remote_options=dict(remote_options),
        context=ADMIN,
    )

    assert result == {"result": "ok"}, result
    assert task.options_seen == [remote_options]
