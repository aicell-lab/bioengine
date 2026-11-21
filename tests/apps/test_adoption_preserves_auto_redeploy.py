"""An adopted application must come back with the ``auto_redeploy`` it was deployed with.

THE HOP THIS GUARDS. ``deploy_app`` resolves ``auto_redeploy`` and records it, and tests already
cover that end. But the value a worker RESTART restores is not that record — the record dies with
the process. It is restored from the ``app_data`` blob the builder stamped onto the
ProxyDeployment, which ``recover_deployed_applications`` reads back. So there are two writes of
this setting on two sides of a restart, and only one of them was guarded:

    builder.py   app_data["auto_redeploy"] = auto_redeploy     <- survives the restart
    manager.py   record["auto_redeploy"]   = app_data[...]     <- read back on adoption

A constant on either side is invisible to every ``deploy_app`` test, because neither runs. Measured
before this file existed: replacing the builder's value with a constant ``True`` left the entire
suite green — 1602 passed, 0 failed — while making every application that opted out come back
auto-redeploying after a restart. Since the default became True, the value an adopted application
inherits is the one that redeploys, so the blind spot is on the dangerous side.

WHAT IS AND IS NOT EXERCISED HERE, stated so the cover is not read as wider than it is.
``recover_deployed_applications`` is driven for real against a fake Serve status and app handle, so
the read-back side is behavioural. The builder's write is NOT driven: ``AppBuilder.build`` is 290
lines of async that needs Ray and Hypha, so its single ``"auto_redeploy"`` occurrence is pinned
statically instead. The join between the two — that both sides name the same key — is asserted
directly, because two independently-correct halves that disagree about the key would still lose the
value.
"""

import ast
import asyncio
import inspect
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import bioengine.apps.builder as builder_mod
import bioengine.apps.manager as manager_mod
from bioengine.apps.manager import AppsManager

APP_ID = "adopted-app"
WORKSPACE = "ws-test"
KEY = "auto_redeploy"


def _app_data(auto_redeploy):
    """The blob the builder stamps onto the ProxyDeployment, with every key recovery requires."""
    return {
        "format_version": "0.6.0",
        "worker_workspace": WORKSPACE,
        "display_name": "Adopted App",
        "description": "an app this worker did not deploy",
        "artifact_id": "bioimage-io/adopted",
        "version": "1.0.0",
        "entry": "entry:EntryApp",
        "application_kwargs": {},
        "application_env_vars": {},
        "disable_gpu": False,
        "max_ongoing_requests": 10,
        "application_resources": {"num_cpus": 1},
        "authorized_users": ["*"],
        "available_methods": ["ping"],
        "started_at": 1.0,
        "last_updated_at": 2.0,
        "last_updated_by": "someone",
        KEY: auto_redeploy,
        "debug": False,
    }


def _manager(app_data):
    m = object.__new__(AppsManager)
    m.logger = logging.getLogger("test.adoption")
    m._deployed_applications = {}
    m.admin_users = []
    m._inject_admin_users = MagicMock(side_effect=lambda users: users)

    handle = MagicMock()
    handle.get_app_data.remote = AsyncMock(return_value=app_data)
    handle.update_authorized_users.remote = AsyncMock(return_value=None)

    status = SimpleNamespace(
        applications={APP_ID: SimpleNamespace(deployments={"ProxyDeployment": object()})}
    )

    async def call_with_reconnect(fn, *args, **kwargs):
        # Dispatch on the serve function the production code asked for, so the stub cannot
        # answer a call the code never makes.
        name = getattr(fn, "__name__", "")
        if name == "status":
            return status
        if name == "get_app_handle":
            return handle
        raise AssertionError(f"unexpected serve call: {name!r}")

    ray_cluster = MagicMock()
    ray_cluster.call_with_reconnect = AsyncMock(side_effect=call_with_reconnect)
    m.ray_cluster = ray_cluster

    m.server = SimpleNamespace(
        config=SimpleNamespace(workspace=WORKSPACE, public_base_url="https://example.invalid")
    )
    return m


def _recovered(m):
    rec = m._deployed_applications.get(APP_ID)
    assert rec is not None, (
        f"nothing was adopted, so this arm would be vacuous; records present: "
        f"{sorted(m._deployed_applications)}"
    )
    # Sentinel, not .get(None): an absent key and a stored False are indistinguishable
    # through .get(), and False is the value this file is about.
    return rec.get(KEY, "<ABSENT>")


# ── the read-back side, driven for real ──────────────────────────────────────


@pytest.mark.asyncio
async def test_an_adopted_app_that_OPTED_OUT_comes_back_opted_out():
    """The dangerous direction: an app that asked not to be redeployed must not come back
    redeploying, because the worker would then restart it forever against its owner's wish."""
    m = _manager(_app_data(False))
    await m.recover_deployed_applications()
    got = _recovered(m)
    assert got is False, (
        f"an application deployed with {KEY}=False was adopted with {got!r}; after a worker "
        f"restart the monitor loop would redeploy an application that opted out"
    )


@pytest.mark.asyncio
async def test_an_adopted_app_that_OPTED_IN_comes_back_opted_in():
    """The mirror, and not symmetry for its own sake: a constant False on either side would
    silently stop protecting every application that asked to be protected, and an arm that only
    checked the False case would pass against it."""
    m = _manager(_app_data(True))
    await m.recover_deployed_applications()
    got = _recovered(m)
    assert got is True, (
        f"an application deployed with {KEY}=True was adopted with {got!r}; it would stop being "
        f"auto-redeployed after a worker restart, silently"
    )


@pytest.mark.asyncio
async def test_the_two_directions_DIFFER_so_the_value_is_carried_not_constant():
    """Instrument control. Each arm above asserts one value, so a constant on the read-back side
    makes exactly one of them fail and the other pass — which reads as a half-working feature.
    Requiring the two to differ is what establishes the value is carried at all."""
    seen = []
    for want in (False, True):
        m = _manager(_app_data(want))
        await m.recover_deployed_applications()
        seen.append(_recovered(m))
    assert len(set(seen)) == 2, (
        f"both directions adopted the same value {seen!r}, so adoption is not carrying "
        f"{KEY} and these arms cannot tell a carried value from a constant"
    )


@pytest.mark.asyncio
async def test_adoption_REFUSES_an_app_data_blob_that_omits_the_setting():
    """A missing key must refuse, not default. Recovery lists `auto_redeploy` in its required
    keys; if it were ever dropped from that set, a blob without it would adopt and the record
    would carry whatever `.get` returned — and `None` is falsy, so the app would silently stop
    being redeployed rather than loudly failing to adopt."""
    blob = _app_data(True)
    del blob[KEY]
    m = _manager(blob)
    await m.recover_deployed_applications()
    assert APP_ID not in m._deployed_applications, (
        f"an app_data blob with no {KEY} was adopted anyway; the setting would then be absent "
        f"from the record instead of the adoption being refused"
    )


# ── the write side, pinned statically because build() cannot be driven here ──


def test_the_BUILDER_stamps_the_RESOLVED_VALUE_into_app_data():
    """Allow-list over the string, the same shape that works in deploy_app's record arm.

    ``AppBuilder.build`` needs Ray and Hypha, so it is not driven here. What is checked instead:
    the string ``"auto_redeploy"`` occurs in ``builder.py`` exactly where it should, as a dict key
    whose value is the resolved NAME. A constant there is the mutation that left all 1602 tests
    green, and it is the one this arm exists for.
    """
    src = inspect.getsource(builder_mod)
    tree = ast.parse(src)
    occurrences = [
        n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == KEY
    ]
    assert occurrences, (
        f"builder.py never mentions {KEY!r}, so the blob an adopted app is restored from no "
        f"longer carries it -- this arm's anchor has moved, not been satisfied"
    )

    keys, other = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == KEY:
                    keys.append((k, v))
        elif isinstance(node, ast.Subscript):
            s = node.slice
            if isinstance(s, ast.Constant) and s.value == KEY:
                other.append(node)

    accounted = len(keys) + len(other)
    assert accounted == len(occurrences), (
        f"{len(occurrences) - accounted} occurrence(s) of {KEY!r} in builder.py sit in neither a "
        f"dict literal nor a subscript; an unclassified mention is what this arm refuses rather "
        f"than ignores (lines {sorted(n.lineno for n in occurrences)})"
    )
    assert len(keys) == 1, (
        f"expected exactly ONE {KEY!r} dict entry in builder.py, found {len(keys)} at lines "
        f"{sorted(k.lineno for k, _ in keys)}"
    )
    key, value = keys[0]
    assert isinstance(value, ast.Name) and value.id == KEY, (
        f"builder.py line {key.lineno} stamps {type(value).__name__} into app_data rather than "
        f"the resolved variable; an adopted application would then come back with that constant "
        f"whatever its owner asked for"
    )
    assert not other, (
        f"builder.py writes {KEY!r} by subscript at line(s) "
        f"{sorted(n.lineno for n in other)}, which can override the dict entry"
    )


# ── the join: both sides must name the same key ──────────────────────────────


def test_the_WRITER_and_the_READER_agree_on_the_key_and_it_is_REQUIRED():
    """Two correct halves that disagree about the key still lose the value.

    The builder stamps a literal; recovery reads a literal and lists it among the keys it
    refuses to adopt without. Renaming either one alone would leave both files internally
    consistent and the setting silently dropped, so assert the agreement directly rather than
    trusting that two separate arms happen to use the same spelling.
    """
    builder_src = inspect.getsource(builder_mod)
    manager_src = inspect.getsource(manager_mod)

    assert f'"{KEY}": {KEY},' in builder_src.replace("'", '"'), (
        f"builder.py does not stamp {KEY!r} as a dict entry holding the same-named variable"
    )

    fn = next(
        (n for n in ast.walk(ast.parse(manager_src))
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "recover_deployed_applications"),
        None,
    )
    assert fn is not None, "manager.py has no recover_deployed_applications"
    seg = ast.get_source_segment(manager_src, fn)
    assert seg is not None, "could not read recover_deployed_applications' source"

    # It must be READ from app_data...
    assert f'app_data["{KEY}"]' in seg, (
        f"recover_deployed_applications does not read {KEY!r} out of app_data, so an adopted "
        f"application is not restored from what it was deployed with"
    )
    # ...and REQUIRED, so a blob without it refuses rather than defaulting.
    required = next(
        (n for n in ast.walk(fn)
         if isinstance(n, ast.Set)
         and any(isinstance(e, ast.Constant) and e.value == "artifact_id" for e in n.elts)),
        None,
    )
    assert required is not None, "could not find recovery's required_keys set"
    names = {e.value for e in required.elts if isinstance(e, ast.Constant)}
    assert KEY in names, (
        f"{KEY!r} is not in recovery's required_keys, so a blob omitting it would adopt and the "
        f"record would carry a default instead of the deployed value"
    )
