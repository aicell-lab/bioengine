"""``get_app_status`` returns a dict KEYED BY application id, for any number of ids.

WHY THIS IS PINNED. The docstring promised, in three places, that a query naming exactly one
application returned that application's status unwrapped. The code never did that — ``output`` is
built as ``{application_id: status}`` and returned, with no single-id branch anywhere in the
method. A client written to the documentation therefore reads every field as absent from a
perfectly healthy worker, and nothing raises: the failure is silent and looks like the worker not
reporting the field.

The documentation is what changed, because the keyed form is what every live worker does and a
shape change would break consumers written from observation. These arms stop the promise coming
back: the behaviour is asserted, and the docstring is asserted not to claim the opposite.
"""

import ast
import asyncio
import inspect
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import bioengine.apps.manager as manager_mod
from bioengine.apps.manager import AppsManager

CONTEXT = {"user": {"id": "tester", "email": "tester@example.org"}}


def _manager(app_ids):
    m = object.__new__(AppsManager)
    m.logger = logging.getLogger("test.shape")
    m.admin_users = ["tester"]
    m._deployed_applications = {a: {} for a in app_ids}
    m._may_read_app_identity = MagicMock(return_value=True)
    # One status object per app, distinguishable so a wrong mapping is visible.
    m._get_app_status = AsyncMock(
        side_effect=lambda application_id, **kw: {"status": "RUNNING", "echo": application_id}
    )
    ray_cluster = MagicMock()
    ray_cluster.check_connection = AsyncMock()
    ray_cluster.call_with_reconnect = AsyncMock(return_value={})
    ray_cluster.proxy_actor_handle.get_serve_instance_details.remote = AsyncMock(return_value={})
    ray_cluster.proxy_actor_handle.get_replica_identities.remote = AsyncMock(return_value={})
    ray_cluster.proxy_actor_handle.get_replica_death_causes.remote = AsyncMock(return_value={})
    m.ray_cluster = ray_cluster
    m.server = SimpleNamespace(config=SimpleNamespace(workspace="ws", client_id="c"))
    return m


@pytest.mark.asyncio
async def test_ONE_id_is_still_keyed_by_that_id():
    """The arm the docstring contradicted. A single id must NOT be unwrapped."""
    m = _manager(["app1"])
    out = await m.get_app_status(application_ids=["app1"], context=CONTEXT)
    assert isinstance(out, dict), f"expected a dict, got {type(out).__name__}"
    assert "app1" in out, (
        f"a one-id query returned something not keyed by that id (keys: {sorted(out)}); a client "
        f"written to the keyed shape reads every field as absent"
    )
    assert out["app1"]["echo"] == "app1"
    # And it must NOT be the bare status: the status fields live one level down.
    assert "status" not in out, (
        "the status was returned unwrapped, so the result is a status and a mapping at once and "
        "no caller can tell which it has"
    )


@pytest.mark.asyncio
async def test_TWO_ids_are_keyed_the_same_way_so_a_caller_never_branches():
    m = _manager(["app1", "app2"])
    out = await m.get_app_status(application_ids=["app1", "app2"], context=CONTEXT)
    assert set(out) == {"app1", "app2"}
    assert out["app1"]["echo"] == "app1" and out["app2"]["echo"] == "app2"


@pytest.mark.asyncio
async def test_the_ONE_and_TWO_id_shapes_are_THE_SAME_shape():
    """The property that actually matters to a caller, asserted as one statement rather than
    inferred from the two arms above: the same access path works for either."""
    one = await _manager(["app1"]).get_app_status(application_ids=["app1"], context=CONTEXT)
    two = await _manager(["app1", "app2"]).get_app_status(
        application_ids=["app1", "app2"], context=CONTEXT
    )
    for out in (one, two):
        assert out["app1"]["echo"] == "app1", (
            "the same access path does not work for both, so a caller has to branch on how many "
            "ids it asked for -- which is the whole defect"
        )


def _docstring() -> str:
    """The DOCSTRING, not the source.

    ``inspect.getsource`` also returns the signature, and these parameters carry ``Field``
    descriptions that repeat the same wording. Asserting a phrase is somewhere in the source
    therefore passes when the phrase has been deleted from the Returns block and survives only
    in a Field description -- measured: removing the Returns statement left all six arms green
    because the Field description still matched. Scope the claim to what a reader of the
    docstring actually sees.
    """
    doc = manager_mod.AppsManager.get_app_status.__doc__
    assert doc, "get_app_status has no docstring at all"
    return " ".join(doc.lower().split())


def test_the_DOCSTRING_does_not_promise_an_unwrapped_single_id():
    """Guards the documentation, not the code. The behaviour arms above would stay green while
    the docstring went back to promising the opposite, and the docstring is what a client reads.
    """
    lowered = _docstring()
    for claim in (
        "not nested in a dictionary",
        "not wrapped in a dictionary",
        "returns status directly",
        "status is returned directly",
    ):
        assert claim not in lowered, (
            f"get_app_status's docstring promises {claim!r} again; the method has no single-id "
            f"branch, so that promise sends a client looking for a shape it never returns"
        )
    # Positive side: it must say what it DOES do, or removing the claim leaves nothing.
    assert "keyed by application id" in lowered, (
        "the docstring no longer states that the result is keyed by application id, so a reader "
        "has to infer the shape from the code"
    )


def test_the_docstring_documents_the_LOGS_nesting_and_that_timezone_is_not_UTC():
    """The second silent-absence trap: `logs` is keyed by replica id, two levels below the app,
    and its stdout/stderr timestamps are in the replica's local zone. Reading `logs` as a string
    or a list yields zero lines from a healthy worker, which is indistinguishable from a quiet
    application."""
    lowered = _docstring()
    for needed in ("replica id", "timezone", "stdout"):
        assert needed in lowered, (
            f"the docstring does not mention {needed!r}, so the logs shape is still undocumented "
            f"on a published surface"
        )
    assert "not necessarily utc" in lowered or "not utc" in lowered, (
        "the docstring does not say the log timezone is not necessarily UTC; a consumer that "
        "assumes UTC mis-dates every line by the offset"
    )


def test_the_method_really_has_no_single_id_branch():
    """Pins the behaviour at its source, so the docstring and the code cannot drift apart again
    in the other direction: if someone later IMPLEMENTS the unwrap, this fails and the docs have
    to be revisited together with it."""
    src = inspect.getsource(manager_mod)
    fn = next(
        (n for n in ast.walk(ast.parse(src))
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "get_app_status"),
        None,
    )
    assert fn is not None, "manager.py has no get_app_status"
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert returns, "get_app_status returns nothing -- the anchor moved"
    # Every return is either the keyed mapping or an empty dict; none indexes a single element.
    for r in returns:
        seg = (ast.get_source_segment(src, r) or "").strip()
        assert "[0]" not in seg, (
            f"a return in get_app_status indexes an element ({seg!r}); that is the unwrap the "
            f"docstring used to promise, and the docs must change with it"
        )
