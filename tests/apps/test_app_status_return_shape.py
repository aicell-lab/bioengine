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
import re
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
    # The unwrap can be spelled many ways -- output[ids[0]], list(output.values())[0],
    # next(iter(output.values())). Review beat a "[0] not in source" check with the third.
    # So allow-list the RETURNED EXPRESSION instead: it must be a bare Name or an empty dict
    # literal, which no unwrap can be.
    for r in returns:
        v = r.value
        ok = (
            isinstance(v, ast.Name)
            or (isinstance(v, ast.Dict) and not v.keys)
        )
        seg = (ast.get_source_segment(src, r) or "").strip()
        assert ok, (
            f"get_app_status returns {type(v).__name__} ({seg!r}) rather than the keyed mapping "
            f"or an empty dict; any expression here is an unwrap or a transform, and the "
            f"documentation has to change with it"
        )


def test_the_docstring_does_NOT_name_one_DIRECTION_for_the_timezone_error():
    """F2. The direction depends on the sign of the offset, so naming one is wrong half the time.

    An earlier wording said ignoring the zone makes lines "look old". For a zone AHEAD of UTC --
    CET, the case this was written for -- a local timestamp read as UTC lands in the FUTURE, so
    the lines look NEWER and a computed age can come out negative. Naming either direction
    contradicts the other case, so the docstring must describe the dependence instead.
    """
    lowered = _docstring()
    for wrong in ("look old", "looks old", "look older", "appear old"):
        assert wrong not in lowered, (
            f"the docstring claims ignoring the timezone makes lines {wrong!r}; for a zone ahead "
            f"of UTC the opposite happens, which is the case this was written for"
        )
    assert "sign" in lowered or "depends on" in lowered, (
        "the docstring names no dependence for the direction of the timezone error, so a reader "
        "cannot tell which way their own cluster's offset shifts things"
    )


def test_the_docstring_does_NOT_claim_timezone_is_ALWAYS_present():
    """F1. It is not, and a consumer told otherwise defaults it to UTC.

    get_actor_logs adds the key only when the zone is known, get_deployment_logs passes None
    when replica_info is missing, and the abnormally-dead-replica entry at the DEFAULT
    n_previous_replica=0 carries neither timezone nor creation_timestamp. So a deployment with a
    crashed replica returns, on default arguments, an entry with no zone.
    """
    lowered = _docstring()
    for overclaim in ("always present", "is always", "every replica record carries a timezone"):
        assert overclaim not in lowered, (
            f"the docstring says {overclaim!r} of timezone; it is conditional in the code, and a "
            f"consumer that believes this labels zone-less entries UTC"
        )
    assert "only when known" in lowered or "may be absent" in lowered, (
        "the docstring does not say timezone can be absent, so the conditional stays invisible"
    )


def test_the_docstring_documents_the_NON_RECORD_entries_in_the_logs_map():
    """F3. Not every value under `logs` is a replica record, and the worked example must survive
    the ones that are not: a string under "error", and an int count under
    "_deliberately_stopped_replicas". A loop that indexes every value as a dict raises on both.
    """
    doc = manager_mod.AppsManager.get_app_status.__doc__ or ""
    normalised = " ".join(doc.split())
    # Assert the DISTINCTIVE forms from the shape enumeration, not bare words. Measured: a
    # mutation deleting the enumeration still passed when this checked for "error" and
    # "_deliberately_stopped_replicas", because the Examples block mentions both -- the same
    # "satisfied by the wrong occurrence" failure that an earlier arm here had.
    for needed in (
        'logs["error"] = "<message>"',
        '"death_cause"',
        '"logs_omitted": True',
        # The counts are SIBLINGS of logs, not entries in it: manager.py lifts
        # _deliberately_stopped_replicas out of the map before returning, exactly so a caller
        # iterating logs never meets an int. Documenting it as a logs entry was wrong, and was
        # corrected by the consumer that reads this surface.
        "deployments[<class_name>].deliberately_stopped_replicas",
        "deployments[<class_name>].death_causes_unavailable",
    ):
        assert " ".join(needed.split()) in normalised, (
            f"the docstring's shape enumeration no longer gives {needed!r}, so a caller writing "
            f"the obvious loop over logs.items() breaks on real payloads"
        )
    lowered = _docstring()
    assert "list[str]" in lowered, (
        "the docstring does not give the stream type; stdout/stderr are list[str] "
        "(stdout.splitlines()), and a consumer assuming a string gets nothing from a list"
    )


def test_the_documented_EXAMPLE_survives_every_shape_the_docstring_names():
    """The example is the part a reader copies, so run it against the awkward shapes.

    Builds the three non-trivial values the docstring enumerates and executes the example's
    own logic over them. A loop that assumed every value were a record would raise here.
    """
    logs = {
        "live-1": {"stdout": ["a"], "stderr": ["b"], "timezone": "CET",
                   "creation_timestamp": 1.0},
        "dead-1": {"stdout": [], "stderr": [], "death_cause": "OOM", "logs_omitted": True},
        "_deliberately_stopped_replicas": 1,
    }
    seen = {}
    for replica_id, rec in logs.items():
        if not isinstance(rec, dict):
            continue
        lines = rec.get("stdout", []) + rec.get("stderr", [])
        seen[replica_id] = (len(lines), rec.get("timezone"))
    assert seen == {"live-1": (2, "CET"), "dead-1": (0, None)}, seen

    # and the error shape, where the VALUE is a string
    err = {"error": "Error retrieving logs: boom"}
    assert isinstance(err.get("error"), str)
    for replica_id, rec in err.items():
        assert not isinstance(rec, dict), (
            "the error entry is a string value; a caller that indexes it as a record raises "
            "TypeError, which is why the docstring has to name this shape"
        )


def test_the_docstring_does_NOT_put_the_COUNTS_inside_the_logs_map():
    """The counts are siblings of `logs`, and saying otherwise sends a caller hunting an int
    inside a map that cannot contain one.

    manager.py rebuilds the logs map WITHOUT `_deliberately_stopped_replicas` and re-exposes it
    as `deployments[<name>].deliberately_stopped_replicas`, specifically so a consumer iterating
    `logs` never meets an int where a record belongs. The proxy's own `get_deployment_logs` does
    still carry the prefixed key, so the distinction is between the two surfaces rather than
    between present and absent.
    """
    normalised = " ".join((manager_mod.AppsManager.get_app_status.__doc__ or "").split())
    assert 'logs["_deliberately_stopped_replicas"]' not in normalised, (
        "the docstring places the count INSIDE the logs map; get_app_status lifts it out before "
        "returning, so a caller told this looks for an int that is never there"
    )
    assert "deliberately_stopped_replicas" in normalised, (
        "the count is not documented at all, so a caller cannot find it on either surface"
    )


# ── doc vs REALITY: run the documented shapes against what the producer EMITS ──
#
# Review's sharpest point about the previous version of this file was not the wrong shape
# itself but WHY these arms could not catch it: one built its logs dict by hand and ran a
# COPY of the documented loop, so it could only ever confirm what the docstring said; and
# the other REQUIRED the false literal, so it would have gone red on the correct fix. Both
# are closed loops around the documentation.
#
# So this arm takes the logs map from the real producer -- AppsManager._get_deployment_status
# via the fixture that already exercises the lift -- and checks the docstring against it.


def _real_logs_map():
    """The logs map as the worker really builds it, including the stopped-count case."""
    from tests.cli.test_apps_logs_command import (
        DEPLOYMENT,
        LIVE_REPLICA,
        PROXY_LOGS_WITH_STOPPED_COUNT,
        _deployments,
    )

    deployments = _deployments(PROXY_LOGS_WITH_STOPPED_COUNT, [LIVE_REPLICA])
    return deployments[DEPLOYMENT]


def test_every_logs_KEY_the_docstring_names_can_ACTUALLY_occur():
    """The docstring must not name a key under `logs` that the producer removes.

    This is the arm whose absence let `_deliberately_stopped_replicas` be documented as a
    logs entry: the worker lifts it out, so no caller of get_app_status can ever meet it
    there, and a reader told otherwise looks for the count, fails to find it, and concludes
    "never scaled down".
    """
    dep = _real_logs_map()
    logs = dep["logs"]
    doc = " ".join((manager_mod.AppsManager.get_app_status.__doc__ or "").split())

    # Every quoted logs[...] key in the docstring must be a key the producer CAN emit.
    quoted = set(re.findall(r'logs\["([^"]+)"\]', doc))
    assert quoted, "the docstring quotes no logs[...] key at all -- this arm would be vacuous"
    emittable = set(logs.keys()) | {"error"}   # "error" comes from the failure branch
    bad = sorted(k for k in quoted if k not in emittable and not k.startswith("<"))
    assert not bad, (
        f"the docstring names logs key(s) {bad} that the producer does not emit; the real map "
        f"here holds {sorted(logs)}. A reader looking there finds nothing and draws the wrong "
        f"conclusion -- which is this issue's own failure mode"
    )

    # And specifically: the lifted count must NOT be claimed as a logs key.
    assert "_deliberately_stopped_replicas" not in quoted, (
        "the docstring claims the stopped count appears under logs; the worker lifts it out "
        "into a sibling field before returning"
    )


def test_the_documented_EXAMPLE_runs_over_the_REAL_map_without_raising():
    """Run the documented loop over the producer's output, not over a hand-built dict.

    The previous arm built its own logs dict, so it validated the example against itself. This
    one takes the real map -- which contains a live replica, an abnormally dead replica with
    death_cause/logs_omitted, and has had the count lifted out -- and executes the documented
    access pattern over it.
    """
    dep = _real_logs_map()
    logs = dep["logs"]
    assert logs, "the producer returned an empty logs map -- this arm would be vacuous"

    seen = {}
    for replica_id, rec in logs.items():
        if not isinstance(rec, dict):
            continue
        lines = rec.get("stdout", []) + rec.get("stderr", [])
        seen[replica_id] = (len(lines), rec.get("timezone"))
    assert seen, "the documented loop extracted nothing from a non-empty real map"

    # the zone-less dead replica must be among them, with no timezone -- the F1 case, live
    zoneless = [r for r, (_, tz) in seen.items() if tz is None]
    assert zoneless, (
        "no entry in the real map lacks a timezone, so the 'ONLY when known' wording is "
        "untested against the producer"
    )

    # and the counts really are siblings, which is what the docstring now says
    assert "deliberately_stopped_replicas" in dep, (
        "the producer does not expose the count as a sibling of logs, so the docstring's "
        "placement is wrong in the other direction"
    )
    assert "_deliberately_stopped_replicas" not in logs


# ── absence is not a value: the two conditional siblings, and the public view ──
#
# The failure mode these arms exist for is the one this module is about: a key that is
# sometimes absent, documented as though it were always present, so a caller reads the
# absence as a number. Both are asserted against the real producer and the real public
# filter rather than against the prose, because prose checked against prose is a closed loop.


def test_the_conditional_SIBLINGS_are_documented_as_conditional():
    """`deliberately_stopped_replicas` is set only when NON-ZERO and
    `death_causes_unavailable` only when the lookup FAILED, so indexing either raises on a
    healthy deployment. The docstring has to say so, or a caller writes `dep[...]`."""
    doc = " ".join((manager_mod.AppsManager.get_app_status.__doc__ or "").split())
    assert "only when NON-ZERO" in doc or "only when non-zero" in doc, (
        "the docstring shows deliberately_stopped_replicas unconditionally; it is present only "
        "when non-zero, so a caller indexing it gets KeyError on a deployment never scaled down"
    )
    assert "only when the cause lookup FAILED" in doc or "when present" in doc, (
        "the docstring does not mark death_causes_unavailable as conditional"
    )


def test_the_count_really_IS_absent_when_nothing_was_stopped():
    """The producer's side of the same claim: no stopped replicas, no key.

    Pins the docstring to behaviour rather than to its own wording — if the producer ever
    started emitting a zero, the documentation would be the thing that is wrong.
    """
    from tests.cli.test_apps_logs_command import (
        DEPLOYMENT,
        LIVE_REPLICA,
        PROXY_LOGS,
        PROXY_LOGS_WITH_STOPPED_COUNT,
        _deployments,
    )

    with_count = _deployments(PROXY_LOGS_WITH_STOPPED_COUNT, [LIVE_REPLICA])[DEPLOYMENT]
    without = _deployments(PROXY_LOGS, [LIVE_REPLICA])[DEPLOYMENT]
    assert "deliberately_stopped_replicas" in with_count, (
        "the fixture that HAS stopped replicas did not produce the count -- this arm would be "
        "vacuous"
    )
    assert "deliberately_stopped_replicas" not in without, (
        "the count is present when nothing was stopped, so the docstring's 'only when non-zero' "
        "is wrong -- and a caller cannot distinguish 'none' from 'not reported'"
    )


def test_the_PUBLIC_view_withholds_logs_and_BOTH_siblings():
    """An unauthorized caller gets none of the three, and their absence means WITHHELD.

    This is the sharpest form of the row's theme: outside the roster, `public_app_status` keeps
    only PUBLIC_DEPLOYMENT_FIELDS, so a client that reads a missing count as "none stopped" or a
    missing `logs` as "no logs" reports a healthy application as idle when it was simply not
    entitled to look.
    """
    status = {
        "status": "RUNNING",
        "deployments": {
            "MyApp": {
                "status": "HEALTHY",
                "message": "",
                "replica_states": {"RUNNING": 1},
                "replicas": [],
                "max_ongoing_requests": 10,
                "logs": {"r1": {"stdout": ["x"], "stderr": [], "timezone": "CET"}},
                "deliberately_stopped_replicas": 3,
                "death_causes_unavailable": True,
            }
        },
    }
    public = manager_mod.public_app_status(status)
    dep = public["deployments"]["MyApp"]
    for withheld in ("logs", "deliberately_stopped_replicas", "death_causes_unavailable"):
        assert withheld not in dep, (
            f"{withheld!r} reached an unauthorized caller; the public view is supposed to "
            f"withhold it"
        )
    # CONTROL: the public fields DO survive, so the arm is not passing on an empty dict
    for kept in ("status", "message", "replica_states", "replicas", "max_ongoing_requests"):
        assert kept in dep, f"the public view dropped {kept!r} too, so this arm proves nothing"

    doc = " ".join((manager_mod.AppsManager.get_app_status.__doc__ or "").split())
    assert "WITHHELD" in doc.upper(), (
        "the docstring does not say that an absence in the public view means withheld rather "
        "than zero, which is the reading that makes a healthy app look idle"
    )
