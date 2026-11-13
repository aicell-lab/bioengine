"""The classifier recognises the refusal a live stale handle actually produces.

Every exception below is the *verbatim* message a real failure produced against
``https://hypha.aicell.io`` with ``hypha_rpc 0.21.40``, only the client ids kept
(#0407). That matters because the shape was not what the module was built for: a
handle resolved by name whose target client has gone does not fail send-side and
does not time out — the server answers with an error envelope, which hypha_rpc
decodes into a ``RemoteException``. ``origin/main`` short-circuited on that type
and returned ``None``, so the re-resolve never ran for the one condition the
module exists for.

The refusal cannot be told from a handler error by its text. A handler is free to
raise ``"Target peer … is not connected"`` itself, and a handler propagating a
*real* stale error puts that text in the remote traceback — the hazard
``test_a_remote_failure_is_never_treated_as_unsent`` in
``tests/apps/test_artifact_manager_reconnect.py`` was written about, and which
still passes unchanged. What separates them is the remote traceback: hypha_rpc
fills it in ``RPC._handle_method`` for anything the handler raised, and a refusal
the server generated has none, because nothing ran to trace.

``test_the_refusal_and_a_handler_error_are_textually_indistinguishable`` is the
discrimination control. Without it, dropping the traceback test and matching the
message would still pass every other case here.
"""
from __future__ import annotations

import pytest
from hypha_rpc.rpc import RemoteException

from bioengine.utils.stale_service import stale_service_kind

# Measured. The server refused to route to a client it no longer knows; the
# handler never ran, confirmed by an invocation log that did not move.
_ROUTING_REFUSAL = RemoteException(
    "RemoteError:Target peer bioimage-io/bdeNa83tqUgQbsxJLsSP2m is not connected\n"
)
# Measured. hypha_rpc's own peer_not_found handler rejects pending calls with
# this instead, when the server notifies rather than replying.
_PEER_NOT_CONNECTED = ConnectionError("Peer bioimage-io/bdeNa83tqUgQbsxJLsSP2m is not connected")
# Measured, and the shape that already classified correctly: the peer dropped
# while the handler was running, so it may have landed.
_DROPPED_MID_CALL = ConnectionError("Client disconnected: bioimage-io/frMataFKYW48hyszts58PT")

_HANDLER_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/usr/local/lib/python3.11/site-packages/hypha_rpc/rpc.py", line 3147,'
    " in _handle_method\n    method_task = self._call_method(...)\n"
    "RuntimeError: ordinary handler failure\n"
)


def _remote(rvalue: str, rtrace: str = "") -> RemoteException:
    """A remote failure exactly as hypha_rpc reconstructs it locally."""
    return RemoteException("RemoteError:" + rvalue + "\n" + rtrace)


def test_the_live_routing_refusal_is_classified_as_never_sent():
    assert stale_service_kind(_ROUTING_REFUSAL) == "never_sent"


def test_the_notified_variant_is_classified_the_same_way():
    assert stale_service_kind(_PEER_NOT_CONNECTED) == "never_sent"


def test_a_drop_while_the_handler_ran_stays_maybe_sent():
    assert stale_service_kind(_DROPPED_MID_CALL) == "maybe_sent"


def test_the_refusal_and_a_handler_error_are_textually_indistinguishable():
    """The control the traceback test exists for.

    A handler is free to raise the server's own wording. Both messages carry the
    marker; only the traceback says which one ran. Matching the text — the whole
    string or just the ``RemoteError:`` value line — classifies this one too, and
    a replayed write is the cost.
    """
    handler_said_it = _remote(
        "Target peer bioimage-io/not-a-real-client is not connected",
        _HANDLER_TRACEBACK,
    )

    assert _ROUTING_REFUSAL.args[0].startswith("RemoteError:Target peer")
    assert handler_said_it.args[0].startswith("RemoteError:Target peer")
    assert "is not connected" in str(handler_said_it).lower()

    assert stale_service_kind(handler_said_it) is None
    assert stale_service_kind(_ROUTING_REFUSAL) == "never_sent"


def test_a_handler_propagating_a_real_stale_error_is_not_retried():
    """Measured: the marker lands in the traceback, the value line is the header."""
    propagated = _remote(
        "Traceback (most recent call last):",
        '  File "/app/broker.py", line 100, in refresh\n'
        "    return await handle.ping()\n"
        "hypha_rpc.rpc.RemoteException: RemoteError:Target peer "
        "bioimage-io/inner-client is not connected\n",
    )

    assert "is not connected" in str(propagated).lower()
    assert stale_service_kind(propagated) is None


def test_an_ordinary_handler_error_is_still_not_a_stale_failure():
    assert stale_service_kind(_remote("KeyError: 'manifest'", _HANDLER_TRACEBACK)) is None


@pytest.mark.parametrize(
    "exc, expected",
    [
        (RuntimeError("Failed to send the request when calling method"), "never_sent"),
        (RuntimeError("WebSocket reconnection timed out"), "never_sent"),
        (RuntimeError("Method call timed out: ws/abc:services.x.read"), "maybe_sent"),
        (KeyError("Service not found: ws/dead:s3-storage@*"), "maybe_sent"),
        (ValueError("artifact not found"), None),
    ],
)
def test_the_markers_that_already_worked_are_unchanged(exc, expected):
    assert stale_service_kind(exc) == expected
