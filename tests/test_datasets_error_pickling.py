"""A failed dataset request has to survive the trip back to the caller.

An app that hits a 404 through ``bioengine.datasets`` runs inside a Ray Serve
replica; the proxy calls it with ``await method.remote(...)`` and Ray pickles
whatever it raised back across the process boundary. ``httpx.HTTPStatusError``
does not survive that — its ``__init__`` requires keyword-only ``request`` and
``response``, which the default reduce never supplies — so the status code, the
URL and the body were replaced by a reconstruction failure at exactly the
moment someone was debugging.

Every assertion here is made on the object the *far side* of a real
``ray.cloudpickle`` round trip hands back. Asserting on the exception where it
is raised would pass against the old code and prove nothing: nothing was ever
wrong with the raising, only with the reconstruction.

Imports are by full module path on purpose — ``from bioengine.datasets import
...`` for a name the package has not bound goes through its ``__getattr__``
delegation and raises ``MissingDataServerError`` instead.
"""

import asyncio

import httpx
import pytest
import ray.cloudpickle as cloudpickle

from bioengine._app.errors import DataServerError
from bioengine.datasets.datasets import BioEngineDatasets
from bioengine.datasets.utils.network import raise_for_data_server_status

SERVER = "http://data-server.invalid"
NOT_FOUND_BODY = '{"detail":"Dataset \'blood-atlas\' does not exist"}'


def datasets_client(handler):
    """A client whose transport is ``handler``, so httpx builds real Response
    objects and ``raise_for_status`` produces a real ``HTTPStatusError``."""
    client = BioEngineDatasets(data_server_url=SERVER, hypha_token="tok")
    client.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def responder(status_code, body=NOT_FOUND_BODY):
    def handler(request):
        return httpx.Response(status_code, text=body, request=request)

    return handler


def across_the_boundary(call):
    """Run ``call``, then hand back what the caller reconstructs.

    ``call`` is a zero-arg coroutine factory. Whatever it raises is pickled and
    unpickled exactly as Ray does when a replica method fails, and the *result*
    of that round trip is returned, so tests can only ever assert on what the
    caller actually received.
    """
    try:
        asyncio.run(call())
    except BaseException as raised:  # noqa: BLE001 — the object under test
        return cloudpickle.loads(cloudpickle.dumps(raised))
    raise AssertionError("expected the dataset call to raise")


# ---------------------------------------------------------------------------
# The reported failure: a 404 through the retrying GET helper
# ---------------------------------------------------------------------------


def test_a_404_reaches_the_caller_as_a_data_server_error():
    client = datasets_client(responder(404))
    received = across_the_boundary(lambda: client.list_files("blood-atlas"))
    assert isinstance(received, DataServerError)


def test_the_caller_can_read_the_status_code():
    client = datasets_client(responder(404))
    received = across_the_boundary(lambda: client.list_files("blood-atlas"))
    assert received.status_code == 404


def test_the_caller_can_read_the_url():
    """The URL is what tells a user their dataset id was wrong."""
    client = datasets_client(responder(404))
    received = across_the_boundary(lambda: client.list_files("blood-atlas"))
    assert received.url.startswith(f"{SERVER}/datasets/blood-atlas/files")


def test_the_caller_can_read_the_body():
    client = datasets_client(responder(404))
    received = across_the_boundary(lambda: client.list_files("blood-atlas"))
    assert "does not exist" in received.body


def test_the_message_the_caller_prints_names_the_status_and_the_url():
    client = datasets_client(responder(404))
    received = across_the_boundary(lambda: client.list_files("blood-atlas"))
    assert "404" in str(received)
    assert "blood-atlas" in str(received)


def test_no_httpx_type_crosses_the_boundary():
    """The point of translating: the public API stops depending on the client
    library, so a caller never needs httpx installed to catch this."""
    client = datasets_client(responder(403))
    received = across_the_boundary(lambda: client.list_files("blood-atlas"))
    assert not isinstance(received, httpx.HTTPError)


# ---------------------------------------------------------------------------
# Every public method that can raise it, not just the one in the report
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda c: c.list_datasets(), id="list_datasets"),
        pytest.param(lambda c: c.list_files("blood-atlas"), id="list_files"),
        pytest.param(lambda c: c.list_saved_files(), id="list_saved_files"),
        pytest.param(lambda c: c.get_saved_file("notes.txt"), id="get_saved_file"),
        pytest.param(lambda c: c.save_file("notes.txt", b"x"), id="save_file"),
        pytest.param(
            lambda c: c.request_dataset_access("blood-atlas"),
            id="request_dataset_access",
        ),
        pytest.param(
            lambda c: c.get_dataset_access_request("blood-atlas"),
            id="get_dataset_access_request",
        ),
        pytest.param(
            lambda c: c.list_dataset_access_requests(), id="list_dataset_access_requests"
        ),
        pytest.param(
            lambda c: c.resolve_dataset_access_request("blood-atlas", "a@b.invalid", "grant"),
            id="resolve_dataset_access_request",
        ),
    ],
)
def test_every_dataset_method_hands_the_caller_a_readable_error(call):
    client = datasets_client(responder(404))
    received = across_the_boundary(lambda: call(client))
    assert isinstance(received, DataServerError)
    assert received.status_code == 404


def test_a_server_error_survives_the_retry_loop():
    """5xx exhausts the retries before it raises — a different raise site from
    the 4xx short-circuit, and it has to translate too."""
    client = datasets_client(responder(503, body="upstream unavailable"))
    received = across_the_boundary(lambda: client.list_files("blood-atlas"))
    assert isinstance(received, DataServerError)
    assert received.status_code == 503


# ---------------------------------------------------------------------------
# What already worked, so a future change cannot quietly take it away
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transport_error",
    [
        httpx.ConnectError("refused"),
        httpx.RemoteProtocolError("truncated"),
        httpx.ReadError("reset"),
        httpx.ConnectTimeout("timed out"),
        httpx.ReadTimeout("timed out"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_the_transport_errors_still_reconstruct(transport_error):
    """These pickle as small module references and always reconstructed; the
    status error was the odd one out. Recorded so it stays that way."""
    received = cloudpickle.loads(cloudpickle.dumps(transport_error))
    assert type(received) is type(transport_error)


# ---------------------------------------------------------------------------
# The error is reachable by the names a caller would reach for
#
# Asserted on module + qualname rather than object identity: the baseline-import
# test re-imports bioengine._app from a stripped sys.modules, which rebinds
# every error class for the rest of the session.
# ---------------------------------------------------------------------------


def named_like(cls):
    return (cls.__module__, cls.__qualname__)


def test_the_error_is_exported_from_the_bioengine_package():
    import bioengine

    assert named_like(bioengine.DataServerError) == named_like(DataServerError)


def test_the_error_is_exported_from_the_datasets_package():
    """``bioengine.datasets`` delegates unknown attributes to a lazily built
    singleton, so an unbound name here fails with MissingDataServerError rather
    than an ImportError. It has to be bound."""
    import bioengine.datasets as datasets_package

    assert named_like(datasets_package.DataServerError) == named_like(DataServerError)


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------


def test_a_2xx_response_passes_through():
    request = httpx.Request("GET", f"{SERVER}/datasets")
    assert raise_for_data_server_status(httpx.Response(200, request=request)) is None


def test_the_httpx_error_is_not_chained_onto_the_translated_one():
    """Chaining would print httpx's traceback at every caller."""
    request = httpx.Request("GET", f"{SERVER}/datasets")
    with pytest.raises(DataServerError) as excinfo:
        raise_for_data_server_status(httpx.Response(404, request=request, text="no"))
    assert excinfo.value.__cause__ is None
