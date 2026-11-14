"""Pin that a replica unregisters its OWN services when another worker runs the same app.

Measured against hypha.aicell.io on 2026-11-13 with hypha-rpc 0.21.40:
``server.unregister_service`` keeps only the part after the first ``:``
(``rpc.py:2216``), so the fully-qualified id a replica registered under arrives
at the server as a bare name, which it expands to ``<workspace>/*:<name>``.
One worker and that wildcard matches a single entry and the call succeeds —
which is why this was invisible until a second worker deployed the same app.
Two workers and it matches both, and the server refuses:

    AssertionError: Multiple services found for <workspace>/*:<name>

The manager service resolves the id it is handed without expanding it, so a
fully-qualified id matches exactly one entry.

``_Hypha`` below implements exactly those two rules, which is what makes these
tests fail on the helper call and pass on the manager call. The server's advice
to disambiguate with ``mode='random'``/``'first'``/``'last'`` is deliberately
not modelled: picking an arbitrary match would let a replica unregister another
worker's service.
"""

from __future__ import annotations

import asyncio

from tests.apps._proxy_double import PROXY_CLS, ProxyDouble

WORKSPACE = "ws"
APP = "app"
MINE = "this-replica"
THEIRS = "other-worker-replica"


def _ids(client: str) -> list[str]:
    """The three ids a proxy replica registers, in ``_deregister_services`` order."""
    return [f"{WORKSPACE}/{client}:{APP}{suffix}" for suffix in ("", "-rtc", "-mcp")]


class _Hypha:
    """The workspace registry and the id resolution the server actually performs."""

    def __init__(self, clients: list[str]):
        self.entries = {sid for client in clients for sid in _ids(client)}

    def unregister(self, service_id: str) -> None:
        if "/" not in service_id:
            service_id = f"{WORKSPACE}/*:{service_id}"
        workspace, rest = service_id.split("/", 1)
        client, name = rest.split(":", 1)
        matches = [
            entry
            for entry in self.entries
            if entry.startswith(f"{workspace}/")
            and entry.split(":", 1)[1] == name
            and client in ("*", entry.split("/", 1)[1].split(":", 1)[0])
        ]
        if not matches:
            raise AssertionError(f"Service not found: {workspace}/{client}:{name}")
        if len(matches) > 1:
            raise AssertionError(
                f"Multiple services found for {workspace}/{client}:{name}, "
                f"e.g. {sorted(matches)}"
            )
        self.entries.discard(matches[0])


class _Manager:
    def __init__(self, hypha: _Hypha, calls: list[str]):
        self._hypha = hypha
        self._calls = calls

    async def unregister_service(self, service_id: str) -> None:
        self._calls.append(service_id)
        self._hypha.unregister(service_id)


class _Rpc:
    def __init__(self, hypha: _Hypha):
        self._hypha = hypha
        self.manager_calls: list[str] = []

    async def get_manager_service(self, config=None, max_retries: int = 20):
        self.max_retries = max_retries
        return _Manager(self._hypha, self.manager_calls)


class _Server:
    def __init__(self, hypha: _Hypha):
        self._hypha = hypha
        self.rpc = _Rpc(hypha)
        self.helper_calls: list[str] = []

    async def unregister_service(self, service_id: str) -> None:
        """hypha-rpc's helper, which discards everything before the first ``:``."""
        self.helper_calls.append(service_id)
        self._hypha.unregister(service_id.split(":")[1])


def _replica(server: _Server) -> ProxyDouble:
    inst = object.__new__(ProxyDouble)
    inst.application_id = APP
    inst.server = server
    (
        inst.websocket_service_id,
        inst.rtc_service_id,
        inst.mcp_service_id,
    ) = _ids(MINE)
    inst._proxy_actor_handle = None
    return inst


def test_deregister_drops_this_replicas_services_and_leaves_the_other_workers() -> None:
    hypha = _Hypha([MINE, THEIRS])
    replica = _replica(_Server(hypha))
    assert len(hypha.entries) == 6

    asyncio.run(PROXY_CLS._deregister_services(replica))

    assert hypha.entries == set(_ids(THEIRS))
    assert (
        replica.websocket_service_id,
        replica.rtc_service_id,
        replica.mcp_service_id,
    ) == (None, None, None)


def test_deregister_sends_the_fully_qualified_id() -> None:
    hypha = _Hypha([MINE, THEIRS])
    server = _Server(hypha)

    asyncio.run(PROXY_CLS._deregister_services(_replica(server)))

    assert server.rpc.manager_calls == _ids(MINE)
    assert server.helper_calls == []


def test_a_sole_registrant_still_unregisters() -> None:
    hypha = _Hypha([MINE])
    replica = _replica(_Server(hypha))

    asyncio.run(PROXY_CLS._deregister_services(replica))

    assert hypha.entries == set()


def test_an_unreachable_manager_is_logged_not_raised() -> None:
    """``_deregister_services`` has a caller that does not wrap it (the health
    loop), and it also runs in the replica's destructor."""
    hypha = _Hypha([MINE, THEIRS])
    server = _Server(hypha)

    async def _refuse(config=None, max_retries: int = 20):
        raise RuntimeError("manager unreachable")

    server.rpc.get_manager_service = _refuse
    replica = _replica(server)

    asyncio.run(PROXY_CLS._deregister_services(replica))

    assert hypha.entries == _Hypha([MINE, THEIRS]).entries
    assert replica.websocket_service_id is None
