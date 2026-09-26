import asyncio
import ipaddress
import logging
import socket
import struct
import time
from copy import copy
from typing import Awaitable, Callable, List, Optional, Tuple, TypeVar

T = TypeVar("T")

# Message fragments of deterministic rejections. hypha_rpc raises every connect
# failure as ConnectionAbortedError — a bad token, a mismatched workspace and a
# client id already in use all arrive as ConnectionError subclasses — so the
# isinstance test below cannot tell them from a refused socket. Measured against
# hypha.aicell.io with hypha-rpc 0.21.x.
_FATAL_CONNECT_MARKERS = (
    "authentication error",
    "failed to authenticate",
    "client already exists",
)

# Message fragments of connection-level failures that are not raised as an
# OSError subclass: a server that is up but not yet serving (503), and the Ray
# client's own connect timeout.
_TRANSIENT_CONNECT_MARKERS = (
    "connect call failed",
    "connection refused",
    "connection timeout",
    "name resolution",
    "temporarily unavailable",
    "timed out",
    "http 502",
    "http 503",
    "http 504",
)


def is_transient_connect_error(error: BaseException) -> bool:
    """Whether a failed connection attempt is worth retrying.

    Authentication, permission and configuration errors are deterministic —
    retrying them only delays the same failure — so anything not recognised
    here is treated as fatal. The fatal markers are checked first because the
    exception type alone does not separate the two cases.
    """
    message = str(error).lower()
    if any(marker in message for marker in _FATAL_CONNECT_MARKERS):
        return False
    if isinstance(error, (ConnectionError, TimeoutError, socket.gaierror)):
        return True
    return any(marker in message for marker in _TRANSIENT_CONNECT_MARKERS)


# Retry budgets for connect_with_retry, split by call site. At startup nothing
# is watching the worker yet, so it can wait out a server restart. A reconnect
# runs inside one monitoring pass, and that pass — registration probe,
# disconnect, retries and everything after — must still finish inside the
# liveness heartbeat's staleness deadline (see bioengine.heartbeat; 120 s at
# the default monitoring interval).
STARTUP_CONNECT_BUDGET_S = 120.0
RECONNECT_BUDGET_S = 15.0


async def connect_with_retry(
    connect: Callable[[], Awaitable[T]],
    description: str,
    logger: logging.Logger,
    total_seconds: float = RECONNECT_BUDGET_S,
    initial_delay: float = 2.0,
    max_delay: float = 15.0,
) -> T:
    """Await ``connect()``, retrying connection-level failures within a budget.

    Both hypha_rpc and the Ray client reconnect an *established* connection,
    but neither retries the initial connect. A worker started inside a server's
    restart window therefore exits on the first refusal; ``STARTUP_CONNECT_BUDGET_S``
    covers the observed ~45 s hypha-server restart gap with margin while
    staying well under the Kubernetes startup probe's own budget.

    Args:
        connect: Zero-argument coroutine function performing the connection.
        description: Used in the retry log line, e.g. "Connection to Hypha".
        logger: Logger for the retry messages.
        total_seconds: Overall budget; the last attempt may start just before it
            expires. Defaults to the reconnect budget; startup callers pass
            ``STARTUP_CONNECT_BUDGET_S``.
        initial_delay: Delay before the second attempt, doubled thereafter.
        max_delay: Cap on the delay between attempts.

    Returns:
        Whatever ``connect()`` returns.

    Raises:
        Exception: The last failure, once the budget is spent or the error is
            not a connection-level one.
    """
    deadline = time.monotonic() + total_seconds
    delay = initial_delay
    while True:
        try:
            return await connect()
        except Exception as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not is_transient_connect_error(error):
                raise
            wait = min(delay, max_delay, remaining)
            logger.warning(
                f"{description} failed: {error}. Retrying in {wait:.1f}s "
                f"({remaining:.0f}s of retry budget left)..."
            )
            await asyncio.sleep(wait)
            delay *= 2


def _enumerate_ipv4_addresses() -> List[str]:
    """Enumerate IPv4 addresses bound to the host's network interfaces.

    Uses ``socket.if_nameindex`` + the ``SIOCGIFADDR`` ioctl, which works
    inside minimal Linux containers (no ``/usr/bin/ip`` required). On
    non-Linux platforms (or if ``fcntl`` is unavailable) returns an empty
    list and the caller falls back to the legacy UDP-connect trick.
    """
    try:
        import fcntl  # Linux/Unix only
    except ImportError:
        return []
    if not hasattr(socket, "if_nameindex"):
        return []

    addresses: List[str] = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _, name in socket.if_nameindex():
            if name == "lo":
                continue
            try:
                packed = fcntl.ioctl(
                    sock.fileno(),
                    0x8915,  # SIOCGIFADDR
                    struct.pack("256s", name[:15].encode()),
                )
                addr = socket.inet_ntoa(packed[20:24])
            except OSError:
                continue
            if addr == "127.0.0.1":
                continue
            addresses.append(addr)
    finally:
        sock.close()
    return addresses


def get_internal_ip() -> str:
    """
    Get the cluster-internal IPv4 address of the host.

    On multi-homed hosts (e.g. HPC login nodes with both a public network
    and a private compute network) prefer the RFC 1918 private address that
    other cluster nodes can route to. Falls back to the legacy
    ``connect-to-8.8.8.8`` trick.

    Returns:
        The internal IPv4 address as a string.
    """
    addresses = _enumerate_ipv4_addresses()
    if addresses:
        private = [a for a in addresses if ipaddress.IPv4Address(a).is_private]
        if private:
            # Prefer 10.x and 172.16-31.x over 192.168.x (compute clusters
            # typically use the larger private blocks).
            private.sort(
                key=lambda a: 0 if not a.startswith("192.168.") else 1
            )
            return private[0]
        return addresses[0]

    # Fallback: open a UDP socket to a public address; the kernel picks the
    # source IP via the routing table.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))  # No data is sent
        return s.getsockname()[0]


def acquire_free_port(
    port: int,
    step: int = 1,
    ip: Optional[str] = "localhost",
    keep_open: bool = False,
) -> Tuple[int, Optional[socket.socket]]:
    """
    Find the next free TCP port starting from a given port number.

    Tries to bind to the port; if unavailable, increments by `step`
    until a free port is found.

    Args:
        port: Starting port number to check.
        step: Increment between port numbers to check.
        ip: IP address to bind to (default: 'localhost').
        keep_open: Whether to keep the socket open after finding a free port.
                   Useful when reserving multiple ports to avoid race conditions.

    Returns:
        (port, socket): The free port number and, if `keep_open` is True,
                        the open socket object (otherwise None).
    """
    port = copy(port)

    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allow quick reuse
        try:
            s.bind((ip, port))
            s.listen(1)  # mark as a passive socket (for good measure)
            _, bound_port = s.getsockname()

            if not keep_open:
                s.close()
                s = None

            return bound_port, s

        except OSError:
            s.close()
            port += step


if __name__ == "__main__":
    print("Internal IP:", get_internal_ip())
    port, s = acquire_free_port(8000)
    print("Free port:", port)
