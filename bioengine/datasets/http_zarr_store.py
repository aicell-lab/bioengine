import asyncio
import logging
import os
from asyncio import Semaphore
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Optional

import httpx

try:
    import zarr
except ImportError as e:
    raise ImportError("zarr>=3.0.8 is required") from e

if zarr.__version__ < "3.0.8":
    raise ImportError(f"zarr>=3.0.8 is required but found {zarr.__version__}")

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype

from bioengine.datasets.chunk_cache import ChunkCache, default_cache
from bioengine.datasets.utils import get_url_with_retry


@dataclass
class HttpZarrStore(Store):
    """
    HTTP-based Zarr store for efficient remote dataset access.

    Implements the Zarr Store interface for accessing datasets over HTTP with
    HTTP range requests and a shared LRU chunk cache.

    Multiple stores opened in the same process share a single ``ChunkCache``
    instance (``default_cache``) by default, so the memory budget is enforced
    across all open zarr files rather than per-store. Pass a custom
    ``chunk_cache`` to override.

    Key Features:
    - Efficient partial data access through HTTP range requests
    - Authentication via token passed on each request
    - Shared LRU chunk data cache across all store instances (configurable)
    - Compatible with standard Zarr API for seamless integration
    - Read-only access with clear error handling for write operations

    Implementation Details:
    The store treats ``base_url`` as the Zarr root and appends ``/{key}`` for
    each chunk request. ``base_url`` may point to a BioEngine local data server
    path (``{service_url}/data/{dataset_id}/{zarr_path}``) or to any
    HTTPS-served Zarr root (e.g. an OME-Zarr URI from the BioImage Archive).
    An optional ``token`` is appended as ``?token=`` for the local-server case.
    """

    dataset_id: Optional[str] = None
    zarr_path: Optional[str] = None
    _read_only: bool = True

    supports_writes: bool = False
    supports_deletes: bool = False
    supports_partial_writes: bool = False
    supports_listing: bool = False

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        chunk_cache: ChunkCache = default_cache,
        max_concurrent_requests: int = int(
            os.getenv("BIOENGINE_DATASETS_ZARR_STORE_CONCURRENT_REQUESTS", 50)
        ),
        max_connections: int = int(
            os.getenv("BIOENGINE_DATASETS_ZARR_STORE_CONNECTIONS", 100)
        ),
        logger: logging.Logger = logging.getLogger("HttpZarrStore"),
        dataset_id: Optional[str] = None,
        zarr_path: Optional[str] = None,
    ):
        """
        Initialize the HTTP-based Zarr store for remote dataset access.

        Args:
            base_url: URL of the Zarr root. Either a local-data-server path of
                the form ``{service_url}/data/{dataset_id}/{zarr_path}`` or an
                arbitrary HTTPS Zarr root (e.g. an OME-Zarr URI from a public
                repository). The store appends ``/{key}`` for each chunk.
            token: Authentication token for access control. When set, appended
                as ``?token=`` to each chunk URL. Public Zarr roots (e.g. BIA)
                don't need it.
            chunk_cache: Shared LRU cache for chunk data. Defaults to the
                process-wide ``default_cache`` so all stores share one budget.
                Pass ``ChunkCache(max_size_gb=0)`` to disable caching.
            max_concurrent_requests: Maximum number of concurrent HTTP requests
            max_connections: Maximum number of HTTP connections in pool
            logger: Logger instance for logging messages
            dataset_id: BioEngine dataset id this store points at, when the
                URL came from ``BioEngineDatasets.get_file`` — exposed for
                downstream consumers that need to round-trip the identifier.
                Left as ``None`` for arbitrary external Zarr roots.
            zarr_path: Zarr path within the dataset, complementary to
                ``dataset_id``. ``None`` for external roots.
        """
        super().__init__(read_only=True)
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.logger = logger
        self._chunk_cache = chunk_cache
        self.dataset_id = dataset_id
        self.zarr_path = zarr_path

        # Concurrency control
        self._request_semaphore = Semaphore(max_concurrent_requests)

        # Configure HTTP/2 and connection pooling for optimal performance
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections // 2,
        )
        self.http_client = httpx.AsyncClient(
            timeout=60,  # seconds
            limits=limits,
            http2=True,
            follow_redirects=True,
        )

    def _build_url(self, key: str) -> str:
        """Build the chunk URL for a key, appending the token if present."""
        url = f"{self.base_url}/{key.lstrip('/')}"
        if self.token:
            url += f"?token={self.token}"
        return url

    def _cache_key(self, key: str, byte_range: ByteRequest | None) -> str:
        """Generate a cache key scoped to this store's base_url."""
        if isinstance(byte_range, RangeByteRequest):
            range_str = f":range:{byte_range.start}-{byte_range.end}"
        elif isinstance(byte_range, OffsetByteRequest):
            range_str = f":offset:{byte_range.offset}"
        elif isinstance(byte_range, SuffixByteRequest):
            range_str = f":suffix:{byte_range.suffix}"
        else:
            range_str = ""
        return f"{self.base_url}/{key}{range_str}"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, HttpZarrStore)
            and self.base_url == other.base_url
            and self.token == other.token
        )

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest = None
    ) -> Buffer | None:
        """
        Get data from the store for a specific key with optional range specification.

        Checks the shared chunk cache first, then fetches from the remote server
        if not cached. Range requests are used for partial data access.

        Args:
            key: Path to the chunk within the zarr store
            prototype: Buffer prototype for creating the return buffer
            byte_range: Optional specification for partial data access

        Returns:
            Buffer containing the requested data, or None if the key doesn't exist
        """
        cache_key = self._cache_key(key, byte_range)
        cached = await self._chunk_cache.get(cache_key)
        if cached is not None:
            return cached

        async with self._request_semaphore:
            url = self._build_url(key)
            headers = {}
            if isinstance(byte_range, RangeByteRequest):
                headers["Range"] = f"bytes={byte_range.start}-{byte_range.end - 1}"
            elif isinstance(byte_range, OffsetByteRequest):
                headers["Range"] = f"bytes={byte_range.offset}-"
            elif isinstance(byte_range, SuffixByteRequest):
                headers["Range"] = f"bytes=-{byte_range.suffix}"

            try:
                response = await get_url_with_retry(
                    url=url,
                    headers=headers,
                    raise_for_status=False,
                    http_client=self.http_client,
                    logger=self.logger,
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                content = response.content
                self.logger.debug(
                    f"Fetched {len(content)} bytes for {key} "
                    f"(range={headers.get('Range')})"
                )
                buffer = prototype.buffer.from_bytes(content)
                await self._chunk_cache.put(cache_key, buffer)
                return buffer

            except Exception as e:
                self.logger.error(
                    f"Failed to fetch {key} (range={headers.get('Range')}) "
                    f"from {url}: {e}"
                )
                raise

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Retrieve multiple chunks in parallel."""
        return await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges))

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the store without fetching its content."""
        url = self._build_url(key)
        response = await self.http_client.get(url, headers={"Range": "bytes=0-0"})
        return response.status_code in (200, 206)

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("Write not supported")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("Delete not supported")

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, bytes]]
    ) -> None:
        raise NotImplementedError("Partial write not supported")

    def list(self) -> AsyncIterator[str]:
        raise NotImplementedError("Listing not supported")

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("Prefix listing not supported")

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("Dir listing not supported")

    def close(self):
        # The shared cache is intentionally not cleared here — other stores
        # may still be using it. Call chunk_cache.clear() explicitly if needed.
        return super().close()
