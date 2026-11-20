"""
Shared LRU chunk cache for HttpZarrStore instances.

A single ChunkCache can be shared across multiple HttpZarrStore instances so
that the memory budget applies to the whole process rather than per-store.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from zarr.core.buffer import Buffer

_DEFAULT_CACHE_SIZE_GB = int(os.getenv("BIOENGINE_DATASETS_ZARR_STORE_CACHE_SIZE", 1))


class ChunkCache:
    """
    Size-limited LRU cache for zarr chunk buffers.

    Thread-safe for use across multiple HttpZarrStore instances in the same
    asyncio event loop. The total memory used by all cached chunks is bounded
    by ``max_size_gb``.
    """

    def __init__(
        self,
        max_size_gb: int = _DEFAULT_CACHE_SIZE_GB,
        logger: logging.Logger = logging.getLogger("ChunkCache"),
    ):
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.logger = logger
        self._cache: dict = {}  # {key: (buffer, size)}
        self._order: list = []  # LRU order, oldest first
        self._total_size: int = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Buffer]:
        async with self._lock:
            if key not in self._cache:
                return None
            buffer, size = self._cache[key]
            # Move to most-recently-used position
            self._order.remove(key)
            self._order.append(key)
            self.logger.debug(f"Cache hit: {key} ({size} bytes)")
            return buffer

    async def put(self, key: str, buffer: Buffer) -> None:
        size = len(buffer.to_bytes()) if hasattr(buffer, "to_bytes") else 0
        if size > self.max_size_bytes:
            self.logger.warning(
                f"Chunk {key} ({size} bytes) exceeds total cache capacity "
                f"({self.max_size_bytes} bytes), not caching"
            )
            return

        async with self._lock:
            # Evict until there is room
            while self._total_size + size > self.max_size_bytes and self._order:
                oldest = self._order.pop(0)
                if oldest in self._cache:
                    _, evicted_size = self._cache.pop(oldest)
                    self._total_size -= evicted_size
                    self.logger.debug(f"Evicted: {oldest} ({evicted_size} bytes)")

            self._cache[key] = (buffer, size)
            self._order.append(key)
            self._total_size += size
            self.logger.debug(
                f"Cached: {key} ({size} bytes), "
                f"total {self._total_size / self.max_size_bytes * 100:.1f}% full"
            )

    async def resize(self, max_size_gb: int) -> None:
        """Change the cache size limit, evicting LRU entries immediately if needed."""
        new_max = max_size_gb * 1024 * 1024 * 1024
        async with self._lock:
            self.max_size_bytes = new_max
            while self._total_size > self.max_size_bytes and self._order:
                oldest = self._order.pop(0)
                if oldest in self._cache:
                    _, evicted_size = self._cache.pop(oldest)
                    self._total_size -= evicted_size
                    self.logger.debug(
                        f"Evicted on resize: {oldest} ({evicted_size} bytes)"
                    )
        self.logger.info(
            f"Cache resized to {max_size_gb} GB "
            f"({self._total_size / max(new_max, 1) * 100:.1f}% full)"
        )

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()
        self._total_size = 0


# Process-wide default cache — shared by all HttpZarrStore instances unless
# a different cache is passed explicitly.
default_cache = ChunkCache()
