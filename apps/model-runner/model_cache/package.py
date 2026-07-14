"""Per-use lock context manager around a cached bioimage.io model package.

A ``BioimageioPackage`` wraps a cached model directory on disk and provides
two pieces of cross-replica coordination:

* an ``.in_use_<replica_id>_<token>`` lock file the cache reads when
  deciding whether the model can be evicted
* a ``.last_access`` timestamp the cache reads when picking the LRU
  victim during ``_ensure_cache_space``

Acquire by ``async with package: …``; the ``__aenter__`` writes the lock
file before the body runs and the ``__aexit__`` removes it + bumps
``.last_access`` afterwards.
"""


import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ray.serve")


class BioimageioPackage:
    """Wrapper for a cached bioimage.io model package with access tracking."""

    def __init__(
        self,
        package_path: Path,
        latest_remote_modified: float,
        replica_id: str,
    ) -> None:
        self.package_path = package_path
        self.source = str(self.package_path / "rdf.yaml")
        self.latest_remote_modified = latest_remote_modified
        self.replica_id = replica_id
        self._lock_file: Optional[Path] = None

    async def __aenter__(self):
        """Create a per-use lock file so the model is not evicted while in use."""
        token = int(time.time() * 1_000_000)
        self._lock_file = self.package_path / f".in_use_{self.replica_id}_{token}"
        try:
            await asyncio.to_thread(self._lock_file.write_text, str(token))
        except (OSError, IOError) as e:
            logger.warning(f"⚠️ Failed to create in-use lock file: {e}")
            self._lock_file = None
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Remove the per-use lock file and update the last-access timestamp."""
        if self._lock_file is not None:
            try:
                await asyncio.to_thread(self._lock_file.unlink, True)
            except (OSError, IOError) as e:
                logger.warning(f"⚠️ Failed to remove in-use lock file: {e}")
        access_file = self.package_path / ".last_access"
        try:
            await asyncio.to_thread(access_file.write_text, str(time.time()))
        except (OSError, IOError) as e:
            logger.warning(f"⚠️ Failed to update access time on exit: {e}")
