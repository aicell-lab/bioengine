"""RuntimeC — date/time operations."""

import datetime
import time

import bioengine

from utils import base_status

logger = bioengine.logger


@bioengine.app(num_cpus=1, num_gpus=0, memory_mb=512, max_ongoing_requests=5)
class RuntimeC:
    @bioengine.async_init
    async def warm_up(self) -> None:
        logger.info("RuntimeC ready")

    @bioengine.method
    async def ping(self) -> str:
        """Liveness ping."""
        return "pong"

    @bioengine.method
    async def get_status(self) -> dict:
        """Self-describe."""
        return base_status(
            "runtime_c",
            current_time=datetime.datetime.now().isoformat(),
        )

    @bioengine.method
    async def time_ops(self, count: int = 5) -> dict:
        """Generate timestamps and formatted date strings."""
        now = datetime.datetime.now()
        timestamps = [
            (now - datetime.timedelta(hours=i)).isoformat() for i in range(count)
        ]
        return {
            "current_timestamp": now.isoformat(),
            "unix_timestamp": time.time(),
            "timestamps": timestamps,
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "count": count,
        }
