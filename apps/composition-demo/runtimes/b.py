"""RuntimeB — stats with numpy at the top of the file (no more inside-method imports)."""

import numpy as np

import bioengine

from utils import base_status

logger = bioengine.logger


@bioengine.app(
    num_cpus=1,
    num_gpus=0,
    memory_mb=512,
    pip=["numpy==1.26.4"],
    max_ongoing_requests=5,
)
class RuntimeB:
    @bioengine.async_init
    async def warm_up(self) -> None:
        logger.info(f"RuntimeB ready (numpy {np.__version__})")

    @bioengine.method
    async def ping(self) -> str:
        """Liveness ping."""
        return "pong"

    @bioengine.method
    async def get_status(self) -> dict:
        """Self-describe."""
        return base_status("runtime_b", numpy_version=np.__version__)

    @bioengine.method
    async def analyze(self, values: list) -> dict:
        """Run statistical analysis on a list of numbers."""
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "sum": float(np.sum(arr)),
            "count": len(arr),
            "sorted": sorted(values),
        }
