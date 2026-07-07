"""RuntimeB — stats backed by numpy.

This file hosts ``@bioengine.app`` so the introspection Ray task imports
it. Heavy deps stay out of the top level — ``numpy`` lives in the
sibling ``numpy_ops`` module which is imported lazily inside method
bodies.
"""

from pathlib import Path

import bioengine

from utils import base_status

logger = bioengine.logger


def _read_pip(name: str) -> list[str]:
    """Load a ``requirements-*.txt`` file next to this module."""
    text = (Path(__file__).parent / name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@bioengine.app(
    num_cpus=1,
    num_gpus=0,
    memory_mb=512,
    pip=_read_pip("requirements-b.txt"),
    max_ongoing_requests=5,
)
class RuntimeB:
    @bioengine.async_init
    async def warm_up(self) -> None:
        from numpy_ops import numpy_version

        logger.info(f"RuntimeB ready (numpy {numpy_version()})")

    @bioengine.method
    async def ping(self) -> str:
        """Liveness ping."""
        return "pong"

    @bioengine.method
    async def get_status(self) -> dict:
        """Self-describe."""
        from numpy_ops import numpy_version

        return base_status("runtime_b", numpy_version=numpy_version())

    @bioengine.method
    async def analyze(self, values: list) -> dict:
        """Run statistical analysis on a list of numbers."""
        from numpy_ops import stats

        return stats(values)
