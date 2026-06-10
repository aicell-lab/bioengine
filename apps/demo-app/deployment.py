"""BioEngine demo application — single-deployment v0.6.0 reference.

Demonstrates the modern authoring model:

* ``@bioengine.app(...)`` instead of ``@serve.deployment(ray_actor_options=...)``
* Lifecycle hooks via decorators (``@async_init``, ``@smoke_test``,
  ``@health_check``) — method names are free.
* API methods via ``@bioengine.method`` instead of ``@schema_method``.
* Multiplexed model loading via ``@bioengine.multiplexed``.
* Module-level access to datasets and the logger.

Import rule: this module contains ``@bioengine.app`` and is loaded by
the worker's introspection Ray task, which runs with only
``bioengine[worker]`` and the standard library installed. ``pandas``
ships to the replica via the decorator's ``pip=`` arg and is imported
lazily inside the smoke test below.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Union

from pydantic import Field

import bioengine

logger = bioengine.logger


@bioengine.app(
    num_cpus=1,
    num_gpus=0,
    memory_mb=512,
    pip=["pandas"],
    env_vars={"EXAMPLE_ENV_VAR": "example_value"},
)
class DemoApp:
    """A small reference app exercising every BioEngine authoring decorator."""

    def __init__(self, test_param: str = "default_value") -> None:
        self.start_time = time.time()
        self.fail_health_check = False

    @bioengine.async_init
    async def warm_up(self) -> None:
        """Optional async init — runs once before traffic is admitted."""
        import asyncio

        await asyncio.sleep(0.01)

    @bioengine.smoke_test
    async def smoke(self) -> None:
        """Optional startup smoke test — fails the replica if it raises."""
        import os

        import pandas  # noqa: F401  # confirms pip=["pandas"] reached this replica

        os.environ["EXAMPLE_ENV_VAR"]
        await self._get_model("test_model")
        await self.ping()
        await self.ascii_art()
        reversed_result = await self.reverse_text(text="hello")
        assert reversed_result["reversed"] == "olleh", reversed_result

    @bioengine.health_check
    async def health(self) -> None:
        """Periodic health check called by Ray Serve."""
        if self.fail_health_check:
            raise Exception("Simulated health check failure.")

    @bioengine.multiplexed(max_models=3)
    async def _get_model(self, model_id: str) -> Any:
        """Multiplexed model loader (mock)."""
        logger.info(f"Loading model with ID: {model_id}")
        return None

    @bioengine.method
    async def ping(self) -> Dict[str, Union[str, float]]:
        """Ping the app to test connectivity.

        Returns:
            Dict[str, Union[str, float]]: status, message, timestamp, timezone, uptime.
        """
        return {
            "status": "ok",
            "message": "Hello from the DemoApp!",
            "timestamp": datetime.now().isoformat(),
            "timezone": time.tzname[0],
            "uptime": time.time() - self.start_time,
        }

    @bioengine.method
    async def ascii_art(self) -> List[str]:
        """Return an ASCII art rendering of 'BioEngine'."""
        return [
            """|================================================================================================|""",
            """|                                                                                                |""",
            """|  oooooooooo.   o8o            oooooooooooo                         o8o                         |""",
            """|  `888'   `Y8b  `''            `888'     `8                         `''                         |""",
            """|   888     888 oooo   .ooooo.   888         ooo. .oo.    .oooooooo oooo  ooo. .oo.    .ooooo.   |""",
            """|   888oooo888' `888  d88' `88b  888oooo8    `888P'Y88b  888' `88b  `888  `888P'Y88b  d88' `88b  |""",
            """|   888    `88b  888  888   888  888          888   888  888   888   888   888   888  888ooo888  |""",
            """|   888    .88P  888  888   888  888       o  888   888  `88bod8P'   888   888   888  888    .o  |""",
            """|  o888bood8P'  o888o `Y8bod8P' o888ooooood8 o888o o888o `8oooooo.  o888o o888o o888o `Y8bod8P'  |""",
            """|                                                        d'     YD                               |""",
            """|                                                        'Y88888P'                               |""",
            """|================================================================================================|""",
        ]

    @bioengine.method
    async def list_datasets(self) -> Dict[str, List[str]]:
        """List available datasets and their files.

        Returns:
            Dict mapping dataset id → list of file names.
        """
        available_datasets = await bioengine.datasets.list_datasets()
        logger.info(f"Available datasets: {list(available_datasets.keys())}")

        data_files: Dict[str, List[str]] = {}
        for dataset_id in available_datasets.keys():
            data_files[dataset_id] = await bioengine.datasets.list_files(dataset_id)
        return data_files

    @bioengine.method
    async def reverse_text(
        self,
        text: str = Field(..., description="Text to reverse"),
    ) -> Dict[str, Union[str, int]]:
        """Reverse a string and return metadata.

        Returns:
            original, reversed, length.
        """
        return {
            "original": text,
            "reversed": text[::-1],
            "length": len(text),
        }

    @bioengine.method
    async def set_fail_health_check(self) -> None:
        """Force the next health check to fail (for testing)."""
        self.fail_health_check = True
