"""Entry app — orchestrates RuntimeA, RuntimeB, RuntimeC via type-hint composition."""

import asyncio
import time

import bioengine
from pydantic import Field

from .runtimes.a import RuntimeA
from .runtimes.b import RuntimeB
from .runtimes.c import RuntimeC
from .utils import assert_pong

logger = bioengine.logger


@bioengine.app(
    num_cpus=0,
    num_gpus=0,
    memory_mb=256,
    max_ongoing_requests=20,
)
class EntryApp:
    def __init__(
        self,
        runtime_a: RuntimeA,
        runtime_b: RuntimeB,
        runtime_c: RuntimeC,
    ) -> None:
        self.runtime_a = runtime_a
        self.runtime_b = runtime_b
        self.runtime_c = runtime_c
        self.start_time = time.time()

    @bioengine.smoke_test
    async def _check_runtimes(self) -> None:
        """Verify all three runtimes are reachable — exercises the no-.remote() proxy."""
        assert_pong(await self.runtime_a.ping())
        assert_pong(await self.runtime_b.ping())
        assert_pong(await self.runtime_c.ping())
        logger.info("EntryApp smoke test passed")

    @bioengine.method
    async def status(self) -> dict:
        """Status snapshot of entry and all runtimes.

        Returns:
            uptime + each runtime's self-description.
        """
        a, b, c = await asyncio.gather(
            self.runtime_a.get_status(),
            self.runtime_b.get_status(),
            self.runtime_c.get_status(),
        )
        return {
            "entry_uptime": time.time() - self.start_time,
            "runtime_a": a,
            "runtime_b": b,
            "runtime_c": c,
        }

    @bioengine.method
    async def process_text(
        self,
        text: str = Field(..., description="Text to process"),
    ) -> dict:
        """Process text via RuntimeA."""
        return await self.runtime_a.process_text(text)

    @bioengine.method
    async def analyze_numbers(
        self,
        values: list = Field(..., description="List of numbers"),
    ) -> dict:
        """Run statistical analysis via RuntimeB."""
        return await self.runtime_b.analyze(values)

    @bioengine.method
    async def time_operations(
        self,
        count: int = Field(5, description="How many timestamps to generate"),
    ) -> dict:
        """Generate time-based strings via RuntimeC."""
        return await self.runtime_c.time_ops(count)

    @bioengine.method
    async def run_all(
        self,
        text: str = Field("hello bioengine", description="Text input for RuntimeA"),
        values: list = Field(None, description="Numbers for RuntimeB (defaults to [1..5])"),
        count: int = Field(3, description="Count for RuntimeC"),
    ) -> dict:
        """Fan out to all three runtimes in parallel and combine the results."""
        if values is None:
            values = [1, 2, 3, 4, 5]
        text_result, data_result, time_result = await asyncio.gather(
            self.runtime_a.process_text(text),
            self.runtime_b.analyze(values),
            self.runtime_c.time_ops(count),
        )
        return {
            "text_result": text_result,
            "data_result": data_result,
            "time_result": time_result,
        }
