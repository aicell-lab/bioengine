"""Composition fixture — entry app referencing two runtimes by type hint."""

import bioengine

from .runtime_a import RuntimeA
from .runtime_b import RuntimeB


@bioengine.app(num_cpus=0, memory_mb=64)
class Entry:
    def __init__(self, runtime_a: RuntimeA, runtime_b: RuntimeB, label: str = "demo"):
        self.runtime_a = runtime_a
        self.runtime_b = runtime_b
        self.label = label

    @bioengine.method
    async def shout(self, text: str) -> str:
        """Forward to RuntimeA — exercises the no-.remote() proxy."""
        return await self.runtime_a.shout(text)

    @bioengine.method
    async def add(self, a: int, b: int) -> int:
        """Forward to RuntimeB."""
        return await self.runtime_b.add(a, b)
