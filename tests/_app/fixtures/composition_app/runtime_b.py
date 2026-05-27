"""Composition fixture — runtime B."""

import bioengine


@bioengine.app(num_cpus=1, memory_mb=64)
class RuntimeB:
    @bioengine.method
    async def add(self, a: int, b: int) -> int:
        """Sum two ints."""
        return a + b
