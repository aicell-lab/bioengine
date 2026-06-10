"""Composition fixture — runtime A."""

import bioengine

from .utils import shout


@bioengine.app(num_cpus=1, memory_mb=64)
class RuntimeA:
    @bioengine.method
    async def shout(self, text: str) -> str:
        """Shout a message via the shared helper."""
        return shout(text)
