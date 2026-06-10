"""Synthetic single-deployment app for bootstrap tests."""

from pydantic import Field

import bioengine


@bioengine.app(num_cpus=1, memory_mb=128)
class FixtureApp:
    def __init__(self, greeting: str = "hi"):
        self.greeting = greeting

    @bioengine.async_init
    async def setup(self):
        self._ready = True

    @bioengine.smoke_test
    async def _smoke(self):
        assert self.greeting

    @bioengine.method
    async def hello(self, name: str = Field(..., description="who to greet")) -> dict:
        """Return a greeting."""
        return {"msg": f"{self.greeting} {name}"}

    @bioengine.method
    async def echo(self, value: str) -> str:
        """Echo back."""
        return value
