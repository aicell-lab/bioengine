"""RuntimeA — pure-stdlib text operations."""

import bioengine

from ..utils import base_status

logger = bioengine.logger


@bioengine.app(num_cpus=1, num_gpus=0, memory_mb=512, max_ongoing_requests=5)
class RuntimeA:
    @bioengine.async_init
    async def warm_up(self) -> None:
        logger.info("RuntimeA ready")

    @bioengine.method
    async def ping(self) -> str:
        """Liveness ping."""
        return "pong"

    @bioengine.method
    async def get_status(self) -> dict:
        """Self-describe."""
        return base_status("runtime_a", capabilities=["text_processing"])

    @bioengine.method
    async def process_text(self, text: str) -> dict:
        """Split, count, transform a string."""
        words = text.split()
        return {
            "word_count": len(words),
            "char_count": len(text),
            "char_count_no_spaces": len(text.replace(" ", "")),
            "words": words,
            "reversed": text[::-1],
            "upper": text.upper(),
            "lower": text.lower(),
            "title": text.title(),
        }
