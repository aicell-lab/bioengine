"""Companion app for the model-runner browser UI.

The UI (frontend/index.html) talks directly to the model-runner service on
the same worker; this entry only exists so the frontend is deployable and
reachable through the worker dashboard's "Open App" card.
"""

import time
from datetime import datetime
from typing import Dict, Union

import bioengine


@bioengine.app(num_cpus=1, memory_mb=256)
class ModelRunnerUI:
    def __init__(self) -> None:
        self.start_time = time.time()

    @bioengine.method
    async def ping(self) -> Dict[str, Union[str, float]]:
        """Liveness check for the UI companion app."""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
        }
