"""BioEngine app wrapper: serve the OME-Zarr view catalog from a Ray Serve replica.

The view server is an HTTP app — viewers fetch Zarr chunks and image tiles over
plain GET — so this registers it with Hypha as an ASGI service rather than as a
set of RPC methods. Hypha then fronts it at
``{server_url}/{workspace}/apps/{service_id}/``, which is a normal URL a browser,
Vizarr, Neuroglancer or a Python client can use unchanged.

The app source itself is untouched by this: ``create_app`` is the same factory
the standalone server uses.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import bioengine
from hypha_rpc import connect_to_server

logger = bioengine.logger

APP_DIR = Path(__file__).resolve().parent
SERVICE_ID = "omezarr-view"


def _read_pip(name: str) -> List[str]:
    path = APP_DIR / name
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


@bioengine.app(
    num_cpus=2,
    memory_mb=4096,
    pip=_read_pip("requirements.txt"),
    max_ongoing_requests=32,
    autoscaling_config={"min_replicas": 1, "initial_replicas": 1, "max_replicas": 1},
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=60.0,
)
class OmeZarrViewApp:
    """Serves the catalog, the OME-Zarr endpoints, tiles and annotations."""

    def __init__(self, datasets_config: Optional[str] = None) -> None:
        # Inside the cluster HYPHA_SERVER_URL is an in-cluster address, which
        # is fine to connect through but must never be advertised: a browser
        # cannot resolve it. The public base is separate and explicit.
        self.server_url = os.environ.get("HYPHA_SERVER_URL",
                                         "https://hypha.aicell.io")
        self.public_server = os.environ.get("HYPHA_PUBLIC_URL",
                                           "https://hypha.aicell.io")
        # Pinning the client id keeps the served URL stable across redeploys;
        # otherwise every replica gets a random id and the URL changes.
        self.client_id = os.environ.get("OMEZARR_VIEW_CLIENT_ID", "omezarr-view")
        self._token = os.environ.get("HYPHA_TOKEN")
        self._config = datasets_config or str(APP_DIR / "datasets.yaml")
        self._asgi = None
        self.service_url: Optional[str] = None

    @bioengine.async_init
    async def _async_init(self) -> None:
        if not self._token:
            raise RuntimeError(
                "HYPHA_TOKEN is not set. Pass hypha_token= to deploy_app; the "
                "replica needs it to register its own service."
            )
        self.hypha_client = await connect_to_server({
            "server_url": self.server_url,
            "token": self._token,
            "client_id": self.client_id,
        })
        workspace = self.hypha_client.config.workspace
        client_id = self.hypha_client.config.client_id

        from omezarr_view.server import create_app

        # Hypha serves this behind its own path prefix, so the app has to build
        # its public URLs from that prefix rather than from the request host.
        # Hypha addresses an ASGI service by CLIENT-scoped id, not by the bare
        # service name.
        public_url = (f"{self.public_server}/{workspace}/apps/"
                      f"{client_id}:{SERVICE_ID}")
        self._asgi = create_app(self._config, public_url=public_url)

        await self.hypha_client.register_service({
            "id": SERVICE_ID,
            "name": "OME-Zarr view",
            "description": "Serves existing image files as lazy OME-Zarr views",
            "type": "asgi",
            "config": {"visibility": "public", "require_context": False},
            "serve": self._serve,
        })
        self.service_url = public_url
        logger.info(f"OME-Zarr view app serving at {public_url}")

    async def _serve(self, args: Dict[str, Any]) -> None:
        """Bridge Hypha's ASGI transport to the FastAPI app."""
        await self._asgi(args["scope"], args["receive"], args["send"])

    @bioengine.method()
    async def get_service_url(self) -> Dict[str, Any]:
        """Where the view catalog is reachable, for clients that only speak RPC."""
        return {
            "service_url": self.service_url,
            "catalog": self.service_url,
            "datasets_api": f"{self.service_url}/api/datasets",
            "note": "zarr endpoints are at {service_url}/zarr/{dataset_id}",
        }

    @bioengine.health_check
    async def _health(self) -> Dict[str, Any]:
        return {"status": "ok" if self._asgi is not None else "starting"}
