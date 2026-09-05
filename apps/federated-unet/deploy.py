"""Deploy or redeploy the three federated-unet instances.

    python deploy.py [--version 0.1.0] [--only site-a site-b pooled]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import dotenv_values
from hypha_rpc import connect_to_server

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_URL = "https://hypha.aicell.io"
ARTIFACT = "bioimage-io/federated-unet"

INSTANCES = {
    "site-a": {
        "worker": "ws-user-github|49943582/bioengine-worker-europa",
        "token_key": "HYPHA_TOKEN",
        "kwargs": {"site_name": "site-a-europa", "role": "participant"},
    },
    "site-b": {
        "worker": "bioimage-io/bioengine-worker-denbi",
        "token_key": "BIOIMAGE_IO_TOKEN",
        "kwargs": {"site_name": "site-b-denbi", "role": "participant"},
    },
    "pooled": {
        "worker": "ws-user-github|49943582/bioengine-worker-europa",
        "token_key": "HYPHA_TOKEN",
        "kwargs": {"site_name": "pooled-oracle-europa", "role": "pooled-oracle"},
    },
}

# caricature: two object classes, an existence proof that flatters federation.
# acquisition: one class split by imaging modality, the honest utility bound.
SPLITS = {
    "caricature": ("dsb2018-fluo", "bbbc010-worms"),
    "acquisition": ("bbbc038-fluo", "bbbc038-histo"),
}


async def resolve_worker(server, prefix: str):
    """Find the worker service whose id starts with ``prefix``.

    Kubernetes bakes the pod name into the service id, so a hardcoded id goes
    stale the moment the worker pod is replaced -- which has already happened
    once here, on de.NBI.
    """
    workspace = prefix.split("/", 1)[0]
    matches = [
        service["id"]
        for service in await server.list_services(workspace)
        if service["id"].startswith(prefix) and service["id"].endswith(":bioengine-worker")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one worker matching {prefix!r}, got {matches}")
    return await server.get_service(matches[0])


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--only", nargs="+", default=list(INSTANCES))
    parser.add_argument("--split", choices=sorted(SPLITS), default="caricature")
    args = parser.parse_args()

    site_a, site_b = SPLITS[args.split]
    datasets = {"site-a": [site_a], "site-b": [site_b], "pooled": [site_a, site_b]}

    env = dotenv_values(REPO_ROOT / ".env")
    for name in args.only:
        spec = INSTANCES[name]
        spec["kwargs"] = {**spec["kwargs"], "datasets": datasets[name]}
        server = await connect_to_server({"server_url": SERVER_URL, "token": env[spec["token_key"]]})
        worker = await resolve_worker(server, spec["worker"])
        app_id = await worker.deploy_app(
            artifact_id=ARTIFACT,
            version=args.version,
            application_id=f"fedunet-{name}",
            application_kwargs={"FederatedUNetSite": spec["kwargs"]},
            # The run artifact lives in bioimage-io, so every instance needs a
            # credential that can write there regardless of which worker it is on.
            hypha_token=env["BIOIMAGE_IO_TOKEN"],
        )
        print(f"{name}: deploying {app_id}", flush=True)
        await server.disconnect()

    print("\nwaiting for RUNNING (env builds take a while on first deploy)", flush=True)
    for _ in range(180):
        await asyncio.sleep(20)
        states = {}
        for name in args.only:
            spec = INSTANCES[name]
            server = await connect_to_server({"server_url": SERVER_URL, "token": env[spec["token_key"]]})
            worker = await resolve_worker(server, spec["worker"])
            record = (await worker.get_app_status([f"fedunet-{name}"])).get(f"fedunet-{name}", {})
            states[name] = record.get("status", "GONE")
            if states[name] not in ("RUNNING", "DEPLOYING", "NOT_STARTED"):
                print(f"\n{name} -> {states[name]}")
                print(json.dumps(record, indent=2, default=str)[:6000])
            await server.disconnect()
        print(f"  {states}", flush=True)
        if all(s == "RUNNING" for s in states.values()):
            print("all RUNNING")
            return
        if any(s not in ("RUNNING", "DEPLOYING", "NOT_STARTED") for s in states.values()):
            sys.exit(1)
    sys.exit("timed out")


if __name__ == "__main__":
    asyncio.run(main())
