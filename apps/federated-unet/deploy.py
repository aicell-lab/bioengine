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
        "worker": "ws-user-github|49943582/bioengine-worker-europa:bioengine-worker",
        "token_key": "HYPHA_TOKEN",
        "kwargs": {"site_name": "site-a-europa", "datasets": ["dsb2018-fluo"], "role": "participant"},
    },
    "site-b": {
        "worker": "bioimage-io/bioengine-worker-denbi-6d8d6dd6d5-zvwxn:bioengine-worker",
        "token_key": "BIOIMAGE_IO_TOKEN",
        "kwargs": {"site_name": "site-b-denbi", "datasets": ["bbbc010-worms"], "role": "participant"},
    },
    "pooled": {
        "worker": "ws-user-github|49943582/bioengine-worker-europa:bioengine-worker",
        "token_key": "HYPHA_TOKEN",
        "kwargs": {
            "site_name": "pooled-oracle-europa",
            "datasets": ["dsb2018-fluo", "bbbc010-worms"],
            "role": "pooled-oracle",
        },
    },
}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--only", nargs="+", default=list(INSTANCES))
    args = parser.parse_args()

    env = dotenv_values(REPO_ROOT / ".env")
    for name in args.only:
        spec = INSTANCES[name]
        server = await connect_to_server({"server_url": SERVER_URL, "token": env[spec["token_key"]]})
        worker = await server.get_service(spec["worker"])
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
            worker = await server.get_service(spec["worker"])
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
