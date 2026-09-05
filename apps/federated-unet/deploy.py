"""Deploy or redeploy the instances of one federation layout.

    python deploy.py --version 0.3.1 --layout acquisition-4site [--only fluo-0 pooled]

A layout names the clients, what each one holds, and which cluster it lands on.
Placement is a capacity decision rather than a scientific one — see PLACEMENT —
but it is recorded per instance because co-located clients have to be disclosed:
"one client per site" is the framing, and two clients sharing a GPU are not two
sites in any physical sense.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import dotenv_values
from hypha_rpc import connect_to_server

from workers import resolve_worker

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_URL = "https://hypha.aicell.io"
ARTIFACT = "bioimage-io/federated-unet"

EUROPA = "ws-user-github|49943582/bioengine-worker-europa"
DENBI = "bioimage-io/bioengine-worker-denbi"

WORKERS = {
    EUROPA: {"token_key": "HYPHA_TOKEN", "description": "Europa single-machine worker, Stockholm"},
    DENBI: {"token_key": "BIOIMAGE_IO_TOKEN", "description": "de.NBI cloud worker, Germany"},
}

#: How many replicas of this app each worker admits, measured rather than
#: assumed. On Europa the binding resource is host RAM, not the GPU: the worker
#: sees one 24576 MB RTX 3090 and advertises it as VRAM_MB, which would allow
#: floor(24576 / 6144) = 4, but its Ray cluster has 30 GiB of memory shared with
#: other apps, so floor(30 / 8) = 3 replicas is the real ceiling. de.NBI is a
#: Kubernetes cluster with no VRAM_MB, so the worker falls back to a GPU fraction
#: of 6144 / 15360 = 0.40 per replica, and only one of its three T4 nodes has
#: free fraction: floor(1 / 0.40) = 2. Both ceilings are set by the declarations,
#: not by the app — a replica measures 0.65 GiB of RAM and ~2.5 GB of VRAM.
PLACEMENT = {EUROPA: 3, DENBI: 2}

LAYOUTS = {
    # Two modalities out of BBBC038, with the fluorescence pool cut into three
    # disjoint clients so the federation has four participants at 3:1 modality
    # representation. Shard 0 of 3 reproduces the two-site run's split exactly,
    # so the test fingerprints — and the earlier numbers — still line up.
    "acquisition-4site": {
        "caveat": (
            "This is the mechanics rig, not the consortium: one fluorescence domain "
            "cut into three disjoint shards plus one histology domain, so three of "
            "the four clients are near-IID with each other and none of them is a "
            "separate lab. It exercises N-client FedAvg and unequal client sizes; it "
            "does not demonstrate cross-site generalisation."
        ),
        "clients": {
            "fluo-0": (EUROPA, ["bbbc038-fluo@0/3"]),
            "fluo-1": (EUROPA, ["bbbc038-fluo@1/3"]),
            "fluo-2": (DENBI, ["bbbc038-fluo@2/3"]),
            "histo": (DENBI, ["bbbc038-histo"]),
        },
        "pooled": (EUROPA, ["bbbc038-fluo@*/3", "bbbc038-histo"]),
    },
    # The original two-site pairings, kept so either earlier run can be rebuilt.
    "caricature": {
        "caveat": (
            "Deliberately a caricature of non-IID: nuclei against C. elegans are far "
            "enough apart that a single-site model collapses out of domain, which "
            "makes federation look very good. An existence proof, not a utility bound."
        ),
        "clients": {
            "site-a": (EUROPA, ["dsb2018-fluo"]),
            "site-b": (DENBI, ["bbbc010-worms"]),
        },
        "pooled": (EUROPA, ["dsb2018-fluo", "bbbc010-worms"]),
    },
    "acquisition": {
        "caveat": (
            "Two clients cut out of one collection by imaging modality, not two labs. "
            "The shift is real but it is the only shift present."
        ),
        "clients": {
            "site-a": (EUROPA, ["bbbc038-fluo"]),
            "site-b": (DENBI, ["bbbc038-histo"]),
        },
        "pooled": (EUROPA, ["bbbc038-fluo", "bbbc038-histo"]),
    },
}


def instances(layout_name: str):
    """Expand a layout into ``{name: (worker, datasets, role)}``, placement checked."""
    layout = LAYOUTS[layout_name]
    expanded = {
        name: (worker, datasets, "participant")
        for name, (worker, datasets) in layout["clients"].items()
    }
    pooled_worker, pooled_datasets = layout["pooled"]
    expanded["pooled"] = (pooled_worker, pooled_datasets, "pooled-oracle")

    for worker, capacity in PLACEMENT.items():
        placed = sum(1 for w, _, _ in expanded.values() if w == worker)
        if placed > capacity:
            raise SystemExit(
                f"layout {layout_name!r} puts {placed} instances on {worker} which "
                f"admits {capacity} at the app's declared resources"
            )
    return expanded


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.3.1")
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="acquisition-4site")
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args()

    layout = instances(args.layout)
    selected = args.only or list(layout)
    env = dotenv_values(REPO_ROOT / ".env")

    for name in selected:
        worker_prefix, datasets, role = layout[name]
        server = await connect_to_server(
            {"server_url": SERVER_URL, "token": env[WORKERS[worker_prefix]["token_key"]]}
        )
        worker = await resolve_worker(server, worker_prefix)
        app_id = await worker.deploy_app(
            artifact_id=ARTIFACT,
            version=args.version,
            application_id=f"fedunet-{name}",
            application_kwargs={
                "FederatedUNetSite": {
                    "site_name": f"{name}-{'europa' if worker_prefix == EUROPA else 'denbi'}",
                    "role": role,
                    "datasets": datasets,
                }
            },
            # The run artifact lives in bioimage-io, so every instance needs a
            # credential that can write there regardless of which worker it is on.
            hypha_token=env["BIOIMAGE_IO_TOKEN"],
        )
        print(f"{name}: deploying {app_id} on {worker_prefix.split('/')[-1]}", flush=True)
        await server.disconnect()

    print("\nwaiting for RUNNING (env builds take a while on first deploy)", flush=True)
    for _ in range(180):
        await asyncio.sleep(20)
        states = {}
        for name in selected:
            worker_prefix = layout[name][0]
            server = await connect_to_server(
                {"server_url": SERVER_URL, "token": env[WORKERS[worker_prefix]["token_key"]]}
            )
            worker = await resolve_worker(server, worker_prefix)
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
