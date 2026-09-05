"""Deploy or redeploy the instances of one federation layout.

    python deploy.py --version 0.4.1 --layout acquisition-4site [--only fluo-0 pooled]

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
#: assumed. Europa's worker sees one 24576 MB RTX 3090 and advertises it as
#: VRAM_MB, so at gpu_memory_mb 5120 the GPU allows floor(24576 / 5120) = 4; its
#: Ray cluster has 30 GiB of memory shared with other apps, so memory_mb 6144
#: allows floor(30 / 6) = 5 and the GPU is the binding resource again. (At the
#: earlier memory_mb of 8192 it was RAM that bound, at 3.) de.NBI is a Kubernetes
#: cluster with no VRAM_MB, so the worker falls back to a GPU fraction of
#: 5120 / 15360 = 0.33 per replica and one T4 takes 3 x 0.33 = 0.99. Both
#: ceilings are set by the declarations, not by the app — a replica measures
#: 0.65 GiB of RAM and ~2.5 GB of VRAM, so both still carry about 2x headroom.
PLACEMENT = {EUROPA: 4, DENBI: 3}

LAYOUTS = {
    # The consortium: one client per public dataset, so a client boundary is an
    # acquisition-setup boundary rather than a slice of one collection. Rostered
    # in bioengine-paper analysis/data/federated_consortium/README.md, frozen
    # against analysis/results/federated-consortium-design.md.
    "consortium": {
        "caveat": (
            "Six clients are six public datasets from six acquisition setups, not "
            "six labs that agreed to federate: the data was gathered by others and "
            "partitioned here. Client size is natural and therefore not randomised, "
            "so it is entangled with client domain and no causal claim about size "
            "follows. Seven instances run on two physical GPUs; see "
            "co_located_clients."
        ),
        # Train sizes are whatever each client has left after the fixed test and
        # val blocks, which is what puts a 16x span on the size axis.
        "n_train": None,
        "clients": {
            "bbbc038-fluo": (EUROPA, ["bbbc038-fluo"]),
            "bbbc039": (EUROPA, ["bbbc039"]),
            "nuinsseg": (EUROPA, ["nuinsseg"]),
            "bbbc038-histo": (DENBI, ["bbbc038-histo"]),
            "cellbindb": (DENBI, ["cellbindb"]),
            "kromp": (DENBI, ["kromp"]),
        },
        # The oracle holds every client's data, which includes the 1.6 GB
        # NuInsSeg archive, so it goes on the worker with the larger disk.
        "pooled": (
            EUROPA,
            ["bbbc038-fluo", "bbbc038-histo", "bbbc039", "cellbindb", "nuinsseg", "kromp"],
        ),
    },
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
        "n_train": 55,
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
        "n_train": 55,
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
        "n_train": 55,
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
    parser.add_argument("--version", default="0.4.1")
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
