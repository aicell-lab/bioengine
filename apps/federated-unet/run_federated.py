"""Drive a federated segmentation experiment across N BioEngine clients.

Every arm consumes the same number of optimiser steps, so the comparison
measures federation rather than compute:

  <client>-only    trained only on that client's images, one arm per client
  fedavg           R rounds of local training, sample-count-weighted state_dict average
  fedavg-uniform   the same merge weighting every client equally (--uniform-arm)
  loso-<client>    federate everyone except <client>, then score on <client> (--loso)
  pooled           an extra instance holding every client's data — the deliberate
                   premise violation, used as the upper bound

Every arm is scored the same way: the checkpoint is pushed to each client and
evaluated there against that client's held-out test split. Test images never
move. This process never receives an image; it reads checkpoints, averages them,
and writes the average back.

Usage:
    python run_federated.py --layout consortium --seeds 0 1 2 3 4 --rounds 60 \\
        --steps 50 --loso --uniform-arm \\
        --pre-registration ../../../bioengine-paper/analysis/results/federated-consortium-design.md
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from dotenv import dotenv_values
from hypha_rpc import connect_to_server

sys.path.insert(0, str(Path(__file__).parent))
from checkpoints import CheckpointStore, TransportLog, ensure_run_artifact  # noqa: E402
# The layout is the single definition of who holds what and where; importing it
# rather than restating it is what stops the driver and the deployer drifting.
from deploy import LAYOUTS, WORKERS, instances  # noqa: E402
from unet import fedavg, signature  # noqa: E402
from workers import resolve_worker  # noqa: E402

SERVER_URL = "https://hypha.aicell.io"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_sites(layout_name: str) -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "worker": worker,
            "application_id": f"fedunet-{name}",
            "token_key": WORKERS[worker]["token_key"],
            "description": WORKERS[worker]["description"],
            "role": role,
            "datasets": datasets,
        }
        for name, (worker, datasets, role) in instances(layout_name).items()
    }


async def resolve(server, worker_prefix: str, application_id: str):
    """Turn an application_id into a callable app service handle."""
    worker = await resolve_worker(server, worker_prefix)
    status = await worker.get_app_status([application_id])
    record = status.get(application_id)
    if not record:
        raise RuntimeError(f"{application_id} is not deployed on {worker_prefix}")
    if record["status"] != "RUNNING":
        raise RuntimeError(f"{application_id} is {record['status']}: {record.get('message')}")
    return await server.get_service(record["service_ids"]["websocket_service_id"]), record


def load_state_dict(payload: bytes) -> Dict[str, torch.Tensor]:
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def dump_state_dict(state_dict: Dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


CONVERGENCE = {
    "primary": {
        "name": "block plateau, per dataset",
        "metric": "validation Dice on one dataset the arm trains on",
        "tolerance": 0.005,
        "rule": (
            "Split the rounds into equal thirds. The arm has converged on that "
            "dataset if the mean validation Dice over the final third exceeds "
            "the mean over the middle third by less than tolerance. Reported "
            "per dataset and never averaged across datasets."
        ),
        "why": (
            "Replaces the window rule below, which was pre-registered for the "
            "20260905-115455 run and failed there: it fires on the first "
            "transient stall in a noisy curve, reporting convergence at round 8 "
            "of 60 for arms whose best value came at round 50+. Averaging the "
            "two datasets also diluted the harder domain by half."
        ),
    },
    "secondary": {
        "name": "first-stall window (superseded, still reported for continuity)",
        "metric": "mean validation Dice over the datasets the arm trains on",
        "window": 5,
        "tolerance": 0.005,
        "rule": (
            "An arm is converged at round r if no round in (r, r+window] exceeds "
            "the best validation Dice seen up to and including r by more than "
            "tolerance. The converged round reported is the smallest such r; an "
            "arm counts as having converged within the run only if "
            "r + window <= last round."
        ),
    },
}


def block_plateau(curve: List[float], tolerance: float = 0.005) -> Dict[str, Any]:
    """Does the last third of training still buy anything over the middle third?

    Noise-robust where `convergence_round` is not: it asks whether more rounds
    would move the number, rather than whether the curve happened to stall once.
    """
    third = len(curve) // 3
    middle = float(np.mean(curve[third : 2 * third]))
    final = float(np.mean(curve[2 * third :]))
    return {
        "middle_third_mean": middle,
        "final_third_mean": final,
        "improvement": final - middle,
        "converged": (final - middle) < tolerance,
    }


def per_dataset_curves(history: List[Dict]) -> Dict[str, List[float]]:
    return {
        dataset: [record["val_dice"][dataset] for record in history]
        for dataset in history[0]["val_dice"]
    }


def convergence_round(curve: List[float], window: int = 5, tolerance: float = 0.005):
    """First round after which `window` further rounds buy less than `tolerance`.

    Pre-registered before the run; see CONVERGENCE. Returns None if the arm was
    still improving at the end, which is the outcome that would say the step
    budget was too short.
    """
    for r in range(window - 1, len(curve) - window):
        best = max(curve[: r + 1])
        if all(value <= best + tolerance for value in curve[r + 1 : r + 1 + window]):
            return r
    return None


async def val_dice(apps, names) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Mean validation Dice per dataset, plus the weights each site scored with."""
    scores: Dict[str, float] = {}
    scored_with: Dict[str, str] = {}
    for name in names:
        for dataset, result in (await apps[name].evaluate(split="val")).items():
            scores[dataset] = float(result["dice_mean"])
            scored_with[name] = result["weights_sha256"]
    return scores, scored_with


async def train_arm(
    app, apps, instance: str, rounds: int, steps: int, lr: float, seed: int, tag: str
) -> List[Dict]:
    """Run rounds x steps of purely local training, scoring validation each round."""
    history = []
    for r in range(rounds):
        record = await app.train(steps=steps, lr=lr, seed=seed * 1000 + r, tag=f"{tag}/r{r:02d}")
        record["val_dice"], _ = await val_dice(apps, [instance])
        history.append(record)
    return history


async def federated_arm(
    apps,
    store,
    run_artifact_id: str,
    participants: List[str],
    eval_on: List[str],
    prefix: str,
    arm: str,
    args,
    seed: int,
    weighting: str,
) -> Dict[str, Any]:
    """One FedAvg arm: R rounds of local training over `participants`, merged each round.

    The same coroutine runs the full-consortium arm, the uniform-weighting arm
    and each leave-one-site-out fold, so a fold cannot silently differ from the
    headline arm in anything but who is in `participants`.
    """
    started = time.time()
    for name in participants:
        await apps[name].pull_weights(run_artifact_id=run_artifact_id, path=f"{prefix}/init.pt")

    round_records = []
    previous_aggregate: str = ""
    previous_scored_with: Dict[str, str] = {}
    for r in range(args.rounds):
        local = await asyncio.gather(
            *(
                apps[name].train(
                    steps=args.steps, lr=args.lr, seed=seed * 1000 + r, tag=f"{arm}/r{r:02d}"
                )
                for name in participants
            )
        )
        state_dicts, counts = [], []
        for name in participants:
            path = f"{prefix}/{arm}/round_{r:02d}/{name}.pt"
            push = await apps[name].push_weights(
                run_artifact_id=run_artifact_id, path=path, note=f"{arm} round {r}"
            )
            state_dicts.append(load_state_dict(await store.get(path)))
            # Uniform weighting is the same merge with every client counted once:
            # the design choice is which of the two is primary, not two merges.
            counts.append(push["n_train_images"] if weighting == "sample-count" else 1)
        merged = fedavg(state_dicts, counts)
        global_path = f"{prefix}/{arm}/round_{r:02d}/global.pt"
        aggregate = await store.put(
            global_path, dump_state_dict(merged), note=f"{arm} round {r} aggregate"
        )
        # Every site that is scored must hold the merged weights, not just the
        # ones that trained. A LOSO fold's held-out client trains on nothing and
        # would otherwise be scored on whatever weights it last happened to
        # hold — which is the one number the fold exists to produce.
        for name in dict.fromkeys([*participants, *eval_on]):
            await apps[name].pull_weights(run_artifact_id=run_artifact_id, path=global_path)
        # Scored after the merge and pull, so this is the aggregate's curve.
        merged_val, scored_with = await val_dice(apps, eval_on)
        # A site whose weights did not move while the aggregate did was scored on
        # stale weights. That failure produces plausible numbers instead of an
        # error — a LOSO fold's held-out client trains on nothing, so a missed
        # pull leaves it reporting a frozen curve that reads as a real result.
        if previous_aggregate and aggregate["sha256"] != previous_aggregate:
            stale = sorted(n for n, h in scored_with.items() if previous_scored_with.get(n) == h)
            if stale:
                raise RuntimeError(
                    f"{arm} round {r}: aggregate changed but {stale} scored on the same "
                    f"weights as round {r - 1} — these sites did not receive the merge"
                )
        previous_aggregate, previous_scored_with = aggregate["sha256"], scored_with
        round_records.append(
            {
                "round": r,
                "merge_weights": dict(zip(participants, counts)),
                "local": [{k: v for k, v in item.items() if k != "loss_curve"} for item in local],
                "val_dice": merged_val,
                "global_sha256": aggregate["sha256"],
                "scored_with": scored_with,
            }
        )
        print(
            f"  {arm} r{r:02d} loss {[round(x['loss_last'], 4) for x in local]} "
            f"val {[round(v, 4) for v in merged_val.values()]}",
            flush=True,
        )

    checkpoint = f"{prefix}/arms/{arm}.pt"
    await apps[participants[0]].push_weights(
        run_artifact_id=run_artifact_id, path=checkpoint, note=f"final {arm} aggregate"
    )
    curve = [float(np.mean(list(rec["val_dice"].values()))) for rec in round_records]
    return {
        "checkpoint": checkpoint,
        "participants": participants,
        "weighting": weighting,
        "rounds": round_records,
        "val_curve": curve,
        "plateau": {
            dataset: block_plateau(values)
            for dataset, values in per_dataset_curves(round_records).items()
        },
        "convergence_round": convergence_round(
            curve, CONVERGENCE["secondary"]["window"], CONVERGENCE["secondary"]["tolerance"]
        ),
        "wall_time_s": time.time() - started,
        "total_steps_per_model": args.rounds * args.steps,
    }


def resolve_pre_registration(path: str) -> dict:
    """Pin the commit that holds this run's predictions, refusing an unstaged one.

    A prediction only counts as pre-registered if it was in git before the first
    optimiser step, so an uncommitted edit is a hard failure rather than a
    warning: the recorded hash would otherwise describe a different document
    than the one the run was designed against.
    """
    if path is None:
        print("WARNING: no --pre-registration; this run's results are exploratory", flush=True)
        return {"declared": False}
    design = Path(path).resolve()
    repo = design.parent
    def git(*cmd):
        return subprocess.run(["git", *cmd], cwd=repo, capture_output=True, text=True).stdout.strip()
    if git("status", "--porcelain", "--", str(design)):
        raise SystemExit(f"{design} has uncommitted changes; commit the predictions before training")
    commit = git("log", "-1", "--format=%H", "--", str(design))
    if not commit:
        raise SystemExit(f"{design} is not committed; commit the predictions before training")
    return {
        "declared": True,
        "file": design.name,
        "commit": commit,
        "committed_at": git("log", "-1", "--format=%cI", "--", str(design)),
        "sha256": hashlib.sha256(design.read_bytes()).hexdigest(),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="acquisition-4site")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50, help="Optimiser steps per round per site")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--previews", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Add one leave-one-site-out fold per client: federate the others, score on the held-out client",
    )
    parser.add_argument(
        "--uniform-arm",
        action="store_true",
        help="Add a second FedAvg arm weighting every client equally instead of by sample count",
    )
    parser.add_argument(
        "--pre-registration",
        default=None,
        help="Path to a design file whose predictions must already be committed",
    )
    args = parser.parse_args()

    pre_registration = resolve_pre_registration(args.pre_registration)

    sites = build_sites(args.layout)
    clients = [name for name, spec in sites.items() if spec["role"] == "participant"]
    print(f"layout {args.layout}: {len(clients)} clients {clients} + pooled oracle", flush=True)

    env = dotenv_values(REPO_ROOT / ".env")
    out_dir = Path(args.out or (REPO_ROOT.parent / "bioengine-paper" / "analysis" / "results" / f"federated-unet-{args.run_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # The run artifact lives in bioimage-io so both workspaces can reach it.
    control = await connect_to_server({"server_url": SERVER_URL, "token": env["BIOIMAGE_IO_TOKEN"]})
    artifact_manager = await control.get_service("public/artifact-manager")
    run_artifact_id = await ensure_run_artifact(
        artifact_manager,
        alias=f"fedunet-run-{args.run_id}",
        manifest={
            "name": f"federated-unet run {args.run_id}",
            "description": "Round checkpoints for a federated U-Net experiment. Weights only.",
        },
    )
    driver_log = TransportLog()
    store = CheckpointStore(artifact_manager, run_artifact_id, driver_log)
    print(f"run artifact: {run_artifact_id}", flush=True)

    servers, apps, app_records = {}, {}, {}
    for name, spec in sites.items():
        servers[name] = await connect_to_server({"server_url": SERVER_URL, "token": env[spec["token_key"]]})
        apps[name], app_records[name] = await resolve(servers[name], spec["worker"], spec["application_id"])
        print(f"{name}: resolved {spec['application_id']}", flush=True)

    site_status = {name: await app.get_status() for name, app in apps.items()}
    if not args.skip_prepare:
        # The layout decides whether train sizes are fixed or natural; the driver
        # does not carry a per-client size table that could drift from it.
        n_train = LAYOUTS[args.layout]["n_train"]
        for name, app in apps.items():
            loaded = await app.prepare_data(n_train=n_train)
            print(f"{name}: {[(k, v['n_train'], v['n_val'], v['n_test']) for k, v in loaded.items()]}", flush=True)
            site_status[name] = await app.get_status()

    loaded = {name: status["datasets_loaded"] for name, status in site_status.items()}
    holders = {
        dataset: [name for name in clients if dataset in loaded[name]]
        for dataset in loaded["pooled"]
    }
    for dataset, names in holders.items():
        if not names:
            raise RuntimeError(f"pooled holds {dataset} but no client does")
        for name in names:
            if loaded[name][dataset]["split_fingerprint"] != loaded["pooled"][dataset]["split_fingerprint"]:
                raise RuntimeError(
                    f"pooled control and {name} disagree on the {dataset} split — "
                    "the arms would not share a test set"
                )
        # Clients on the same domain must hold disjoint images, or the
        # federation is duplicating one site rather than joining several.
        shards = [loaded[name][dataset]["shard_fingerprint"] for name in names]
        if len(set(shards)) != len(shards):
            raise RuntimeError(
                f"clients {names} report the same {dataset} training shard — "
                "they hold the same images"
            )

    # One client per domain does the scoring: with a domain spread over several
    # clients they all hold the identical test split, so scoring on each would
    # only repeat the same number under the same key.
    eval_sites = {dataset: names[0] for dataset, names in holders.items()}
    print(f"scoring each domain on: {eval_sites}", flush=True)

    results: Dict[str, Any] = {}
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        prefix = f"seed_{seed}"

        # One common starting point, distributed rather than re-derived, so the
        # transport log shows every instance loading the same digest.
        origin = clients[0]
        init = await apps[origin].init_model(seed=seed)
        init_entry = await apps[origin].push_weights(
            run_artifact_id=run_artifact_id, path=f"{prefix}/init.pt", note="common initialisation"
        )
        for name in (n for n in sites if n != origin):
            await apps[name].pull_weights(run_artifact_id=run_artifact_id, path=f"{prefix}/init.pt")
        print(f"init: {init['n_parameters']} params, sha256 {init_entry['sha256'][:12]}", flush=True)

        arms: Dict[str, Dict[str, Any]] = {}
        scored_on = sorted(set(eval_sites.values()))

        async def fed(arm: str, participants: List[str], weighting: str) -> None:
            arms[arm] = await federated_arm(
                apps, store, run_artifact_id, participants,
                scored_on, prefix, arm, args, seed, weighting,
            )

        # --- Arm: federated, every client -----------------------------------
        await fed("fedavg", clients, "sample-count")
        if args.uniform_arm:
            await fed("fedavg-uniform", clients, "uniform")

        # --- Arms: leave one site out ----------------------------------------
        # The held-out unit is a whole client, i.e. a whole domain — never a
        # shard of one — so the fold asks what a site that never joined would
        # get from the federation, not what a site with half its own data would.
        if args.loso:
            for held_out in clients:
                if len(sites[held_out]["datasets"]) != 1 or "@" in sites[held_out]["datasets"][0]:
                    raise SystemExit(
                        f"--loso needs one whole domain per client; {held_out} holds "
                        f"{sites[held_out]['datasets']}"
                    )
                await fed(
                    f"loso-{held_out}", [n for n in clients if n != held_out], "sample-count"
                )

        # --- Arms: single-site and pooled ----------------------------------
        single_site_arms = [(f"{name}-only", name) for name in clients]
        for arm, instance in single_site_arms + [("pooled", "pooled")]:
            started = time.time()
            await apps[instance].pull_weights(run_artifact_id=run_artifact_id, path=f"{prefix}/init.pt")
            history = await train_arm(
                apps[instance], apps, instance, args.rounds, args.steps, args.lr, seed, arm
            )
            await apps[instance].push_weights(
                run_artifact_id=run_artifact_id, path=f"{prefix}/arms/{arm}.pt", note=f"{arm} final"
            )
            curve = [float(np.mean(list(h["val_dice"].values()))) for h in history]
            arms[arm] = {
                "checkpoint": f"{prefix}/arms/{arm}.pt",
                "history": [{k: v for k, v in h.items() if k != "loss_curve"} for h in history],
                "val_curve": curve,
                "plateau": {
                    dataset: block_plateau(values)
                    for dataset, values in per_dataset_curves(history).items()
                },
                "convergence_round": convergence_round(
                    curve, CONVERGENCE["secondary"]["window"], CONVERGENCE["secondary"]["tolerance"]
                ),
                "wall_time_s": time.time() - started,
                "total_steps_per_model": args.rounds * args.steps,
            }
            print(
                f"  {arm}: loss {history[0]['loss_first']:.4f} -> {history[-1]['loss_last']:.4f}, "
                f"val {curve[0]:.4f} -> {curve[-1]:.4f}, converged at round "
                f"{arms[arm]['convergence_round']}",
                flush=True,
            )

        # --- Evaluation: every arm, on both sites' held-out test splits -----
        for arm, record in arms.items():
            record["evaluation"] = {}
            for name in sorted(set(eval_sites.values())):
                await apps[name].pull_weights(run_artifact_id=run_artifact_id, path=record["checkpoint"])
                record["evaluation"][name] = await apps[name].evaluate(split="test")
            summary = {
                dataset: round(scores["dice_mean"], 4)
                for site in record["evaluation"].values()
                for dataset, scores in site.items()
            }
            print(f"  {arm} test dice: {summary}", flush=True)

        results[f"seed_{seed}"] = arms
        # Flushed per seed: a transient transport failure on seed 4 should cost
        # one seed, not the four that already finished.
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

        if args.previews:
            for name in sorted(set(eval_sites.values())):
                for arm in arms:
                    await apps[name].pull_weights(run_artifact_id=run_artifact_id, path=arms[arm]["checkpoint"])
                    status = await apps[name].get_status()
                    for dataset in status["datasets_loaded"]:
                        preview = await apps[name].preview(dataset=dataset, split="test", n=3)
                        (out_dir / f"preview_{prefix}_{arm}_{name}_{dataset}.png").write_bytes(
                            base64.b64decode(preview["png_base64"])
                        )

    transport = {name: await app.get_transport_log() for name, app in apps.items()}
    transport["driver"] = driver_log.dump()

    provenance = {
        "run_id": args.run_id,
        "generated_at": time.time(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "app_artifact": app_records[clients[0]].get("artifact_id"),
        "app_version": app_records[clients[0]].get("version"),
        "run_artifact_id": run_artifact_id,
        "aggregation_rule": "sample-count-weighted FedAvg over the full state_dict",
        "arm_structure": {
            "fedavg": "all clients, sample-count weighted — the primary aggregate",
            "fedavg-uniform": "all clients, every client weighted equally" if args.uniform_arm else None,
            "loso-<client>": (
                "federate every client except <client>, then score on <client>'s held-out "
                "test split; the held-out unit is a whole domain, never a shard"
            ) if args.loso else None,
            "<client>-only": "that client's own data alone, the baseline LOSO is compared against",
            "pooled": "one instance holding every client's data — the premise violation, an upper bound",
        },
        "train_sizes": "natural" if LAYOUTS[args.layout]["n_train"] is None else LAYOUTS[args.layout]["n_train"],
        "normalisation": "GroupNorm (no running statistics, so the merge averages weights only)",
        "convergence_criterion": CONVERGENCE,
        "pre_registration": pre_registration,
        "config": {
            "seeds": args.seeds,
            "rounds": args.rounds,
            "steps_per_round": args.steps,
            "total_steps_per_model_per_arm": args.rounds * args.steps,
            "lr": args.lr,
            "batch_size": 8,
            "crop": 256,
        },
        "layout": args.layout,
        # What the layout is not. A table of four clients reads as four labs
        # unless the record says otherwise, so it travels with every run rather
        # than living only in the message that announced the layout.
        "layout_caveat": LAYOUTS[args.layout]["caveat"],
        "clients": clients,
        "scored_on": eval_sites,
        # "one client per site" is the framing, so clients that share a worker
        # share a physical GPU and are not independent sites in any hardware
        # sense. Disclosed here rather than left to be inferred from the table.
        "co_located_clients": {
            worker: [n for n in clients if sites[n]["worker"] == worker]
            for worker in {sites[n]["worker"] for n in clients}
        },
        "sites": {
            name: {
                "description": sites[name]["description"],
                "worker_service_id": sites[name]["worker"],
                "application_id": sites[name]["application_id"],
                "role": sites[name]["role"],
                "dataset_specs": sites[name]["datasets"],
                "status": site_status[name],
                "app_record": {k: v for k, v in app_records[name].items() if k != "deployments"},
            }
            for name in sites
        },
        "architecture_signature": signature(
            load_state_dict(await store.get(f"seed_{args.seeds[0]}/init.pt"))
        ),
    }

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    (out_dir / "transport_audit.json").write_text(json.dumps(transport, indent=2, default=str))
    print(f"\nwrote {out_dir}", flush=True)

    for server in list(servers.values()) + [control]:
        await server.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
