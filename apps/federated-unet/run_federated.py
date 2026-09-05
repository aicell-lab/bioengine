"""Drive a four-arm federated segmentation experiment across two BioEngine sites.

The four arms all consume the same number of optimiser steps, so the comparison
measures federation rather than compute:

  site-a-only   trained only on site A's images
  site-b-only   trained only on site B's images
  fedavg        R rounds of local training, sample-count-weighted state_dict average
  pooled        a third instance holding both datasets — the deliberate premise
                violation, used as the upper bound

Every arm is scored the same way: the checkpoint is pushed to each site and
evaluated there against that site's held-out test split. Test images never move.
This process never receives an image; it reads checkpoints, averages them, and
writes the average back.

Usage:
    python run_federated.py --seeds 0 --rounds 10 --steps 50 --out <dir>
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
from typing import Any, Dict, List

import numpy as np
import torch
from dotenv import dotenv_values
from hypha_rpc import connect_to_server

sys.path.insert(0, str(Path(__file__).parent))
from checkpoints import CheckpointStore, TransportLog, ensure_run_artifact  # noqa: E402
from unet import fedavg, signature  # noqa: E402

SERVER_URL = "https://hypha.aicell.io"
REPO_ROOT = Path(__file__).resolve().parents[2]

SITES = {
    "site-a": {
        "worker": "ws-user-github|49943582/bioengine-worker-europa:bioengine-worker",
        "application_id": "fedunet-site-a",
        "token_key": "HYPHA_TOKEN",
        "description": "Europa single-machine worker, Stockholm",
    },
    "site-b": {
        "worker": "bioimage-io/bioengine-worker-denbi-6d8d6dd6d5-zvwxn:bioengine-worker",
        "application_id": "fedunet-site-b",
        "token_key": "BIOIMAGE_IO_TOKEN",
        "description": "de.NBI cloud worker, last free GPU node",
    },
    "pooled": {
        "worker": "ws-user-github|49943582/bioengine-worker-europa:bioengine-worker",
        "application_id": "fedunet-pooled",
        "token_key": "HYPHA_TOKEN",
        "description": "Pooled-oracle control on Europa; holds both domains",
    },
}


async def resolve(server, worker_service_id: str, application_id: str):
    """Turn an application_id into a callable app service handle."""
    worker = await server.get_service(worker_service_id)
    status = await worker.get_app_status([application_id])
    record = status.get(application_id)
    if not record:
        raise RuntimeError(f"{application_id} is not deployed on {worker_service_id}")
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


async def val_dice(apps, names) -> Dict[str, float]:
    """Mean validation Dice per dataset, across one or more instances."""
    scores: Dict[str, float] = {}
    for name in names:
        for dataset, result in (await apps[name].evaluate(split="val")).items():
            scores[dataset] = float(result["dice_mean"])
    return scores


async def train_arm(
    app, apps, instance: str, rounds: int, steps: int, lr: float, seed: int, tag: str
) -> List[Dict]:
    """Run rounds x steps of purely local training, scoring validation each round."""
    history = []
    for r in range(rounds):
        record = await app.train(steps=steps, lr=lr, seed=seed * 1000 + r, tag=f"{tag}/r{r:02d}")
        record["val_dice"] = await val_dice(apps, [instance])
        history.append(record)
    return history


def resolve_pre_registration(path: str) -> dict:
    """Pin the commit that holds this run's predictions, refusing an unstaged one.

    A prediction only counts as pre-registered if it was in git before the first
    optimiser step, so an uncommitted edit is a hard failure rather than a
    warning: the recorded hash would otherwise describe a different document
    than the one the run was designed against.
    """
    if path is None:
        return {}
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
        "file": design.name,
        "commit": commit,
        "committed_at": git("log", "-1", "--format=%cI", "--", str(design)),
        "sha256": hashlib.sha256(design.read_bytes()).hexdigest(),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50, help="Optimiser steps per round per site")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--previews", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--pre-registration",
        default=None,
        help="Path to a design file whose predictions must already be committed",
    )
    args = parser.parse_args()

    pre_registration = resolve_pre_registration(args.pre_registration)

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
    for name, spec in SITES.items():
        servers[name] = await connect_to_server({"server_url": SERVER_URL, "token": env[spec["token_key"]]})
        apps[name], app_records[name] = await resolve(servers[name], spec["worker"], spec["application_id"])
        print(f"{name}: resolved {spec['application_id']}", flush=True)

    site_status = {name: await app.get_status() for name, app in apps.items()}
    if not args.skip_prepare:
        for name, app in apps.items():
            loaded = await app.prepare_data()
            print(f"{name}: {[(k, v['n_train'], v['n_val'], v['n_test']) for k, v in loaded.items()]}", flush=True)
            site_status[name] = await app.get_status()

    fingerprints = {
        name: {k: v["split_fingerprint"] for k, v in status["datasets_loaded"].items()}
        for name, status in site_status.items()
    }
    for dataset in set(fingerprints["pooled"]):
        site = "site-a" if dataset in fingerprints["site-a"] else "site-b"
        if fingerprints["pooled"][dataset] != fingerprints[site][dataset]:
            raise RuntimeError(
                f"pooled control and {site} disagree on the {dataset} split — "
                "the arms would not share a test set"
            )

    results: Dict[str, Any] = {}
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        prefix = f"seed_{seed}"

        # One common starting point, distributed rather than re-derived, so the
        # transport log shows all three instances loading the same digest.
        init = await apps["site-a"].init_model(seed=seed)
        init_entry = await apps["site-a"].push_weights(
            run_artifact_id=run_artifact_id, path=f"{prefix}/init.pt", note="common initialisation"
        )
        for name in ("site-b", "pooled"):
            await apps[name].pull_weights(run_artifact_id=run_artifact_id, path=f"{prefix}/init.pt")
        print(f"init: {init['n_parameters']} params, sha256 {init_entry['sha256'][:12]}", flush=True)

        arms: Dict[str, Dict[str, Any]] = {}

        # --- Arm: federated ------------------------------------------------
        started = time.time()
        round_records = []
        for r in range(args.rounds):
            local = await asyncio.gather(
                apps["site-a"].train(steps=args.steps, lr=args.lr, seed=seed * 1000 + r, tag=f"fedavg/r{r:02d}"),
                apps["site-b"].train(steps=args.steps, lr=args.lr, seed=seed * 1000 + r, tag=f"fedavg/r{r:02d}"),
            )
            state_dicts, counts = [], []
            for name, site_key in (("site-a", "a"), ("site-b", "b")):
                path = f"{prefix}/round_{r:02d}/site-{site_key}.pt"
                push = await apps[name].push_weights(
                    run_artifact_id=run_artifact_id, path=path, note=f"fedavg round {r}"
                )
                state_dicts.append(load_state_dict(await store.get(path)))
                counts.append(push["n_train_images"])
            merged = fedavg(state_dicts, counts)
            global_path = f"{prefix}/round_{r:02d}/global.pt"
            await store.put(global_path, dump_state_dict(merged), note=f"fedavg round {r} aggregate")
            for name in ("site-a", "site-b"):
                await apps[name].pull_weights(run_artifact_id=run_artifact_id, path=global_path)
            # Scored after the merge and pull, so this is the aggregate's curve.
            merged_val = await val_dice(apps, ("site-a", "site-b"))
            round_records.append(
                {
                    "round": r,
                    "merge_weights": counts,
                    "local": [{k: v for k, v in item.items() if k != "loss_curve"} for item in local],
                    "val_dice": merged_val,
                    "global_sha256": driver_log.dump()["entries"][-1]["sha256"],
                }
            )
            print(
                f"  fedavg r{r:02d} loss {[round(x['loss_last'], 4) for x in local]} "
                f"val {[round(v, 4) for v in merged_val.values()]}",
                flush=True,
            )
        await apps["site-a"].push_weights(
            run_artifact_id=run_artifact_id, path=f"{prefix}/arms/fedavg.pt", note="final fedavg aggregate"
        )
        fedavg_curve = [float(np.mean(list(rec["val_dice"].values()))) for rec in round_records]
        arms["fedavg"] = {
            "checkpoint": f"{prefix}/arms/fedavg.pt",
            "rounds": round_records,
            "val_curve": fedavg_curve,
            "plateau": {
                dataset: block_plateau(curve)
                for dataset, curve in per_dataset_curves(round_records).items()
            },
            "convergence_round": convergence_round(
                fedavg_curve, CONVERGENCE["secondary"]["window"], CONVERGENCE["secondary"]["tolerance"]
            ),
            "wall_time_s": time.time() - started,
            "total_steps_per_model": args.rounds * args.steps,
        }

        # --- Arms: single-site and pooled ----------------------------------
        for arm, instance in (("site-a-only", "site-a"), ("site-b-only", "site-b"), ("pooled", "pooled")):
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
            for name in ("site-a", "site-b"):
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
            for name in ("site-a", "site-b"):
                for arm in ("site-a-only", "site-b-only", "fedavg", "pooled"):
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
        "app_artifact": app_records["site-a"].get("artifact_id"),
        "app_version": app_records["site-a"].get("version"),
        "run_artifact_id": run_artifact_id,
        "aggregation_rule": "sample-count-weighted FedAvg over the full state_dict",
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
        "sites": {
            name: {
                "description": SITES[name]["description"],
                "worker_service_id": SITES[name]["worker"],
                "application_id": SITES[name]["application_id"],
                "status": site_status[name],
                "app_record": {k: v for k, v in app_records[name].items() if k != "deployments"},
            }
            for name in SITES
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
