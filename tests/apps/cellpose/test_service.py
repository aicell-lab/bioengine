"""Client script to use a deployed Cellpose Fine-Tuning service.

This is intended to be run *after* deployment (for example via
apps/cellpose-finetuning/scripts/deploy_local_cellpose.py).

Examples:
  export HYPHA_TOKEN=...
  python tests/cellpose_legacy_scripts/test_service.py \
    --dataset-artifact ri-scale/cellpose-test \
    --train-images 'train/*_image.ome.tif' \
    --train-annotations 'train/*_mask.ome.tif' \
    --test-images 'test/*_image.ome.tif' \
    --test-annotations 'test/*_mask.ome.tif'

Resume monitoring an existing session:
    python tests/cellpose_legacy_scripts/test_service.py \
        --dataset-artifact ri-scale/cellpose-test \
        --train-images 'train/*_image.ome.tif' \
        --train-annotations 'train/*_mask.ome.tif' \
        --session <session-id>

Export a completed session (and optionally download cover):
    python tests/cellpose_legacy_scripts/test_service.py \
        --dataset-artifact ri-scale/cellpose-test \
        --train-images 'train/*_image.ome.tif' \
        --train-annotations 'train/*_mask.ome.tif' \
        --session <session-id> \
        --export \
        --export-collection 'ri-scale/ai-model-hub' \
        --download-cover

The default service id is <your-workspace>/cellpose-finetuning.
You can override with --service-id.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

# from dotenv import load_dotenv
from hypha_rpc import connect_to_server, login

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService

# load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def get_cellpose_service(
    server: RemoteService,
    *,
    service_id: str | None,
    application_id: str,
) -> RemoteService:
    workspace = server.config.workspace
    logger.info("Using workspace: %s", workspace)

    resolved_service_id = service_id or f"{workspace}/{application_id}"
    logger.info("Using cellpose service id: %s", resolved_service_id)

    cellpose_service = await server.get_service(resolved_service_id)
    logger.info("Obtained Cellpose Fine-Tuning service")
    return cellpose_service


async def infer(
    cellpose_service: RemoteService,
    *,
    dataset_artifact: str,
    image_paths: list[str] | None,
    use_numpy: bool,
    diameter: float,
    model: str | None,
) -> None:
    kwargs = {
        "diameter": diameter,
    }
    if model:
        kwargs["model"] = model

    if image_paths and not use_numpy:
        kwargs["artifact"] = dataset_artifact
        kwargs["image_paths"] = image_paths
        inference_result = await cellpose_service.infer(**kwargs)
    else:
        import numpy as np

        test_image = np.random.randint(0, 255, (3, 512, 512), dtype=np.uint8)
        kwargs["input_arrays"] = [test_image]
        inference_result = await cellpose_service.infer(**kwargs)

    logger.info("Inference done (showing first item).")
    first = inference_result[0]
    out = first.get("output")
    if out is not None:
        logger.info("Output shape: %s", getattr(out, "shape", None))


async def export_model(
    *,
    cellpose_service: RemoteService,
    server: RemoteService,
    session_id: str,
    model_name: str,
    collection: str,
    download_cover: bool,
) -> dict:
    """Export a trained model and optionally download the cover image."""
    logger.info("Exporting model from session: %s", session_id)

    export_result = await cellpose_service.export_model(
        session_id=session_id,
        model_name=model_name,
        collection=collection,
    )

    artifact_id = export_result.get("artifact_id")
    artifact_url = export_result.get("artifact_url")
    download_url = export_result.get("download_url")

    logger.info("Model exported successfully")
    logger.info("  Artifact ID: %s", artifact_id)
    logger.info("  Artifact URL: %s", artifact_url)
    logger.info("  Download URL: %s", download_url)

    if download_cover and artifact_id:
        artifact_manager = await server.get_service("public/artifact-manager")
        cover_url = await artifact_manager.get_file(artifact_id, "cover.png")
        try:
            import httpx
        except Exception as e:
            raise RuntimeError(
                "httpx is required to download the cover image; install with 'pip install httpx'"
            ) from e

        async with httpx.AsyncClient() as client:
            resp = await client.get(cover_url)
            resp.raise_for_status()
            cover_path = f"exported_cover_{session_id[:8]}.png"
            with open(cover_path, "wb") as f:
                f.write(resp.content)
        logger.info("Downloaded cover image: %s", cover_path)

    return export_result


def _format_metrics(m: dict) -> str:
    keys = ["pixel_accuracy", "precision", "recall", "f1", "iou"]
    parts = []
    for k in keys:
        if k in m and m[k] is not None:
            parts.append(f"{k}={float(m[k]):.4f}")
    return ", ".join(parts)


def write_training_metrics_file(
    *,
    session_id: str,
    train_losses: list[float],
    test_losses: list[float | None] | None,
    test_metrics: list[dict] | None,
) -> str:
    """Write per-epoch losses/metrics to a TSV file.

    Mirrors the behavior in the original test_service.py, with optional
    additional columns for validation performance metrics if available.

    Returns:
        Path to the written file.
    """
    metrics_path = f"training_metrics_{session_id[:8]}.tsv"
    headers = [
        "Epoch",
        "Train Loss",
        "Val Loss",
        "Val PixelAcc",
        "Val Precision",
        "Val Recall",
        "Val F1",
        "Val IoU",
    ]

    def _metric_at(idx: int) -> dict | None:
        if not test_metrics or idx >= len(test_metrics):
            return None
        m = test_metrics[idx]
        return m if m else None

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for epoch_index, train_loss in enumerate(train_losses, start=1):
            # Validation loss is None for epochs without evaluation.
            val_loss_str = "N/A"
            if test_losses is not None and (epoch_index - 1) < len(test_losses):
                candidate = test_losses[epoch_index - 1]
                if candidate is not None:
                    val_loss_str = str(float(candidate))

            m = _metric_at(epoch_index - 1)
            row = [
                str(epoch_index),
                str(float(train_loss)),
                val_loss_str,
                str(m.get("pixel_accuracy", "N/A")) if m else "N/A",
                str(m.get("precision", "N/A")) if m else "N/A",
                str(m.get("recall", "N/A")) if m else "N/A",
                str(m.get("f1", "N/A")) if m else "N/A",
                str(m.get("iou", "N/A")) if m else "N/A",
            ]
            f.write("\t".join(row) + "\n")

    return metrics_path


async def monitor_training(
    cellpose_service: RemoteService,
    session_id: str,
    *,
    poll_s: float,
) -> dict:
    status: dict | None = None
    current_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{current_time}] Monitoring training...")  # noqa: T201

    while True:
        status = await cellpose_service.get_training_status(session_id)

        msg_parts = [f"[{status['status_type']}]", status.get("message", "")]  # type: ignore[index]

        if "current_epoch" in status and "total_epochs" in status:
            msg_parts.append(
                f"| Epoch: {status['current_epoch']}/{status['total_epochs']}"
            )

        if "elapsed_seconds" in status:
            msg_parts.append(f"| Time: {float(status['elapsed_seconds']):.1f}s")

        if status.get("train_losses"):
            losses = [float(loss) for loss in status["train_losses"] if float(loss) > 0]
            if losses:
                msg_parts.append(f"| Train Loss: {losses[-1]:.4f}")

        if status.get("test_losses"):
            losses = [float(loss) for loss in status["test_losses"] if loss is not None]
            if losses:
                msg_parts.append(f"| Val Loss: {losses[-1]:.4f}")

        if status.get("test_metrics"):
            metrics_list = status["test_metrics"]
            last_metrics = next((m for m in reversed(metrics_list) if m), None)
            if last_metrics:
                msg_parts.append(f"| Val: {_format_metrics(last_metrics)}")

        message = f"\r{' '.join(msg_parts)}"
        print(message, end="")  # noqa: T201

        if status.get("instance_metrics"):
            im = status["instance_metrics"]
            msg_parts.append(f"| Instance AP@0.5={im.get('ap_0_5', 0):.4f}")

        if status.get("status_type") in ("completed", "failed"):
            print()  # noqa: T201

            # Print instance metrics summary if available
            if status.get("instance_metrics"):
                im = status["instance_metrics"]
                print(  # noqa: T201
                    f"Instance segmentation metrics: "
                    f"AP@0.5={im.get('ap_0_5', 'N/A')}, "
                    f"AP@0.75={im.get('ap_0_75', 'N/A')}, "
                    f"AP@0.9={im.get('ap_0_9', 'N/A')}, "
                    f"n_true={im.get('n_true', 'N/A')}, "
                    f"n_pred={im.get('n_pred', 'N/A')}"
                )

            # Ported behavior: write per-epoch metrics to a file on completion.
            if status.get("status_type") == "completed" and status.get("train_losses"):
                train_losses = [float(loss) for loss in status["train_losses"]]
                test_losses_raw = status.get("test_losses")
                test_losses = (
                    [
                        float(loss) if loss is not None else None
                        for loss in test_losses_raw
                    ]
                    if isinstance(test_losses_raw, list)
                    else None
                )
                test_metrics_raw = status.get("test_metrics")
                test_metrics = (
                    test_metrics_raw if isinstance(test_metrics_raw, list) else None
                )

                out_path = write_training_metrics_file(
                    session_id=session_id,
                    train_losses=train_losses,
                    test_losses=test_losses,
                    test_metrics=test_metrics,
                )
                logger.info("Wrote training metrics: %s", out_path)

            return status

        await asyncio.sleep(poll_s)


async def start_training(
    cellpose_service: RemoteService,
    *,
    dataset_artifact: str,
    train_images: str,
    train_annotations: str,
    test_images: str | None,
    test_annotations: str | None,
    n_epochs: int,
    n_samples: int | None,
    learning_rate: float,
    min_train_masks: int,
    validation_interval: int | None,
) -> str:
    kwargs: dict = {
        "artifact": dataset_artifact,
        "train_images": train_images,
        "train_annotations": train_annotations,
        "test_images": test_images,
        "test_annotations": test_annotations,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "n_samples": n_samples,
        "min_train_masks": min_train_masks,
    }
    if validation_interval is not None:
        kwargs["validation_interval"] = validation_interval
    session_status = await cellpose_service.start_training(**kwargs)
    session_id = session_status["session_id"]
    logger.info("Started training session: %s", session_id)
    return session_id


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Use a deployed cellpose-finetuning service"
    )
    p.add_argument(
        "--server-url",
        default=os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io"),
    )
    p.add_argument("--token", default=os.environ.get("HYPHA_TOKEN"))
    p.add_argument(
        "--workspace",
        default=os.environ.get("HYPHA_WORKSPACE"),
        help="Optional workspace override",
    )
    p.add_argument(
        "--service-id",
        default=None,
        help="Override service id (default: <workspace>/cellpose-finetuning)",
    )
    p.add_argument("--application-id", default="cellpose-finetuning")

    p.add_argument(
        "--session",
        "-s",
        default=None,
        help=(
            "Existing training session ID to monitor. If provided, no new training is started."
        ),
    )

    p.add_argument("--dataset-artifact", required=True)
    p.add_argument("--train-images", required=True)
    p.add_argument("--train-annotations", required=True)
    p.add_argument("--test-images", default=None)
    p.add_argument("--test-annotations", default=None)

    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--min-train-masks", type=int, default=5)
    p.add_argument(
        "--validation-interval",
        type=int,
        default=None,
        help="Epochs between validation evaluations (default: every 10 epochs). Set to 1 for every epoch.",
    )
    p.add_argument("--poll-s", type=float, default=0.3)

    p.add_argument(
        "--infer-image",
        action="append",
        default=None,
        help="Artifact-relative image path(s) to run inference on",
    )
    p.add_argument(
        "--infer-numpy",
        action="store_true",
        help="Run inference using a random numpy array input.",
    )
    p.add_argument(
        "--skip-infer",
        action="store_true",
        help="Skip inference after training/monitoring completes.",
    )
    p.add_argument("--infer-diameter", type=float, default=40.0)
    p.add_argument("--infer-model", default=None)

    p.add_argument(
        "--export",
        action="store_true",
        help="Export the model after completion (requires completed session).",
    )
    p.add_argument(
        "--export-collection",
        default="ri-scale/ai-model-hub",
        help="Target collection for export_model().",
    )
    p.add_argument(
        "--export-model-name",
        default=None,
        help="Model name for export (default: test-model-<session8>).",
    )
    p.add_argument(
        "--download-cover",
        action="store_true",
        help="Download exported cover.png to the current directory.",
    )

    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    token = args.token or os.environ.get("HYPHA_TOKEN")
    print(
        f"DEBUG: Token starts with {token[:10] if token else 'None'}.. ends with {token[-10:] if token else 'None'}"
    )
    print(f"DEBUG: Workspace arg: {args.workspace}")

    if not token:
        token = await login({"server_url": args.server_url})

    async with connect_to_server(
        {
            "server_url": args.server_url,
            "token": token,
            "workspace": args.workspace,
        }
    ) as server:  # type: ignore[generalTypeIssues]
        cellpose_service = await get_cellpose_service(
            server,
            service_id=args.service_id,
            application_id=args.application_id,
        )

        if args.session:
            session_id = args.session
            logger.info("Using existing session: %s", session_id)
        else:
            session_id = await start_training(
                cellpose_service,
                dataset_artifact=args.dataset_artifact,
                train_images=args.train_images,
                train_annotations=args.train_annotations,
                test_images=args.test_images,
                test_annotations=args.test_annotations,
                n_epochs=args.n_epochs,
                n_samples=args.n_samples,
                learning_rate=args.learning_rate,
                min_train_masks=args.min_train_masks,
                validation_interval=args.validation_interval,
            )

        final_status = await monitor_training(
            cellpose_service,
            session_id,
            poll_s=args.poll_s,
        )

        if final_status.get("status_type") == "completed":
            if args.export:
                export_name = args.export_model_name or f"test-model-{session_id[:8]}"
                await export_model(
                    cellpose_service=cellpose_service,
                    server=server,
                    session_id=session_id,
                    model_name=export_name,
                    collection=args.export_collection,
                    download_cover=args.download_cover,
                )

            if not args.skip_infer:
                await infer(
                    cellpose_service,
                    dataset_artifact=args.dataset_artifact,
                    image_paths=args.infer_image,
                    use_numpy=bool(args.infer_numpy) or (args.infer_image is None),
                    diameter=args.infer_diameter,
                    model=session_id if args.infer_model is None else args.infer_model,
                )


if __name__ == "__main__":
    asyncio.run(main())
