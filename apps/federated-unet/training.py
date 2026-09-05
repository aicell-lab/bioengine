"""Local training and evaluation for one federation site.

Deliberately step-based rather than epoch-based: the pooled-oracle arm sees
twice the data of a single-site arm, so matching epochs would hand it twice the
gradient steps and the comparison would measure compute, not federation.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

Pair = Tuple[np.ndarray, np.ndarray]


def _random_crop_batch(
    pairs: Sequence[Pair], batch_size: int, crop: int, rng: np.random.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    images, masks = [], []
    for _ in range(batch_size):
        image, mask = pairs[rng.integers(len(pairs))]
        pad_y = max(0, crop - image.shape[0])
        pad_x = max(0, crop - image.shape[1])
        if pad_y or pad_x:
            image = np.pad(image, ((0, pad_y), (0, pad_x)), mode="reflect")
            mask = np.pad(mask, ((0, pad_y), (0, pad_x)), mode="reflect")
        y = rng.integers(image.shape[0] - crop + 1)
        x = rng.integers(image.shape[1] - crop + 1)
        image = image[y : y + crop, x : x + crop]
        mask = mask[y : y + crop, x : x + crop]
        if rng.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if rng.random() < 0.5:
            image, mask = image[::-1], mask[::-1]
        images.append(np.ascontiguousarray(image))
        masks.append(np.ascontiguousarray(mask))
    return (
        torch.from_numpy(np.stack(images)).unsqueeze(1),
        torch.from_numpy(np.stack(masks)).unsqueeze(1),
    )


def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (probs.sum() + target.sum() + 1.0)
    return bce + dice


def train_steps(
    model: nn.Module,
    pairs: Sequence[Pair],
    steps: int,
    lr: float,
    batch_size: int,
    crop: int,
    device: torch.device,
    seed: int,
) -> Dict[str, object]:
    """Run exactly ``steps`` optimiser steps and report the loss trajectory."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        images, masks = _random_crop_batch(pairs, batch_size, crop, rng)
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model(images), masks)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    window = max(1, steps // 10)
    return {
        "steps": steps,
        "loss_first": float(np.mean(losses[:window])) if losses else None,
        "loss_last": float(np.mean(losses[-window:])) if losses else None,
        "loss_curve": [float(x) for x in losses[:: max(1, steps // 50)]],
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    pairs: Sequence[Pair],
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Score whole images (not crops) and return per-image numbers.

    Per-image scores are returned, not just the mean, so the aggregate can be
    recomputed independently from the record.
    """
    model.to(device).eval()
    per_image: List[Dict[str, float]] = []
    for index, (image, mask) in enumerate(pairs):
        height, width = image.shape
        pad_y = (-height) % 16
        pad_x = (-width) % 16
        padded = np.pad(image, ((0, pad_y), (0, pad_x)), mode="reflect")
        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
        probs = torch.sigmoid(model(tensor))[0, 0, :height, :width].cpu().numpy()
        prediction = probs >= threshold
        truth = mask > 0.5
        intersection = float(np.logical_and(prediction, truth).sum())
        union = float(np.logical_or(prediction, truth).sum())
        per_image.append(
            {
                "index": index,
                "dice": (2.0 * intersection / (prediction.sum() + truth.sum()))
                if (prediction.sum() + truth.sum()) > 0
                else 1.0,
                "iou": (intersection / union) if union > 0 else 1.0,
                "pred_foreground_fraction": float(prediction.mean()),
                "true_foreground_fraction": float(truth.mean()),
            }
        )
    return {
        "n_images": len(per_image),
        "dice_mean": float(np.mean([r["dice"] for r in per_image])),
        "iou_mean": float(np.mean([r["iou"] for r in per_image])),
        # A model that has collapsed to all-background scores a deceptively
        # non-zero dice on sparse targets; this is how that shows up.
        "pred_foreground_fraction_mean": float(
            np.mean([r["pred_foreground_fraction"] for r in per_image])
        ),
        "true_foreground_fraction_mean": float(
            np.mean([r["true_foreground_fraction"] for r in per_image])
        ),
        "per_image": per_image,
    }
