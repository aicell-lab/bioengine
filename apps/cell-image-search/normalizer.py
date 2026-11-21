"""
Microscopy image normalization for DINOv2 embedding.

Handles:
- uint8 / uint16 / float32 inputs
- 1, 2, 3, 4, 5 channel fluorescence images
- Percentile stretch to [0, 255] uint8
- 5-channel Cell Painting → RGB (AGP, ER, DNA)
- Shot-noise-robust percentile clipping
"""

from __future__ import annotations

import numpy as np


# JUMP Cell Painting channel order (1-indexed, standard acquisition)
# ch1=DNA(DAPI), ch2=ER, ch3=RNA(SYTO), ch4=AGP, ch5=Mito
JUMP_CH_DNA = 0  # index into 0-based array
JUMP_CH_ER = 1
JUMP_CH_RNA = 2
JUMP_CH_AGP = 3
JUMP_CH_MITO = 4

# Standard Cell Painting RGB composite: R=AGP, G=ER, B=DNA
JUMP_RGB_CHANNELS = [JUMP_CH_AGP, JUMP_CH_ER, JUMP_CH_DNA]

# ImageNet mean/std for DINOv2 input normalisation (applied after uint8 → float [0,1])
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def percentile_stretch(
    img: np.ndarray,
    plow: float = 1.0,
    phigh: float = 99.0,
    out_dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """Stretch a single-channel image to [0, 255] using percentile clipping.

    Robust to shot noise and bright outlier pixels.
    Works on any numeric dtype.
    """
    lo = np.percentile(img, plow)
    hi = np.percentile(img, phigh)
    if hi <= lo:
        hi = lo + 1.0
    stretched = (img.astype(np.float32) - lo) / (hi - lo)
    stretched = np.clip(stretched, 0.0, 1.0) * 255.0
    return stretched.astype(out_dtype)


def normalize_multichannel(
    img: np.ndarray,
    plow: float = 1.0,
    phigh: float = 99.0,
) -> np.ndarray:
    """Normalise each channel independently with percentile stretch.

    Args:
        img: (H, W) or (H, W, C) or (C, H, W) array, any dtype.
        plow: lower percentile clip.
        phigh: upper percentile clip.

    Returns:
        uint8 array same shape as input, channels normalised independently.
    """
    if img.ndim == 2:
        return percentile_stretch(img, plow, phigh)

    # Ensure channel-last (H, W, C)
    if img.shape[0] <= 7 and img.shape[0] < img.shape[-1]:
        img = np.moveaxis(img, 0, -1)

    result = np.zeros_like(img, dtype=np.uint8)
    for c in range(img.shape[-1]):
        result[..., c] = percentile_stretch(img[..., c], plow, phigh)
    return result


def to_rgb_uint8(
    img: np.ndarray,
    rgb_channels: list[int] | None = None,
    plow: float = 1.0,
    phigh: float = 99.0,
) -> np.ndarray:
    """Convert any microscopy image to (H, W, 3) uint8 RGB.

    Handles:
    - Grayscale (H, W)  → replicate to 3 channels
    - 3-channel RGB     → normalise each channel
    - 5-channel CP      → select AGP/ER/DNA as R/G/B
    - N-channel         → select first 3 channels (or rgb_channels override)

    Args:
        img: Input image array.
        rgb_channels: Explicit 0-based channel indices to map to R, G, B.
                      Overrides default channel selection.
        plow: Lower percentile for stretch.
        phigh: Upper percentile for stretch.

    Returns:
        (H, W, 3) uint8 array ready for DINOv2 preprocessing.
    """
    # Canonicalize to (H, W, C) or (H, W)
    if img.ndim == 3 and img.shape[0] <= 7 and img.shape[0] < img.shape[2]:
        img = np.moveaxis(img, 0, -1)  # (C, H, W) → (H, W, C)

    if img.ndim == 2:
        # Grayscale
        ch = percentile_stretch(img, plow, phigh)
        return np.stack([ch, ch, ch], axis=-1)

    n_ch = img.shape[-1]

    if rgb_channels is not None:
        selected = np.stack([img[..., c] for c in rgb_channels], axis=-1)
    elif n_ch == 1:
        ch = percentile_stretch(img[..., 0], plow, phigh)
        return np.stack([ch, ch, ch], axis=-1)
    elif n_ch == 2:
        # Two-channel: use ch0, ch1, and mean as B
        selected = np.stack(
            [
                img[..., 0],
                img[..., 1],
                (img[..., 0].astype(np.float32) + img[..., 1]) / 2,
            ],
            axis=-1,
        )
    elif n_ch == 3:
        selected = img
    elif n_ch >= 5:
        # Cell Painting: R=AGP(3), G=ER(1), B=DNA(0)
        selected = np.stack(
            [img[..., JUMP_CH_AGP], img[..., JUMP_CH_ER], img[..., JUMP_CH_DNA]],
            axis=-1,
        )
    else:
        # 4-channel: use first 3
        selected = img[..., :3]

    # Normalise each channel
    rgb = np.zeros((*selected.shape[:2], 3), dtype=np.uint8)
    for c in range(3):
        rgb[..., c] = percentile_stretch(selected[..., c], plow, phigh)
    return rgb


def to_dinov2_tensor(img_rgb_uint8: np.ndarray, size: int = 224):
    """Convert (H, W, 3) uint8 RGB to a normalised torch float32 tensor.

    Returns:
        torch.Tensor of shape (1, 3, size, size) normalised with ImageNet stats.
    """
    import torch
    from PIL import Image

    pil = Image.fromarray(img_rgb_uint8, mode="RGB")
    pil = pil.resize((size, size), Image.BICUBIC)
    arr = np.array(pil, dtype=np.float32) / 255.0  # [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD  # ImageNet normalise
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    return tensor.unsqueeze(0)  # (1, 3, H, W)


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode bytes (TIFF/PNG/JPG) into a numpy array (H, W[, C]).

    Tries tifffile first (for microscopy TIFFs), falls back to PIL.
    """
    import io

    try:
        import tifffile

        img = tifffile.imread(io.BytesIO(data))
        return img
    except Exception:
        pass
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    return np.array(img)
