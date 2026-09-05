"""Public segmentation datasets, one per federation site.

The two defaults were picked so a failure to generalise is visible by eye after
a couple of minutes of training: both are bright objects on a dark background,
so the only thing that separates them is object *shape* — many small round
nuclei versus a handful of long thin worms. A U-Net trained on one produces
obvious garbage on the other.

Neither download needs credentials.
"""

import hashlib
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen

import numpy as np

DATASETS: Dict[str, Dict[str, str]] = {
    "dsb2018-fluo": {
        "url": "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip",
        "objects": "fluorescence cell nuclei (many small round objects)",
        "source": "BBBC038v1 (Kaggle 2018 Data Science Bowl), fluorescence subset "
                  "as repackaged by StarDist (repo BSD-3-Clause)",
        "licence": "CC0",
        "citation": "Caicedo et al., Nature Methods, 2019",
    },
    "bbbc010-worms": {
        "url": "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip",
        "mask_url": "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground.zip",
        "objects": "C. elegans worms (few long thin objects)",
        "source": "BBBC010 v1/v2, Broad Bioimage Benchmark Collection",
        "licence": "CC0",
        "citation": "Ljosa et al., Nature Methods, 2012",
    },
    # The acquisition-split pair: one object class, two imaging modalities, cut
    # out of a single archive so nothing but the modality differs between them.
    "bbbc038-fluo": {
        "url": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip",
        "slot": "bbbc038",
        "modality": "fluorescence",
        "objects": "cell nuclei imaged by fluorescence",
        "source": "BBBC038v1 (2018 Data Science Bowl), fluorescence subset",
        "licence": "CC0",
        "citation": "Caicedo et al., Nature Methods, 2019",
    },
    "bbbc038-histo": {
        "url": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip",
        "slot": "bbbc038",
        "modality": "histology",
        "objects": "cell nuclei in H&E-stained histology",
        "source": "BBBC038v1 (2018 Data Science Bowl), histology subset",
        "licence": "CC0",
        "citation": "Caicedo et al., Nature Methods, 2019",
    },
}


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(url, timeout=300) as response, open(tmp, "wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    tmp.rename(dest)
    return dest


def _extract(archive: Path, target: Path) -> Path:
    marker = target / ".complete"
    if marker.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = [
            n for n in zf.namelist()
            if "__MACOSX" not in n and "/.svn/" not in n and not n.endswith("/")
        ]
        zf.extractall(target, members=members)
    marker.write_text("ok")
    return target


def _normalize(image: np.ndarray) -> np.ndarray:
    """Percentile-normalise to [0, 1] so intensity scale never carries the signal."""
    image = image.astype(np.float32)
    if image.ndim == 3:
        image = image.mean(axis=-1)
    lo, hi = np.percentile(image, (1.0, 99.8))
    if hi <= lo:
        return np.zeros_like(image)
    # np.percentile returns float64 scalars, which under NumPy 2's NEP 50 promote
    # the whole array; nothing downstream calls .float() on the tensor, so a
    # float64 image reaches a float32 model and raises. Harmless under the pinned
    # numpy 1.26, load-bearing the moment that pin moves.
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _load_dsb2018(root: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    import tifffile

    image_dir = root / "dsb2018" / "train" / "images"
    pairs = []
    for image_path in sorted(image_dir.glob("*.tif")):
        mask_path = root / "dsb2018" / "train" / "masks" / image_path.name
        if not mask_path.exists():
            continue
        pairs.append((_normalize(tifffile.imread(image_path)),
                      (tifffile.imread(mask_path) > 0).astype(np.float32)))
    return pairs


def _load_bbbc010(root: Path, mask_root: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    import imageio.v3 as iio
    import tifffile

    # w1 is the channel the foreground annotation was drawn on; w2 has roughly
    # half its object/background contrast.
    by_well = {}
    for image_path in root.glob("*_w1_*.tif"):
        match = re.search(r"_([A-Z]\d{2})_w1_", image_path.name)
        if match:
            by_well[match.group(1)] = image_path
    pairs = []
    for mask_path in sorted(mask_root.glob("*_binary.png")):
        well = mask_path.name.split("_")[0]
        if well not in by_well:
            continue
        mask = iio.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        pairs.append((_normalize(tifffile.imread(by_well[well])),
                      (mask > 0).astype(np.float32)))
    return pairs


def _bbbc038_modality(rgb: np.ndarray) -> str:
    """Which imaging modality produced this image.

    Read off the pixels rather than out of metadata.xlsx, whose stain_type is
    per project group with no mapping to an ImageId. The two classes separate
    absolutely on this collection: saturation is exactly zero up to the 80th
    percentile, every one of the 108 coloured images has a bright background,
    and only 2 of 670 images sit anywhere in the ambiguous band.
    """
    saturation = float((rgb.max(-1) - rgb.min(-1)).mean())
    if saturation > 0.02:
        return "histology"
    # Greyscale on a bright background is brightfield -- a third modality, and
    # the only 1024x1024 images here. Assigning it to either side would put two
    # modalities on one side of a two-site modality split.
    return "brightfield" if float(np.median(rgb.mean(-1))) > 0.35 else "fluorescence"


def _signal(rgb: np.ndarray, modality: str) -> np.ndarray:
    """The physically correct 'amount of signal' for the modality.

    Fluorescence is emission, so intensity is the signal. Histology is
    absorbance, so optical density is, and it is linear in stain concentration.
    Both make nuclei the bright class, which matters: the raw greyscale polarity
    is inverted between the two (objects are brighter than background in 100% of
    fluorescence images and 0% of histology ones), and feeding both through one
    intensity pipeline would turn an acquisition shift into a sign flip.
    """
    if modality == "histology":
        return -np.log10((rgb * 255.0 + 1.0) / 256.0).mean(-1)
    return rgb.mean(-1)


def _load_bbbc038(root: Path, modality: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    import imageio.v3 as iio

    pairs = []
    for image_dir in sorted(root.iterdir()):
        image_paths = list((image_dir / "images").glob("*.png"))
        if not image_paths:
            continue
        rgb = iio.imread(image_paths[0])[..., :3].astype(np.float32) / 255.0
        if _bbbc038_modality(rgb) != modality:
            continue
        # One PNG per nucleus; the task here is foreground, not instances.
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        for mask_path in (image_dir / "masks").glob("*.png"):
            mask |= iio.imread(mask_path) > 0
        pairs.append((_normalize(_signal(rgb, modality)), mask.astype(np.float32)))
    return pairs


def load_dataset(
    name: str,
    cache_dir: Path,
    n_train: int = 55,
    n_val: int = 15,
    n_test: int = 25,
    split_seed: int = 20260905,
) -> Dict[str, object]:
    """Download, cache and deterministically split a dataset into train/val/test.

    The test split is the held-out set every arm of the experiment is scored on,
    so it is cut once from a fixed permutation that does not depend on the
    training seed — otherwise the four arms would not share a test set and the
    comparison between them would be meaningless.
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    spec = DATASETS[name]
    # The two BBBC038 modalities come out of one archive, so they share a slot
    # and the pooled instance downloads it once rather than twice.
    slot = cache_dir / spec.get("slot", name)
    archive = _download(spec["url"], slot / "images.zip")
    extracted = _extract(archive, slot / "images")

    if name == "dsb2018-fluo":
        pairs = _load_dsb2018(extracted)
    elif "modality" in spec:
        pairs = _load_bbbc038(extracted, spec["modality"])
    else:
        mask_archive = _download(spec["mask_url"], slot / "masks.zip")
        mask_root = _extract(mask_archive, slot / "masks") / "BBBC010_v1_foreground"
        pairs = _load_bbbc010(extracted, mask_root)

    if not pairs:
        raise RuntimeError(f"dataset {name!r} extracted but no image/mask pairs were found")
    if len(pairs) < n_train + n_val + n_test:
        raise RuntimeError(
            f"dataset {name!r} has {len(pairs)} pairs, too few for "
            f"n_train={n_train} + n_val={n_val} + n_test={n_test}"
        )

    order = np.random.default_rng(split_seed).permutation(len(pairs))
    cuts = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val : n_train + n_val + n_test],
    }
    splits = {k: [pairs[i] for i in idx] for k, idx in cuts.items()}
    return {
        "name": name,
        **splits,
        "n_available": len(pairs),
        "objects": spec["objects"],
        "source": spec["source"],
        "licence": spec["licence"],
        "citation": spec["citation"],
        "mean_foreground_fraction": float(np.mean([m.mean() for _, m in splits["train"]])),
        # Recorded per arm; a mismatch between arms means they were not scored
        # on the same images and the comparison has to be thrown away.
        "split_fingerprint": hashlib.sha256(
            (
                f"{name}|{split_seed}|{len(pairs)}|"
                + ",".join(str(int(i)) for i in cuts["test"])
            ).encode("utf-8")
        ).hexdigest()[:16],
    }
