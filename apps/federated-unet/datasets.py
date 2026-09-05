"""Public segmentation datasets, one per federation site.

Three groups live here. ``dsb2018-fluo`` and ``bbbc010-worms`` are the
caricature pair, picked so a failure to generalise is visible by eye: nuclei
against worms differ in object *shape* alone, and a U-Net trained on one
produces obvious garbage on the other. ``bbbc038-fluo`` and ``bbbc038-histo``
are the acquisition pair, one collection cut by imaging modality. The rest are
the consortium clients — one public dataset per acquisition setup, all doing 2D
nuclei segmentation, rostered in bioengine-paper
``analysis/data/federated_consortium/README.md``.

No download needs credentials, and every licence is CC0 or CC BY 4.0.
"""

import csv
import hashlib
import io
import re
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

from overlap import BBBC038_DUPLICATE_IDS, BBBC039_DUPLICATE_FILES, BBBC039_EMPTY_MASKS

BBBC039 = "https://data.broadinstitute.org/bbbc/BBBC039"
BIOSTUDIES = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD"

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
    # The consortium clients. One per acquisition setup, so a client boundary is
    # an instrument boundary rather than a slice of one collection.
    "bbbc039": {
        "url": f"{BBBC039}/images.zip",
        "mask_url": f"{BBBC039}/masks.zip",
        "objects": "U2OS nuclei, Hoechst, DNA channel of the BBBC022 Cell Painting screen",
        "source": "BBBC039v1, Broad Bioimage Benchmark Collection, ImageXpress Micro",
        "licence": "CC0",
        "citation": "Caicedo et al., Cytometry A 95, 952-965 (2019)",
    },
    "cellbindb": {
        "url": f"{BIOSTUDIES}/538/S-BIAD1538/Files",
        "objects": "nuclei in ssDNA-stained whole-slide tiles",
        # The Zenodo mirror (record 15370205) carries a record-level CC0 stamp
        # over re-bundled CC BY-NC data from other groups. This is the deposit
        # holding only BGI's own images.
        "source": "S-BIAD1538 CellBinDB, ssDNA subset, BGI Research STOmics slide scanner",
        "licence": "CC0",
        "citation": "CellBinDB, BioImage Archive S-BIAD1538, DOI 10.6019/S-BIAD1538",
    },
    "nuinsseg": {
        "url": "https://zenodo.org/records/10518968/files/NuInsSeg.zip",
        "objects": "nuclei in H&E-stained sections across 31 organs",
        "source": "NuInsSeg, Zenodo 10518968, TissueFAXS / Axio Imager Z1",
        "licence": "CC BY 4.0",
        "citation": "Mahbod et al., Sci. Data 11, 295 (2024), doi 10.1038/s41597-024-03117-2",
    },
    "kromp": {
        # Deposited twice; S-BIAD634 is the 2023 re-deposit carrying the CC0
        # tag, S-BSST265 is the same data with no licence attribute at all.
        "url": f"{BIOSTUDIES}/634/S-BIAD634/Files",
        "objects": "nuclei in clinical specimens — neuroblastoma, bone-marrow cytospin, touch imprint",
        "source": "S-BIAD634, Children's Cancer Research Institute Vienna, DAPI",
        "licence": "CC0",
        "citation": "Kromp et al., Sci. Data 7, 262 (2020), doi 10.1038/s41597-020-00608-w",
    },
}


def _download(url: str, dest: Path) -> Path:
    """Fetch one file, retrying on transient refusals.

    CellBinDB is fetched file by file, so the pooled oracle makes a few hundred
    requests to the same host in a row and BioStudies refuses some of them. One
    refused file out of hundreds must not fail a whole client's load.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            with urlopen(url, timeout=300) as response, open(tmp, "wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
            break
        except (URLError, OSError):
            tmp.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2**attempt)
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


def _fetch_text(url: str) -> str:
    with urlopen(url, timeout=300) as response:
        return response.read().decode("utf-8", "replace")


def _load_dsb2018(slot: Path, spec: Dict[str, str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    import tifffile

    root = _extract(_download(spec["url"], slot / "images.zip"), slot / "images")
    image_dir = root / "dsb2018" / "train" / "images"
    pairs = []
    for image_path in sorted(image_dir.glob("*.tif")):
        mask_path = root / "dsb2018" / "train" / "masks" / image_path.name
        if not mask_path.exists():
            continue
        pairs.append((_normalize(tifffile.imread(image_path)),
                      (tifffile.imread(mask_path) > 0).astype(np.float32)))
    return pairs


def _load_bbbc010(slot: Path, spec: Dict[str, str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    import imageio.v3 as iio
    import tifffile

    root = _extract(_download(spec["url"], slot / "images.zip"), slot / "images")
    mask_root = (
        _extract(_download(spec["mask_url"], slot / "masks.zip"), slot / "masks")
        / "BBBC010_v1_foreground"
    )
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


def _load_bbbc038(slot: Path, spec: Dict[str, str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    import imageio.v3 as iio

    modality = spec["modality"]
    root = _extract(_download(spec["url"], slot / "images.zip"), slot / "images")
    pairs = []
    for image_dir in sorted(root.iterdir()):
        # The 43 fields shared with BBBC039 are all fluorescence, but the check
        # is unconditional so it cannot silently stop applying if the modality
        # classifier ever moves one of them.
        if image_dir.name in BBBC038_DUPLICATE_IDS:
            continue
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


def _load_bbbc039(slot: Path, spec: Dict[str, str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    import imageio.v3 as iio
    import tifffile

    images = _extract(_download(spec["url"], slot / "images.zip"), slot / "images")
    masks = _extract(_download(spec["mask_url"], slot / "masks.zip"), slot / "masks")
    pairs = []
    for image_path in sorted(images.rglob("*.tif")):
        if image_path.name in BBBC039_DUPLICATE_FILES:
            continue
        if image_path.name.startswith(BBBC039_EMPTY_MASKS):
            continue
        mask_paths = list(masks.rglob(f"{image_path.stem}.png"))
        if not mask_paths:
            continue
        # The masks are not instance labels: they are RGBA colour maps holding a
        # 3-4 value class index in the red channel, arranged so two touching
        # nuclei differ. Foreground is red > 0.
        mask = iio.imread(mask_paths[0])[..., 0] > 0
        pairs.append((_normalize(tifffile.imread(image_path)), mask.astype(np.float32)))
    return pairs


def _load_nuinsseg(slot: Path, spec: Dict[str, str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    import imageio.v3 as iio

    # 48,399 entries of which 1,330 are wanted, so the archive is read in place
    # rather than extracted.
    archive = _download(spec["url"], slot / "nuinsseg.zip")
    pairs = []
    with zipfile.ZipFile(archive) as zf:
        for member in sorted(
            n for n in zf.namelist()
            if "/tissue images/" in n and n.lower().endswith(".png")
        ):
            organ, _, stem = member.split("/")[0], None, Path(member).stem
            # "label masks modify" is the authors' own evaluation mask, labelled
            # consecutively 1..N; plain "label masks" gives overlaps their own id.
            label = f"{organ}/label masks modify/{stem}.tif"
            if label not in zf.namelist():
                continue
            rgb = iio.imread(io.BytesIO(zf.read(member)))[..., :3].astype(np.float32) / 255.0
            mask = iio.imread(io.BytesIO(zf.read(label))) > 0
            pairs.append((_normalize(_signal(rgb, "histology")), mask.astype(np.float32)))
    return pairs


def _load_cellbindb(slot: Path, spec: Dict[str, str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    import tifffile

    base = spec["url"]
    rows = csv.DictReader(
        io.StringIO(_fetch_text(f"{base}/CellBinDB_Archive.tsv")), delimiter="\t"
    )
    # ssDNA over the larger DAPI subset: 276 tiles from 127 slides against 203
    # from 14, so it is both bigger and about seven times more independent.
    members = sorted(
        row["Files"]
        for row in rows
        if row["Type"] == "Original Image" and row["Staining Type"].strip() == "ssDNA"
    )
    pairs = []
    for member in members:
        stem = Path(member).stem.replace("-img", "")
        image = _download(f"{base}/{member}", slot / "images" / f"{stem}.tif")
        label = _download(
            f"{base}/{member.replace('-img.tif', '-instancemask.tif')}",
            slot / "masks" / f"{stem}.tif",
        )
        # Tiles are 8-bit L, little-endian I;16 and big-endian I;16B mixed inside
        # this one client, so the per-image percentile normalisation is doing
        # real work here rather than only rescaling.
        pairs.append(
            (_normalize(tifffile.imread(image)), (tifffile.imread(label) > 0).astype(np.float32))
        )
    return pairs


def _load_kromp(slot: Path, spec: Dict[str, str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    import tifffile

    base = spec["url"]
    rows = csv.DictReader(
        io.StringIO(_fetch_text(f"{base}/file_list_raw_images.tsv")), delimiter="\t"
    )
    # One of the 80 rows carries the literal string "null" in every metadata
    # column and is not one of the 42 train / 37 test annotated fields.
    members = sorted(
        row["Files"] for row in rows if row.get("Files") and row.get("Diagnosis") != "null"
    )
    pairs = []
    for member in members:
        stem = Path(member).stem
        image = _download(f"{base}/{member}", slot / "images" / f"{stem}.tif")
        label = _download(
            f"{base}/dataset/groundtruth/{stem}.tif", slot / "masks" / f"{stem}.tif"
        )
        raw = tifffile.imread(image)
        # L and RGB files are mixed in this client and the RGB ones carry the
        # DAPI signal in blue alone, so averaging the channels would dilute it
        # by three against the greyscale files.
        signal = raw[..., 2] if raw.ndim == 3 else raw
        pairs.append((_normalize(signal), (tifffile.imread(label) > 0).astype(np.float32)))
    return pairs


LOADERS: Dict[str, Callable[[Path, Dict[str, str]], List[Tuple[np.ndarray, np.ndarray]]]] = {
    "dsb2018-fluo": _load_dsb2018,
    "bbbc010-worms": _load_bbbc010,
    "bbbc038-fluo": _load_bbbc038,
    "bbbc038-histo": _load_bbbc038,
    "bbbc039": _load_bbbc039,
    "cellbindb": _load_cellbindb,
    "nuinsseg": _load_nuinsseg,
    "kromp": _load_kromp,
}


def parse_spec(spec: str) -> Tuple[str, List[int], int]:
    """Read a dataset spec: ``name``, ``name@k/n`` or ``name@*/n``.

    ``@k/n`` gives a site one of n disjoint slices of the training pool, so two
    sites can hold the same domain without holding the same images. ``@*/n`` is
    the union of all n, which is what the pooled oracle needs to remain an upper
    bound once a domain is spread over several sites.
    """
    name, _, shard_part = spec.partition("@")
    if not shard_part:
        return name, [0], 1
    which, _, total = shard_part.partition("/")
    n_shards = int(total)
    if n_shards < 1:
        raise ValueError(f"{spec!r}: shard count must be >= 1")
    if which == "*":
        return name, list(range(n_shards)), n_shards
    index = int(which)
    if not 0 <= index < n_shards:
        raise ValueError(f"{spec!r}: shard {index} is outside 0..{n_shards - 1}")
    return name, [index], n_shards


def load_dataset(
    dataset_spec: str,
    cache_dir: Path,
    n_train: Optional[int] = 55,
    n_val: int = 15,
    n_test: int = 25,
    split_seed: int = 20260905,
) -> Dict[str, object]:
    """Download, cache and deterministically split a dataset into train/val/test.

    ``n_train=None`` takes everything left after the fixed test and val blocks,
    which is what the consortium runs on: equal-sized readouts per client and
    naturally unequal training sets, so client size is a real axis rather than a
    number chosen here.

    The test split is the held-out set every arm of the experiment is scored on,
    so it is cut once from a fixed permutation that does not depend on the
    training seed — otherwise the four arms would not share a test set and the
    comparison between them would be meaningless. It is cut from a position that
    does not move with the shard count either, so a run that spreads a domain
    over several sites is still scored on exactly the images an earlier two-site
    run was scored on.
    """
    name, shards, n_shards = parse_spec(dataset_spec)
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    spec = DATASETS[name]
    # The two BBBC038 modalities come out of one archive, so they share a slot
    # and the pooled instance downloads it once rather than twice.
    slot = cache_dir / spec.get("slot", name)
    pairs = LOADERS[name](slot, spec)

    if not pairs:
        raise RuntimeError(f"dataset {name!r} extracted but no image/mask pairs were found")
    if n_train is None:
        # Natural size: every image the client has that is not held out. Test and
        # val stay at fixed counts so no client's readout is noisier than
        # another's, and what is left over is the size axis the experiment needs.
        n_train = (len(pairs) - n_test) // n_shards - n_val
        if n_train < 1:
            raise RuntimeError(
                f"dataset {name!r} has {len(pairs)} pairs, too few for {n_shards} site(s) "
                f"of n_val={n_val} plus n_test={n_test}"
            )
    block = n_train + n_val
    if len(pairs) < n_test + n_shards * block:
        raise RuntimeError(
            f"dataset {name!r} has {len(pairs)} pairs, too few for {n_shards} disjoint "
            f"site(s) of n_train={n_train} + n_val={n_val} plus n_test={n_test}"
        )

    order = np.random.default_rng(split_seed).permutation(len(pairs))
    # The test block sits where a single unsharded site would put it, and the
    # per-site blocks are cut from what is left. Shard 0 of 1 therefore
    # reproduces the unsharded split exactly, indices included.
    test_idx = order[block : block + n_test]
    rest = np.concatenate([order[:block], order[block + n_test :]])
    cuts = {
        "train": np.concatenate([rest[k * block : k * block + n_train] for k in shards]),
        "val": np.concatenate([rest[k * block + n_train : (k + 1) * block] for k in shards]),
        "test": test_idx,
    }
    splits = {k: [pairs[i] for i in idx] for k, idx in cuts.items()}
    return {
        "name": name,
        "spec": dataset_spec,
        "shards": shards,
        "n_shards": n_shards,
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
        # Two sites on the same domain must not hold the same images; comparing
        # these across sites is how that is checked rather than assumed.
        "shard_fingerprint": hashlib.sha256(
            (
                f"{name}|{split_seed}|{len(pairs)}|"
                + ",".join(str(int(i)) for i in cuts["train"])
            ).encode("utf-8")
        ).hexdigest()[:16],
    }
