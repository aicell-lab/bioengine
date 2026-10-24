"""Tests for ``preflight_training_dataset`` and its pure-numpy CLAHE.

Two things here are worth pinning. The validation predicates decide whether a
caller commits to a multi-hour run, and their contract is that a bad pair is
*reported*, not raised — one 404 in a hundred must not sink the batch. And
``_clahe_numpy`` is a reimplementation of ``cv2.createCLAHE().apply()``, written
because adding ``opencv-python-headless`` to the CPU entry dragged in a numpy
that crashed ``EntryApp.__init__`` on deploy (numpy.dtype ABI mismatch against
the pandas imported at init, measured on the deNBI verify of 0.21.0). A
reimplementation with no reference implementation to diff against is exactly the
kind of code that silently degrades to "does something", so the tests below
check that it is *adaptive* — not merely that it changes pixels.

Plain pytest, no network, no GPU, no Ray: the external sources are stubbed, so
what is under test is the preflight logic rather than the BioImage Archive.
"""

from __future__ import annotations

import numpy as np
import pytest

import entry


def _method_of(name: str, instance):
    """The real coroutine function off the Serve-wrapped class.

    ``entry.EntryApp`` is a ``ray.serve.deployment.Deployment`` once
    ``@bioengine.app`` has run, so its methods are not reachable through it.
    """
    return getattr(entry.EntryApp.func_or_class, name).__get__(instance)


# --- the pure-numpy CLAHE ------------------------------------------------


def _faint_blobs(rng: np.random.Generator) -> np.ndarray:
    """A low-contrast frame: faint discs on a slowly varying background, which is
    the fluorescence case CLAHE is being reintroduced for."""
    yy, xx = np.mgrid[0:128, 0:128]
    background = 90 + 20 * (xx / 127.0)
    img = background.copy()
    for cy, cx in ((32, 32), (96, 40), (64, 100)):
        img += 8.0 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 12.0**2)))
    img += rng.normal(0, 0.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_clahe_raises_contrast_on_a_low_contrast_frame() -> None:
    """Magnitudes here are deliberately modest, and that is CLAHE, not a weak
    reimplementation: at ``clip_limit=3`` most of a low-contrast tile's histogram
    mass is clipped and redistributed, so the curve stays close to linear. Checked
    against the reference on this exact frame — ``cv2.createCLAHE(3.0, (16,16))``
    takes std 5.87 to 6.65 where this takes it to 6.29, mean absolute difference
    6.9/255. Anything claiming a 2x std gain would be a different transform.
    """
    img = _faint_blobs(np.random.default_rng(0))

    out = entry._clahe_numpy(img)

    assert out.std() > img.std()
    assert out.ptp() > img.ptp() * 1.4
    assert out.dtype == np.uint8 and out.shape == img.shape


def test_clahe_leaves_a_flat_frame_flat() -> None:
    """The scope control. An equalisation that manufactures structure out of a
    constant frame is amplifying noise that is not there — and the clipped-CDF
    normalisation divides by a range that is zero here, so this is also the
    division-by-zero guard."""
    flat = np.full((64, 64), 120, dtype=np.uint8)

    out = entry._clahe_numpy(flat)

    assert np.isfinite(out).all()
    assert out.ptp() == 0


def test_clahe_is_adaptive_not_one_global_curve() -> None:
    """The discrimination control, and the one that separates CLAHE from plain
    histogram equalisation. A single global LUT maps every occurrence of an input
    value to the same output. Here the same grey sits in a dark half and a bright
    half; if the two map to the same output, the per-tile LUTs are not being
    applied and the implementation has quietly degraded to a global curve.
    """
    img = np.empty((128, 128), dtype=np.uint8)
    img[:, :64] = 40
    img[:, 64:] = 200
    img[30:40, 20:30] = 128   # the same grey, in the dark half
    img[30:40, 84:94] = 128   # ...and in the bright half

    out = entry._clahe_numpy(img)

    assert out[30:40, 20:30].mean() != pytest.approx(out[30:40, 84:94].mean(), abs=1.0)


@pytest.mark.parametrize("shape", [(64, 64), (64, 64, 1), (64, 64, 3), (3, 64, 64)])
def test_apply_clahe_normalises_any_layout_to_one_grayscale_frame(shape) -> None:
    """``_resolve_image`` yields (H,W), (H,W,C) and (C,H,W) frames depending on the
    source, and the staged array has to be one predictable thing regardless."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, shape, dtype=np.uint8)

    out = entry._apply_clahe(img)

    assert out.ndim == 2 and out.shape == (64, 64) and out.dtype == np.uint8


def test_apply_clahe_rescales_a_non_uint8_frame() -> None:
    """BioImage Archive TIFFs are routinely uint16, and the histogram has 256 bins.

    The input is a row-major ramp, so the output must climb from dark to bright
    without ever collapsing. Neither ``dtype == uint8`` nor ``ptp() > 0`` nor even
    a first-row/last-row gap catches the failure that matters: casting uint16
    straight to uint8 truncates modulo 256, which still gives a full-range uint8
    frame with a brighter bottom — but one in which the ramp has been shredded
    into ~15 sawtooth cycles. The collapse at each wrap is what separates the two.
    Measured on this frame: the worst row-to-row drop is -6.7 with the rescale
    (CLAHE's own per-tile wiggle) against -191.8 without it.
    """
    img = (np.linspace(0, 4000, 64 * 64).reshape(64, 64)).astype(np.uint16)

    out = entry._apply_clahe(img)
    row_means = out.mean(axis=1)

    assert out.dtype == np.uint8 and out.ptp() > 0
    assert np.diff(row_means).min() > -50
    assert row_means[0] < 30 and row_means[-1] > 200


# --- preflight validation -------------------------------------------------


class _Replica:
    """Stands in for the entry replica: external sources resolved from a dict,
    staging recorded rather than uploaded."""

    def __init__(self, images: dict, labels: dict):
        self._images, self._labels = images, labels
        self.staged: list = []

    async def _resolve_image(self, src):
        value = self._images[src]
        if isinstance(value, Exception):
            raise value
        return value

    async def _resolve_label_array(self, src, shape):
        value = self._labels[src]
        if isinstance(value, Exception):
            raise value
        return value

    async def _save_npy_to_temp(self, arr):
        self.staged.append(np.asarray(arr))
        return f"https://temp.invalid/staged-{len(self.staged)}.npy"


def _pair(h=32, w=32, n_objects=3, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    lbl = np.zeros((h, w), dtype=np.uint16)
    for k in range(1, n_objects + 1):
        lbl[k * 4:k * 4 + 3, 2:6] = k
    return img, lbl


async def _run(replica, **kwargs):
    return await _method_of("preflight_training_dataset", replica)(**kwargs)


@pytest.mark.asyncio
async def test_unequal_image_and_label_counts_are_refused() -> None:
    with pytest.raises(ValueError):
        await _run(_Replica({}, {}), train_images=["a", "b"], train_labels=["a"])


@pytest.mark.asyncio
async def test_validation_images_and_labels_must_come_together() -> None:
    img, lbl = _pair()
    r = _Replica({"i": img}, {"l": lbl})
    with pytest.raises(ValueError):
        await _run(r, train_images=["i"], train_labels=["l"], val_images=["i"])


@pytest.mark.asyncio
async def test_one_unreachable_source_does_not_sink_the_batch() -> None:
    """The contract that makes preflight worth calling: a dataset with one dead
    file reports which one, instead of raising and telling you nothing."""
    img, lbl = _pair()
    r = _Replica({"i0": img, "i1": RuntimeError("404 Not Found")},
                 {"l0": lbl, "l1": lbl})

    out = await _run(r, train_images=["i0", "i1"], train_labels=["l0", "l1"])

    assert out["ok"] is False
    assert out["train_pair_count"] == 2 and out["train_pair_ok"] == 1
    assert out["pairs"][0]["ok"] is True
    assert out["pairs"][1]["ok"] is False and "404" in out["pairs"][1]["error"]


@pytest.mark.asyncio
async def test_an_image_and_label_of_different_sizes_is_caught() -> None:
    img, _ = _pair(h=32, w=32)
    _, lbl = _pair(h=16, w=16)
    r = _Replica({"i": img}, {"l": lbl})

    out = await _run(r, train_images=["i"], train_labels=["l"])

    assert out["ok"] is False and "mismatch" in out["pairs"][0]["error"]


@pytest.mark.asyncio
async def test_an_empty_label_map_is_rejected() -> None:
    """An all-background mask reads as a valid array and would train on nothing."""
    img, _ = _pair()
    r = _Replica({"i": img}, {"l": np.zeros((32, 32), dtype=np.uint16)})

    out = await _run(r, train_images=["i"], train_labels=["l"])

    assert out["ok"] is False and "no foreground" in out["pairs"][0]["error"]


@pytest.mark.asyncio
async def test_a_clean_dataset_is_staged_into_start_training_urls() -> None:
    img, lbl = _pair()
    r = _Replica({"i": img}, {"l": lbl})

    out = await _run(r, train_images=["i"], train_labels=["l"])

    assert out["ok"] is True and out["staged"] is True
    assert len(out["train_images"]) == 1 and len(out["train_labels"]) == 1
    assert len(r.staged) == 2


@pytest.mark.asyncio
async def test_stage_false_validates_without_uploading_anything() -> None:
    img, lbl = _pair()
    r = _Replica({"i": img}, {"l": lbl})

    out = await _run(r, train_images=["i"], train_labels=["l"], stage=False)

    assert out["ok"] is True and out["staged"] is False
    assert "train_images" not in out
    assert r.staged == []


@pytest.mark.asyncio
async def test_clahe_forces_staging_and_reaches_the_staged_array() -> None:
    """``enable_clahe`` with ``stage=False`` is incoherent — the enhanced pixels
    would be computed and thrown away — so staging is forced. The second assertion
    is what stops the flag being cosmetic: the bytes that get staged must differ
    from the source.
    """
    img, lbl = _pair(seed=5)
    r = _Replica({"i": img}, {"l": lbl})

    out = await _run(r, train_images=["i"], train_labels=["l"],
                     stage=False, enable_clahe=True)

    assert out["staged"] is True and out["clahe_applied"] is True
    assert "train_images" in out
    assert not np.array_equal(r.staged[0], img)
