"""A region annotation records which slice it was drawn on, and a bad one is refused here.

Why this is checked at import and not only at read time: the reader already
refuses an axis of size greater than one with no plane recorded, and that
refusal protects the PIXELS. It cannot protect the RECORD. A plane naming an
axis the store does not have, or an index past its end, would otherwise be
written into the image manifest and surface only when somebody trained on it —
reported as a training failure, long after the annotation was saved, with
nothing pointing back at the import that was wrong.

The ruling this implements: the annotator records the plane; a frontend
convention is not promoted to a contract. An explicit record is checkable and
survives a change to the viewer's default slice, and annotations saved without
one stay refused rather than being guessed at.
"""

from __future__ import annotations

import pathlib

import pytest

import broker_core as core


# ── the shape of a real multi-plane store ─────────────────────────────────────
TCZYX = ["t", "c", "z", "y", "x"]
SHAPE = [1, 3, 9, 1000, 2000]        # z has NINE slices: the case that must be recorded


def test_no_planes_is_None_not_an_empty_mapping():
    """An absent plane and an empty one must not mean two different things to a
    consumer, so both normalise to None."""
    for empty in (None, {}):
        assert core.normalise_planes(empty, TCZYX, SHAPE) is None


@pytest.mark.parametrize("falsy", [0, "", False, [], ()])
def test_a_FALSY_NON_MAPPING_is_refused_rather_than_read_as_no_planes(falsy):
    """`0`, `""`, `False` and `[]` are falsy but are not "no plane given" -- they
    are a caller passing the wrong type, and swallowing them returned a silent
    no-op instead of saying so."""
    with pytest.raises(ValueError) as e:
        core.normalise_planes(falsy, TCZYX, SHAPE)
    assert "object of axis name to index" in str(e.value)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_NON_FINITE_index_is_refused_naming_the_axis(bad):
    """Before this, `inf` raised OverflowError and `nan` a ValueError naming
    neither the axis nor the problem. Nothing was written either way, but the
    message is the whole point of refusing."""
    with pytest.raises(ValueError) as e:
        core.normalise_planes({"z": bad}, TCZYX, SHAPE)
    assert "'z'" in str(e.value) and "finite" in str(e.value)


def test_a_PARTIAL_mapping_is_refused_at_import_not_at_training_time():
    """SHAPE has z=9 and c=3; t is singleton. Naming only z leaves nothing
    missing, but on a store with two multi-slice non-spatial axes a partial
    record would be accepted and then refused by the reader far away from here."""
    two_stacks = [5, 3, 9, 1000, 2000]          # t=5 AND z=9
    roles = {"c": 1, "y": 3, "x": 4}
    with pytest.raises(ValueError) as e:
        core.normalise_planes({"z": 4}, TCZYX, two_stacks, roles)
    assert "'t'" in str(e.value) and "unset" in str(e.value)
    # naming both is accepted
    assert core.normalise_planes({"z": 4, "t": 1}, TCZYX, two_stacks, roles) == {
        "z": 4, "t": 1
    }
    # and naming none is still fine -- the reader refuses that legibly
    assert core.normalise_planes(None, TCZYX, two_stacks, roles) is None


def test_a_named_plane_is_recorded_as_given():
    assert core.normalise_planes({"z": 4}, TCZYX, SHAPE) == {"z": 4}
    assert core.normalise_planes({"z": 0, "t": 0}, TCZYX, SHAPE) == {"z": 0, "t": 0}


def test_the_last_valid_index_is_accepted_and_the_next_one_is_not():
    """The boundary, both sides. An off-by-one here is the whole bug class."""
    assert core.normalise_planes({"z": 8}, TCZYX, SHAPE) == {"z": 8}
    with pytest.raises(ValueError) as e:
        core.normalise_planes({"z": 9}, TCZYX, SHAPE)
    assert "0..8" in str(e.value), str(e.value)


@pytest.mark.parametrize("bad", [-1, 9, 100])
def test_an_index_outside_the_axis_is_REFUSED_naming_the_range(bad):
    with pytest.raises(ValueError) as e:
        core.normalise_planes({"z": bad}, TCZYX, SHAPE)
    msg = str(e.value)
    assert "'z'" in msg and "9 element" in msg, msg


def test_an_axis_the_store_does_not_have_is_REFUSED_naming_the_axes():
    """A caller sending 'slice' or 'Z' has the wrong vocabulary, and silently
    dropping the key would record an annotation with no plane at all — which the
    reader would then refuse far away from here."""
    for name in ("slice", "Z", "depth", "zz"):
        with pytest.raises(ValueError) as e:
            core.normalise_planes({name: 1}, TCZYX, SHAPE)
        assert name in str(e.value) and "does not have" in str(e.value)


@pytest.mark.parametrize("bad", ["two", "4", None, [1], {"a": 1}, 1.5, True, False])
def test_a_non_integer_index_is_REFUSED(bad):
    """Strictly, and `1.5` is the one that matters: `int(1.5)` is 1, so a
    permissive validator turns a fractional index into slice 1 with no
    complaint — the wrong-slice bug this field exists to prevent, arriving by a
    different route. `True` is an int in Python and would become slice 1 too.
    `"4"` is refused rather than coerced: a caller sending a string has a bug,
    and silently parsing it hides the bug instead of the index."""
    with pytest.raises(ValueError):
        core.normalise_planes({"z": bad}, TCZYX, SHAPE)


def test_an_integral_float_is_accepted_because_JSON_has_one_number_type():
    """4.0 over the wire is 4. Refusing it would break a caller that is right."""
    assert core.normalise_planes({"z": 4.0}, TCZYX, SHAPE) == {"z": 4}


def test_a_plane_on_a_source_with_no_recorded_layout_is_REFUSED():
    """Not silently accepted. Without axis names and a shape there is nothing to
    check an index against, and a record that cannot be checked is exactly what
    this validator exists to keep out of the manifest."""
    for axes, shape in ((None, SHAPE), (TCZYX, None), ([], []), (None, None)):
        with pytest.raises(ValueError) as e:
            core.normalise_planes({"z": 1}, axes, shape)
        assert "re-import" in str(e.value) or "does not describe itself" in str(e.value)


def test_a_layout_that_contradicts_itself_is_REFUSED():
    """Axis names and shape of different lengths cannot be zipped, and guessing
    the alignment is how an index lands on the wrong axis."""
    with pytest.raises(ValueError) as e:
        core.normalise_planes({"z": 1}, TCZYX, [1, 3, 9])
    assert "does not describe itself" in str(e.value)


def test_planes_are_not_required_for_a_singleton_axis():
    """The common case: a slide scanner's pyramid is 5-D with t and z of size 1,
    and must not be made to carry a plane it has no choice about."""
    singleton = [1, 3, 1, 1000, 2000]
    assert core.normalise_planes(None, TCZYX, singleton) is None
    assert core.normalise_planes({"z": 0}, TCZYX, singleton) == {"z": 0}
    with pytest.raises(ValueError):
        core.normalise_planes({"z": 1}, TCZYX, singleton)


# ── the entry actually written into the manifest ─────────────────────────────


def _entry(planes):
    return core.remote_image_entry(
        "https://host/img.zarr/",
        None,
        200,
        150,
        dtype="|u1",
        planes=planes,
        layout={"dataset_paths": ["0"], "axis_roles": {"y": 3, "x": 4, "c": 1},
                "level0_shape": SHAPE, "axes": TCZYX},
    )


def test_the_plane_reaches_the_manifest_entry_under_source():
    """Where the reader looks. If this key moves, training stops resolving the
    slice and starts refusing every multi-plane source again."""
    e = _entry({"z": 4})
    assert e["source"]["planes"] == {"z": 4}


def test_an_entry_without_a_plane_carries_no_planes_key_at_all():
    """Rather than a null or an empty object: the reader distinguishes 'nothing
    recorded' from 'recorded as nothing', and only the first is refusable."""
    e = _entry(None)
    assert "planes" not in e["source"]


# ── only NON-SPATIAL axes may carry a plane ──────────────────────────────────

ROLES = {"t": None, "c": 1, "z": None, "y": 3, "x": 4}


@pytest.mark.parametrize("name,role", [("y", "y"), ("x", "x"), ("c", "c")])
def test_a_plane_on_a_SPATIAL_or_CHANNEL_axis_is_refused(name, role):
    """Measured consequence of accepting one: the reader keeps y and x whole
    because they are the crop, and keeps the channel axis whole on purpose, so
    `region_index` with planes={'z':4,'c':2,'y':5} returns
    (0, slice(None), 4, slice(0,10), slice(0,10)) — the record says channel 2
    and training reads every channel, with nothing reporting the difference."""
    with pytest.raises(ValueError) as e:
        core.normalise_planes({name: 0}, TCZYX, SHAPE, ROLES)
    msg = str(e.value)
    assert name in msg and role in msg, msg
    assert "ignored" in msg or "read whole" in msg, msg


def test_a_plane_on_a_non_spatial_axis_is_still_accepted_with_roles_given():
    """THE CONTROL for the three above. Without it, the refusal could be the
    role check rejecting every plane rather than only the wrong ones.

    SHAPE is t=1, c=3, z=9, so naming z is complete (t is singleton) while
    naming only t is NOT -- z would be left unset. That asymmetry is the
    completeness rule working, and this test asserted the old lenient behaviour
    until that rule landed."""
    assert core.normalise_planes({"z": 4}, TCZYX, SHAPE, ROLES) == {"z": 4}
    assert core.normalise_planes({"z": 4, "t": 0}, TCZYX, SHAPE, ROLES) == {
        "z": 4, "t": 0
    }
    with pytest.raises(ValueError) as e:
        core.normalise_planes({"t": 0}, TCZYX, SHAPE, ROLES)
    assert "'z'" in str(e.value) and "unset" in str(e.value)


def test_a_mixed_mapping_is_refused_whole_rather_than_partly_kept():
    """{'z': 4, 'c': 1} must not quietly become {'z': 4}: a caller that sent a
    channel believes the channel was recorded."""
    with pytest.raises(ValueError):
        core.normalise_planes({"z": 4, "c": 1}, TCZYX, SHAPE, ROLES)


def test_absent_roles_forbid_nothing():
    """A source that never recorded which axis is y cannot have a plane refused
    for being y — the check is only as good as the layout, and inventing a role
    would refuse legitimate planes on older imports."""
    assert core.normalise_planes({"y": 5}, TCZYX, SHAPE, None) == {"y": 5}
    assert core.normalise_planes({"c": 1}, TCZYX, SHAPE, {"c": None}) == {"c": 1}
