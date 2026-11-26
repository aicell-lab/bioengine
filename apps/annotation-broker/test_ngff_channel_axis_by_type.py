"""The channel axis is whatever OME-NGFF says it is, not whatever is called "c".

OME-NGFF declares what each axis IS — ``{"name": "ch", "type": "channel"}`` —
and nothing in the spec requires the channel axis to be named ``c``. This app
kept only the axis NAMES out of ``multiscales`` and then did ``axes.index("c")``,
so a store that called its channel axis ``ch`` was reported as having no channel
axis at all.

WHY THAT IS WORSE THAN A WRONG LABEL. The viewer's plane record comes from the
broker's roles, and model-finetune's remote region reader takes its channel axis
from those same roles. So all three agreed — on the wrong axis. A user annotated
one plane of a multichannel image and the model trained on another, with nothing
anywhere disagreeing, which is the one shape no amount of cross-checking between
the three can catch.

THE DISCRIMINATING CASE, and the reason a test on a ``ch`` store alone is not
enough: a store can have an axis *named* ``c`` that is NOT the channel axis. The
test below gives one axis the name ``c`` and the type ``space`` and another the
name ``ch`` and the type ``channel``. A name-only reader returns the first; only
a type-first reader returns the second. A fixture whose channel axis is merely
renamed is passed by any implementation that happens to look at types *or* at
nothing in particular.

THE TRAINER'S HALF IS ASSERTED HERE, ACROSS THE TWO APPS, on purpose.
``apps/model-finetune/ngff_region.py`` is loaded by path and fed the roles this
app produced, because the clause to satisfy is that the planes the viewer shows
and the planes the trainer reads are the SAME — and two tests in two app
directories each pinning the number 1 would satisfy that clause without
establishing it. The import is cheap: that module's only module-level imports
are ``re`` and ``typing``.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

import broker_core as core

# ── the fixture store: a channel axis named "ch", and a SPATIAL axis named "c" ──

# Shape [1, 3, 64, 64] over t, ch, y, x. The `c`-named axis is `t`'s neighbour
# and is NOT channels, which is what makes a name-only reader wrong rather than
# merely unlucky.
CH_STORE = {
    "multiscales": [
        {
            "name": "sample",
            "datasets": [{"path": "0"}, {"path": "1"}],
            "axes": [
                {"name": "c", "type": "space"},
                {"name": "ch", "type": "channel"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
        }
    ]
}
CH_SHAPE = [1, 3, 64, 64]
CHANNEL_INDEX = 1  # the "ch" axis
NAME_ONLY_INDEX = 0  # what a name-only reader returns: the "c"-named SPACE axis


def _ngff_region():
    """``apps/model-finetune/ngff_region.py``, loaded by path.

    Not a `skipif`: the two apps live in one repository, so this file being
    absent is a real problem and must fail rather than quietly remove the only
    cross-app assertion in the suite.
    """
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "model-finetune"
        / "ngff_region.py"
    )
    assert path.is_file(), f"the trainer's region reader is not at {path}"
    spec = importlib.util.spec_from_file_location("_ngff_region_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _labelled_channels():
    """[1, 3, 64, 64] over t, ch, y, x, with each channel filled with its own
    value, so a test can tell WHICH channels came back and not merely how
    many."""
    array = np.zeros(CH_SHAPE, dtype=np.uint8)
    for k in range(CH_SHAPE[1]):
        array[0, k] = k + 1
    return array


# ── the parse: types come back, aligned with the names ────────────────────


def test_the_TYPES_come_back_beside_the_names():
    parsed = core.parse_ngff_multiscales(CH_STORE)
    assert parsed["axes"] == ["c", "ch", "y", "x"]
    assert parsed["axis_types"] == ["space", "channel", "space", "space"]
    assert len(parsed["axis_types"]) == len(parsed["axes"])


def test_an_UNDECLARED_type_is_None_and_not_a_guess():
    store = {
        "multiscales": [
            {
                "datasets": [{"path": "0"}],
                "axes": [{"name": "ch"}, {"name": "y"}, {"name": "x"}],
            }
        ]
    }
    parsed = core.parse_ngff_multiscales(store)
    assert parsed["axis_types"] == [None, None, None]


def test_a_PRE_v04_store_declares_no_types_at_all():
    """`axes` was optional before v0.4 and the order was a convention about
    NAMES, so there is nothing to read types from and the name fallback is the
    only evidence there is."""
    parsed = core.parse_ngff_multiscales({"multiscales": [{"datasets": ["0"]}]})
    assert parsed["axes"] == list(core.NGFF_IMPLICIT_AXES)
    assert parsed["axis_types"] == [None] * len(core.NGFF_IMPLICIT_AXES)


# ── the channel axis: type first, name second ─────────────────────────────


def test_the_TYPE_DECIDES_even_when_another_axis_is_named_c():
    """THE DISCRIMINATING CASE. A name-only reader returns 0 here."""
    axes = ["c", "ch", "y", "x"]
    types = ["space", "channel", "space", "space"]

    assert core.ngff_channel_axis(axes, types) == CHANNEL_INDEX
    assert (
        core.ngff_channel_axis(axes, None) == NAME_ONLY_INDEX
    ), "without types the name is all there is, and this records what that costs"


def test_the_NAME_is_the_fallback_when_no_type_is_declared():
    assert core.ngff_channel_axis(["t", "c", "y", "x"], [None] * 4) == 1
    assert core.ngff_channel_axis(["t", "c", "y", "x"]) == 1


def test_a_channel_axis_with_NO_TYPE_and_a_NON_c_NAME_is_honestly_unknown():
    """Not a gap being papered over. Nothing in such a store says which axis is
    channels, and inventing an answer is worse than reporting none."""
    assert core.ngff_channel_axis(["t", "ch", "y", "x"], [None] * 4) == -1


def test_NO_channel_axis_is_minus_one():
    assert core.ngff_channel_axis(["y", "x"], ["space", "space"]) == -1


def test_TWO_axes_typed_channel_are_REFUSED_rather_than_guessed():
    """OME-NGFF allows at most one, so this store is malformed. Picking the
    first would be the silent wrong answer this whole change is about: the
    viewer and the trainer would still agree with each other and both be wrong
    about the image."""
    with pytest.raises(core.NgffSourceError) as exc:
        core.ngff_channel_axis(
            ["a", "b", "y", "x"], ["channel", "channel", "space", "space"]
        )
    assert "at most one" in str(exc.value)
    assert "'a'" in str(exc.value) and "'b'" in str(exc.value)


def test_a_type_that_is_not_channel_is_not_a_channel_axis():
    for t in ("space", "time", "CHANNELS", "chan", ""):
        assert core.ngff_channel_axis(["q", "y", "x"], [t, "space", "space"]) == -1, t


# ── the roles the viewer and the trainer both read ────────────────────────


def test_the_ROLES_carry_the_typed_channel_axis():
    parsed = core.parse_ngff_multiscales(CH_STORE)
    roles = core.ngff_axis_roles(parsed["axes"], len(CH_SHAPE), parsed["axis_types"])
    assert roles == {"y": 2, "x": 3, "c": CHANNEL_INDEX}


def test_Y_and_X_are_still_found_by_NAME_and_that_is_not_an_oversight():
    """OME-NGFF types BOTH of them `space`, so the type says which two
    dimensions are spatial and cannot say which is which."""
    parsed = core.parse_ngff_multiscales(CH_STORE)
    spatial = [i for i, t in enumerate(parsed["axis_types"]) if t == "space"]
    assert spatial == [0, 2, 3], "the type alone cannot pick y and x out of these"
    roles = core.ngff_axis_roles(parsed["axes"], len(CH_SHAPE), parsed["axis_types"])
    assert (roles["y"], roles["x"]) == (2, 3)


def test_the_ROLES_FALL_BACK_when_the_axes_cannot_be_trusted():
    """The pre-existing fallback is untouched: a declared axis list of the
    wrong length, or one with no y/x, still yields the two fastest-varying
    dimensions and NO channel axis."""
    roles = core.ngff_axis_roles(["a", "b"], 4, ["channel", "space"])
    assert roles == {"y": 2, "x": 3, "c": None}


# ── clause 1, across the two apps: the trainer reads the SAME axis ─────────


def test_the_TRAINER_READS_THE_SAME_CHANNEL_AXIS_THE_BROKER_REPORTS():
    """Clause 1. The roles this app computes, handed to the trainer's reader as
    the stored layout hands them over, must put the channels on the same axis.

    The assertion is the SHAPE the trainer ends up with: 3 channels, off the
    `ch` axis, for a store whose `c`-named axis has length 1.
    """
    region = _ngff_region()
    parsed = core.parse_ngff_multiscales(CH_STORE)
    roles = core.ngff_axis_roles(parsed["axes"], len(CH_SHAPE), parsed["axis_types"])

    # The layout the broker stores on the image entry, which is what both the
    # viewer and the trainer read.
    layout = {
        "dataset_paths": parsed["dataset_paths"],
        "axis_roles": roles,
        "level0_shape": CH_SHAPE,
        "axes": parsed["axes"],
    }
    assert layout["axis_roles"]["c"] == CHANNEL_INDEX

    # The REAL pipeline, not just the transpose: `region_index` drops every
    # axis except y, x and c -- keeping the channel axis WHOLE -- and
    # `to_hw_or_hwc` then puts it last. Calling only the transpose would have
    # handed it a 4-D array it returns untouched, which passes nothing.
    array = _labelled_channels()
    index = region.region_index(
        layout["axis_roles"],
        len(CH_SHAPE),
        CH_SHAPE,
        {"x": 0, "y": 0, "width": 8, "height": 8},
        axes=layout["axes"],
    )
    out = region.to_hw_or_hwc(array[index], layout["axis_roles"], len(CH_SHAPE))

    assert out.shape == (8, 8, 3), out.shape
    # Each channel kept its own values, so the axis was not merely the right
    # length by accident.
    assert [int(out[0, 0, k]) for k in range(3)] == [1, 2, 3]


def test_THE_NAME_ONLY_READ_MAKES_THE_TRAINER_REFUSE_THE_WHOLE_REGION():
    """Clause 2's mutant as a test, and it measures something DIFFERENT from
    what I expected, so the expectation is recorded here rather than quietly
    replaced.

    I went looking for the trainer silently reading channel 0 of three. It does
    not: with the channel axis read by NAME, `ch` is not a role axis, so
    `region_index` sees a 3-element axis with no plane given and REFUSES, naming
    the axis and saying that index 0 is not assumed. That guard is doing its
    job. The harm on this store is therefore not "trains on the wrong channel"
    but "a perfectly good multichannel image cannot be trained on at all, and
    the error tells the user to pick ONE channel of an image the viewer drew as
    multichannel".

    Which is what makes clause 3's "both halves together" the binding part: the
    divergence the row warns about is not in today's behaviour, it is what a
    one-sided fix would create.
    """
    region = _ngff_region()
    parsed = core.parse_ngff_multiscales(CH_STORE)

    name_only = core.ngff_axis_roles(parsed["axes"], len(CH_SHAPE), None)
    assert name_only["c"] == NAME_ONLY_INDEX

    with pytest.raises(region.RegionSourceError) as exc:
        region.region_index(
            name_only,
            len(CH_SHAPE),
            CH_SHAPE,
            {"x": 0, "y": 0, "width": 8, "height": 8},
            axes=parsed["axes"],
        )

    message = str(exc.value)
    assert "'ch'" in message, message
    assert "3 elements" in message
    # And with the TYPED read the same call needs no plane at all, which is the
    # difference the fix makes on this store.
    typed = core.ngff_axis_roles(parsed["axes"], len(CH_SHAPE), parsed["axis_types"])
    region.region_index(
        typed,
        len(CH_SHAPE),
        CH_SHAPE,
        {"x": 0, "y": 0, "width": 8, "height": 8},
        axes=parsed["axes"],
    )


def test_NAMING_A_PLANE_ON_THE_CHANNEL_AXIS_IS_STILL_REFUSED():
    """The pre-existing rule, confirmed against the NEW channel axis: a channel
    axis is kept whole, so a caller cannot ask for one channel of it as though
    it were a z or t plane. Before the fix `ch` was not a channel axis, so this
    request was ACCEPTED -- it is the route by which a user would have trained
    on a single channel."""
    region = _ngff_region()
    parsed = core.parse_ngff_multiscales(CH_STORE)
    typed = core.ngff_axis_roles(parsed["axes"], len(CH_SHAPE), parsed["axis_types"])

    with pytest.raises(region.RegionSourceError):
        region.region_index(
            typed,
            len(CH_SHAPE),
            CH_SHAPE,
            {"x": 0, "y": 0, "width": 8, "height": 8},
            axes=parsed["axes"],
            planes={"ch": 1},
        )

    name_only = core.ngff_axis_roles(parsed["axes"], len(CH_SHAPE), None)
    index = region.region_index(
        name_only,
        len(CH_SHAPE),
        CH_SHAPE,
        {"x": 0, "y": 0, "width": 8, "height": 8},
        axes=parsed["axes"],
        planes={"ch": 1},
    )
    array = _labelled_channels()
    out = region.to_hw_or_hwc(array[index], name_only, len(CH_SHAPE))
    assert out.size == 64, f"{out.shape} is not a single plane"
    assert (
        int(np.asarray(out).reshape(-1)[0]) == 2
    ), "before the fix, planes={'ch': 1} trained on channel 1 ALONE"
