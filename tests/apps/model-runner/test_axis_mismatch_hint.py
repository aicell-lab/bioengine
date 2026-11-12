"""A grayscale image refused by a 3-channel model must say so, and say what to do.

bioimageio raises ``ValueError("Array shape (...) does not map to axes [AxisInfo(...)]")``.
It is accurate and unactionable: it reprs the model's axis objects and leaves the caller to
work out which axis is wrong. A workshop participant hit exactly that on 2026-11-11 with a
grayscale image against Cellpose-SAM (myco #0308).

THE FIXTURE IS REPRODUCED, NOT TRANSCRIBED. The production log that prompted this issue
shows the message elided with "..."; a parser written against the elided form would have
been a guess, so it was not. The text below is the full message as the real code path
emits it, captured by running ``create_sample_for_model`` — the call ``runtime.py`` makes
at the top of its dispatch — against a ``(1, 1, 807, 696)`` float32 array and the live
``idealistic-eagle`` (Cellpose-SAM) description inside
``ghcr.io/aicell-lab/model-runner:2.11.6`` (bioimageio.core 0.11.0, spec 0.5.12.0).

Two details of that capture are worth keeping, because guessing either one wrong is how
this fixture rots into fiction:

* The space axes carry ``step=None``, not ``step=16``. Cellpose-SAM declares a bare
  ``size: 256`` for ``y`` and ``x``, and bioimageio reprs a *fixed* size as
  ``AxisSize(min=256, max=None, step=None)`` — a lower bound, so the error understates
  its own constraint. An earlier revision of this fixture guessed ``step=16``; the parser
  is insensitive to the value (it accepts ``\\d+|None``) and produced a byte-identical
  hint either way, which is exactly why the error survived unnoticed.
* A fixed ``size: 256`` still reprs *as* an ``AxisSize(...)``. Had it reprd as a bare
  int, ``_AXIS_INFO_RE`` would have matched 2 of the 4 axes, the length check in
  ``_axis_mismatch_hint`` would have returned ``None``, and the hint would never have
  fired in production while every test here still passed.

The helper is a pure function parsed out of ``runtime.py`` with the AST loader this
directory already uses, so none of bioengine, ray or bioimageio is imported here.
"""

import ast
import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parents[3] / "apps" / "model-runner"
RUNTIME = APP / "runtime.py"

# The exact text, as captured in the image. Kept on one line because that is how it
# arrives — reflowing it would be a different string from the one the parser must handle.
REAL = (
    "Array shape (1, 1, 807, 696) does not map to axes ["
    "AxisInfo(id='batch', type='batch', size=AxisSize(min=1, max=None, step=None)), "
    "AxisInfo(id='channel', type='channel', size=AxisSize(min=3, max=3, step=None)), "
    "AxisInfo(id='y', type='space', size=AxisSize(min=256, max=None, step=None)), "
    "AxisInfo(id='x', type='space', size=AxisSize(min=256, max=None, step=None))]"
)

# The SAME capture, for a plain 2-D grayscale array — what ``iio.imread`` of a grayscale
# PNG actually yields, and the shape this issue's own live arm logged on 2026-11-11.
# Captured in the same run as REAL above, not derived from it; the integrity test below
# asserts the two differ only in the shape tuple.
REAL_2D = (
    "Array shape (807, 696) does not map to axes ["
    "AxisInfo(id='batch', type='batch', size=AxisSize(min=1, max=None, step=None)), "
    "AxisInfo(id='channel', type='channel', size=AxisSize(min=3, max=3, step=None)), "
    "AxisInfo(id='y', type='space', size=AxisSize(min=256, max=None, step=None)), "
    "AxisInfo(id='x', type='space', size=AxisSize(min=256, max=None, step=None))]"
)


def _load(path, *names):
    """Exec just the named top-level defs/assignments out of ``path``."""
    tree = ast.parse(path.read_text())
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    found = {
        n.name if hasattr(n, "name") else n.targets[0].id for n in wanted
    }
    if found != set(names):
        pytest.fail(f"missing {sorted(set(names) - found)} in {path.name}")
    # ONE namespace, used as both globals and locals. With two, the module-level regex
    # assignments land in locals while the function's __globals__ is the other dict, and
    # calling it raises NameError on its own constants — which is how this first ran.
    ns = {"re": __import__("re"), "Optional": __import__("typing").Optional}
    exec(  # noqa: S102 — the house pattern in this directory
        compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), ns
    )
    return ns


@pytest.fixture(scope="module")
def hint():
    ns = _load(RUNTIME, "_ARRAY_SHAPE_RE", "_AXIS_INFO_RE", "_axis_mismatch_hint")
    return ns["_axis_mismatch_hint"]


def test_the_real_workshop_message_names_the_channel_axis(hint):
    """The whole point: the caller is told which axis and what to do about it."""
    got = hint(REAL)
    assert got is not None, "the message that prompted #0308 was not recognised"
    assert "'channel' axis must be exactly 3 but the array supplies 1" in got
    assert "grayscale" in got
    assert "np.repeat(arr, 3, axis=1)" in got, "the remedy must be runnable, not described"


def test_the_original_error_is_kept_verbatim(hint):
    """Actionable must not mean lossy — the accurate part is still there to debug with."""
    got = hint(REAL)
    assert REAL in got


def test_a_message_that_is_not_an_axis_error_is_left_alone(hint):
    """Anything unrecognised falls through, so this can never make an error worse."""
    assert hint("CUDA error: device-side assert triggered") is None
    assert hint("") is None


def test_a_shape_that_satisfies_every_axis_is_not_reported(hint):
    """The discriminator. If this returned a hint, the parser would be inventing one.

    Same axes, a conforming (1, 3, 807, 696) array. bioimageio would not raise for it;
    the point here is that the helper's own test is "does any axis fall outside its
    bounds", not "did the string contain the word channel".
    """
    conforming = REAL.replace("Array shape (1, 1, 807, 696)", "Array shape (1, 3, 807, 696)")
    assert hint(conforming) is None


def test_a_space_axis_below_its_minimum_is_reported_without_the_channel_remedy(hint):
    """#0293's shape, which is a different axis — the hint must follow the data.

    A hint that always said "channel" would pass the first test and be wrong here, so
    this is what holds the two apart.
    """
    small = REAL.replace("Array shape (1, 1, 807, 696)", "Array shape (1, 3, 100, 696)")
    got = hint(small)
    assert got is not None
    assert "'y' axis must be at least 256 but the array supplies 100" in got
    assert "grayscale" not in got
    assert "np.repeat" not in got


def test_the_step_field_is_parsed_but_does_not_change_the_verdict(hint):
    """Stated on purpose, because this insensitivity already hid a wrong fixture.

    ``_AXIS_INFO_RE`` accepts ``step=<int>`` and ``step=None`` and the bounds check
    ignores the value, so a guessed step produces a byte-identical hint — which is how
    an earlier revision of this fixture carried ``step=16`` against a production
    ``step=None`` without any arm here reddening. Keeping the property asserted means a
    future change that *starts* reading ``step`` has to come past this test, and the
    fixture's accuracy stops being load-bearing for the parser.
    """
    stepped = REAL.replace("min=256, max=None, step=None", "min=256, max=None, step=16")
    assert stepped != REAL, "the substitution must actually change the string"
    assert hint(stepped) is not None
    assert hint(stepped).split("Original error")[0] == hint(REAL).split("Original error")[0]


def test_a_length_mismatch_names_no_axis_but_still_states_the_requirement(hint):
    """Position only means something when the two sequences line up — but that only
    forbids NAMING an axis, not saying anything at all.

    bioimageio tries permutations and singleton insertions before giving up, so pairing
    a 3-entry shape against 4 axes would name an axis at random. The required sizes and
    the shape that arrived are still true, though, and they are the actionable half.

    This test used to assert ``hint(ragged) is None`` and so pinned the gap shut: the
    2-D grayscale upload — this issue's own reproduction — went through this branch and
    came out byte-identical to the unfixed code. Keeping the "no axis is named" property
    asserted preserves what the guard is actually for.
    """
    ragged = REAL.replace("Array shape (1, 1, 807, 696)", "Array shape (1, 807, 696)")
    got = hint(ragged)
    assert got is not None
    assert "axis must be" not in got, "no axis may be named on a rank mismatch"
    assert "3 dimensions but this model declares 4 axes" in got
    assert "channel=3" in got and "y>=256" in got, "the requirement is still stated"
    assert REAL.split("does not map")[1] in got, "the original error is still carried"


def test_the_real_2d_grayscale_upload_is_given_the_requirement_and_a_remedy(hint):
    """The variant the refusal was about: a grayscale file, uploaded the ordinary way.

    ``imageio`` yields ``(H, W)`` for a grayscale PNG and nothing between the upload and
    ``create_sample_for_model`` adds an axis, so this — not the 4-D form — is what the
    workshop participant's own client produced. CAPTURED, not transcribed: run against
    the live ``idealistic-eagle`` description inside ``ghcr.io/aicell-lab/model-runner:2.11.2``
    (bioimageio.core 0.11.0, spec 0.5.12.0 — the versions all three ``requirements-*.txt``
    pin), with ``(1, 3, 807, 696)`` as a control that the library does NOT raise for a
    conforming array.
    """
    got = hint(REAL_2D)
    assert got is not None, "the shape this issue actually reproduced was not recognised"
    assert "axis must be" not in got, "2 sizes against 4 axes cannot name an axis"
    assert "2 dimensions but this model declares 4 axes" in got
    assert "carries no channel axis at all" in got
    assert "np.repeat(arr[np.newaxis, np.newaxis], 3, axis=1)" in got, (
        "the remedy must be runnable: two leading axes added, then the channel replicated"
    )
    assert REAL_2D in got


def test_the_2d_fixture_differs_from_the_4d_one_only_in_the_shape(hint):
    """An integrity check on the two captured strings, not a behavioural assertion.

    Both were captured from the same call in the same image, so the axis list must be
    identical; if a future recapture changes one and not the other, the two tests above
    would be measuring different models and nothing would say so.
    """
    assert REAL_2D == REAL.replace("Array shape (1, 1, 807, 696)", "Array shape (807, 696)")
    assert REAL_2D != REAL
