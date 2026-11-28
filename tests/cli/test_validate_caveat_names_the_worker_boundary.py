"""`apps validate` must not promise a refusal the user's worker may not perform.

`apps validate` runs OFFLINE: its text comes from the installed CLI and says nothing about the
worker the upload will go to. The caveat used to state the non-application refusal
unconditionally, so a reader with a current CLI and an older worker was told they were
protected when that worker would delete their artifact instead. The wrong direction.

These arms pin the two halves that make the sentence honest: the worker boundary is NAMED, and
the reader is told to read the worker's version rather than trust the CLI's wording.
"""

import os
import re

import pytest
from click.testing import CliRunner

from bioengine.cli.apps import apps_group

# The boundary is asserted as LITERALS below, not via the new constants. Importing those at
# module scope turns this whole file into a collection ERROR against unfixed code -- and an
# error is not a failure, so the arms would be credited with a RED they never produced. The
# one arm that genuinely tests the constants imports them INSIDE itself, where an ImportError
# is a test failure like any other.
FIRST_REFUSING_WORKER = "0.16.142"
LAST_DELETING_WORKER = "0.16.141"

# --- the unqualified-promise scan, shared by the caveat arm and the tree-wide arm -----------
#
# Three things about the real text defeated a simpler version of this scan, so each is handled
# deliberately rather than by widening a window until it passed:
#
# 1. The verb appears as "refused" AND as "refuses". A pattern keyed to one misses the other,
#    and the CLI's own retired wording was the "refuses" case.
# 2. The qualifier may come BEFORE the phrase ("a worker at X or later refuses outright, at any
#    manifest version") or AFTER it ("refused at any manifest version by a worker at X"). So the
#    window has to be the whole SENTENCE, not a suffix of the match.
# 3. A sentence end cannot be "the next period": version numbers contain periods, so `[^.]`
#    stops inside "0.16.142" -- truncating the window just before the qualifier that makes the
#    sentence honest. SENTENCE_END therefore ignores a period that follows a digit.
PROMISE = re.compile(r"refus\w*[^.]{0,240}?at any (?:manifest )?version", re.I)
QUALIFIER = re.compile(
    r"0\.16\.1[0-9]{2}|worker at|bioengine_version|depends on the WORKER", re.I
)
SENTENCE_END = re.compile(r"(?<![0-9])\.(?=\s|\Z)")


def _flatten(text: str, is_python: bool) -> str:
    """One line, with Python string-literal scaffolding reduced to the prose it renders."""
    flat = re.sub(r"\s+", " ", text.replace("\n", " "))
    if is_python:
        flat = flat.replace("\\n", " ")  # a literal \n inside a string literal
        flat = re.sub(r'"\s*f?"', " ", flat)  # f-string concatenation seams
        flat = re.sub(r"\s+", " ", flat)
    return flat


def _sentence_around(flat: str, start: int, end: int) -> str:
    left = 0
    for match in SENTENCE_END.finditer(flat, 0, start):
        left = match.end()
    match = SENTENCE_END.search(flat, end)
    return flat[left : (match.end() if match else len(flat))].strip()


def _unqualified_promises(text: str, is_python: bool = False) -> list:
    """Every "refused at any version" claim whose own sentence names no worker boundary."""
    flat = _flatten(text, is_python)
    return [
        sentence
        for match in PROMISE.finditer(flat)
        if not QUALIFIER.search(
            sentence := _sentence_around(flat, match.start(), match.end())
        )
    ]


# The three sentences this issue is actually about, copied from the tree BEFORE the fix. Using
# the real text as the control is the point: a scan is only as good as the hardest case it
# catches, and a short invented "refused" sentence proves only the case that already worked.
RETIRED_DOCS = (
    "An upload onto such an id is **refused**, at any version: the artifact is left "
    "alone and the error names it and tells you to change the `id`."
)
RETIRED_SKILL = (  # 225 chars to its full stop -- a fixed 200-char window misses this one
    "An upload onto such an id is **refused at any version**, with an error naming the "
    "artifact and telling you to change the `id`; the version-monotonicity error you may "
    "be expecting never appears, because that check only runs for an artifact the upload "
    "keeps."
)
RETIRED_CLI = (  # "refuses", not "refused" -- a pattern keyed to one verb form misses this
    "If something that is NOT an application holds it - a dataset, a model - upload "
    "refuses outright, at any version, rather than replacing it."
)


@pytest.fixture(scope="module")
def caveat() -> str:
    result = CliRunner().invoke(apps_group, ["validate", "apps/model-runner"])
    assert result.exit_code == 0, result.output
    return result.output


def test_the_caveat_NAMES_the_first_refusing_worker(caveat):
    assert FIRST_REFUSING_WORKER in caveat


def test_the_caveat_NAMES_the_last_deleting_worker(caveat):
    assert LAST_DELETING_WORKER in caveat


def test_the_caveat_says_an_older_worker_DELETES(caveat):
    # The direction that matters: the reader must learn the old behaviour destroys the
    # artifact, not merely that the new one is better.
    assert "DELETES" in caveat


def test_the_caveat_points_at_the_WORKER_version_not_the_CLI(caveat):
    assert "bioengine_version" in caveat


def test_the_UNCONDITIONAL_claim_is_gone(caveat):
    """The caveat must state no refusal the reader's worker may not perform.

    Keyed to the CLASS of claim, not to the one retired sentence. An arm that asserts the
    absence of an exact string is satisfied by any rewording of the same falsehood, so it
    stops being evidence the moment the text is edited -- which is precisely when it is read.
    """
    assert not _unqualified_promises(caveat), (
        "the caveat promises a refusal without naming the worker boundary:\n"
        + "\n".join(_unqualified_promises(caveat))
    )


def test_the_SCAN_catches_all_three_retired_sentences():
    """Two-sided control for the scan the two tree/caveat arms depend on.

    Without this, a clean result from either of those arms is unfalsifiable: a pattern that
    quietly stopped matching reports exactly the same empty list as a clean tree. Each of
    these three breaks the scan in a different way -- verb form, sentence length, and the
    period inside a version number -- so they are kept separate rather than collapsed into
    one representative string.
    """
    for label, sentence in (
        ("docs", RETIRED_DOCS),
        ("skill (225 chars to its full stop)", RETIRED_SKILL),
        ("cli ('refuses', not 'refused')", RETIRED_CLI),
    ):
        assert _unqualified_promises(
            sentence
        ), f"scan no longer flags the retired {label} text"

    # And the other side: the QUALIFIED forms must come back clean, or the arms above would
    # fail on correct text and the fix would be to weaken the scan.
    for label, sentence in (
        (
            "qualifier after the phrase",
            "An upload onto such an id is **refused at any manifest version by a worker at "
            "0.16.142 or later** (anything at or below 0.16.141 DELETES it instead), with an "
            "error naming the artifact.",
        ),
        (
            "qualifier before the phrase",
            "Whether the upload is refused depends on the WORKER you upload to: a worker at "
            "0.16.142 or later refuses outright, at any manifest version, and leaves the "
            "artifact alone.",
        ),
    ):
        assert not _unqualified_promises(
            sentence
        ), f"scan flags correctly qualified text ({label}); it would reject the fix"


def test_the_boundary_constants_are_adjacent_releases():
    """Pins the pair, so bumping one without the other cannot pass.

    If the refusal ever moves to a different release, both constants change together or this
    arm fails -- the sentence would otherwise name a gap or an overlap.
    """
    from bioengine.cli.apps import _FIRST_REFUSING_WORKER, _LAST_DELETING_WORKER

    # Also pins the literals above to the module's own constants, so the two cannot drift.
    assert _FIRST_REFUSING_WORKER == FIRST_REFUSING_WORKER
    assert _LAST_DELETING_WORKER == LAST_DELETING_WORKER
    first = tuple(int(p) for p in _FIRST_REFUSING_WORKER.split("."))
    last = tuple(int(p) for p in _LAST_DELETING_WORKER.split("."))
    assert first[:2] == last[:2]
    assert first[2] == last[2] + 1


def test_the_caveat_names_a_RUNNABLE_route_to_the_worker_version(caveat):
    """ "Read the worker's bioengine_version" is only advice if a command prints it.

    No bioengine CLI command does: the field is in the worker's public get_status allowlist
    and on the website's worker cards, but `cluster status --json` prints cluster and nodes
    only. So the caveat has to name the route, or it tells the reader to check something they
    cannot reach from the CLI they are already holding.
    """
    assert "get_status" in caveat, caveat
    assert "bioengine call" in caveat, caveat


def test_NO_OTHER_doc_promises_the_refusal_without_the_worker_qualifier():
    """The CLI caveat and the prose docs must not contradict each other.

    The defect is a reader reassured of a protection their worker may not have. Fixing it in
    the CLI while `docs/apps-guide.md` and the maintainer skill still say "refused at any
    version" leaves the reassurance in place and makes the tree self-contradictory. This arm
    is tree-wide on purpose: the same fix-one-file-and-stop mistake has now happened three
    times on this work.

    Reads the WHOLE SENTENCE around each promise, because the qualifier may fall on either
    side of the phrase. See the notes on PROMISE/SENTENCE_END above for the three properties
    of the real text that a suffix-of-the-match window got wrong; the scan's own two-sided
    control is test_the_SCAN_catches_all_three_retired_sentences.
    """
    offenders = []
    for base in ("docs", ".claude/skills", "bioengine"):
        for dirpath, _dirnames, filenames in os.walk(base):
            for name in filenames:
                if not name.endswith((".md", ".py")):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8") as handle:
                    body = handle.read()
                for sentence in _unqualified_promises(body, name.endswith(".py")):
                    offenders.append(f"{path}: {sentence[:160]}")

    assert not offenders, "unqualified refusal promises:\n" + "\n".join(offenders)
