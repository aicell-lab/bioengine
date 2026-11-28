"""`apps validate` must not promise a refusal the user's worker may not perform.

`apps validate` runs OFFLINE: its text comes from the installed CLI and says nothing about the
worker the upload will go to. The caveat used to state the non-application refusal
unconditionally, so a reader with a current CLI and an older worker was told they were
protected when that worker would delete their artifact instead. The wrong direction.

These arms pin the two halves that make the sentence honest: the worker boundary is NAMED, and
the reader is told to read the worker's version rather than trust the CLI's wording.
"""

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
    # The retired wording. Asserting its ABSENCE is the point: a caveat that confidently
    # states a protection the worker may not have is worse than a vague one.
    assert "refuses outright, at any version, rather than replacing it" not in caveat


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
