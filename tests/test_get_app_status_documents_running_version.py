"""``get_app_status`` must say on its PUBLIC surface what ``running_version: None`` means.

``running_version`` is published: it is in ``PUBLIC_APP_STATUS_FIELDS``, so an unauthenticated
caller receives it. It is ``None`` whenever no replica of the ENTRY deployment is in state
RUNNING, which is the normal state for an app whose entry deployment has scaled to zero while
idle. A caller reading ``None`` without that explanation concludes "nothing is running", which is
wrong twice over -- the app is healthy, and ``version`` holds the deployed version.

The worker's internals already explain this, in a comment at the site that builds the status
dict. That comment is not reachable by anyone calling the service, which is the gap these arms
close: they assert on the DOCSTRING, because that is what the RPC schema publishes.

They are deliberately about the published text and not about behaviour. Nothing in this change
alters a value, so an arm asserting behaviour would pass identically before and after and prove
nothing.
"""

import inspect
import re

from bioengine.apps.manager import PUBLIC_APP_STATUS_FIELDS, AppsManager


def _doc() -> str:
    doc = inspect.getdoc(AppsManager.get_app_status)
    assert doc, "get_app_status has no docstring at all"
    return re.sub(r"\s+", " ", doc)


def test_running_version_is_actually_published():
    """The premise. If it were not public, documenting it for callers would be moot."""
    assert "running_version" in PUBLIC_APP_STATUS_FIELDS


def test_the_docstring_NAMES_running_version():
    """RED on main, where the public docstring mentions it zero times."""
    assert "running_version" in _doc(), (
        "get_app_status publishes running_version but its docstring never mentions it, so a "
        "caller has no way to learn what the field means"
    )


def test_the_docstring_says_None_means_NO_ENTRY_REPLICA_RUNNING():
    """The specific condition, not just the field name.

    Naming the field while leaving ``None`` unexplained would satisfy the arm above and still
    leave the caller with the wrong reading, so this pins the cause: an entry deployment with no
    RUNNING replica.
    """
    doc = _doc()
    assert re.search(r"None", doc), "the docstring says nothing about None"
    # Case-insensitive on purpose: the prose emphasises ENTRY in capitals, and an arm that
    # pinned the capitalisation would fail on a correct rewording. The words are the claim.
    assert re.search(
        r"entry deployment is in state ``RUNNING``", doc, re.I
    ), "the docstring does not say that None means no ENTRY-deployment replica is RUNNING"


def test_the_docstring_says_this_is_NORMAL_for_a_scaled_to_zero_app():
    """Without this, None reads as a defect rather than an idle app."""
    doc = _doc()
    assert (
        "scaled to zero" in doc
    ), "the docstring does not mention the scale-to-zero case"
    assert re.search(
        r"NORMAL state", doc
    ), "the docstring does not say the scaled-to-zero case is normal"


def test_the_docstring_points_the_caller_at_version_and_status_instead():
    """A warning with no remedy leaves the caller stuck.

    ``version`` answers "which version is deployed" and ``status`` answers liveness; the whole
    reason None misleads is that readers want one of those two.
    """
    doc = _doc()
    assert re.search(
        r'answer to "which version is deployed"', doc
    ), "the docstring does not say which field DOES answer the deployed-version question"
    assert re.search(
        r"``status``\s*for liveness", doc
    ), "the docstring does not redirect liveness questions to status"


def test_the_docstring_warns_that_version_verified_can_be_TRUE_beside_a_None():
    """The asymmetry that makes the pair look contradictory.

    ``version_verified`` accumulates over every deployment's replicas while
    ``running_version`` comes only from the entry deployment's, so the two disagree on a
    perfectly healthy app. A caller who treats the pair as inconsistent will chase a
    non-problem.
    """
    doc = _doc()
    assert re.search(r"``True`` while ``running_version`` is ``None``", doc), (
        "the docstring does not warn that version_verified can be True while "
        "running_version is None"
    )


def test_every_version_field_in_the_public_allowlist_is_described():
    """Guards against describing three of the four and leaving one to be guessed at.

    Keyed to the allowlist rather than to a hand-written list, so a version field published
    later without a description fails here instead of passing by omission.
    """
    doc = _doc()
    version_fields = [
        f
        for f in PUBLIC_APP_STATUS_FIELDS
        if f in ("version", "running_version", "version_verified", "code_verified")
    ]
    assert (
        len(version_fields) == 4
    ), f"the allowlist's version fields changed: {version_fields}"
    missing = [f for f in version_fields if f"``{f}``" not in doc]
    assert (
        not missing
    ), f"published version fields with no description in the docstring: {missing}"
