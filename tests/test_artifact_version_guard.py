"""Unit tests for the version-monotonicity guard in stage_artifact.

These tests exercise the pure helper ``_enforce_version_increases`` directly
so they need no Hypha server. See ``test_artifact_version.py`` for the live
end-to-end coverage of ``upload_app``.
"""

import pytest

from bioengine.utils.artifact_utils import _enforce_version_increases


def test_missing_version_rejected():
    with pytest.raises(ValueError, match="missing a 'version' field"):
        _enforce_version_increases(
            new_version=None,
            existing_versions=["1.0.0"],
            artifact_id="ws/app",
        )


def test_empty_existing_versions_passes():
    _enforce_version_increases(
        new_version="0.0.1",
        existing_versions=[],
        artifact_id="ws/app",
    )


def test_strictly_greater_passes():
    _enforce_version_increases(
        new_version="1.1.0",
        existing_versions=["1.0.0", "1.0.5"],
        artifact_id="ws/app",
    )


def test_patch_bump_passes():
    _enforce_version_increases(
        new_version="1.0.6",
        existing_versions=["1.0.5"],
        artifact_id="ws/app",
    )


def test_equal_to_highest_rejected():
    with pytest.raises(ValueError, match="strictly greater"):
        _enforce_version_increases(
            new_version="1.0.0",
            existing_versions=["1.0.0"],
            artifact_id="ws/app",
        )


def test_lower_than_highest_rejected():
    with pytest.raises(ValueError, match="strictly greater"):
        _enforce_version_increases(
            new_version="0.9.0",
            existing_versions=["1.0.0", "1.1.0"],
            artifact_id="ws/app",
        )


def test_highest_is_not_last_in_list():
    """List order doesn't matter — comparison is against the parsed max."""
    with pytest.raises(ValueError, match="strictly greater"):
        _enforce_version_increases(
            new_version="1.1.0",
            existing_versions=["1.2.0", "1.0.0"],
            artifact_id="ws/app",
        )


def test_non_pep440_exact_match_rejected():
    with pytest.raises(ValueError, match="already exists"):
        _enforce_version_increases(
            new_version="demo",
            existing_versions=["demo", "v2"],
            artifact_id="ws/app",
        )


def test_non_pep440_different_string_passes():
    """If parsing fails and the new tag isn't an exact match, accept it."""
    _enforce_version_increases(
        new_version="rc-3",
        existing_versions=["demo", "v2"],
        artifact_id="ws/app",
    )
