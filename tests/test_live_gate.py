"""The gate that keeps a stray .env from pointing the suite at production.

`tests/conftest.py` calls `load_dotenv()` unconditionally, so whether a Hypha
credential resolves depends on which checkout pytest was started from. These
tests pin the two consequences of that: live tests are deselected unless
``--live`` is passed, and a resolved credential is announced before the first
test runs.

The sub-runs below use ``--collect-only``, so nothing here contacts a cluster.
"""

import base64
import functools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from tests.conftest import (
    HYPHA_SERVER_URL,
    _token_workspace,
    live_exposure_banner,
    resolved_live_targets,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# One live test in tests/apps/model-runner/, by node name.
A_LIVE_TEST = "test_search_models_returns_list"


def _fake_jwt(workspace: str) -> str:
    claims = {"scope": f"ws:{workspace}#a wid:{workspace}", "sub": "auth0|test"}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode()
    return f"header.{payload.rstrip('=')}.signature"


def _collect(extra_args: List[str], env_overrides: Dict[str, str]) -> str:
    return _collect_cached(tuple(extra_args), tuple(sorted(env_overrides.items())))


@functools.lru_cache(maxsize=None)
def _collect_cached(
    extra_args: Tuple[str, ...], overrides: Tuple[Tuple[str, str], ...]
) -> str:
    """Collect tests/apps/model-runner in a sub-run and return its output.

    Both token variables are always set explicitly: python-dotenv does not
    override a variable that is already in the environment, so passing an
    empty string is what actually neutralises the repo-root .env.
    """
    env = {
        **os.environ,
        "HYPHA_TOKEN": "",
        "BIOIMAGE_IO_TOKEN": "",
        **dict(overrides),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            "tests/apps/model-runner",
            *extra_args,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    # 5 is EXIT_NOTESTSCOLLECTED, which is exactly what a fully deselected
    # scope produces.
    assert result.returncode in (0, 5), result.stdout + result.stderr
    return result.stdout


def test_token_workspace_reads_the_wid_scope() -> None:
    assert _token_workspace(_fake_jwt("bioimage-io")) == "bioimage-io"


def test_token_workspace_tolerates_a_non_jwt() -> None:
    assert _token_workspace("not-a-jwt") == ""


def test_resolved_live_targets_is_empty_without_a_credential() -> None:
    assert resolved_live_targets({"HYPHA_TOKEN": "", "BIOIMAGE_IO_TOKEN": ""}) == []


def test_resolved_live_targets_names_every_credential_in_scope() -> None:
    assert resolved_live_targets(
        {
            "HYPHA_TOKEN": _fake_jwt("ws-user-github|1"),
            "BIOIMAGE_IO_TOKEN": _fake_jwt("bioimage-io"),
        }
    ) == [("HYPHA_TOKEN", "ws-user-github|1"), ("BIOIMAGE_IO_TOKEN", "bioimage-io")]


def test_banner_names_the_server_the_workspace_and_the_count() -> None:
    banner = live_exposure_banner([("BIOIMAGE_IO_TOKEN", "bioimage-io")], 25, False)
    assert HYPHA_SERVER_URL in banner
    assert "bioimage-io" in banner
    assert "25" in banner


def test_banner_distinguishes_deselected_from_about_to_run() -> None:
    targets = [("BIOIMAGE_IO_TOKEN", "bioimage-io")]
    assert "WILL RUN" in live_exposure_banner(targets, 25, True)
    assert "WILL RUN" not in live_exposure_banner(targets, 25, False)


def test_every_resolved_credential_is_announced_before_the_first_test() -> None:
    stdout = _collect(
        [],
        {
            "HYPHA_TOKEN": _fake_jwt("ws-user-github|1"),
            # The model-runner gate prefers this one, so a run can reach
            # production even with HYPHA_TOKEN scoped somewhere harmless.
            "BIOIMAGE_IO_TOKEN": _fake_jwt("bioimage-io"),
        },
    )
    assert "live cluster credentials resolved" in stdout
    assert "HYPHA_TOKEN -> workspace ws-user-github|1" in stdout
    assert "BIOIMAGE_IO_TOKEN -> workspace bioimage-io" in stdout
    assert HYPHA_SERVER_URL in stdout
    assert "25 collected test(s) can reach" in stdout


def test_nothing_is_announced_when_no_credential_resolves() -> None:
    stdout = _collect([], {})
    assert "live cluster credentials resolved" not in stdout


def test_live_tests_are_deselected_by_default() -> None:
    stdout = _collect([], {})
    assert A_LIVE_TEST not in stdout
    assert "25 deselected" in stdout


def test_live_tests_are_selected_with_the_flag() -> None:
    stdout = _collect(["--live"], {})
    assert A_LIVE_TEST in stdout
    assert "deselected" not in stdout
