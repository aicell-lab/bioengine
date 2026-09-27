"""The gate that keeps a stray .env from pointing the suite at production.

`tests/conftest.py` calls `load_dotenv()` unconditionally and `find_dotenv()`
walks *up*, so a credential can arrive without anyone asking for it. These
tests pin the three consequences: live tests are deselected unless ``--live``
is passed, every resolved credential is announced before the first test with
the count it enables, and that announcement survives the xdist worker that
``pytest.ini``'s ``addopts`` puts collection inside.

Sub-runs either deselect everything or use ``--collect-only``, and the tokens
are synthetic, so nothing here contacts a cluster.
"""

import ast
import base64
import functools
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from tests.conftest import (
    HYPHA_SERVER_URL,
    _token_workspace,
    live_exposure_banner,
    resolved_live_targets,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# One live test in tests/apps/model-runner/, by node name.
A_LIVE_TEST = "test_search_models_returns_list"

# pytest.ini's addopts carries --numprocesses=1, so a sub-run that honours it
# needs xdist. It is in requirements-test.txt; ad-hoc runners sometimes
# install a narrower set.
needs_shipped_addopts = pytest.mark.skipif(
    importlib.util.find_spec("xdist") is None,
    reason="pytest-xdist absent, so pytest.ini's addopts cannot be honoured",
)


def _fake_jwt(workspace: str) -> str:
    claims = {"scope": f"ws:{workspace}#a wid:{workspace}", "sub": "auth0|test"}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode()
    return f"header.{payload.rstrip('=')}.signature"


def _collect(extra_args: List[str], env_overrides: Dict[str, str]) -> str:
    """A sub-run with addopts neutralised -- the fast path, for selection logic."""
    return _run_cached(
        ("-o", "addopts=", "--collect-only", "-q", *extra_args),
        tuple(sorted(env_overrides.items())),
    )


def _run_with_default_addopts(
    extra_args: List[str], env_overrides: Dict[str, str]
) -> str:
    """A sub-run under pytest.ini as shipped.

    `addopts` carries `--numprocesses=1`, so collection happens inside an
    xdist worker whose terminalreporter output is discarded. Passing
    `-o addopts=` -- as every other sub-run here does -- hides that entirely,
    which is why this variant exists.
    """
    return _run_cached(tuple(extra_args), tuple(sorted(env_overrides.items())))


@functools.lru_cache(maxsize=None)
def _run_cached(
    extra_args: Tuple[str, ...], overrides: Tuple[Tuple[str, str], ...]
) -> str:
    """Run pytest over tests/apps/model-runner and return its combined output.

    Both token variables are always set explicitly: python-dotenv does not
    override a variable that is already in the environment, so passing an
    empty string is what actually neutralises the repo-root .env -- which
    `load_dotenv()` finds by walking up, even from a worktree.
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
            "-p",
            "no:cacheprovider",
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
    return result.stdout + result.stderr


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


@needs_shipped_addopts
def test_the_banner_survives_the_default_addopts() -> None:
    """pytest.ini runs collection inside an xdist worker; the banner must
    still reach the controller, count and all."""
    output = _run_with_default_addopts([], {"HYPHA_TOKEN": _fake_jwt("bioimage-io")})
    assert "live cluster credentials resolved" in output
    assert "HYPHA_TOKEN -> workspace bioimage-io" in output
    assert "25 collected test(s) can reach" in output


@needs_shipped_addopts
def test_the_wall_clock_summary_states_the_exposure_even_with_the_flag() -> None:
    """--live stops nothing being reported: the count and the runtime are the
    only audit trail left once someone has opted in on purpose."""
    output = _run_with_default_addopts(
        ["--live", "--collect-only"], {"HYPHA_TOKEN": _fake_jwt("bioimage-io")}
    )
    assert "25 collected test(s) can reach" in output
    assert "live cluster exposure:" in output
    assert "--live GIVEN" in output


def test_the_artifact_version_tests_are_gated() -> None:
    """Four tests that call upload_app and delete artifacts on
    bioimage-io/bioengine-worker. Their fixture asserts on the token rather
    than skipping, so without the gate they error rather than skip."""
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
            "tests/test_artifact_version.py",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "HYPHA_TOKEN": "", "BIOIMAGE_IO_TOKEN": ""},
        capture_output=True,
        text=True,
    )
    assert "4 deselected" in result.stdout, result.stdout


def _module_pytestmarks(relpath: str) -> List[str]:
    tree = ast.parse((REPO_ROOT / relpath).read_text())
    return [
        ast.unparse(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
    ]


@pytest.mark.parametrize(
    "relpath",
    [
        # Drives a browser at the deployed app and injects HYPHA_TOKEN into
        # its localStorage. Uses only the `page` fixture, so the marker is
        # the only thing that gates it. Checked on the source rather than by
        # collection because playwright is not installed here -- which is
        # also why it collects as an import error, and why nobody noticed.
        "tests/apps/cellpose/test_ui_e2e.py",
        # Calls upload_app and deletes artifacts on bioimage-io/bioengine-worker.
        # The fixture-name rule catches these too, but only because the file
        # happens to call its own fixture `hypha_client`; rename it and the
        # gate silently stops applying. The marker is what makes it deliberate,
        # so assert on the marker and not just on the deselection below.
        "tests/test_artifact_version.py",
    ],
)
def test_the_modules_a_marker_alone_can_gate_carry_one(relpath: str) -> None:
    assert _module_pytestmarks(relpath) == ["pytest.mark.live"]


def test_a_malformed_token_does_not_abort_the_session() -> None:
    """A payload decoding to a JSON scalar used to raise AttributeError out of
    a collection hook, which pytest turns into INTERNALERROR."""
    scalar_payload = base64.urlsafe_b64encode(b"123").decode().rstrip("=")
    output = _collect([], {"HYPHA_TOKEN": f"header.{scalar_payload}.sig"})
    assert "INTERNALERROR" not in output
    assert "HYPHA_TOKEN -> workspace <unreadable token>" in output
