"""``BIOENGINE_SERVER_URL`` must actually reach the worker process.

The container launcher passes the server as an environment variable and the
image's entrypoint is ``python -m bioengine.worker``, so the variable arrives in
the process environment. Nothing read it: ``--server-url`` was declared without
an ``envvar``, the unset flag was dropped before construction, and the worker
fell through to its own ``hypha.aicell.io`` default — registering against the
wrong server while reporting healthy, with no stage at which the operator is
told their setting was ignored.

**Two halves, and they cover different failures.** The resolver's *behaviour*
is exercised below in a subprocess that runs the same functions in the same
order. That does NOT cover the *wiring*, because the probe calls those
functions itself — mutation-checked: deleting the ``resolve_server_url`` call
from the entrypoint, which is precisely the shape of the original bug, left
every behavioural test green.

The wiring is therefore asserted separately, against the module source. It has
to be: the resolution chain lives in a bare ``if __name__ == "__main__":``
block, so it cannot be imported and called — which is also why it never had a
test, and why a declared-but-unread option could sit there unnoticed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bioengine.worker.__main__ import (
    create_parser,
    get_args_by_group,
    resolve_server_url,
)

CHECKOUT_ROOT = Path(__file__).resolve().parents[2]

ENV_URL = "https://env.example.invalid"
FLAG_URL = "https://flag.example.invalid"
DEFAULT_URL = "https://hypha.aicell.io"

# Resolves exactly as the real entrypoint does — same functions, same order —
# then prints the result instead of starting a worker.
_PROBE = """
import json, sys
from bioengine.worker.__main__ import (
    create_parser, get_args_by_group, resolve_token, resolve_server_url,
)

configs = resolve_token(get_args_by_group(create_parser()))
configs = resolve_server_url(configs)
print(json.dumps({"server_url": configs["Hypha Options"].get("server_url")}))
"""


def _resolve_in_subprocess(argv, env_url):
    """Run the real resolution chain in a fresh interpreter."""
    env = dict(os.environ)
    env.pop("BIOENGINE_SERVER_URL", None)
    env.pop("HYPHA_TOKEN", None)
    if env_url is not None:
        env["BIOENGINE_SERVER_URL"] = env_url
    env["PYTHONPATH"] = str(CHECKOUT_ROOT)

    result = subprocess.run(
        [sys.executable, "-c", _PROBE, *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(CHECKOUT_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])["server_url"]


def _resolve(argv, monkeypatch, env_url):
    monkeypatch.delenv("BIOENGINE_SERVER_URL", raising=False)
    if env_url is not None:
        monkeypatch.setenv("BIOENGINE_SERVER_URL", env_url)
    monkeypatch.setattr(sys, "argv", ["bioengine.worker", *argv])
    return resolve_server_url(get_args_by_group(create_parser())).get(
        "Hypha Options", {}
    ).get("server_url")


# ===== the defect =====


def test_the_environment_variable_reaches_the_worker() -> None:
    """The regression: set the variable, pass no flag, and it must be used."""
    assert _resolve_in_subprocess(["--mode", "single-machine"], ENV_URL) == ENV_URL


def test_the_flag_still_wins_over_the_environment(monkeypatch) -> None:
    """Precedence matches the token's: explicit argument beats ambient."""
    resolved = _resolve(
        ["--mode", "single-machine", "--server-url", FLAG_URL], monkeypatch, ENV_URL
    )
    assert resolved == FLAG_URL


def test_neither_source_leaves_the_key_absent(monkeypatch) -> None:
    """Absent, not empty — ``get_args_by_group`` drops ``None`` so the worker's
    own default applies. Writing an empty string here would override it."""
    assert _resolve(["--mode", "single-machine"], monkeypatch, None) is None


def test_an_empty_environment_variable_is_not_a_url(monkeypatch) -> None:
    """An exported-but-empty variable is how a shell passes "unset", and the
    CLI's own passthrough drops falsey values for exactly that reason."""
    assert _resolve(["--mode", "single-machine"], monkeypatch, "") is None


def test_an_empty_flag_is_rejected_like_an_empty_token_file(monkeypatch) -> None:
    """``--server-url ""`` is a misconfiguration, not a request to fall back.

    ``resolve_token`` rejects an empty ``--token-file`` because an unset Helm
    value renders as ``--token-file=`` rather than omitting the flag. The same
    rendering applies here, and falling back to the environment on an empty flag
    would point the worker at whatever the host exports — silently, which is the
    failure this whole change exists to prevent.
    """
    with pytest.raises(ValueError, match="empty URL"):
        _resolve(["--mode", "single-machine", "--server-url", ""], monkeypatch, ENV_URL)

    # ...and with no environment set either, so it cannot be mistaken for a
    # precedence question.
    with pytest.raises(ValueError, match="empty URL"):
        _resolve(["--mode", "single-machine", "--server-url", "   "], monkeypatch, None)


def test_resolution_is_idempotent(monkeypatch) -> None:
    """Called twice it must not change its own answer — the chain in __main__
    is easy to reorder, and a resolver that rewrites a resolved value would
    make the flag lose to the environment on the second pass."""
    monkeypatch.delenv("BIOENGINE_SERVER_URL", raising=False)
    monkeypatch.setenv("BIOENGINE_SERVER_URL", ENV_URL)
    monkeypatch.setattr(
        sys, "argv", ["bioengine.worker", "--mode", "single-machine", "--server-url", FLAG_URL]
    )
    configs = get_args_by_group(create_parser())
    once = resolve_server_url(configs)
    twice = resolve_server_url(once)
    assert twice["Hypha Options"]["server_url"] == FLAG_URL


# ===== the wiring, which the behavioural tests above cannot see =====


# Runs the REAL entrypoint — ``if __name__ == "__main__":`` and all — with
# BioEngineWorker replaced, so the assertion is on what the worker is actually
# constructed with. ``SystemExit`` is a BaseException, so the entrypoint's own
# ``except Exception`` cannot swallow it and mask a failure as a clean run.
_ENTRYPOINT_PROBE = """
import json, runpy, sys
import bioengine.worker as pkg

class _Stub:
    def __init__(self, **kwargs):
        print("CAPTURED " + json.dumps({"server_url": kwargs.get("server_url", "<ABSENT>")}))
        raise SystemExit(0)

pkg.BioEngineWorker = _Stub
sys.argv = ["bioengine.worker", *sys.argv[1:]]
runpy.run_module("bioengine.worker", run_name="__main__")
"""


def _server_url_at_the_boundary(argv, env_url):
    """What BioEngineWorker is really handed, running the real entrypoint."""
    env = dict(os.environ)
    env.pop("BIOENGINE_SERVER_URL", None)
    env.pop("HYPHA_TOKEN", None)
    if env_url is not None:
        env["BIOENGINE_SERVER_URL"] = env_url
    env["PYTHONPATH"] = str(CHECKOUT_ROOT)

    result = subprocess.run(
        [sys.executable, "-c", _ENTRYPOINT_PROBE, *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(CHECKOUT_ROOT),
        timeout=180,
    )
    captured = [
        line for line in result.stdout.splitlines() if line.startswith("CAPTURED ")
    ]
    assert captured, f"entrypoint never reached BioEngineWorker:\n{result.stdout}\n{result.stderr}"
    return json.loads(captured[-1][len("CAPTURED ") :])["server_url"]


@pytest.mark.parametrize(
    "argv, env_url, expected",
    [
        (["--mode", "single-machine"], ENV_URL, ENV_URL),
        (["--mode", "single-machine", "--server-url", FLAG_URL], ENV_URL, FLAG_URL),
        (["--mode", "single-machine"], None, "<ABSENT>"),
        (["--mode", "single-machine"], "", "<ABSENT>"),
    ],
    ids=["env-only", "flag-beats-env", "neither", "empty-env-is-unset"],
)
def test_the_entrypoint_hands_the_worker_the_right_server(argv, env_url, expected) -> None:
    """The wiring, asserted at the boundary rather than by reading source.

    The original bug was a declared option nothing read, and a resolver that is
    never *called* reproduces it exactly. Deleting the call from the entrypoint
    makes the first case below return ``<ABSENT>`` — so this kills that mutation
    by observing behaviour, where a source-level string match would pass on any
    textual hit and false-fail on a harmless rename.

    Credit to the review for this technique; my first version grepped the file.
    """
    assert _server_url_at_the_boundary(argv, env_url) == expected


# ===== the documentation the operator reads =====


def test_the_help_text_names_the_variable() -> None:
    """The variable was documented in the user-facing skill as the knob while
    the worker ignored it. Whatever else is true, ``--help`` must agree with
    the code about where the value can come from."""
    parser = create_parser()
    action = next(a for a in parser._actions if "--server-url" in a.option_strings)
    assert "BIOENGINE_SERVER_URL" in (action.help or "")
