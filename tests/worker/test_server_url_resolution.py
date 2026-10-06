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


def test_the_entrypoint_actually_calls_the_resolver() -> None:
    """The original bug was a declared option nothing read. A resolver that is
    never called reproduces it exactly, and every behavioural test stays green
    — verified by mutation, which is why this assertion exists at all.

    Source-level because the chain sits in ``if __name__ == "__main__":`` and
    cannot be imported. If that block is ever lifted into a function, replace
    this with a call to it.
    """
    source = (CHECKOUT_ROOT / "bioengine" / "worker" / "__main__.py").read_text()
    entrypoint = source.split('if __name__ == "__main__":', 1)
    assert len(entrypoint) == 2, "entrypoint block not found — has it been refactored?"
    body = entrypoint[1]

    assert "resolve_server_url(group_configs)" in body, (
        "the entrypoint does not call resolve_server_url, so BIOENGINE_SERVER_URL "
        "is ignored again exactly as it was before this fix"
    )
    # Order matters: resolving before the args exist would read an empty config.
    assert body.index("get_args_by_group") < body.index("resolve_server_url")


# ===== the documentation the operator reads =====


def test_the_help_text_names_the_variable() -> None:
    """The variable was documented in the user-facing skill as the knob while
    the worker ignored it. Whatever else is true, ``--help`` must agree with
    the code about where the value can come from."""
    parser = create_parser()
    action = next(a for a in parser._actions if "--server-url" in a.option_strings)
    assert "BIOENGINE_SERVER_URL" in (action.help or "")
