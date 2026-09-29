"""The Hypha token must not reach the worker process's own argv.

``/proc/<pid>/cmdline`` is mode 0444 while ``/proc/<pid>/environ`` is 0400, so
a token given with ``--token`` is readable by every uid that can run a process
beside the worker — and any routine "what is this running with?" diagnostic
copies it into its own output. ``--token-file`` and ``HYPHA_TOKEN`` must keep
the value out of the process table entirely.

The assertions below read the launched process's real ``/proc/<pid>/cmdline``
rather than checking which parameter was used, so a future code path that put
the value back into argv would fail them regardless of how it got there.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bioengine.worker.__main__ import resolve_token

# Placeholders, not credentials. Distinct per source so precedence is provable.
FILE_TOKEN = "PLACEHOLDER-FROM-FILE-NOT-A-CREDENTIAL"
FLAG_TOKEN = "PLACEHOLDER-FROM-FLAG-NOT-A-CREDENTIAL"
ENV_TOKEN = "PLACEHOLDER-FROM-ENV-NOT-A-CREDENTIAL"

# Resolves the token exactly as ``python -m bioengine.worker`` does, publishes
# the result, then stays alive so the parent can read its /proc entry.
_PROBE = """
import json, os, sys, time
from bioengine.worker.__main__ import create_parser, get_args_by_group, resolve_token

configs = resolve_token(get_args_by_group(create_parser()))
with open(os.environ["PROBE_OUT"], "w") as fh:
    json.dump(configs["Hypha Options"], fh)
time.sleep(120)
"""


def _launch(tmp_path, worker_args, env_token=None):
    """Run the probe as a real process; return (resolved Hypha Options, its cmdline)."""
    probe_out = tmp_path / "resolved.json"
    env = {k: v for k, v in os.environ.items() if k != "HYPHA_TOKEN"}
    env["PROBE_OUT"] = str(probe_out)
    if env_token is not None:
        env["HYPHA_TOKEN"] = env_token

    process = subprocess.Popen(
        [sys.executable, "-c", _PROBE, "--mode", "single-machine", *worker_args],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 60
        while not probe_out.exists():
            if process.poll() is not None:
                pytest.fail(f"probe exited early with code {process.returncode}")
            if time.time() > deadline:
                pytest.fail("probe did not resolve a token within 60s")
            time.sleep(0.05)

        cmdline = Path(f"/proc/{process.pid}/cmdline").read_bytes().decode()
    finally:
        process.terminate()
        process.wait(timeout=30)

    return json.loads(probe_out.read_text()), cmdline


def _token_file(tmp_path, value):
    path = tmp_path / "hypha-token"
    path.write_text(value + "\n")
    return path


# ── What the launched process exposes ────────────────────────────────────────


def test_token_file_value_is_absent_from_cmdline(tmp_path):
    path = _token_file(tmp_path, FILE_TOKEN)

    resolved, cmdline = _launch(tmp_path, ["--token-file", str(path)])

    # Positive control: this really is the probe's command line.
    assert "--token-file" in cmdline
    assert str(path) in cmdline

    assert FILE_TOKEN not in cmdline
    assert resolved["token"] == FILE_TOKEN


def test_env_token_value_is_absent_from_cmdline(tmp_path):
    resolved, cmdline = _launch(tmp_path, [], env_token=ENV_TOKEN)

    assert "--mode" in cmdline  # positive control
    assert ENV_TOKEN not in cmdline
    assert resolved["token"] == ENV_TOKEN


def test_token_flag_value_is_visible_in_cmdline(tmp_path):
    """The exposure --token warns about, pinned so the warning cannot go stale."""
    _, cmdline = _launch(tmp_path, ["--token", FLAG_TOKEN])

    assert FLAG_TOKEN in cmdline


# ── Precedence: --token-file > --token > HYPHA_TOKEN ─────────────────────────


def _resolve(hypha_options):
    return resolve_token({"Hypha Options": dict(hypha_options)})["Hypha Options"]


def test_token_file_beats_both_other_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPHA_TOKEN", ENV_TOKEN)
    path = _token_file(tmp_path, FILE_TOKEN)

    resolved = _resolve({"token_file": str(path), "token": FLAG_TOKEN})

    assert resolved["token"] == FILE_TOKEN


def test_token_flag_beats_the_environment(monkeypatch):
    monkeypatch.setenv("HYPHA_TOKEN", ENV_TOKEN)

    assert _resolve({"token": FLAG_TOKEN})["token"] == FLAG_TOKEN


def test_environment_is_used_when_no_flag_is_given(monkeypatch):
    monkeypatch.setenv("HYPHA_TOKEN", ENV_TOKEN)

    assert _resolve({})["token"] == ENV_TOKEN


def test_no_source_leaves_no_token(monkeypatch):
    """BioEngineWorker falls through to interactive login only when nothing is set."""
    monkeypatch.delenv("HYPHA_TOKEN", raising=False)

    assert "token" not in _resolve({})


def test_token_file_is_not_forwarded_to_the_worker(tmp_path, monkeypatch):
    """BioEngineWorker takes no token_file argument, so it must not survive."""
    monkeypatch.delenv("HYPHA_TOKEN", raising=False)
    path = _token_file(tmp_path, FILE_TOKEN)

    assert "token_file" not in _resolve({"token_file": str(path)})


# ── Failure modes and the --token warning ────────────────────────────────────


def test_missing_token_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Could not read --token-file"):
        _resolve({"token_file": str(tmp_path / "absent")})


def test_empty_token_file_is_rejected(tmp_path):
    path = _token_file(tmp_path, "")

    with pytest.raises(ValueError, match="is empty"):
        _resolve({"token_file": str(path)})


def test_token_flag_warns_about_the_process_table(capsys):
    _resolve({"token": FLAG_TOKEN})

    stderr = capsys.readouterr().err
    assert "/proc/<pid>/cmdline" in stderr
    assert FLAG_TOKEN not in stderr


def test_token_file_warns_that_it_overrides_the_flag(tmp_path, capsys):
    path = _token_file(tmp_path, FILE_TOKEN)

    _resolve({"token_file": str(path), "token": FLAG_TOKEN})

    stderr = capsys.readouterr().err
    assert "--token is ignored" in stderr
    assert FLAG_TOKEN not in stderr
    assert FILE_TOKEN not in stderr


def test_token_file_alone_warns_about_nothing(tmp_path, capsys):
    path = _token_file(tmp_path, FILE_TOKEN)

    _resolve({"token_file": str(path)})

    assert capsys.readouterr().err == ""
