"""``error_exit`` must not bury the cause under a remote traceback (myco #0446)."""
import traceback

import pytest

from bioengine.cli.utils import condense_traceback, error_exit

MARKER = "Traceback (most recent call last):"
CAUSE = (
    "Cannot upload 'bioimage-io/probe' at version '0.1.0': 0.1.0 already "
    "exists; bump the version in manifest.yaml, or choose a different id instead"
)


def _wire_payload(message: str = CAUSE) -> str:
    """The string hypha_rpc hands the CLI for a remote failure.

    ``rpc.py`` builds it as ``"RemoteError:" + _rvalue + "\\n" + _rtrace``
    where ``_rvalue`` is already the entire server-side traceback and
    ``_rtrace`` is that same traceback once more — hence two copies of
    everything, every time, before the CLI adds its own prefix.
    """
    try:
        try:
            raise ValueError(message)
        except ValueError as inner:
            raise RuntimeError(f"Failed to create/update artifact: {inner}")
    except RuntimeError as exc:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return "RemoteError:" + trace + "\n" + "Exception: " + trace


def test_wire_payload_really_is_doubled():
    """Guards the fixture: a payload with one traceback proves nothing about the fix."""
    assert _wire_payload().count(MARKER) == 4


def test_frames_are_dropped_and_the_cause_survives():
    out = condense_traceback(f"Upload failed: {_wire_payload()}")
    assert MARKER not in out
    assert 'File "' not in out
    assert out.startswith("Upload failed: ")
    assert out.endswith(CAUSE)
    assert out.count(CAUSE) == 1


def test_a_multi_line_cause_is_kept_whole():
    out = condense_traceback(_wire_payload("first line\nsecond line"))
    assert out.endswith("first line\nsecond line")


def test_a_message_without_a_traceback_is_untouched():
    msg = "Invalid manifest at /x/manifest.yaml:\n\nmissing required field 'entry'"
    assert condense_traceback(msg) == msg


def test_error_exit_writes_the_cause_in_its_last_lines(capsys):
    with pytest.raises(SystemExit):
        error_exit(f"Upload failed: {_wire_payload()}")
    err = capsys.readouterr().err
    assert MARKER not in err
    assert CAUSE in "\n".join(err.splitlines()[-2:])


def test_the_full_traceback_is_still_reachable(capsys, monkeypatch):
    monkeypatch.setenv("BIOENGINE_CLI_TRACEBACK", "1")
    with pytest.raises(SystemExit):
        error_exit(f"Upload failed: {_wire_payload()}")
    assert capsys.readouterr().err.count(MARKER) == 4


def test_a_callers_own_hint_is_not_replaced(capsys):
    with pytest.raises(SystemExit):
        error_exit(f"Upload failed: {_wire_payload()}", hint="run `bioengine apps list`")
    assert "Hint:  run `bioengine apps list`" in capsys.readouterr().err
