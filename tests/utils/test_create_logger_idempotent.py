"""create_logger must be idempotent per name (#0644, github aicell-lab/bioengine#232).

logging.getLogger(name) returns the same object for a given name, so attaching a handler
unconditionally grew the handler list by one per call -- two with log_file -- and every line was
then emitted once per handler. Measured on origin/main before the fix: five calls left FIVE
handlers, one warning() printed three lines after three calls, and the log_file path leaked one
file descriptor per call (4 -> 9 fds over five calls).

The clauses this file covers, and why each arm has to be here:

  1  N calls leave ONE handler                        -- the report's own reproduction
  2  CONTROL: a fresh logger still gets its handler   -- a fix that just stopped attaching
                                                         handlers would satisfy clause 1 and
                                                         silence the whole package
  3  log_file: handlers bounded AND fds stable        -- removeHandler alone detaches without
                                                         closing, so the descriptor leaks and
                                                         only the handler count looks fixed
  4  a caller's OWN handler survives                  -- the fix must not clear handlers it did
                                                         not attach
  5  a later level still applies                      -- an early-return implementation passes
                                                         1-4 and silently ignores this; worker.py
                                                         passes DEBUG or INFO from --debug
"""
import logging
import os

import pytest

from bioengine.utils import create_logger


def _fresh(name):
    lg = logging.getLogger(name)
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    return lg


def test_repeat_calls_leave_exactly_one_handler():
    _fresh("IdempotentProbe")
    for _ in range(5):
        create_logger("IdempotentProbe")
    assert len(logging.getLogger("IdempotentProbe").handlers) == 1


def test_a_line_is_emitted_once_after_repeat_calls(capsys):
    _fresh("EmitOnceProbe")
    for _ in range(3):
        lg = create_logger("EmitOnceProbe")
    lg.warning("exactly once please")
    emitted = capsys.readouterr().err.count("exactly once please")
    assert emitted == 1, "the line was emitted %d times, one per leaked handler" % emitted


def test_control_a_fresh_logger_still_gets_its_handler_and_emits(capsys):
    """CONTROL: the fix must not be bought by never attaching anything."""
    _fresh("FreshProbe")
    lg = create_logger("FreshProbe")
    assert len(logging.getLogger("FreshProbe").handlers) == 1
    lg.warning("must appear")
    assert capsys.readouterr().err.count("must appear") == 1


def test_log_file_path_is_bounded_and_does_not_leak_descriptors(tmp_path):
    _fresh("FdProbe")
    target = tmp_path / "fd.log"
    create_logger("FdProbe", log_file=str(target))
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(5):
        create_logger("FdProbe", log_file=str(target))
    after = len(os.listdir("/proc/self/fd"))
    assert len(logging.getLogger("FdProbe").handlers) == 2, "console + file, not 2 per call"
    # The load-bearing assertion: removeHandler WITHOUT close() keeps the descriptor open, so
    # the handler count would read fixed while the leak continued.
    assert after == before, "leaked %d descriptors over 5 calls" % (after - before)


def test_a_handler_the_caller_attached_is_not_removed():
    lg = _fresh("CallerHandlerProbe")
    mine = logging.NullHandler()
    lg.addHandler(mine)
    create_logger("CallerHandlerProbe")
    assert mine in logging.getLogger("CallerHandlerProbe").handlers, (
        "create_logger removed a handler it did not attach"
    )
    assert len(logging.getLogger("CallerHandlerProbe").handlers) == 2


def test_a_later_call_can_still_change_the_level():
    """An early-return fix passes every arm above and fails this one."""
    _fresh("LevelProbe")
    create_logger("LevelProbe", level=logging.INFO)
    assert logging.getLogger("LevelProbe").level == logging.INFO
    create_logger("LevelProbe", level=logging.DEBUG)
    assert logging.getLogger("LevelProbe").level == logging.DEBUG
