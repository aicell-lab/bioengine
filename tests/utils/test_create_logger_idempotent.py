"""create_logger must be idempotent per name (github aicell-lab/bioengine#232).

logging.getLogger(name) returns the same object for a given name, so attaching a handler
unconditionally grew the handler list by one per call -- two with log_file -- and every line was
then emitted once per handler. Measured on origin/main before the fix: five calls left FIVE
handlers, one warning() printed three lines after three calls, and the log_file path leaked one
file descriptor per call (4 -> 9 fds over five calls).

This file's ARMS, and why each has to be here. These are numbered for reference within this
file and are deliberately NOT the tracker's clause numbers -- they did not correspond, and a
reader who assumed they did would look for the wrong guarantee:

  A  N calls leave ONE handler                        -- the report's own reproduction
  B  CONTROL: a fresh logger still gets its handler   -- a fix that just stopped attaching
                                                         handlers would satisfy arm A and
                                                         silence the whole package
  C  log_file: handlers bounded AND fds stable        -- removeHandler alone detaches without
                                                         closing. Holding the detached handlers
                                                         alive is what makes this arm bite; see
                                                         the comment in the arm itself
  D  a caller's OWN handler survives                  -- the fix must not clear handlers it did
                                                         not attach
  E  a repeat call REPLACES destinations              -- pins the documented contract: a later
                                                         call without log_file closes the file
                                                         handler, so file logging stops
  F  a later level still applies                      -- an early-return implementation passes
                                                         A-E and silently ignores this; worker.py
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
    # Every handler this loop is about to detach is kept ALIVE in `kept`, and that is what makes
    # the assertion below load-bearing rather than decorative.
    #
    # Without it the arm passes against a close()-less implementation and guards nothing:
    # CPython refcounting finalises the detached FileHandler's stream the instant removeHandler
    # drops the last reference, so the descriptor closes even though close() was never called
    # and the count stays flat. Measured on the close()-less mutant: 6 passed, fd delta 0, with
    # only a `ResourceWarning: unclosed file` per call to show for it. Holding a reference
    # suppresses that finalisation, so the missing close() shows up as a real descriptor.
    #
    # The handler COUNT cannot substitute: removeHandler fixes the count whether or not the
    # descriptor is released, which is the whole reason this arm exists separately.
    kept = []
    for _ in range(5):
        kept.extend(logging.getLogger("FdProbe").handlers)
        create_logger("FdProbe", log_file=str(target))
    after = len(os.listdir("/proc/self/fd"))
    assert len(logging.getLogger("FdProbe").handlers) == 2, "console + file, not 2 per call"
    # Load-bearing, and verified to be so: this fails on a mutant whose only change is
    # handler.close() -> pass, and passes on the fix.
    assert after == before, "leaked %d descriptors over 5 calls" % (after - before)
    assert kept, "the guard reference list stayed empty, so nothing was held alive"


def test_a_repeat_call_without_log_file_replaces_the_file_destination(tmp_path):
    """Arm E. Pins the docstring's contract rather than leaving it to be discovered.

    This behaviour is a real change from the pre-fix function, where the file handler simply
    accumulated and so survived a later no-log_file call. Measured both ways on the two trees:
    'file still written after a no-log_file call' was True before the fix and is False after.
    No in-repo caller does this -- every create_logger name is constructed once in production --
    but a downstream caller can, so the change is pinned here and stated in the docstring
    instead of being silent.
    """
    _fresh("ReplaceDestProbe")
    target = tmp_path / "replace.log"
    lg = create_logger("ReplaceDestProbe", log_file=str(target))
    lg.warning("while the file destination is attached")
    assert "while the file destination is attached" in target.read_text(), (
        "the file destination never worked, so this arm cannot test replacing it"
    )

    lg = create_logger("ReplaceDestProbe")
    assert len(logging.getLogger("ReplaceDestProbe").handlers) == 1, (
        "a no-log_file repeat call must leave the console handler alone"
    )
    before = target.read_text()
    lg.warning("after the destination was replaced")
    assert target.read_text() == before, (
        "file logging continued after a repeat call without log_file, so the destination was "
        "carried forward and the result now depends on call history"
    )


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
