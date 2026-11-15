"""The destructor must not print a second traceback after a failed startup (#0584).

The defect: __init__ raises before `self.start_time` is assigned, CPython still calls __del__ on
the half-constructed object, and the bare `self.start_time` raises AttributeError. Python prints
`Exception ignored in:` plus a traceback AFTER the real error, so the misleading text is the LAST
thing a reader sees.

Both directions are pinned here on purpose. A fix that simply deleted the warning would satisfy
the silence test alone, so `test_warning_still_fires...` is the control that keeps the guard
honest: it asserts the warning STILL happens for an object that really is being collected
half-initialised, which is the thing __del__ exists to say.
"""
import logging

from bioengine.worker.worker import BioEngineWorker


def _del_output(obj):
    """Run __del__ directly and return whatever it raised, or None.

    Called explicitly rather than via `del obj` + gc: an exception inside __del__ is SWALLOWED by
    the interpreter (printed to stderr as "Exception ignored in"), so a test that relied on
    garbage collection could not see it fail and would pass against the unfixed code.
    """
    try:
        obj.__del__()
    except BaseException as exc:  # noqa: BLE001 - the defect IS an exception escaping here
        return exc
    return None


def test_del_is_silent_when_init_failed_before_start_time():
    """A worker whose __init__ died early must not raise out of __del__."""
    worker = BioEngineWorker.__new__(BioEngineWorker)  # never ran __init__, so no attributes
    assert not hasattr(worker, "start_time")
    assert _del_output(worker) is None


def test_del_is_silent_when_logger_is_also_missing():
    """start_time set but logger missing must not raise either.

    An earlier shape of this fix guarded only start_time, which moved the AttributeError from one
    attribute to the next instead of removing it.
    """
    worker = BioEngineWorker.__new__(BioEngineWorker)
    worker.start_time = 12345.0
    assert not hasattr(worker, "logger")
    assert _del_output(worker) is None


def test_warning_still_fires_for_a_half_initialised_worker(caplog):
    """CONTROL: the guard must not turn the real warning into silence."""
    worker = BioEngineWorker.__new__(BioEngineWorker)
    worker.start_time = 12345.0
    worker.logger = logging.getLogger("test_0584_control")

    with caplog.at_level(logging.WARNING, logger="test_0584_control"):
        assert _del_output(worker) is None

    assert any(
        "garbage collected with partially/completely initialized components" in r.message
        for r in caplog.records
    ), "the destructor's warning disappeared -- the guard skipped real work"
