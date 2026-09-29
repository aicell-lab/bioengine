"""Replica setup moves the process, and that must not outlive the test.

The restoration test below is the contract. The one above it is a tripwire
over current behaviour and deliberately not a contract -- see its docstring.
"""

import os
from pathlib import Path

import bioengine

_IMPORT_CWD = os.getcwd()


def test_the_process_is_still_moved_after_user_init_returns(tmp_path, monkeypatch):
    """Tripwire over current behaviour, NOT a requirement. Retarget it freely.

    Nothing in this repository reads the working directory after replica
    setup, so this asserts more than anything in-tree needs: it fails for a
    restore inside ``_ensure_working_directory`` and equally for one in a
    ``finally`` in ``wrap_init``, which would hold the directory across the
    user's ``__init__`` and leak nothing. If a change makes this red, decide
    whether the new behaviour is wrong -- do not assume it is.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    @bioengine.app(num_cpus=0)
    class App:
        def __init__(self) -> None:
            pass

    App.func_or_class()

    assert Path(os.getcwd()).resolve() == tmp_path.resolve()


def test_cwd_is_restored_for_the_next_test():
    assert os.getcwd() == _IMPORT_CWD
