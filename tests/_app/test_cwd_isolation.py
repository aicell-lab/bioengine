"""Replica setup chdirs the process, and that must not outlive the test.

``_ensure_working_directory`` anchors a replica at its app directory for the
replica's whole life, so the first test below asserts the process really did
move. The second asserts the process is back where this module was imported,
which is the property the rest of the suite depends on: a leaked cwd makes a
later test resolve relative paths -- or a child process resolve ``import
bioengine`` -- against ``$HOME`` instead of the checkout.
"""

import os
from pathlib import Path

import bioengine

_IMPORT_CWD = os.getcwd()


def test_replica_setup_chdirs_the_process(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    @bioengine.app(num_cpus=0)
    class App:
        def __init__(self) -> None:
            pass

    App.func_or_class()

    assert Path(os.getcwd()).resolve() == tmp_path.resolve()


def test_cwd_is_restored_for_the_next_test():
    assert os.getcwd() == _IMPORT_CWD
