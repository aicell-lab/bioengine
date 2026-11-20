"""A directory option that cannot be created must name the option.

The option list is DERIVED FROM THE PARSER rather than typed here, per the row's clause 1: a
hand-written list silently stops covering an option the moment somebody adds one, and the test
would keep passing. `test_the_parsers_dir_option_set_has_not_changed` is what turns that
derivation into a tripwire.
"""
import argparse

import pytest

from bioengine.worker.__main__ import create_option_dirs, create_parser

# Only --workspace-dir is created by THIS process. The other two name paths at the far end of a
# cluster -- see create_option_dirs' docstring for the measurement. Kept as data so the tripwire
# below can compare the parser against it.
LOCAL_DIR_OPTIONS = {"--workspace-dir"}
REMOTE_DIR_OPTIONS = {"--ray-workspace-dir", "--worker-workspace-dir"}


def _parser_dir_options():
    """Every --*-dir option the argparse surface accepts, read off the parser itself."""
    found = set()
    for action in create_parser()._actions:
        for opt in action.option_strings:
            if opt.endswith("-dir"):
                found.add(opt)
    return found


def test_the_parsers_dir_option_set_has_not_changed():
    """TRIPWIRE: a new --*-dir option must be classified local or remote deliberately.

    This is the clause-1 guarantee. If someone adds a directory option and does not decide which
    side of the boundary it falls on, this reddens instead of the suite quietly covering one
    option fewer.
    """
    assert _parser_dir_options() == LOCAL_DIR_OPTIONS | REMOTE_DIR_OPTIONS


def test_uncreatable_workspace_dir_names_the_option():
    with pytest.raises(ValueError) as exc:
        create_option_dirs({"Core Options": {"workspace_dir": "/proc/cannot-create-here/x"}})
    msg = str(exc.value)
    assert "--workspace-dir" in msg, f"the message does not name the option: {msg!r}"
    # The OS reason must survive too -- naming the option is an ADDITION, not a replacement.
    assert "Not a directory" in msg or "No such file" in msg or "denied" in msg, msg


def test_creatable_workspace_dir_is_created_and_does_not_raise(tmp_path):
    """CONTROL (clause 3): valid input still works, and the directory really is made."""
    target = tmp_path / "nested" / "ws"
    assert not target.exists()
    out = create_option_dirs({"Core Options": {"workspace_dir": str(target)}})
    assert target.is_dir()
    assert out["Core Options"]["workspace_dir"] == str(target)


def test_an_existing_workspace_dir_is_accepted(tmp_path):
    """exist_ok: re-running against an already-created directory is not an error."""
    target = tmp_path / "ws"
    target.mkdir()
    create_option_dirs({"Core Options": {"workspace_dir": str(target)}})
    assert target.is_dir()


@pytest.mark.parametrize("key", ["ray_workspace_dir", "worker_workspace_dir"])
def test_remote_dir_options_are_not_created_locally(key):
    """CONTROL: the cluster-side options must NOT be pre-created here.

    They name paths on a filesystem this process may not have -- the remote Ray cluster's, or a
    SLURM compute node's. Pre-creating them would refuse a value that is valid at the far end.
    An uncreatable value must therefore pass through this step untouched.
    """
    cfg = {"Core Options": {"workspace_dir": None, key: "/proc/cannot-create-here/x"}}
    create_option_dirs(cfg)  # must not raise


def test_absent_workspace_dir_is_left_to_the_component_default():
    """get_args_by_group drops None values so component defaults survive; blame nothing then."""
    create_option_dirs({"Core Options": {}})
    create_option_dirs({})
