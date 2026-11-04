"""The example in each shell-string flag's help must actually parse.

``--further-slurm-args`` and ``--further-apptainer-args`` take one value that
normally starts with a dash. argparse only accepts such a value when it also
contains a space, so a two-directive example parses while the single-directive
form an operator needs for ``--reservation=safe`` does not. The help text used
to show the space-separated shape, which works by accident and silently stops
working as soon as the value has one directive in it.

These tests take the example out of the help string, split it the way a shell
would, and feed it back to the parser — so the help cannot drift back to a
shape that does not run.
"""

import shlex

import pytest

from bioengine.worker.__main__ import create_parser

SHELL_STRING_FLAGS = ("--further-slurm-args", "--further-apptainer-args")


def _action_for(flag):
    parser = create_parser()
    for action in parser._actions:
        if flag in action.option_strings:
            return action
    raise AssertionError(f"{flag} is not declared on the parser")


def _example_tokens(flag):
    """The shell tokens the flag's own help tells an operator to type."""
    help_text = _action_for(flag).help
    marker = "Example: "
    assert marker in help_text, f"{flag} help carries no example to check"
    example = help_text.split(marker, 1)[1].strip().rstrip(".")
    tokens = shlex.split(example)
    assert tokens, f"{flag} example is empty"
    return tokens


@pytest.mark.parametrize("flag", SHELL_STRING_FLAGS)
def test_help_example_parses(flag):
    tokens = _example_tokens(flag)
    args = create_parser().parse_args(["--mode", "slurm", *tokens])
    value = getattr(args, flag.lstrip("-").replace("-", "_"))
    assert value, f"{flag} example parsed but set no value"


@pytest.mark.parametrize("flag", SHELL_STRING_FLAGS)
def test_help_example_survives_a_single_directive(flag):
    """The example must not depend on carrying two directives.

    Truncating the value to its first directive is what an operator does when
    they only need one. It is the case the old examples failed.
    """
    tokens = _example_tokens(flag)
    assert tokens[0].startswith(f"{flag}="), (
        f"{flag} example must attach its value with '=' so a single "
        f"directive parses too; got {tokens[0]!r}"
    )
    first_directive = tokens[0].split("=", 1)[1].split()[0]
    args = create_parser().parse_args(
        ["--mode", "slurm", f"{flag}={first_directive}"]
    )
    value = getattr(args, flag.lstrip("-").replace("-", "_"))
    assert value == first_directive


@pytest.mark.parametrize("flag", SHELL_STRING_FLAGS)
def test_space_separated_single_directive_is_still_rejected(flag):
    """Discrimination control for the two tests above.

    If argparse ever accepted a lone dash-leading value, those tests would
    pass no matter what the help said. This fails the moment that changes,
    which is the signal to revisit the '=' advice rather than trust it.
    """
    with pytest.raises(SystemExit):
        create_parser().parse_args(
            ["--mode", "slurm", flag, "--reservation=safe"]
        )
