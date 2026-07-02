"""Pin the log-level split for the download-token permission failure.

Personal-workspace workers routinely deploy artifacts from other workspaces
(e.g. chiron-platform/chiron-manager, bioimage-io/model-runner). The
pre-introspect generate_token call fails with a Hypha PermissionError in
that case — expected, benign, artifact is still readable anonymously.

Before this fix ``builder.AppBuilder.build`` logged the full stacktrace at
WARNING for every such deploy. Operators read it as "something is broken"
and asked whether the worker was misconfigured (Chiron field report).

Now: permission-mismatch is logged at INFO with a one-line message.
Genuine failures (network, unexpected server errors) stay at WARNING so
they don't get lost in the noise.
"""
from __future__ import annotations

import inspect

from bioengine.apps import builder as builder_module


def test_permission_branch_logs_at_info() -> None:
    src = inspect.getsource(builder_module.AppBuilder.build)
    # Match on the substring the fix uses to detect the expected case.
    assert 'any permission for workspace' in src, (
        "expected the fix to detect the Hypha PermissionError message "
        "and downgrade its log level."
    )
    # The permission-mismatch branch must be INFO, not WARNING.
    idx = src.find('any permission for workspace')
    branch = src[idx:idx + 500]
    assert 'logger.info' in branch, (
        "expected the permission-mismatch branch to log at INFO — "
        "it's the expected outcome for public cross-workspace artifacts, "
        "not a warning."
    )
    # The message must NOT re-include the full exception (which would drag
    # the stacktrace back in). Presence of "exc}" inside the info branch
    # would defeat the purpose.
    info_line = branch.split('logger.info')[1].split(')')[0]
    assert '{exc' not in info_line, (
        "the INFO branch must not interpolate the full exception (that "
        "brings back the stacktrace the operator wanted removed)."
    )


def test_non_permission_failures_still_warn() -> None:
    """Genuine failures (network, unexpected server errors) must remain at
    WARNING with the exception attached so they don't get lost in the noise."""
    src = inspect.getsource(builder_module.AppBuilder.build)
    # There must still be a logger.warning branch that includes the exc.
    assert 'logger.warning' in src
    # The warning branch must still interpolate the exception — the
    # remaining failure modes are unexpected and the operator needs the
    # detail to diagnose.
    warn_idx = src.find('logger.warning')
    warn_line = src[warn_idx:warn_idx + 500]
    assert '{exc' in warn_line, (
        "the WARNING branch must keep interpolating exc so genuine "
        "failures still show enough detail to diagnose."
    )
