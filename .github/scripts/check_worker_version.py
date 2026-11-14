#!/usr/bin/env python3
"""Gate a commit range that touches the worker on its ``pyproject.toml`` version.

Two legs, both run whenever a gated path changed between the two refs:

1. ``version`` in ``pyproject.toml`` is strictly greater (PEP 440) at the head
   ref than at the base ref.
2. ``__version__`` in ``bioengine/_version.py`` equals that version.

Leg 2 runs even when leg 1 is skipped, so a stray edit to ``_version.py`` alone
cannot drift the two files apart.

``version-check.yml`` applies the same two legs to a pull request, comparing
against ``origin/main``. This script exists so the identical comparison can be
made over an arbitrary commit range -- in particular ``github.event.before`` to
``github.event.after`` on a push, where there is no pull request to hang a
required check on.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

from packaging.version import InvalidVersion, Version

# The paths that drive the published worker image, as version-check.yml matches
# them. docker/worker-ray-overlay.Dockerfile is a user-facing recipe rather than
# part of the canonical image build, so it is deliberately not here.
GATED = re.compile(
    r"^(bioengine/"
    r"|docker/worker\.Dockerfile$"
    r"|docker/datasets\.Dockerfile$"
    r"|requirements[^/]*\.txt$"
    r"|pyproject\.toml$"
    r"|\.dockerignore$)"
)

PYPROJECT_VERSION = re.compile(r"^version\s*=\s*\"(.*?)\"", re.M)
DUNDER_VERSION = re.compile(r"^__version__\s*=\s*[\"'](.*?)[\"']", re.M)


class GateError(Exception):
    """A read failed, so the gate reached no verdict and must not pass."""


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise GateError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def show(ref: str, path: str) -> str | None:
    """``path`` at ``ref``, or None if it does not exist there."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    return result.stdout


def pyproject_version(ref: str) -> str | None:
    blob = show(ref, "pyproject.toml")
    if blob is None:
        return None
    match = PYPROJECT_VERSION.search(blob)
    if match is None:
        raise GateError(f"no 'version' field in pyproject.toml at {ref}")
    return match.group(1)


def dunder_version(ref: str) -> str | None:
    blob = show(ref, "bioengine/_version.py")
    if blob is None:
        return None
    match = DUNDER_VERSION.search(blob)
    if match is None:
        raise GateError(f"no '__version__' in bioengine/_version.py at {ref}")
    return match.group(1)


def check_increase(base_ref: str, head_ref: str) -> list[str]:
    new_version = pyproject_version(head_ref)
    if new_version is None:
        raise GateError(f"pyproject.toml is missing at {head_ref}")
    old_version = pyproject_version(base_ref)
    if old_version is None:
        print(f"ℹ️  pyproject.toml is new at {head_ref}; skipping the increase check")
        return []

    try:
        increased = Version(new_version) > Version(old_version)
    except InvalidVersion as error:
        raise GateError(f"pyproject.toml: {error}") from error

    if increased:
        print(f"✅ worker: version increased {old_version} -> {new_version}")
        return []
    return [
        f"pyproject.toml version ({new_version}) is not greater than {base_ref}'s "
        f"({old_version}). Bump pyproject.toml and bioengine/_version.py together."
    ]


def check_lockstep(head_ref: str) -> list[str]:
    """``_version.py`` must match ``pyproject.toml`` at the head ref.

    Ray Serve replicas import ``bioengine`` through ``runtime_env.py_modules``,
    which ships module source but no ``.dist-info``, so ``_version.py`` is the
    only version string a replica can read.
    """
    project_version = pyproject_version(head_ref)
    module_version = dunder_version(head_ref)
    if project_version is None:
        raise GateError(f"pyproject.toml is missing at {head_ref}")
    if module_version is None:
        raise GateError(f"bioengine/_version.py is missing at {head_ref}")
    if module_version != project_version:
        return [
            f"bioengine/_version.py __version__ ({module_version}) does not match "
            f"pyproject.toml version ({project_version}). Bump both together."
        ]
    print(f"✅ worker: bioengine/_version.py in sync with pyproject.toml "
          f"({project_version})")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True, help="range start, for the diff")
    parser.add_argument("--head-ref", required=True, help="range end, and the ref read")
    args = parser.parse_args()

    try:
        print(f"Comparing {args.base_ref}...{args.head_ref}")
        changed = git(
            "diff", "--name-only", args.base_ref, args.head_ref
        ).splitlines()
        print("Changed files:")
        for path in changed:
            print(f" - {path}")

        # Leg 2 runs regardless, so a lone edit to _version.py is still caught.
        failures = check_lockstep(args.head_ref)

        if any(GATED.match(path) for path in changed):
            failures += check_increase(args.base_ref, args.head_ref)
        else:
            print("✅ No gated worker file changed. Skipping the increase check.")
    except GateError as error:
        print(f"❌ {error}")
        return 1

    if failures:
        print()
        for failure in failures:
            print(f"❌ {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
