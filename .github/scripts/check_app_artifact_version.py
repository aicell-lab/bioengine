#!/usr/bin/env python3
"""Gate PRs that touch a published BioEngine app on its manifest version.

For each app in ``GATED_APPS`` whose files the PR touches, require BOTH:

1. ``version`` in ``manifest.yaml`` is strictly greater (PEP 440) than the
   same field on ``main``.
2. That exact version is already present in the published Hypha artifact's
   version list.

Leg 2 is a *membership* test, not "greater than published", and that is the
reverse of the GHCR rule in ``version-check.yml``. The app workflow uploads
the real ``X.Y.Z`` to Hypha while the PR is still a draft and only then marks
it ready, so by gate time the manifest version equals the newest published
one. A "strictly greater than published" rule here would reject every
correctly prepared PR.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request

import yaml
from packaging.version import InvalidVersion, Version

# app directory under apps/ -> published Hypha artifact id
GATED_APPS = {
    "model-runner": "bioimage-io/model-runner",
    "model-finetune": "bioimage-io/model-finetune",
}

# Not read from HYPHA_SERVER_URL: that variable is set to a different server on
# developer machines, and an ambient value silently pointing the gate at a host
# that does not hold these artifacts is how a required check stops checking.
DEFAULT_SERVER_URL = "https://hypha.aicell.io"
HTTP_TIMEOUT = 30


class GateError(Exception):
    """A condition the gate must never resolve as pass."""


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise GateError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def manifest_version(ref: str, app: str) -> str | None:
    """``version`` from the app's manifest at ``ref``, or None if absent there."""
    path = f"apps/{app}/manifest.yaml"
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{path}"], capture_output=True
    )
    if exists.returncode != 0:
        return None
    manifest = yaml.safe_load(git("show", f"{ref}:{path}"))
    version = (manifest or {}).get("version")
    if not version:
        raise GateError(f"no 'version' field in {path} at {ref}")
    return str(version)


def published_versions(artifact_id: str, server_url: str) -> list[str] | None:
    """Versions published for ``artifact_id``, or None if it does not exist.

    Anything other than a 200 or a 404 is an error, never an empty list: a
    timeout or 5xx read as "version absent" would fail the gate toward green
    on the next run, and read as "present" would skip the check entirely.
    """
    server_url = server_url.rstrip("/")
    workspace, alias = artifact_id.split("/", 1)
    url = f"{server_url}/{workspace}/artifacts/{alias}"
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise GateError(f"GET {url} returned HTTP {error.code}") from error
    except (urllib.error.URLError, OSError) as error:
        raise GateError(f"GET {url} failed: {error}") from error

    if status != 200:
        raise GateError(f"GET {url} returned HTTP {status}")
    try:
        payload = json.loads(body)
        versions = [entry["version"] for entry in payload["versions"]]
    except (ValueError, KeyError, TypeError) as error:
        raise GateError(f"GET {url} returned an unreadable payload: {error}") from error
    return versions


def is_published(version: str, published: list[str]) -> bool:
    """Membership, comparing normalised PEP 440 versions where parseable."""
    try:
        wanted = Version(version)
    except InvalidVersion:
        return version in published
    for candidate in published:
        if candidate == version:
            return True
        try:
            if Version(candidate) == wanted:
                return True
        except InvalidVersion:
            continue
    return False


def newest(published: list[str]) -> str:
    """Newest PEP 440 version in the list, for the failure message only."""
    parseable = []
    for candidate in published:
        try:
            parseable.append(Version(candidate))
        except InvalidVersion:
            continue
    return str(max(parseable)) if parseable else "none"


def check_app(
    app: str, artifact_id: str, main_ref: str, head_ref: str, server_url: str
) -> list[str]:
    """Run both legs for one app; return the failure messages (empty = pass)."""
    new_version = manifest_version(head_ref, app)
    if new_version is None:
        raise GateError(f"apps/{app}/manifest.yaml is missing at {head_ref}")
    old_version = manifest_version(main_ref, app)

    failures = []
    if old_version is None:
        print(f"ℹ️  apps/{app} is new on {main_ref}; skipping the increase check")
    else:
        try:
            increased = Version(new_version) > Version(old_version)
        except InvalidVersion as error:
            raise GateError(f"apps/{app}/manifest.yaml: {error}") from error
        if increased:
            print(f"✅ apps/{app}: version increased {old_version} -> {new_version}")
        else:
            failures.append(
                f"apps/{app}/manifest.yaml version ({new_version}) is not greater "
                f"than {main_ref}'s ({old_version}). Bump it."
            )

    published = published_versions(artifact_id, server_url)
    if published is None:
        failures.append(
            f"artifact {artifact_id} does not exist, so apps/{app} version "
            f"{new_version} cannot be published"
        )
    elif is_published(new_version, published):
        print(f"✅ apps/{app}: version {new_version} is published to {artifact_id}")
    else:
        failures.append(
            f"apps/{app} version {new_version} is not published to {artifact_id} "
            f"(newest published: {newest(published)}). Upload it before marking "
            f"this PR ready."
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True, help="PR base, for the file diff")
    parser.add_argument("--head-ref", required=True, help="PR head")
    parser.add_argument(
        "--main-ref",
        default="origin/main",
        help="ref the manifest version must exceed (default: origin/main)",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"Hypha server holding the artifacts (default: {DEFAULT_SERVER_URL})",
    )
    args = parser.parse_args()

    try:
        print(f"Comparing {args.base_ref}...{args.head_ref}")
        changed = git("diff", "--name-only", args.base_ref, args.head_ref).splitlines()
        print("Changed files:")
        for path in changed:
            print(f" - {path}")

        gated = {
            app: artifact_id
            for app, artifact_id in GATED_APPS.items()
            if any(path.startswith(f"apps/{app}/") for path in changed)
        }
        if not gated:
            print("✅ No gated app changed. Skipping app version check.")
            return 0

        failures = []
        for app, artifact_id in sorted(gated.items()):
            print(f"\n--- apps/{app} ({artifact_id}) ---")
            failures += check_app(
                app, artifact_id, args.main_ref, args.head_ref, args.server_url
            )
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
