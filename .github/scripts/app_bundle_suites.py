#!/usr/bin/env python3
"""Decide which app bundles' in-bundle suites CI runs, and refuse on a surprise.

Each app under ``apps/`` ships its tests inside the bundle published to Hypha,
and ``pytest.ini`` sets ``testpaths = tests``, so ``pytest tests/`` never
collects them (backlog #0247). The apps CI runs cannot simply be ``apps/*``:
some apps' suites need that app's own image, which the CI runner does not have.

So the set is declared in ``.github/app-bundle-suites.json`` and RECONCILED
against what is on disk. An app that ships ``test_*.py`` and is named in
neither table fails the job, rather than having its tests collected by nobody
while every step reports success (backlog #0423). The registry is one file, so
the list CI iterates and the floors it asserts cannot drift apart.

  reconcile   refuse unless the registry and the tree agree
  list        print the apps to run, one per line (reconciles first)
  assert-ran  read junit-app-*.xml and refuse below each app's floor

``--root`` points all three at a different tree; the registry is read from
inside it, which is what lets the self-test exercise this against synthetic
layouts without a second way to supply the tables.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

REGISTRY = ".github/app-bundle-suites.json"
APPS_DIR = "apps"

# pytest.ini sets python_files = test_*.py, so *_test.py is NOT collected and
# must not be discovered here either: claiming an app ships tests that pytest
# would not collect would refuse a PR over a file CI could never run.
TEST_GLOB = "test_*.py"


class Refusal(Exception):
    """A condition this gate must never resolve as pass."""


def load_registry(root: str) -> tuple[dict[str, int], dict[str, str]]:
    path = os.path.join(root, REGISTRY)
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise Refusal(f"cannot read {REGISTRY}: {e}")
    run, skip = data.get("run", {}), data.get("skip", {})
    if not isinstance(run, dict) or not isinstance(skip, dict):
        raise Refusal(f"{REGISTRY} must hold two objects, 'run' and 'skip'")
    for app, floor in run.items():
        if not isinstance(floor, int) or floor < 1:
            raise Refusal(f"{REGISTRY}: run['{app}'] must be a floor >= 1, got {floor!r}")
    for app, reason in skip.items():
        if not isinstance(reason, str) or not reason.strip():
            raise Refusal(f"{REGISTRY}: skip['{app}'] must say WHY CI cannot run it")
    return run, skip


def discover(root: str) -> set[str]:
    """Apps under apps/ shipping at least one file pytest would collect."""
    apps_root = os.path.join(root, APPS_DIR)
    if not os.path.isdir(apps_root):
        raise Refusal(f"no {APPS_DIR}/ directory under {root}")
    found = set()
    for app in sorted(os.listdir(apps_root)):
        if not os.path.isdir(os.path.join(apps_root, app)):
            continue
        pattern = os.path.join(apps_root, app, "**", TEST_GLOB)
        if glob.glob(pattern, recursive=True):
            found.add(app)
    return found


def reconcile(root: str) -> list[str]:
    run, skip = load_registry(root)
    declared, found = set(run) | set(skip), discover(root)

    both = sorted(set(run) & set(skip))
    if both:
        raise Refusal(
            f"{', '.join(both)} appear(s) in both 'run' and 'skip' in {REGISTRY} — "
            "an app is either run with a floor or skipped with a reason, not both"
        )

    undeclared = sorted(found - declared)
    if undeclared:
        raise Refusal(
            f"{', '.join(undeclared)} ship(s) in-bundle {TEST_GLOB} that no CI step "
            f"runs. Add each to {REGISTRY}: to 'run' with a floor on the number of "
            "tests that must execute (measure it), or to 'skip' with the reason CI "
            "cannot run it. This refusal is the point of backlog #0423 — without it "
            "those tests are collected by nobody and the job still reports success."
        )

    stale = sorted(declared - found)
    if stale:
        raise Refusal(
            f"{REGISTRY} names {', '.join(stale)}, which ship(s) no in-bundle "
            f"{TEST_GLOB} — drop the entry, or the registry attests to a suite that "
            "does not exist"
        )

    print(
        f"OK: {len(found)} app(s) ship in-bundle tests; "
        f"run={sorted(run) or '-'} skip={sorted(skip) or '-'}"
    )
    for app, reason in sorted(skip.items()):
        print(f"  not run: {app} — {reason}")
    return sorted(run)


def assert_ran(root: str, reports_dir: str) -> None:
    run, _ = load_registry(root)
    reports = sorted(glob.glob(os.path.join(reports_dir, "junit-app-*.xml")))
    if len(reports) != len(run):
        raise Refusal(
            f"{len(reports)} app report(s), expected {len(run)} "
            f"({', '.join(sorted(run))})"
        )
    bad = []
    for path in reports:
        app = os.path.basename(path)[len("junit-app-") : -len(".xml")]
        try:
            root_el = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as e:
            bad.append(f"{app}: no readable JUnit report, so nothing attests a run: {e}")
            continue
        suites = root_el.findall("testsuite") or [root_el]
        total = sum(int(s.get("tests", 0)) for s in suites)
        skipped = sum(int(s.get("skipped", 0)) for s in suites)
        failed = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
        ran = total - skipped
        print(
            f"{app}: collected={total} skipped={skipped} ran={ran} "
            f"failures+errors={failed}"
        )
        if app not in run:
            bad.append(f"{app}: no floor recorded in {REGISTRY}")
            continue
        if ran < run[app]:
            bad.append(f"{app}: {ran} test(s) ran, below the floor of {run[app]}")
        if failed:
            bad.append(f"{app}: {failed} test(s) failed or errored")
    if bad:
        raise Refusal("\n".join(bad))
    print(f"OK: {len(reports)} app suite(s) ran at or above their floors")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("reconcile", "list", "assert-ran")
    )
    parser.add_argument(
        "--root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
    )
    parser.add_argument("--reports-dir", default=".")
    args = parser.parse_args()
    root = os.path.normpath(args.root)

    try:
        if args.command == "list":
            for app in reconcile_quiet(root):
                print(app)
        elif args.command == "reconcile":
            reconcile(root)
        else:
            assert_ran(root, args.reports_dir)
    except Refusal as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 1
    return 0


def reconcile_quiet(root: str) -> list[str]:
    """``list`` reconciles too, so the loop cannot run off a stale registry."""
    stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        return reconcile(root)
    finally:
        sys.stdout = stdout


if __name__ == "__main__":
    sys.exit(main())
