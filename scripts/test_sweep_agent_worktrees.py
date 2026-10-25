#!/usr/bin/env python3
"""Positive and discrimination controls for scripts/sweep_agent_worktrees.py.

On the real repository the sweep currently retires nothing, because every
candidate fails at least one gate. That result is indistinguishable from a
sweep that is simply broken, so this builds a scratch repository where each
gate is the *only* thing standing between a worktree and removal, and proves:

  * a worktree that passes all four gates IS removed and its branch deleted
  * breaking exactly one gate flips that worktree to NOT-DONE and spares it
  * a branch holding work absent from origin/main is never deleted
  * a branch squash-landed under a different sha is still recognised as landed
  * a stale registration is pruned without taking its branch or its commits

Run it directly; it needs nothing but git and prints a line per assertion.

    scripts/test_sweep_agent_worktrees.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "sweep_agent_worktrees.py")

FAILURES = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"\n          {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def git(repo, *args, check_rc=True):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check_rc and p.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} in {repo} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def build_fixture(root):
    """A clone whose worktrees differ from each other in exactly one gate."""
    upstream, repo = os.path.join(root, "upstream.git"), os.path.join(root, "clone")
    seed = os.path.join(root, "seed")
    # git 2.25 has no `init -b`, which is what this machine ships.
    subprocess.run(["git", "init", "--quiet", "--bare", upstream], check=True)
    subprocess.run(["git", "init", "--quiet", seed], check=True)
    for r in (upstream, seed):
        git(r, "symbolic-ref", "HEAD", "refs/heads/main")
    git(seed, "config", "user.email", "sweep-test@example.invalid")
    git(seed, "config", "user.name", "sweep test")

    write(os.path.join(seed, "base.txt"), "base\n")
    git(seed, "add", "base.txt")
    git(seed, "commit", "--quiet", "-m", "base")
    base = git(seed, "rev-parse", "HEAD")

    # The commit a later branch will reproduce under a different sha, which is
    # what a GitHub squash-merge looks like from the worktree's side.
    write(os.path.join(seed, "landed.txt"), "landed change\n")
    git(seed, "add", "landed.txt")
    git(seed, "commit", "--quiet", "-m", "the squashed landing of the same change")
    git(seed, "remote", "add", "origin", upstream)
    git(seed, "push", "--quiet", "origin", "main")

    subprocess.run(["git", "clone", "--quiet", upstream, repo], check=True)
    git(repo, "config", "user.email", "sweep-test@example.invalid")
    git(repo, "config", "user.name", "sweep test")

    agents = os.path.join(repo, ".claude", "worktrees")
    os.makedirs(agents)

    def worktree(name, start):
        git(repo, "worktree", "add", "--quiet", "-b", name, os.path.join(agents, name), start)
        return os.path.join(agents, name)

    paths = {}
    # 1. passes every gate: identical to origin/main, clean, unowned, unnamed.
    paths["agent-clean"] = worktree("agent-clean", "origin/main")

    # 2. gate 1 via patch equivalence: a distinct sha carrying the same diff.
    p = worktree("agent-squashed", base)
    write(os.path.join(p, "landed.txt"), "landed change\n")
    git(p, "add", "landed.txt")
    git(p, "commit", "--quiet", "-m", "the pre-squash commit of the same change")
    paths["agent-squashed"] = p

    # 3. gate 1 fails: work that is nowhere in origin/main.
    p = worktree("agent-unmerged", "origin/main")
    write(os.path.join(p, "only-here.txt"), "work that exists nowhere else\n")
    git(p, "add", "only-here.txt")
    git(p, "commit", "--quiet", "-m", "unmerged work")
    paths["agent-unmerged"] = p

    # 4. gate 2 fails: an untracked file.
    p = worktree("agent-dirty", "origin/main")
    write(os.path.join(p, "scratch.tmp"), "uncommitted\n")
    paths["agent-dirty"] = p

    # 5. gate 3 fails: an index lock says an operation is in flight. The lock
    # lives in the linked worktree's real git dir, not under <worktree>/.git.
    p = worktree("agent-owned", "origin/main")
    write(os.path.join(git(p, "rev-parse", "--absolute-git-dir"), "index.lock"), "")
    paths["agent-owned"] = p

    # 6. gate 4 fails: a watcher hardcodes the branch name.
    paths["agent-watched"] = worktree("agent-watched", "origin/main")
    watchers = os.path.join(root, "watchers")
    write(os.path.join(watchers, "check.sh"), "#!/bin/sh\ngit checkout agent-watched\n")

    # 7. a registration left behind by a directory someone already deleted.
    # It carries unmerged work, to prove a prune does not take the branch with it.
    p = worktree("agent-stale", "origin/main")
    write(os.path.join(p, "stale-only.txt"), "work only this branch has\n")
    git(p, "add", "stale-only.txt")
    git(p, "commit", "--quiet", "-m", "unmerged work on a worktree that got deleted")
    shutil.rmtree(p)
    paths["agent-stale"] = p

    return repo, watchers, paths


def sweep(repo, watchers, *extra):
    p = subprocess.run(
        [sys.executable, SWEEP, "--repo", repo, "--owned-within", "0",
         "--watcher-root", watchers, *extra],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise SystemExit(f"sweep exited {p.returncode}:\n{p.stdout}\n{p.stderr}")
    return p.stdout


def verdicts(report):
    """name -> DONE / STALE / NOT-DONE / EXCLUDED, parsed from the rendered sections."""
    out, section = {}, None
    for line in report.splitlines():
        if line.startswith("== WORKTREES "):
            section = line.split()[2]
        elif line.startswith("== SESSIONS "):
            section = None
        elif section and line.startswith("  ["):
            out[line.split("] ", 1)[1].split()[0]] = section
    return out


def notify(root, repo, watchers, fail_first, attempts):
    """Run --notify against a `myco` whose first `fail_first` sends fail.

    Returns the finished process and how many times `session send` was called,
    so a run that succeeded by retrying is distinguishable from one that
    succeeded first time.
    """
    bindir = os.path.join(root, f"shim-{fail_first}-{attempts}")
    os.makedirs(bindir, exist_ok=True)
    calls = os.path.join(bindir, "calls")
    shim = os.path.join(bindir, "myco")
    write(shim, '#!/bin/sh\n'
                'if [ "$1" = session ] && [ "$2" = send ]; then\n'
                '  cat > /dev/null\n'
                f'  n=$(cat "{calls}" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "{calls}"\n'
                f'  [ "$n" -le {fail_first} ] && {{ echo "names no host" >&2; exit 1; }}\n'
                'fi\n'
                'exit 0\n')
    os.chmod(shim, 0o755)
    env = {**os.environ, "PATH": bindir + os.pathsep + os.environ["PATH"],
           "SWEEP_NOTIFY_ATTEMPTS": str(attempts), "SWEEP_NOTIFY_RETRY_SECONDS": "0"}
    p = subprocess.run(
        [sys.executable, SWEEP, "--repo", repo, "--owned-within", "0",
         "--watcher-root", watchers, "--notify", "no-such-session"],
        capture_output=True, text=True, env=env,
    )
    n = int(open(calls).read().strip()) if os.path.exists(calls) else 0
    return p, n


def main():
    with tempfile.TemporaryDirectory(prefix="sweep-control-") as root:
        repo, watchers, paths = build_fixture(root)

        print("report (nothing should be removed yet)")
        before = verdicts(sweep(repo, watchers))
        expected = {
            "agent-clean": "DONE",
            "agent-squashed": "DONE",
            "agent-unmerged": "NOT-DONE",
            "agent-dirty": "NOT-DONE",
            "agent-owned": "NOT-DONE",
            "agent-watched": "NOT-DONE",
            "agent-stale": "STALE",
        }
        for name, want in expected.items():
            check(f"{name} is {want}", before.get(name) == want, f"got {before.get(name)!r}")
        check(
            "the main checkout is excluded from the sweepable bands",
            before.get(os.path.basename(repo)) == "EXCLUDED",
            f"got {before.get(os.path.basename(repo))!r}",
        )
        check(
            "a report run removes nothing",
            all(os.path.isdir(p) for n, p in paths.items() if n != "agent-stale"),
            "a dry report deleted a worktree",
        )

        check(
            "a report run leaves the stale registration in place",
            "agent-stale" in git(repo, "worktree", "list"),
        )

        print("notify (a flaky registry lookup must be retried, a dead one must still fail)")
        p, calls = notify(root, repo, watchers, fail_first=0, attempts=3)
        check("a delivery that works is sent once", p.returncode == 0 and calls == 1,
              f"exit {p.returncode}, {calls} call(s): {p.stderr.strip()}")
        p, calls = notify(root, repo, watchers, fail_first=2, attempts=3)
        check("two transient misses are retried and the report still lands",
              p.returncode == 0 and calls == 3, f"exit {p.returncode}, {calls} call(s)")
        p, calls = notify(root, repo, watchers, fail_first=2, attempts=1)
        check("the same two misses DO fail when only one attempt is allowed",
              p.returncode == 1 and calls == 1, f"exit {p.returncode}, {calls} call(s)")
        p, calls = notify(root, repo, watchers, fail_first=9, attempts=3)
        check("a recipient that never answers still fails the run",
              p.returncode == 1 and calls == 3 and "after 3 attempts" in p.stderr,
              f"exit {p.returncode}, {calls} call(s): {p.stderr.strip()}")

        print("apply (only the two that pass every gate should go)")
        sweep(repo, watchers, "--apply")
        live = git(repo, "worktree", "list")
        branches = git(repo, "branch", "--format=%(refname:short)").split()

        for name in ("agent-clean", "agent-squashed"):
            check(f"{name} worktree removed", not os.path.isdir(paths[name]))
            check(f"{name} branch deleted", name not in branches, f"branches: {branches}")
        for name in ("agent-unmerged", "agent-dirty", "agent-owned", "agent-watched"):
            check(f"{name} worktree survived", os.path.isdir(paths[name]))
            check(f"{name} branch survived", name in branches, f"branches: {branches}")
            check(f"{name} still listed as a worktree", paths[name] in live)

        check("stale registration pruned", "agent-stale" not in live, live)
        check("stale branch NOT deleted by the prune", "agent-stale" in branches, f"branches: {branches}")
        stale_sha = git(repo, "rev-parse", "agent-stale")
        check(
            "the stale branch's unmerged commit survived the prune",
            git(repo, "cat-file", "-t", stale_sha) == "commit",
        )

        unmerged_sha = git(os.path.join(repo, ".claude", "worktrees", "agent-unmerged"), "rev-parse", "HEAD")
        check(
            "the unmerged commit is still reachable after the sweep",
            git(repo, "cat-file", "-t", unmerged_sha) == "commit",
        )

        print("re-run (idempotent: nothing left to remove)")
        after = verdicts(sweep(repo, watchers))
        check(
            "nothing is removable on the second pass",
            not ({"DONE", "STALE"} & set(after.values())), f"{after}",
        )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} control(s) FAILED: " + ", ".join(FAILURES))
        return 1
    print("all controls passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
