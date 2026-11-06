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
  * --summary prints the digest and invokes no transport at all
  * gate 3 asks every workspace, and refuses outright when one cannot be read

`myco` and `svamp` are stubbed throughout, so the verdicts depend on the
fixture and nothing else. They used to be the live ones, which made every
assertion here a hostage of whatever the mesh happened to answer -- and gate 3
reads the mesh, so a workspace that merely went slow could turn the whole suite
red (#0178).

Run it directly; it needs nothing but git and prints a line per assertion.

    scripts/test_sweep_agent_worktrees.py
"""

import json
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

    # 7. gate 4 must NOT fail on the sweep's own output: this worktree is named
    # only inside the escalation sink's evidence bundle and the workflow
    # runner's log, both of which live inside the watcher root and quote the
    # whole report back. Without the exclusion, one tick's alert fails gate 4
    # for everything the next tick looks at.
    paths["agent-recorded"] = worktree("agent-recorded", "origin/main")
    write(os.path.join(watchers, "bundles", "20260101T000000-sweep.md"),
          "  [crew] agent-recorded  agent-recorded  0000000\n      all four gates pass\n")
    write(os.path.join(watchers, ".runs", "worktree-sweep-report.jsonl"),
          '{"workflow":"worktree-sweep-report","triggerContext":"agent-recorded"}\n')
    write(os.path.join(watchers, ".escalations", "sweep.issue"), "agent-recorded\n")

    # 8. a registration left behind by a directory someone already deleted.
    # It carries unmerged work, to prove a prune does not take the branch with it.
    p = worktree("agent-stale", "origin/main")
    write(os.path.join(p, "stale-only.txt"), "work only this branch has\n")
    git(p, "add", "stale-only.txt")
    git(p, "commit", "--quiet", "-m", "unmerged work on a worktree that got deleted")
    shutil.rmtree(p)
    paths["agent-stale"] = p

    return repo, watchers, paths


WS_A = "ws-sweep-alpha-0a0a0a"
WS_B = "ws-sweep-beta-0b0b0b"


def session_row(directory, sid, active=True, archived=False):
    return {
        "sessionId": sid,
        "directory": directory,
        "active": active,
        "metadata": {"archived": archived, "friendlyName": sid},
    }


def myco_shim(root, tag, *, workspaces=(WS_A, WS_B), sessions=None, fail=()):
    """A stub `myco`/`svamp` pair on PATH, and the env that reaches them.

    Answers `workspaces` from a file and `session list --json` per
    MYCOLAB_WORKSPACE, so a workspace the sweep does not ask for is a workspace
    whose sessions it cannot see -- which is the behaviour under test. Every
    call is logged as `<MYCOLAB_WORKSPACE>|<argv>`, so the log says both what
    was asked and where.
    """
    bindir = os.path.join(root, tag)
    os.makedirs(bindir, exist_ok=True)
    log = os.path.join(bindir, "argv")
    write(os.path.join(bindir, "workspaces.txt"),
          "attached workspaces:\n" + "".join(f"  {w}  (europa)\n" for w in workspaces))
    for ws in workspaces:
        write(os.path.join(bindir, f"sessions-{ws}.json"),
              json.dumps((sessions or {}).get(ws, [])))
    for ws in fail:
        write(os.path.join(bindir, f"fail-{ws}"), "")

    shim = os.path.join(bindir, "myco")
    write(shim, f"""#!/bin/sh
D="{bindir}"
echo "${{MYCOLAB_WORKSPACE:-none}}|$@" >> "{log}"
if [ "$1" = "workspaces" ]; then cat "$D/workspaces.txt"; exit 0; fi
if [ "$1" = "session" ] && [ "$2" = "list" ]; then
  ws="${{MYCOLAB_WORKSPACE:-{workspaces[0] if workspaces else 'none'}}}"
  if [ -f "$D/fail-$ws" ]; then echo "not attached to $ws" >&2; exit 1; fi
  if [ -f "$D/sessions-$ws.json" ]; then cat "$D/sessions-$ws.json"; else echo '[]'; fi
  exit 0
fi
exit 0
""")
    os.chmod(shim, 0o755)
    svamp = os.path.join(bindir, "svamp")
    write(svamp, '#!/bin/sh\necho "[]"\nexit 0\n')
    os.chmod(svamp, 0o755)
    return {**os.environ, "PATH": bindir + os.pathsep + os.environ["PATH"]}, bindir


def calls(bindir):
    """(workspace, argv) for every stub call made, in order."""
    log = os.path.join(bindir, "argv")
    if not os.path.exists(log):
        return []
    return [tuple(l.split("|", 1)) for l in open(log).read().splitlines() if "|" in l]


def sweep(repo, watchers, *extra, env=None):
    p = subprocess.run(
        [sys.executable, SWEEP, "--repo", repo, "--owned-within", "0",
         "--watcher-root", watchers, *extra],
        capture_output=True, text=True, env=env,
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


def shimmed(root, repo, watchers, *extra, tag="shim"):
    """Run the sweep against a stub `myco` and return it plus every call made.

    The sweep must not deliver anything itself (#0171), and the only way to
    show that is to make a transport available and prove it was never reached.
    The stub answers the session reads properly, because one that answers
    nothing leaves gate 3 permanently gapped and every verdict NOT-DONE --
    which a `Removable with` assertion cannot tell apart from a working sweep
    as long as one stale registration is around to satisfy it (#0178).
    """
    env, bindir = myco_shim(root, tag)
    p = subprocess.run(
        [sys.executable, SWEEP, "--repo", repo, "--owned-within", "0",
         "--watcher-root", watchers, *extra],
        capture_output=True, text=True, env=env,
    )
    return p, calls(bindir)


def main():
    with tempfile.TemporaryDirectory(prefix="sweep-control-") as root:
        repo, watchers, paths = build_fixture(root)
        base, base_bin = myco_shim(root, "myco-base")

        print("report (nothing should be removed yet)")
        before = verdicts(sweep(repo, watchers, env=base))
        expected = {
            "agent-clean": "DONE",
            "agent-squashed": "DONE",
            "agent-unmerged": "NOT-DONE",
            "agent-dirty": "NOT-DONE",
            "agent-owned": "NOT-DONE",
            "agent-watched": "NOT-DONE",
            "agent-recorded": "DONE",
            "agent-stale": "STALE",
        }
        for name, want in expected.items():
            check(f"{name} is {want}", before.get(name) == want, f"got {before.get(name)!r}")

        # agent-recorded being DONE above is only meaningful if those three files
        # really would have failed gate 4 — otherwise the exclusion is untested
        # and the pass is free. Copy one of them to a path the exclusion does not
        # cover and the same text must fail the gate.
        decoy = os.path.join(watchers, "sweep-report-copy.md")
        shutil.copyfile(os.path.join(watchers, "bundles", "20260101T000000-sweep.md"), decoy)
        check(
            "the same text OUTSIDE a generated-record dir still fails gate 4",
            verdicts(sweep(repo, watchers, env=base)).get("agent-recorded") == "NOT-DONE",
            "gate 4 ignored a bundle-shaped file sitting beside the watchers",
        )
        os.remove(decoy)
        check(
            "and removing it puts agent-recorded back to DONE",
            verdicts(sweep(repo, watchers, env=base)).get("agent-recorded") == "DONE",
            "the decoy left a residue",
        )
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

        print("gate 3 asks every workspace, not the ambient one (#0178)")
        asked = {ws for ws, a in calls(base_bin) if a.startswith("session list")}
        check(
            "the baseline run asked every workspace myco lists",
            asked == {WS_A, WS_B}, f"asked: {sorted(asked)}",
        )

        # The defect: a session working in the worktree, registered in a
        # workspace that is not the ambient one. Gate 3 has to find it.
        owner = {WS_B: [session_row(paths["agent-clean"], "owner-in-second-ws")]}
        env, _ = myco_shim(root, "ws-owner", sessions=owner)
        report = sweep(repo, watchers, env=env)
        check(
            "a session in the SECOND workspace holds agent-clean",
            verdicts(report).get("agent-clean") == "NOT-DONE",
            f"got {verdicts(report).get('agent-clean')!r}",
        )
        check(
            "and the report names that session as the owner",
            "owner-in-second-ws" in report and "works here" in report,
            [l for l in report.splitlines() if "agent-clean" in l],
        )
        # The discriminator for that pass: with only one workspace in the
        # registry -- the pre-#0178 reach -- the same session is invisible and
        # the same worktree is cleared for removal.
        env, _ = myco_shim(root, "ws-owner-unreachable", workspaces=(WS_A,), sessions=owner)
        check(
            "the same owner is invisible when only one workspace is read",
            verdicts(sweep(repo, watchers, env=env)).get("agent-clean") == "DONE",
            "something other than the second workspace is sparing agent-clean",
        )
        # ...and gate 3 must still be capable of clearing: an archived session
        # in the same place does not block, so the control above is the session
        # being found and not merely a session row existing.
        env, _ = myco_shim(root, "ws-owner-archived", sessions={
            WS_B: [session_row(paths["agent-clean"], "archived-in-second-ws", archived=True)]})
        check(
            "an ARCHIVED session in the second workspace does not block",
            verdicts(sweep(repo, watchers, env=env)).get("agent-clean") == "DONE",
            "gate 3 now blocks on any session row, archived or not",
        )

        # A workspace that cannot be read is the case that must not be guessed:
        # it is indistinguishable from one holding the owner.
        env, _ = myco_shim(root, "ws-unreadable", fail=(WS_B,))
        report = sweep(repo, watchers, env=env)
        v = verdicts(report)
        check(
            "an unreadable workspace makes every worktree NOT-DONE",
            "DONE" not in set(v.values()), f"{v}",
        )
        check(
            "and the report says gate 3 could not be established",
            "could not be established" in report and WS_B in report,
            [l for l in report.splitlines() if "ownership" in l][:2],
        )
        check(
            "a stale registration is still reported when a workspace is unreadable",
            v.get("agent-stale") == "STALE", f"got {v.get('agent-stale')!r}",
        )
        p, _ = shimmed(root, repo, watchers, "--summary", tag="shim-gap")
        check(
            "the healthy --summary does NOT carry the gate 3 warning",
            "could not be established" not in p.stdout, p.stdout[:300],
        )
        env, _ = myco_shim(root, "ws-unreadable-summary", fail=(WS_B,))
        gap = subprocess.run(
            [sys.executable, SWEEP, "--repo", repo, "--owned-within", "0",
             "--watcher-root", watchers, "--summary"],
            capture_output=True, text=True, env=env,
        )
        check(
            "--summary states the gap rather than reporting a clean tick",
            "GATE 3 COULD NOT BE ESTABLISHED" in gap.stdout, gap.stdout[:300],
        )
        check(
            "--summary with a gap offers no worktree for removal",
            "all four gates pass" not in gap.stdout, gap.stdout[:300],
        )

        print("summary (the digest is data on stdout, and nothing is delivered)")
        p, made = shimmed(root, repo, watchers, "--summary", tag="shim-summary")
        check("--summary exits 0 and prints the digest",
              p.returncode == 0 and "worktrees against origin/main" in p.stdout,
              f"exit {p.returncode}: {p.stdout.strip()[:200]} {p.stderr.strip()[:200]}")
        check("--summary prints the compact form, not the long report",
              "== WORKTREES DONE" not in p.stdout, p.stdout[:200])
        check("--summary names the retirable worktrees it found",
              "Removable with" in p.stdout, p.stdout[:200])
        # The point of the row: a transport was on PATH and the sweep never used it.
        check("--summary calls no transport at all",
              not [a for _, a in made if a.startswith("session send")],
              f"myco was called with: {made}")
        check("--summary removes nothing",
              all(os.path.isdir(x) for n, x in paths.items() if n != "agent-stale"))
        # A control for the control: the same shim DOES record a send when one is made,
        # so the empty result above is a measurement and not a shim that records nothing.
        subprocess.run([os.path.join(root, "shim-summary", "myco"), "session", "send", "x"],
                       capture_output=True)
        logged = open(os.path.join(root, "shim-summary", "argv")).read()
        check("the shim would have recorded a send if one had been made",
              "session send x" in logged, f"recorded: {logged!r}")
        p, _ = shimmed(root, repo, watchers, "--summary", "--apply", tag="shim-apply")
        check("--summary --apply is refused, so a reporter can never remove",
              p.returncode == 2 and "mutually exclusive" in p.stderr,
              f"exit {p.returncode}: {p.stderr.strip()[:200]}")
        p, _ = shimmed(root, repo, watchers, "--summary", "--json", tag="shim-json")
        check("--summary --json is refused rather than silently picking one",
              p.returncode == 2, f"exit {p.returncode}: {p.stderr.strip()[:200]}")
        p, _ = shimmed(root, repo, watchers, "--notify", "no-such-session", tag="shim-notify")
        check("the removed --notify fails loudly instead of claiming a delivery",
              p.returncode == 2 and "--notify" in p.stderr,
              f"exit {p.returncode}: {p.stderr.strip()[:200]}")

        print("apply (only the three that pass every gate should go)")
        sweep(repo, watchers, "--apply", env=base)
        live = git(repo, "worktree", "list")
        branches = git(repo, "branch", "--format=%(refname:short)").split()

        for name in ("agent-clean", "agent-squashed", "agent-recorded"):
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
        after = verdicts(sweep(repo, watchers, env=base))
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
