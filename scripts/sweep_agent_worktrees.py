#!/usr/bin/env python3
"""Report (and optionally retire) finished agent worktrees and spawned sessions.

Read-only by default. ``--apply`` removes only entries that pass all four
safety gates from BioEngine backlog issue #0049; everything else is listed
with the evidence for its verdict and left alone.

    scripts/sweep_agent_worktrees.py                 # report
    scripts/sweep_agent_worktrees.py --json          # machine-readable report
    scripts/sweep_agent_worktrees.py --apply         # retire the DONE entries
    scripts/sweep_agent_worktrees.py --notify <id>   # report into a session's chat

The four gates, all of which must pass before anything is removed:

  1. no commits absent from a freshly fetched ``origin/main``
  2. no uncommitted or untracked changes
  3. no live session, running process or lock file owns it
  4. no watcher script, workflow or crontab entry names it

A run that removes nothing because every gate failed is a pass, not a failure.
Idle is not done: a hibernating session may be blocked on a human answer, so a
session only counts as retirable once its working directory is already gone.

To run it on a schedule, arm a workflow that reads the script out of
``origin/main`` rather than off a working tree -- a checkout can sit a hundred
commits behind, and a watcher must run the landed code. Say where the repo and
the reporting session are; both are machine-local and neither belongs in here::

    myco workflow add worktree-sweep-report --on schedule --cron '37 7 * * *' \\
      --run 'set -e; r=<your checkout>; d="$HOME/.cache/bioengine-sweep";
             mkdir -p "$d"; git -C "$r" fetch -q origin;
             git -C "$r" show origin/main:scripts/sweep_agent_worktrees.py > "$d/sweep.py";
             timeout 900 python3 "$d/sweep.py" --repo "$r" --notify <your session id>'

--notify is required for a scheduled run: a workflow step's stdout reaches
nobody, so a tick that only printed would report nothing and still look green.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

# Bands that are never removed: the main checkout and the sibling worktrees,
# which are named after open PRs and open issues.
BAND_SWEEPABLE = ("agent", "named", "crew")

# Watcher scripts live outside the repo and their location differs per machine.
# Discovery below finds most of them via the workflow files that invoke them;
# name any that nothing schedules in SWEEP_WATCHER_ROOTS (os.pathsep-separated)
# or with --watcher-root. Gate 4 is only as wide as what this reaches, so the
# report prints how many files were searched.
SWEEP_WATCHER_ROOTS_ENV = "SWEEP_WATCHER_ROOTS"
SCRIPT_REF = re.compile(r"(/[\w./@+-]+\.(?:sh|py|js|ts))")
MAX_WATCHER_FILE_BYTES = 2 * 1024 * 1024
OWNED_IF_TOUCHED_WITHIN = 24 * 3600

# The registry lookup behind `myco session send` is flaky per-process: a session
# that resolves one minute answers "names no host" the next, so one miss is not
# evidence the recipient is gone. Retries do not cover every case, though — a
# session on the legacy node-daemon plane is invisible to the mesh permanently,
# and six sends 15s apart all failed identically. So exhausting the attempts is
# a plumbing fault, not a sweep finding: the report goes to the fallback file
# and the run still succeeds. Only a real gate failure may fail the run.
NOTIFY_ATTEMPTS = int(os.environ.get("SWEEP_NOTIFY_ATTEMPTS", "3"))
NOTIFY_RETRY_SECONDS = float(os.environ.get("SWEEP_NOTIFY_RETRY_SECONDS", "20"))
NOTIFY_FALLBACK_DEFAULT = os.path.join(
    os.path.expanduser("~"), ".cache", "bioengine-worktree-sweep", "last-report.md")


def run(cmd, cwd=None, check=False):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed ({p.returncode}): {p.stderr.strip()}")
    return p


def json_cmd(cmd):
    p = run(cmd)
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout)
    except ValueError:
        return None


# ---------------------------------------------------------------- enumeration


def worktrees(repo):
    out = run(["git", "-C", repo, "worktree", "list", "--porcelain"], check=True).stdout
    entries, cur = [], {}
    for line in out.splitlines():
        if not line.strip():
            if cur:
                entries.append(cur)
                cur = {}
            continue
        key, _, val = line.partition(" ")
        cur[key] = val
    if cur:
        entries.append(cur)

    rows = []
    for e in entries:
        path = e["worktree"]
        branch = e.get("branch", "")
        rows.append(
            {
                "path": path,
                "name": os.path.basename(path),
                "branch": branch[len("refs/heads/") :] if branch.startswith("refs/heads/") else None,
                "head": e.get("HEAD"),
                "band": band_of(path, repo),
            }
        )
    return rows


def band_of(path, repo):
    if os.path.realpath(path) == os.path.realpath(repo):
        return "main"
    if "/.claude/worktrees/" in path:
        return "agent" if os.path.basename(path).startswith("agent-") else "named"
    if "/.dev/worktree/" in path:
        return "crew"
    return "sibling"


def sessions():
    """Every spawned session the two daemons know about, normalised."""
    rows = []
    for s in json_cmd(["myco", "session", "list", "--json"]) or []:
        meta = s.get("metadata") or {}
        rows.append(
            {
                "cli": "myco",
                "id": s.get("sessionId"),
                "directory": s.get("directory") or meta.get("path"),
                "active": bool(s.get("active")),
                "archived": meta.get("archived") is True,
                "title": meta.get("customTitle") or meta.get("friendlyName"),
            }
        )
    for s in json_cmd(["svamp", "session", "list", "--json"]) or []:
        rows.append(
            {
                "cli": "svamp",
                "id": s.get("sessionId") or s.get("handle"),
                "directory": s.get("directory") or s.get("path"),
                "active": bool(s.get("active")),
                "archived": False,
                "title": s.get("name"),
            }
        )
    return rows


# --------------------------------------------------------------------- gates


def fetch_baseline(repo):
    """Gate 1 is only meaningful against a remote fetched just now."""
    p = run(["git", "-C", repo, "fetch", "origin"])
    if p.returncode != 0:
        raise SystemExit(
            "ABORT: `git fetch origin` failed, so origin/main cannot be trusted and no\n"
            "gate may be evaluated against it. Nothing was inspected or removed.\n"
            + p.stderr.strip()
        )
    sha = run(["git", "-C", repo, "rev-parse", "origin/main"], check=True).stdout.strip()
    return sha


def gate_merged(repo, row, baseline):
    """Gate 1 — the worktree holds nothing absent from origin/main.

    Ancestry alone is too strict: `main` squash-merges pull requests, so landed
    work reappears there under a different sha. `git cherry` compares patch ids,
    which clears a branch whose change is already in origin/main under another
    commit. It never clears a change that is not there at all.
    """
    refs = [row["branch"] or row["head"]]
    unmerged = []
    for ref in refs:
        p = run(["git", "-C", repo, "log", "--oneline", f"{baseline}..{ref}"])
        if p.returncode != 0:
            return False, f"cannot compare {ref} against origin/main: {p.stderr.strip()}"
        ahead = [l for l in p.stdout.splitlines() if l.strip()]
        if not ahead:
            continue
        c = run(["git", "-C", repo, "cherry", baseline, ref])
        if c.returncode != 0 or any(l.startswith("+") for l in c.stdout.splitlines()):
            unmerged += ahead
    if unmerged:
        uniq = sorted(set(unmerged))
        return False, (
            f"{len(uniq)} commit(s) absent from origin/main: "
            + "; ".join(uniq[:4])
            + f" [{origin_reachability(repo, refs)}]"
        )
    return True, "contained in origin/main (by ancestry or identical patch)"


def origin_reachability(repo, refs):
    """Evidence, not a gate: would deleting this branch destroy the commits?

    A tip still reachable from some origin/* ref survives local deletion; a tip
    reachable from none is only here. Gate 1 fails either way — this only tells
    a reader how expensive getting it wrong would be.
    """
    for ref in refs:
        p = run(["git", "-C", repo, "branch", "-r", "--contains", ref])
        remotes = [l.strip() for l in p.stdout.splitlines() if l.strip()]
        if remotes:
            return "tip still on " + remotes[0]
    return "tip is on NO origin ref — local only"


def gate_clean(row):
    """Gate 2 — no uncommitted or untracked changes."""
    if not os.path.isdir(row["path"]):
        return False, "worktree directory is missing from disk"
    p = run(["git", "-C", row["path"], "status", "--porcelain", "--untracked-files=all"])
    if p.returncode != 0:
        return False, f"git status failed: {p.stderr.strip()}"
    dirty = [l for l in p.stdout.splitlines() if l.strip()]
    if dirty:
        return False, f"{len(dirty)} dirty/untracked path(s): " + ", ".join(d[3:] for d in dirty[:4])
    return True, "clean, no untracked files"


def recent_mtime(path, window):
    """First file touched inside the window, or None. Stops at the first hit."""
    if window <= 0:
        return None
    p = run(["find", path, "-newermt", f"-{window} seconds", "-print", "-quit"])
    hit = p.stdout.strip().splitlines()
    return hit[0] if hit else None


def process_cwds():
    """pid -> (cwd, cmdline) for every process this user can inspect."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        out[pid] = (cwd, cmd)
    return out


def gate_unowned(row, sess, procs, self_cwd, owned_within):
    """Gate 3 — no live session, process or lock claims the worktree."""
    path = os.path.realpath(row["path"])
    reasons = []

    if self_cwd == path or self_cwd.startswith(path + os.sep):
        reasons.append("this sweep is running inside it")

    for s in sess:
        d = s.get("directory")
        if not d:
            continue
        d = os.path.realpath(d)
        if d != path and not d.startswith(path + os.sep):
            continue
        if s["archived"]:
            continue
        state = "active" if s["active"] else "hibernating"
        reasons.append(f"{s['cli']} session {s['id']} ({state}) works here")

    for pid, (cwd, cmd) in procs.items():
        if cwd == path or cwd.startswith(path + os.sep):
            reasons.append(f"pid {pid} has its cwd here ({cmd[:60]})")
            break

    # A linked worktree's `.git` is a FILE pointing at <repo>/.git/worktrees/<name>,
    # so joining ".git" onto the worktree path finds nothing and the check never fires.
    gitdir = run(["git", "-C", row["path"], "rev-parse", "--absolute-git-dir"]).stdout.strip()
    if gitdir:
        for lock in ("index.lock", "MERGE_HEAD", "rebase-merge", "rebase-apply"):
            if os.path.exists(os.path.join(gitdir, lock)):
                reasons.append(f"{lock} present in {gitdir} — an operation is in flight")

    touched = recent_mtime(row["path"], owned_within) if os.path.isdir(row["path"]) else None
    if touched:
        reasons.append(
            f"modified within {owned_within}s ({os.path.relpath(touched, row['path'])})"
        )

    if reasons:
        return False, "; ".join(reasons)
    return True, "no session, process, lock or recent write"


def watcher_roots_from_env():
    raw = os.environ.get(SWEEP_WATCHER_ROOTS_ENV, "")
    return [p for p in raw.split(os.pathsep) if p.strip()]


def watcher_corpus(repo, override_roots=None):
    """Every file that could hardcode a worktree path, branch or session id."""
    deep = list(override_roots) if override_roots else watcher_roots_from_env()
    shallow, files = [], []

    wf_dirs = []
    if not override_roots:
        # Siblings of the repo and the home directory: a workflow lives in
        # <somewhere>/.mycolab/workflows, and the checkouts it drives are
        # usually parked beside this one.
        for base in (os.path.dirname(os.path.abspath(repo)), os.path.expanduser("~")):
            p = run(
                ["find", base, "-maxdepth", "3", "-type", "d",
                 "(", "-path", "*/.mycolab/workflows", "-o", "-path", "*/.svamp/workflows", ")"]
            )
            wf_dirs += [d for d in p.stdout.splitlines() if d.strip()]
        deep += wf_dirs

    # A workflow step names the script it runs; that script is the real watcher,
    # and its own directory usually holds the config it reads. One level only —
    # a .svamp/ root walked two deep swallows every session's inbox.json.
    for d in wf_dirs:
        for entry in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            fp = os.path.join(d, entry)
            if os.path.isfile(fp):
                files.append(fp)
                try:
                    with open(fp, errors="replace") as fh:
                        text = fh.read(MAX_WATCHER_FILE_BYTES)
                except OSError:
                    continue
                for ref in SCRIPT_REF.findall(text):
                    if os.path.isfile(ref):
                        files.append(ref)
                        shallow.append(os.path.dirname(ref))

    for depth, roots in (("2", deep), ("1", shallow)):
        for root in dict.fromkeys(roots):
            if not os.path.isdir(root):
                continue
            p = run(["find", root, "-maxdepth", depth, "-type", "f", "-size", "-2M"])
            files += [f for f in p.stdout.splitlines() if f.strip()]

    corpus = []
    for fp in dict.fromkeys(files):
        try:
            with open(fp, errors="replace") as fh:
                corpus.append((fp, fh.read(MAX_WATCHER_FILE_BYTES)))
        except OSError:
            continue

    cron = run(["crontab", "-l"])
    if cron.returncode == 0 and cron.stdout.strip():
        corpus.append(("crontab -l", cron.stdout))
    return corpus


def gate_unwatched(row, corpus, sess):
    """Gate 4 — nothing scheduled names this worktree, branch or owning session."""
    terms = {row["path"], row["name"]}
    if row["branch"]:
        terms.add(row["branch"])
    path = os.path.realpath(row["path"])
    for s in sess:
        d = s.get("directory")
        if d and os.path.realpath(d) == path and s["id"]:
            terms.add(s["id"])

    hits = []
    for term in sorted(terms):
        if len(term) < 6:  # too short to be an identifying reference
            continue
        pat = re.compile(r"(?<![\w./-])" + re.escape(term) + r"(?![\w-])")
        for fp, text in corpus:
            m = pat.search(text)
            if m:
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{term} named in {fp}:{line}")
                break
    if hits:
        return False, "; ".join(hits[:3])
    return True, f"not named in {len(corpus)} watcher/workflow/cron file(s)"


# ------------------------------------------------------------------ sessions


def session_verdict(s, self_id):
    """A spawned session is retirable only once its working directory is gone.

    Idle is not the signal (#0049): a session can sit idle for days because it
    is blocked on a human answer, and archiving on inactivity discards that.
    A directory that no longer exists is the one piece of evidence that its
    work has already been taken away from it.
    """
    if s["archived"]:
        return "EXCLUDED", "already archived"
    if s["id"] == self_id:
        return "NOT-DONE", "this session"
    if s["active"]:
        return "NOT-DONE", "active"
    d = s.get("directory")
    if not d:
        return "NOT-DONE", "no working directory recorded"
    if os.path.isdir(d):
        return "NOT-DONE", f"idle, but its working directory still exists ({d}) — idle is not done"
    return "DONE", f"working directory {d} no longer exists"


# -------------------------------------------------------------------- report


def evaluate(repo, baseline, owned_within=OWNED_IF_TOUCHED_WITHIN, watcher_roots=None):
    sess = sessions()
    procs = process_cwds()
    corpus = watcher_corpus(repo, watcher_roots)
    self_cwd = os.path.realpath(os.getcwd())
    self_id = os.environ.get("MYCOLAB_SESSION_ID")

    rows = []
    for row in worktrees(repo):
        gates = {}
        if not os.path.isdir(row["path"]):
            # A registration whose directory is already gone. Pruning it deletes
            # no files and no branch, so it is reported apart from the gates
            # rather than being held back by the ones it cannot satisfy.
            row["verdict"] = "STALE"
            row["why"] = "registration points at a directory that no longer exists; the branch is untouched by a prune"
        elif row["band"] not in BAND_SWEEPABLE:
            row["verdict"] = "EXCLUDED"
            row["why"] = f"{row['band']} band is never swept automatically"
        else:
            gates["1-merged"] = gate_merged(repo, row, baseline)
            gates["2-clean"] = gate_clean(row)
            gates["3-unowned"] = gate_unowned(row, sess, procs, self_cwd, owned_within)
            gates["4-unwatched"] = gate_unwatched(row, corpus, sess)
            passed = all(ok for ok, _ in gates.values())
            row["verdict"] = "DONE" if passed else "NOT-DONE"
            row["why"] = "all four gates pass" if passed else "; ".join(
                f"gate {k} FAILED: {why}" for k, (ok, why) in sorted(gates.items()) if not ok
            )
        row["gates"] = {k: {"pass": ok, "evidence": why} for k, (ok, why) in gates.items()}
        rows.append(row)

    srows = []
    for s in sess:
        verdict, why = session_verdict(s, self_id)
        srows.append(dict(s, verdict=verdict, why=why))

    return rows, srows, {"watcher_files": len(corpus), "processes": len(procs)}


def render(rows, srows, baseline, counted):
    out = []
    out.append(f"origin/main = {baseline[:7]} (fetched {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})")
    out.append(
        f"{len(rows)} worktrees, {len(srows)} spawned sessions, "
        f"{counted['watcher_files']} watcher/workflow/cron files, {counted['processes']} processes inspected"
    )
    out.append("")
    for verdict in ("DONE", "STALE", "NOT-DONE", "EXCLUDED"):
        band = [r for r in rows if r["verdict"] == verdict]
        out.append(f"== WORKTREES {verdict} ({len(band)}) ==")
        for r in sorted(band, key=lambda r: (r["band"], r["name"])):
            out.append(f"  [{r['band']}] {r['name']}  {r['branch'] or '(detached)'}  {(r['head'] or '')[:7]}")
            out.append(f"      {r['why']}")
        out.append("")
    for verdict in ("DONE", "NOT-DONE", "EXCLUDED"):
        band = [s for s in srows if s["verdict"] == verdict]
        out.append(f"== SESSIONS {verdict} ({len(band)}) ==")
        for s in sorted(band, key=lambda s: (s["cli"], s["id"] or "")):
            out.append(f"  [{s['cli']}] {s['id']}  {s['why']}")
        out.append("")
    return "\n".join(out)


def summarise(rows, srows, baseline):
    """The compact form the recurring reporter sends into a session's chat."""
    done = [r for r in rows if r["verdict"] == "DONE"]
    stale = [r for r in rows if r["verdict"] == "STALE"]
    nd = [r for r in rows if r["verdict"] == "NOT-DONE"]
    ex = [r for r in rows if r["verdict"] == "EXCLUDED"]
    sdone = [s for s in srows if s["verdict"] == "DONE"]
    lines = [
        "Worktree/session sweep (report only, nothing was removed).",
        f"{len(rows)} worktrees against origin/main {baseline[:7]}: "
        f"{len(done)} pass all four gates, {len(stale)} are stale registrations, "
        f"{len(nd)} fail a gate, {len(ex)} excluded by band.",
        f"{len(srows)} spawned sessions: {len(sdone)} retirable (working directory gone).",
    ]
    if done or stale:
        lines.append("")
        lines.append("Removable with `scripts/sweep_agent_worktrees.py --apply`:")
        for r in sorted(done, key=lambda r: r["name"]):
            lines.append(f"  {r['path']}  [{r['branch'] or 'detached'}]  all four gates pass")
        for r in sorted(stale, key=lambda r: r["name"]):
            lines.append(f"  {r['path']}  [{r['branch'] or 'detached'}]  stale registration, directory already gone")
    if sdone:
        lines.append("")
        lines.append("Sessions whose working directory is gone — archive with `myco session archive <id>`:")
        for s in sdone:
            lines.append(f"  {s['id']}  ({s['why']})")
    if not done and not stale and not sdone:
        lines.append("")
        lines.append("Nothing is retirable this tick. That is a pass, not a failure.")
    return "\n".join(lines)


# --------------------------------------------------------------------- apply


def apply_removals(repo, rows, baseline):
    removed = []
    for r in sorted([r for r in rows if r["verdict"] == "DONE"], key=lambda r: r["name"]):
        # `worktree remove` without --force refuses a dirty tree: gate 2 proved
        # it clean, so a refusal here means the tree changed under us.
        p = run(["git", "-C", repo, "worktree", "remove", r["path"]])
        if p.returncode != 0:
            removed.append((r, f"worktree remove REFUSED: {p.stderr.strip()}"))
            continue
        note = "worktree removed"
        br = r["branch"]
        if br and br not in ("main", "master"):
            # Re-run gate 1 against the branch itself, by the same definition
            # that cleared it, in case the ref moved since the report.
            landed, why = gate_merged(repo, {"branch": br, "head": None}, baseline)
            if landed:
                d = run(["git", "-C", repo, "branch", "-D", br])
                note += f"; branch {br} deleted" if d.returncode == 0 else f"; branch {br} kept: {d.stderr.strip()}"
            else:
                note += f"; branch {br} KEPT ({why})"
        removed.append((r, note))

    if any(r["verdict"] == "STALE" for r in rows):
        planned = run(["git", "-C", repo, "worktree", "prune", "--dry-run", "--verbose"]).stdout.strip()
        run(["git", "-C", repo, "worktree", "prune"])
        for line in planned.splitlines():
            if line.strip():
                removed.append(({"path": line.strip()}, "stale registration pruned (no files, no branch)"))
    return removed


def default_repo():
    """The checkout to sweep, when --repo is not given.

    Deliberately not derived from ``__file__``: the recurring reporter runs a
    copy of this script out of a cache directory, where the repo is nowhere up
    the path. $BIOENGINE_REPO is how that caller names it.
    """
    env = os.environ.get("BIOENGINE_REPO")
    if env:
        return env
    p = run(["git", "rev-parse", "--show-toplevel"])
    return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=default_repo(),
                    help="main checkout that owns the worktrees (default: $BIOENGINE_REPO, "
                         "else the checkout containing the current directory)")
    ap.add_argument("--apply", action="store_true", help="remove the entries that pass all four gates")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument("--notify", metavar="SESSION_ID",
                    help="send the summary to a session with `myco session send`; never removes anything")
    ap.add_argument("--fallback-file", metavar="PATH", default=NOTIFY_FALLBACK_DEFAULT,
                    help="where --notify writes the report when delivery fails "
                         f"(default: {NOTIFY_FALLBACK_DEFAULT})")
    # The two knobs below exist so the gates can be exercised on a scratch repo
    # (scripts/test_sweep_agent_worktrees.py). Lowering --owned-within weakens
    # gate 3; narrowing --watcher-root weakens gate 4.
    ap.add_argument("--owned-within", type=int, default=OWNED_IF_TOUCHED_WITHIN, metavar="SECONDS",
                    help="treat a worktree written to inside this window as owned (default 86400)")
    ap.add_argument("--watcher-root", action="append", metavar="DIR",
                    help="search only these directories for watchers instead of discovering them")
    args = ap.parse_args()

    if args.apply and args.notify:
        ap.error("--apply and --notify are mutually exclusive: the recurring reporter never removes anything")
    if not args.repo:
        ap.error("no checkout to sweep: pass --repo, set $BIOENGINE_REPO, or run from inside one")

    baseline = fetch_baseline(args.repo)
    rows, srows, counted = evaluate(args.repo, baseline, args.owned_within, args.watcher_root)

    if args.notify:
        body = summarise(rows, srows, baseline)
        cmd = ["myco", "session", "send", args.notify, "--subject",
               "Worktree sweep: %d of %d retirable"
               % (sum(1 for r in rows if r["verdict"] in ("DONE", "STALE")), len(rows)),
               "--body-file", "-"]
        for attempt in range(1, NOTIFY_ATTEMPTS + 1):
            p = subprocess.run(cmd, input=body, text=True, capture_output=True)
            if p.returncode == 0:
                print(f"delivered to {args.notify}: {len(body)} bytes (attempt {attempt})")
                return 0
            print(f"delivery to {args.notify} failed (attempt {attempt}/{NOTIFY_ATTEMPTS}): "
                  f"{p.stderr.strip()}", file=sys.stderr)
            if attempt < NOTIFY_ATTEMPTS:
                time.sleep(NOTIFY_RETRY_SECONDS)
        try:
            os.makedirs(os.path.dirname(args.fallback_file) or ".", exist_ok=True)
            with open(args.fallback_file, "w") as fh:
                fh.write(body)
        except OSError as exc:
            print(f"FAILED to deliver the report to {args.notify} after {NOTIFY_ATTEMPTS} "
                  f"attempts, and the fallback write to {args.fallback_file} failed: {exc}",
                  file=sys.stderr)
            return 1
        print(f"undelivered to {args.notify} after {NOTIFY_ATTEMPTS} attempts; "
              f"report written to {args.fallback_file} ({len(body)} bytes)", file=sys.stderr)
        return 0

    if args.json:
        print(json.dumps({"origin_main": baseline, "worktrees": rows, "sessions": srows, "counted": counted}, indent=2))
    else:
        print(render(rows, srows, baseline, counted))

    if args.apply:
        results = apply_removals(args.repo, rows, baseline)
        print(f"== APPLIED ({len(results)}) ==")
        for r, note in results:
            print(f"  {r['path']}: {note}")
        if not results:
            print("  nothing passed all four gates; nothing was removed (this is a pass)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
