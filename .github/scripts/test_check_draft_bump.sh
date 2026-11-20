#!/usr/bin/env bash
# Re-runnable demonstration for check_draft_bump.sh.
#
# The behaviour under test is a DISTINCTION, so arms are paired: the same bump passes
# when fresh and fails when aged, and the grace threshold is moved in both directions
# to show it is what decides. An arm showing only the failing case would be satisfied
# by a check that flags everything, which is the defect this script replaced.
#
# The BEHIND-MAIN arms exist because an earlier version of this harness got them
# backwards: it asserted that a draft behind main errors, which enshrined a false RED
# on the most common input there is. `base.sha` is main's CURRENT tip, so a draft that
# has not merged main in reads LOWER than base without having bumped anything.
#
# Builds throwaway git repositories in a temp dir. Touches nothing else.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUT="$HERE/check_draft_bump.sh"
T=$(mktemp -d "${TMPDIR:-/tmp}/draftbump-XXXXXX")
ran=0; fail=0

say()   { printf '%s\n' "$*"; }
check() {
  ran=$((ran + 1))
  if [ "$2" = "$3" ]; then say "  ok   $1"
  else say "  FAIL $1: want rc=$2 got rc=$3"; fail=$((fail + 1)); fi
}

[ -x "$SUT" ] || { say "FAIL: $SUT is not executable"; exit 1; }

export GIT_AUTHOR_NAME=harness GIT_AUTHOR_EMAIL=harness@example.com
export GIT_COMMITTER_NAME=harness GIT_COMMITTER_EMAIL=harness@example.com

cd "$T" || exit 1
git init -q .
printf 'version = "0.1.0"\n' > pyproject.toml
git add pyproject.toml
GIT_COMMITTER_DATE="2026-01-01T00:00:00Z" git commit -qm base --date="2026-01-01T00:00:00Z"
BASE=$(git rev-parse HEAD)

printf 'x\n' > other.txt
git add other.txt
GIT_COMMITTER_DATE="2026-01-01T01:00:00Z" git commit -qm other --date="2026-01-01T01:00:00Z"
NOBUMP=$(git rev-parse HEAD)

printf 'version = "0.1.1"\n' > pyproject.toml
git add pyproject.toml
GIT_COMMITTER_DATE="2026-01-02T00:00:00Z" git commit -qm bump --date="2026-01-02T00:00:00Z"
BUMP=$(git rev-parse HEAD)
BUMP_EPOCH=$(git show -s --format=%ct "$BUMP")

# A draft that edited pyproject.toml WITHOUT touching the version, long ago, and is
# behind a main that has since bumped.
git checkout -q "$BASE" -b depedit
printf 'version = "0.1.0"\ndeps = ["x"]\n' > pyproject.toml
git add pyproject.toml
GIT_COMMITTER_DATE="2026-01-01T02:00:00Z" git commit -qm "dependency edit" --date="2026-01-01T02:00:00Z"
DEPEDIT=$(git rev-parse HEAD)
git checkout -q "$BUMP"

# A later edit to pyproject.toml that leaves the version alone. Selecting the bump
# commit by FILE would make this the clock, hiding a rung that is still held.
git checkout -q -B depslater "$BUMP"
printf 'version = "0.1.1"\ndeps = ["x"]\n' > pyproject.toml
git add pyproject.toml
GIT_COMMITTER_DATE="2026-01-09T00:00:00Z" git commit -qm "deps, not the version" --date="2026-01-09T00:00:00Z"
DEPSLATER=$(git rev-parse HEAD)
git checkout -q "$BUMP"

# A rung claimed by RESOLVING A MERGE, which is the shape `git log -G` cannot see
# without -m: the draft carries no bump commit of its own, and the version is settled
# in the merge itself.
git checkout -q -B mainline "$BASE"
printf 'version = "0.1.1"\n' > pyproject.toml
git add pyproject.toml
GIT_COMMITTER_DATE="2026-01-05T00:00:00Z" git commit -qm "base branch moves to 0.1.1" --date="2026-01-05T00:00:00Z"
MAINLINE=$(git rev-parse HEAD)
git checkout -q -B mergepr "$BASE"
printf 'y\n' > work.txt
git add work.txt
GIT_COMMITTER_DATE="2026-01-02T00:00:00Z" git commit -qm "work, no version change" --date="2026-01-02T00:00:00Z"
GIT_COMMITTER_DATE="2026-01-09T00:00:00Z" git merge -q --no-edit "$MAINLINE" >/dev/null 2>&1 || true
printf 'version = "0.1.2"\n' > pyproject.toml
git add pyproject.toml
GIT_COMMITTER_DATE="2026-01-09T00:00:00Z" git commit -q --amend --no-edit --date="2026-01-09T00:00:00Z"
MERGEHEAD=$(git rev-parse HEAD)
MERGE_EPOCH=$(git show -s --format=%ct "$MERGEHEAD")
git checkout -q "$BUMP"

# A base branch that does NOT touch pyproject.toml, so a draft can merge it cleanly
# and keep a rung that is still above it. This is the shape that pins --first-parent:
# without it, -m diffs the merge against the base side too, where the version line
# differs, and the merge is then mistaken for the moment the rung was claimed.
git checkout -q -B cleanbase "$BASE"
printf 'unrelated\n' > notes.txt
git add notes.txt
GIT_COMMITTER_DATE="2026-01-05T00:00:00Z" git commit -qm "base branch moves, no version change" --date="2026-01-05T00:00:00Z"
CLEANBASE=$(git rev-parse HEAD)
git checkout -q -B cleanmerge "$BUMP"
GIT_COMMITTER_DATE="2026-01-09T00:00:00Z" git merge -q --no-edit "$CLEANBASE" >/dev/null 2>&1
CLEANMERGE=$(git rev-parse HEAD)
CLEANMERGE_EPOCH=$(git show -s --format=%ct "$CLEANMERGE")
git checkout -q "$BUMP"

run() { NOW_S="$1" DRAFT_BUMP_GRACE_S="$2" bash "$SUT" "$3" "$4" "${5:-true}" >"$T/out.txt" 2>&1; echo $?; }

say "=== a draft BEHIND main reserves nothing (the false-RED class this replaced) ==="
rc=$(run "$((BUMP_EPOCH + 86400))" 14400 "$BUMP" "$NOBUMP")
check "behind main, never bumped -> silent"                                       0 "$rc"
rc=$(run "$((BUMP_EPOCH + 86400))" 14400 "$BUMP" "$DEPEDIT")
check "behind main, edited pyproject but not the version -> silent"               0 "$rc"
grep -q "BEHIND main" "$T/out.txt"
check "and it says so, rather than being silent for an unstated reason"           0 "$?"

say
say "=== a non-draft RUNS and reports clear, rather than being skipped ==="
say "    The job has no draft filter: a skipped job reports completed/skipped on the"
say "    head that merges, which looks green and ran nothing."
rc=$(run "$((BUMP_EPOCH + 86400))" 14400 "$BASE" "$BUMP" false)
check "an aged bump on a NON-draft is clear (it is the bump it will merge with)"   0 "$rc"
grep -q "not a draft" "$T/out.txt"
check "and it SAYS it ran, rather than being silently skipped"                     0 "$?"
rc=$(run "$((BUMP_EPOCH + 86400))" 14400 "$BASE" "$BUMP" true)
check "CONTROL: the SAME bump, same age, IS flagged when it is a draft"            1 "$rc"

say
say "=== the distinction: the same bump, fresh vs sat ==="
rc=$(run "$((BUMP_EPOCH + 600))" 14400 "$BASE" "$BUMP")
check "a bump made 10 min ago is NOT flagged (the prescribed bump-last sequence)"  0 "$rc"
rc=$(run "$((BUMP_EPOCH + 86400))" 14400 "$BASE" "$BUMP")
check "the SAME bump, 24 h old, IS flagged"                                        1 "$rc"

say
say "=== the grace knob is what decides, and nothing else ==="
rc=$(run "$((BUMP_EPOCH + 86400))" 999999 "$BASE" "$BUMP")
check "CONTROL: the 24 h-old bump passes when the grace is raised above its age"   0 "$rc"
rc=$(run "$((BUMP_EPOCH + 600))" 60 "$BASE" "$BUMP")
check "CONTROL: the 10 min bump fails when the grace is lowered below its age"     1 "$rc"

say
say "=== a later non-version edit must NOT reset the clock ==="
say "    The rung is selected by a change to a version LINE, not by the file."
rc=$(run "$((BUMP_EPOCH + 605400))" 14400 "$BASE" "$DEPSLATER")
check "rung held 7 days, with a later deps-only commit on top -> still flagged"    1 "$rc"
rc=$(run "$((BUMP_EPOCH + 600))" 14400 "$BASE" "$DEPSLATER")
check "CONTROL: the same history 10 min after the BUMP -> clear"                   0 "$rc"
grep -q "claimed by" "$T/out.txt"
check "and the line names the commit it timed from"                                0 "$?"

say
say "=== a rung claimed by resolving a MERGE is seen, and timed from the merge ==="
say "    git log -G does not diff merge commits without -m: without it this rung is"
say "    invisible, which fails closed on a rung claimed moments ago."
rc=$(run "$((MERGE_EPOCH + 600))" 14400 "$MAINLINE" "$MERGEHEAD")
check "a fresh rung set in the merge commit is clear, not a fail-closed error"     0 "$rc"
grep -q "claimed by $MERGEHEAD" "$T/out.txt"
check "and it is timed from the MERGE, not from an older commit"                   0 "$?"
rc=$(run "$((MERGE_EPOCH + 605400))" 14400 "$MAINLINE" "$MERGEHEAD")
check "CONTROL: the same merge-claimed rung, 7 days on, IS flagged"                1 "$rc"

say
say "=== merging the base branch in must NOT reset the draft's own clock ==="
say "    Pins --first-parent. Without it the merge is diffed against the base side,"
say "    where the version line differs, and an old rung reads as claimed just now."
say "    MEASURED: this pair discriminates on git >= 2.31 only. git 2.31 changed how"
say "    -m combines with --first-parent; on 2.25.1 the mutant passes because the"
say "    defect does not arise there. Mutate on a modern git or this proves nothing."
say "    git here: $(git --version)"
rc=$(run "$((CLEANMERGE_EPOCH + 600))" 14400 "$CLEANBASE" "$CLEANMERGE")
check "a 7-day-old rung survives a later CLEAN merge of the base -> still flagged"  1 "$rc"
grep -q "claimed by $BUMP" "$T/out.txt"
check "and it is still timed from the original bump, not from the merge"            0 "$?"

say
say "=== no bump at all ==="
rc=$(run "$((BUMP_EPOCH + 86400))" 14400 "$BASE" "$NOBUMP")
check "equal versions are silent however long the draft has been open"             0 "$rc"

say
say "=== fails closed where the answer is genuinely unknown ==="
rc=$(run "$((BUMP_EPOCH + 600))" 14400 "$BASE" "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
check "an unreadable head version is an error, not 'no bump'"                      1 "$rc"
grep -q "FAILS CLOSED" "$T/out.txt"
check "and it says FAILS CLOSED rather than failing for an unstated reason"        0 "$?"

say
say "=== KNOWN GAP, pinned so it cannot be forgotten: committer time resets ==="
say "    The header says the rebase case is NOT covered. This arm asserts the gap as"
say "    it currently is, so that closing it turns this arm RED and forces an update."
git checkout -q -B hoarder "$BUMP"
GIT_COMMITTER_DATE="2026-06-01T00:00:00Z" git commit -q --amend --no-edit --date="2026-01-02T00:00:00Z"
REBASED=$(git rev-parse HEAD)
REB_EPOCH=$(git show -s --format=%ct "$REBASED")
rc=$(run "$((REB_EPOCH + 600))" 14400 "$BASE" "$REBASED")
check "an aged rung whose commit was rewritten reads FRESH (gap, not a pass)"      0 "$rc"

say
say "=== checks_ran=$ran pass=$((ran - fail)) fail=$fail ==="
cd / && rm -rf "$T"
[ "$fail" -eq 0 ]
