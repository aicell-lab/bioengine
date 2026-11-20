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

run() { NOW_S="$1" DRAFT_BUMP_GRACE_S="$2" bash "$SUT" "$3" "$4" >"$T/out.txt" 2>&1; echo $?; }

say "=== a draft BEHIND main reserves nothing (the false-RED class this replaced) ==="
rc=$(run "$((BUMP_EPOCH + 86400))" 14400 "$BUMP" "$NOBUMP")
check "behind main, never bumped -> silent"                                       0 "$rc"
rc=$(run "$((BUMP_EPOCH + 86400))" 14400 "$BUMP" "$DEPEDIT")
check "behind main, edited pyproject but not the version -> silent"               0 "$rc"
grep -q "BEHIND main" "$T/out.txt"
check "and it says so, rather than being silent for an unstated reason"           0 "$?"

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
