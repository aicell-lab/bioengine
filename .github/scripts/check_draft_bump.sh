#!/usr/bin/env bash
# Flag a version rung that has SAT on a draft pull request.
#
# A draft holding a bump reserves a rung nobody else can use, so every other open
# pull request has to re-bump the moment one of them merges. Flagging ANY bump on a
# draft is too blunt: bump LAST means the bump is pushed while the pull request is
# still a draft and it is marked ready a moment later, so that version of the check
# fired on the one commit the rule asks for, every time.
#
# TWO conditions have to hold before anything is said:
#
#   1. The head claims a rung ABOVE the base. `base.sha` is main's CURRENT tip, so a
#      draft that is merely behind main reads LOWER than base without having bumped
#      anything -- which is most open drafts here, most of the time. Only a head
#      version strictly greater than base's reserves a rung anyone else wants.
#   2. That rung has been held longer than a grace period. A bump pushed minutes
#      before `gh pr ready` is the prescribed workflow, not a reservation.
#
# WHAT THIS DOES NOT CATCH, stated because the check's own wording would otherwise
# overclaim:
#
#   * "How long the rung has been held" is approximated by the bump commit's
#     COMMITTER time, which is the only clock available to a workflow without extra
#     API calls. Rebase and amend reset it, so a draft that is kept current by
#     rebasing reads as freshly bumped however long it has really held the rung. A
#     bump committed locally hours before being pushed reads as stale for the same
#     reason. Merging main in resets it too, but only when main itself touched a
#     non-version line of pyproject.toml: the merge commit then becomes the newest
#     commit in range that touched the file. When main changed only the version line
#     the original bump is still found. The approximation is deliberate; none of
#     these history-rewriting or merge cases is covered.
#   * It can only judge when an event fires (opened, synchronize, reopened,
#     ready_for_review, converted_to_draft). A draft that bumps and then sits with no
#     further push is judged once, at age ~0, and never again -- which is exactly the
#     pure reservation case. It is caught only if the author pushes again after the
#     grace period.
#
# Usage:  check_draft_bump.sh <base_sha> <head_sha>
# Env:    DRAFT_BUMP_GRACE_S  how long a rung may be held before it is flagged (default 4h)
#         NOW_S               override "now", for the test harness only
#
# Exit 0 = nothing to say. Exit 1 = flagged, or could not be determined. This FAILS
# CLOSED only where the answer is genuinely unknown: an unreadable version is an
# error, while a head below base is a confident "no reservation".
set -uo pipefail

BASE_SHA="${1:-}"
HEAD_SHA="${2:-}"
GRACE_S="${DRAFT_BUMP_GRACE_S:-14400}"
NOW_S="${NOW_S:-$(date -u +%s)}"

if [ -z "$BASE_SHA" ] || [ -z "$HEAD_SHA" ]; then
  echo "::error::usage: check_draft_bump.sh <base_sha> <head_sha>"
  exit 1
fi

read_version() {
  # Prints nothing and returns non-zero if the side cannot be read, so an
  # unreadable ref is never mistaken for "no bump".
  git show "$1:pyproject.toml" 2>/dev/null | grep -m1 '^version' | sed 's/.*"\(.*\)".*/\1/'
}

BASE_V=$(read_version "$BASE_SHA")
HEAD_V=$(read_version "$HEAD_SHA")

if [ -z "$BASE_V" ] || [ -z "$HEAD_V" ]; then
  echo "::error::Could not read pyproject.toml version on one side"
  echo "  base $BASE_SHA -> '${BASE_V:-<unreadable>}'"
  echo "  head $HEAD_SHA -> '${HEAD_V:-<unreadable>}'"
  echo "This check FAILS CLOSED: an unreadable version is not evidence of no bump."
  exit 1
fi

echo "base version: $BASE_V"
echo "head version: $HEAD_V"

if [ "$BASE_V" = "$HEAD_V" ]; then
  echo "No bump on this draft. A draft that merged main in will match the base"
  echo "version, which is correct and is not a reservation."
  exit 0
fi

# Strictly-greater, version-ordered. Anything else means this draft is simply behind
# main, which reserves nothing -- the common case, and the one a tip comparison alone
# gets wrong.
HIGHER=$(printf '%s\n%s\n' "$BASE_V" "$HEAD_V" | sort -V | tail -1)
if [ "$HIGHER" != "$HEAD_V" ]; then
  echo "This draft is BEHIND main ($HEAD_V < $BASE_V) and reserves nothing. A rung"
  echo "below main cannot be claimed by anyone, so there is nothing to report."
  exit 0
fi

# The newest commit in the pull request's own range that touched pyproject.toml. Its
# committer time is when this rung was claimed, with the caveat in the header.
BUMP_SHA=$(git log --format=%H -1 "$BASE_SHA..$HEAD_SHA" -- pyproject.toml 2>/dev/null)
BUMP_AT=""
[ -n "$BUMP_SHA" ] && BUMP_AT=$(git show -s --format=%ct "$BUMP_SHA" 2>/dev/null)

if [ -z "$BUMP_SHA" ] || [ -z "$BUMP_AT" ]; then
  echo "::error::This draft claims $HEAD_V, above base's $BASE_V, but the commit that"
  echo "changed pyproject.toml in this range could not be identified, so how long the"
  echo "rung has been held is unknown."
  echo "  range     $BASE_SHA..$HEAD_SHA"
  echo "  bump sha  '${BUMP_SHA:-<none found>}'"
  echo "This check FAILS CLOSED: an unknown age is not evidence of a fresh bump."
  exit 1
fi

AGE_S=$(( NOW_S - BUMP_AT ))
echo "rung $HEAD_V claimed by $BUMP_SHA, ${AGE_S}s ago (grace ${GRACE_S}s)"

if [ "$AGE_S" -le "$GRACE_S" ]; then
  echo "This bump is FRESH, so it reads as the prescribed bump-last sequence: bump,"
  echo "push, then mark ready. Nothing is being reserved and nothing is owed here."
  exit 0
fi

echo "::error::This draft has held version $HEAD_V for ${AGE_S}s (over the ${GRACE_S}s grace)."
echo ""
echo "The rule is bump LAST. While this draft holds $HEAD_V, no other pull request can"
echo "use that rung, so each of them has to re-bump the moment one merges."
echo "Drop the bump, keep iterating, and add it back in the same sitting as"
echo "'gh pr ready'."
echo ""
echo "This job is advisory and does not block a merge."
exit 1
