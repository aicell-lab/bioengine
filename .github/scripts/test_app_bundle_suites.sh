#!/usr/bin/env bash
# Re-runnable demonstration of the app-bundle discovery gate (backlog #0423).
#
# Case 2 is the one this gate exists for: a third app that ships in-bundle
# tests and is named in neither table REFUSES, where the hardcoded two-app loop
# it replaces exited 0 and ran nothing. Cases 3, 4, 7 and 9 are its controls —
# the same tree with that app declared, or shipping files pytest would not
# collect, must stay green, or the gate would just fail on every new app.
#
# Each case builds a throwaway tree under a temp dir and runs the real script
# against it with --root. Nothing touches the repo. Case 1 is the exception and
# runs against the committed tree, so the registry and apps/ are checked as
# they actually are.
#
#   bash .github/scripts/test_app_bundle_suites.sh

set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
GATE=$REPO/.github/scripts/app_bundle_suites.py
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
PASSED=0
FAILED=0

# A throwaway tree: $1 is the registry JSON, the rest are paths to create
# relative to the tree root ("" creates only the directory).
tree_with() {
  local registry=$1; shift
  local dir
  dir=$(mktemp -d "$WORK/tree.XXXXXX")
  mkdir -p "$dir/.github" "$dir/apps"
  printf '%s\n' "$registry" > "$dir/.github/app-bundle-suites.json"
  local path
  for path in "$@"; do
    mkdir -p "$dir/$(dirname "$path")"
    [ "$(basename "$path")" = "." ] || echo "def test_x(): pass" > "$dir/$path"
  done
  echo "$dir"
}

# $1 label, $2 expected exit status, $3 text the combined output must contain
# ("" to assert nothing about it), then the command.
check() {
  local label=$1 want=$2 needle=$3; shift 3
  local out rc
  out=$("$@" 2>&1); rc=$?
  if [ "$rc" -ne "$want" ]; then
    echo "FAIL  $label: exit $rc, expected $want"
    echo "      output: $(echo "$out" | head -3 | tr '\n' '|')"
    FAILED=$((FAILED + 1)); return
  fi
  if [ -n "$needle" ] && ! printf '%s' "$out" | grep -qF -- "$needle"; then
    echo "FAIL  $label: exit $rc as expected, but output lacks '$needle'"
    echo "      output: $(echo "$out" | head -3 | tr '\n' '|')"
    FAILED=$((FAILED + 1)); return
  fi
  echo "ok    $label"
  PASSED=$((PASSED + 1))
}

# A JUnit report with $2 tests, $3 skipped, $4 failures, for app $1 in dir $5.
junit() {
  printf '<testsuite tests="%s" skipped="%s" failures="%s" errors="0"/>\n' \
    "$2" "$3" "$4" > "$5/junit-app-$1.xml"
}

ONE='{"run": {"alpha": 1}, "skip": {}}'

echo "== the committed tree =="
check "1a  reconcile accepts the repo as committed" 0 "OK:" \
  python3 "$GATE" reconcile
LISTED=$(python3 "$GATE" list 2>/dev/null | tr '\n' ' ')
if [ "$LISTED" = "annotation-broker model-finetune " ]; then
  echo "ok    1b  list prints exactly the two apps CI runs"
  PASSED=$((PASSED + 1))
else
  echo "FAIL  1b  list printed '$LISTED'"
  FAILED=$((FAILED + 1))
fi

echo "== an app shipping tests that nothing runs =="
T=$(tree_with "$ONE" apps/alpha/test_a.py apps/newapp/test_b.py)
check "2   THE DISCRIMINATOR: undeclared app refuses" 1 "newapp" \
  python3 "$GATE" reconcile --root "$T"
check "2b  and list refuses too, so the loop cannot run" 1 "newapp" \
  python3 "$GATE" list --root "$T"

T=$(tree_with '{"run": {"alpha": 1}, "skip": {"newapp": "needs the app image"}}' \
  apps/alpha/test_a.py apps/newapp/test_b.py)
check "3   declared in skip: accepted" 0 "not run: newapp" \
  python3 "$GATE" reconcile --root "$T"
LISTED=$(python3 "$GATE" list --root "$T" 2>/dev/null | tr '\n' ' ')
if [ "$LISTED" = "alpha " ]; then
  echo "ok    3b  a skipped app is not in the run list"
  PASSED=$((PASSED + 1))
else
  echo "FAIL  3b  list printed '$LISTED', expected 'alpha '"
  FAILED=$((FAILED + 1))
fi

T=$(tree_with '{"run": {"alpha": 1, "newapp": 2}, "skip": {}}' \
  apps/alpha/test_a.py apps/newapp/test_b.py)
check "4   declared in run: accepted" 0 "OK:" \
  python3 "$GATE" reconcile --root "$T"
LISTED=$(python3 "$GATE" list --root "$T" 2>/dev/null | tr '\n' ' ')
if [ "$LISTED" = "alpha newapp " ]; then
  echo "ok    4b  list prints both apps"
  PASSED=$((PASSED + 1))
else
  echo "FAIL  4b  list printed '$LISTED', expected 'alpha newapp '"
  FAILED=$((FAILED + 1))
fi

echo "== registry and tree disagreeing the other way =="
T=$(tree_with '{"run": {"alpha": 1, "ghost": 1}, "skip": {}}' apps/alpha/test_a.py)
check "5   a registry entry with no tests on disk refuses" 1 "ghost" \
  python3 "$GATE" reconcile --root "$T"

T=$(tree_with '{"run": {"alpha": 1}, "skip": {"alpha": "x"}}' apps/alpha/test_a.py)
check "6   an app in both tables refuses" 1 "both" \
  python3 "$GATE" reconcile --root "$T"

echo "== controls: what must NOT refuse =="
T=$(tree_with "$ONE" apps/alpha/test_a.py apps/newapp/app.py)
check "7   a new app shipping NO tests is accepted" 0 "OK:" \
  python3 "$GATE" reconcile --root "$T"

T=$(tree_with "$ONE" apps/alpha/test_a.py apps/newapp/thing_test.py)
check "8   *_test.py is not discovered (pytest would not collect it)" 0 "OK:" \
  python3 "$GATE" reconcile --root "$T"

T=$(tree_with "$ONE" apps/alpha/test_a.py apps/newapp/suite/test_b.py)
check "9   a test NESTED in a new app is still discovered" 1 "newapp" \
  python3 "$GATE" reconcile --root "$T"

echo "== a registry that cannot be trusted =="
T=$(tree_with 'not json at all' apps/alpha/test_a.py)
check "10  unparseable registry refuses" 1 "cannot read" \
  python3 "$GATE" reconcile --root "$T"

T=$(tree_with '{"run": {"alpha": 0}, "skip": {}}' apps/alpha/test_a.py)
check "11  a floor of 0 refuses" 1 "floor >= 1" \
  python3 "$GATE" reconcile --root "$T"

T=$(tree_with '{"run": {"alpha": 1}, "skip": {"newapp": "  "}}' \
  apps/alpha/test_a.py apps/newapp/test_b.py)
check "12  a blank skip reason refuses" 1 "must say WHY" \
  python3 "$GATE" reconcile --root "$T"

echo "== assert-ran reads the reports =="
T=$(tree_with '{"run": {"alpha": 10, "beta": 10}, "skip": {}}' \
  apps/alpha/test_a.py apps/beta/test_b.py)
junit alpha 12 0 0 "$T"; junit beta 12 0 0 "$T"
check "13  both at floor, no failures: accepted" 0 "OK:" \
  python3 "$GATE" assert-ran --root "$T" --reports-dir "$T"

junit alpha 12 5 0 "$T"
check "14  skips counted out drop it below the floor: refuses" 1 "below the floor" \
  python3 "$GATE" assert-ran --root "$T" --reports-dir "$T"

junit alpha 12 0 0 "$T"; junit beta 12 0 1 "$T"
check "15  a failing case refuses" 1 "failed or errored" \
  python3 "$GATE" assert-ran --root "$T" --reports-dir "$T"

rm "$T/junit-app-beta.xml"
check "16  a missing report refuses rather than passing on one" 1 "expected 2" \
  python3 "$GATE" assert-ran --root "$T" --reports-dir "$T"

echo
echo "passed=$PASSED failed=$FAILED"
[ "$FAILED" -eq 0 ]
