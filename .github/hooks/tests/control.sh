#!/usr/bin/env bash
# Discrimination control for the pre-push hook.
#
# Six policy arms, plus the arm that matters most: with the hook REMOVED, the
# forbidden push must SUCCEED. Without that, a harness that refuses everything
# for an unrelated reason would score a perfect 6/6 and mean nothing.
set -uo pipefail

ROOT=${TMPDIR:-/tmp}/bioengine-prepush-control
HOOK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pre-push"
rm -rf "$ROOT"; mkdir -p "$ROOT"

pass=0; fail=0
note() { printf '  [%s] %-58s %s\n' "$1" "$2" "${3:-}"; }

setup() {
    # fresh origin + clone for every arm, so arms cannot contaminate each other
    rm -rf "$ROOT/origin" "$ROOT/work"
    git init -q --bare "$ROOT/origin"
    git clone -q "$ROOT/origin" "$ROOT/work" 2>/dev/null
    cd "$ROOT/work"
    git config user.name t; git config user.email t@t
    git config core.hooksPath "$ROOT/hooks"
    mkdir -p "$ROOT/hooks"
    if [ "${1:-with-hook}" = "with-hook" ]; then
        cp "$HOOK" "$ROOT/hooks/pre-push"
        cp "$(dirname "$HOOK")/match-pr-only.sh" "$ROOT/hooks/"
        cp "$(dirname "$HOOK")/pr-only-paths.txt" "$ROOT/hooks/"
        chmod +x "$ROOT/hooks/pre-push" "$ROOT/hooks/match-pr-only.sh"
    else
        rm -f "$ROOT/hooks/pre-push"
    fi
    # seed main so later pushes are incremental rather than initial
    mkdir -p docs && echo seed > docs/seed.md
    git add -A && git commit -q -m seed
    git push -q origin HEAD:refs/heads/main 2>/dev/null
}

# arm: <name> <file> <target-ref> <expect: REFUSE|ALLOW> [with-hook|no-hook]
arm() {
    local name="$1" file="$2" ref="$3" expect="$4" mode="${5:-with-hook}"
    setup "$mode"
    mkdir -p "$(dirname "$file")"; echo change > "$file"
    git add -A && git commit -q -m "touch $file"
    git push origin "HEAD:$ref" >/dev/null 2>"$ROOT/err.txt"
    local rc=$?
    local got; [ $rc -eq 0 ] && got=ALLOW || got=REFUSE
    if [ "$got" = "REFUSE" ] && [ "$expect" = "REFUSE" ] \
       && ! grep -q "pre-push: REFUSED" "$ROOT/err.txt"; then
        fail=$((fail+1)); note FAIL "$name" "-> refused, but NOT by the policy"
        sed 's/^/        /' "$ROOT/err.txt" | head -3
    elif [ "$got" = "$expect" ]; then
        pass=$((pass+1)); note ok "$name" "-> $got"
    else
        fail=$((fail+1)); note FAIL "$name" "-> $got (wanted $expect)"
        sed 's/^/        /' "$ROOT/err.txt" | head -4
    fi
}

echo "=== policy arms (hook installed) ==="
arm "bioengine/** to main is refused"            "bioengine/x.py"            refs/heads/main    REFUSE
arm "pyproject.toml to main is refused"          "pyproject.toml"            refs/heads/main    REFUSE
arm "requirements*.txt to main is refused"       "requirements-dev.txt"      refs/heads/main    REFUSE
arm "apps/model-runner to main is refused"       "apps/model-runner/e.py"    refs/heads/main    REFUSE
arm "docs/** to main is ALLOWED"                 "docs/guide.md"             refs/heads/main    ALLOW
arm "apps/other-app to main is ALLOWED"          "apps/other-app/e.py"       refs/heads/main    ALLOW
arm "bioengine/** to a FEATURE branch is ALLOWED" "bioengine/x.py"           refs/heads/fix/s   ALLOW

echo
echo "=== control: the refusal must come from the HOOK ==="
arm "bioengine/** to main, hook REMOVED -> allowed" "bioengine/x.py"         refs/heads/main    ALLOW   no-hook

echo
echo "RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
