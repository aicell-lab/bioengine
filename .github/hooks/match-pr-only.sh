#!/usr/bin/env bash
# Print the PR-only paths among those given on stdin, one per line.
# The single place the policy in pr-only-paths.txt is interpreted.
set -uo pipefail

LIST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pr-only-paths.txt"
[ -r "$LIST" ] || { echo "match-pr-only: cannot read $LIST" >&2; exit 2; }

patterns=()
while IFS= read -r line; do
    line="${line%%#*}"; line="${line// /}"
    [ -n "$line" ] && patterns+=("$line")
done < "$LIST"
[ "${#patterns[@]}" -gt 0 ] || { echo "match-pr-only: no patterns" >&2; exit 2; }

while IFS= read -r path; do
    [ -z "$path" ] && continue
    for p in "${patterns[@]}"; do
        if [ "${p: -1}" = "/" ]; then
            [[ "$path" == "$p"* ]] && { echo "$path"; break; }
        else
            # shellcheck disable=SC2053
            [[ "$path" == $p ]] && { echo "$path"; break; }
        fi
    done
done
