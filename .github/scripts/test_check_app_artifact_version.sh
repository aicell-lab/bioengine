#!/usr/bin/env bash
# Re-runnable demonstration of the apps/ version gate (backlog #0124).
#
# Each case builds throwaway commits with ``git commit-tree`` — no index, no
# hooks, no branch, nothing pushed — then runs the real gate script against
# them and asserts both the exit status and the message. Cases 1-5 hit the
# live Hypha server; cases 6-9 point --server-url at a local stub to show that
# a 404, a 5xx, an unreadable body and an unreachable host all fail the gate
# instead of reading as "version absent" or "version present".
#
#   bash .github/scripts/test_check_app_artifact_version.sh

set -uo pipefail
STUB_URL=
STUB_PID=
cd "$(dirname "$0")/../.."

GATE=.github/scripts/check_app_artifact_version.py
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"; [ -n "$STUB_PID" ] && kill "$STUB_PID" 2>/dev/null' EXIT
PASSED=0
FAILED=0

# A commit identical to $1 except that apps/$2/manifest.yaml has `version: $3`
# (and, when $4 is given, that extra line appended so the file differs even
# when the version does not).
synth() {
  local base=$1 app=$2 version=$3 extra=${4:-}
  local path="apps/$app/manifest.yaml" blob tree
  {
    git show "$base:$path" | sed -E "s/^version:.*/version: $version/"
    [ -n "$extra" ] && echo "$extra"
  } > "$WORK/manifest.yaml"
  blob=$(git hash-object -w "$WORK/manifest.yaml")
  GIT_INDEX_FILE="$WORK/index" git read-tree "$base"
  GIT_INDEX_FILE="$WORK/index" git update-index --cacheinfo "100644,$blob,$path"
  tree=$(GIT_INDEX_FILE="$WORK/index" git write-tree)
  git commit-tree "$tree" -p "$base" -m "synthetic: $app $version"
}

# A commit identical to $1 except that CLAUDE.md differs — no apps/ change.
synth_unrelated() {
  local base=$1 blob tree
  git show "$base:CLAUDE.md" > "$WORK/CLAUDE.md"
  echo "" >> "$WORK/CLAUDE.md"
  blob=$(git hash-object -w "$WORK/CLAUDE.md")
  GIT_INDEX_FILE="$WORK/index" git read-tree "$base"
  GIT_INDEX_FILE="$WORK/index" git update-index --cacheinfo "100644,$blob,CLAUDE.md"
  tree=$(GIT_INDEX_FILE="$WORK/index" git write-tree)
  git commit-tree "$tree" -p "$base" -m "synthetic: unrelated"
}

# expect <name> <wanted-status: pass|fail> <wanted-substring> -- <gate args...>
expect() {
  local name=$1 wanted=$2 needle=$3 status out
  shift 4
  out=$(python3 "$GATE" ${STUB_URL:+--server-url "$STUB_URL"} "$@" 2>&1)
  status=$?
  local got=pass
  [ "$status" -ne 0 ] && got=fail
  if [ "$got" = "$wanted" ] && echo "$out" | grep -qF -- "$needle"; then
    echo "  ok   $name (exit $status, matched: $needle)"
    PASSED=$((PASSED + 1))
  else
    echo "  FAIL $name — wanted $wanted containing '$needle', got $got (exit $status):"
    echo "$out" | sed 's/^/       | /'
    FAILED=$((FAILED + 1))
  fi
}

stub() { # stub <404|500|garbage>
  rm -f "$WORK/port"   # a leftover port file makes the wait below return the DEAD stub
  python3 - "$1" "$WORK/port" <<'PY' &
import http.server, pathlib, sys
mode, portfile = sys.argv[1], sys.argv[2]
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if mode == "404":
            self.send_error(404, "artifact not found")
            return
        if mode == "500":
            self.send_error(500, "internal error")
            return
        body = b'{"versions": "not-a-list"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass
srv = http.server.HTTPServer(("127.0.0.1", 0), H)
pathlib.Path(portfile).write_text(str(srv.server_port))
srv.serve_forever()
PY
  STUB_PID=$!
  for _ in $(seq 50); do [ -s "$WORK/port" ] && break; sleep 0.1; done
  STUB_URL="http://127.0.0.1:$(cat "$WORK/port")"
}

stub_stop() { kill "$STUB_PID" 2>/dev/null; wait "$STUB_PID" 2>/dev/null; STUB_PID=; STUB_URL=; }

HEAD_SHA=$(git rev-parse HEAD)
echo "Repository HEAD: $HEAD_SHA"
echo "apps/model-runner:   $(git show "$HEAD_SHA:apps/model-runner/manifest.yaml" | sed -nE 's/^version: //p')"
echo "apps/model-finetune: $(git show "$HEAD_SHA:apps/model-finetune/manifest.yaml" | sed -nE 's/^version: //p')"

echo
echo "Live Hypha server:"

# 1. Unbumped: manifest edited, version left alone. Leg 1 must fail.
UNBUMPED=$(synth "$HEAD_SHA" model-finetune 0.27.15 "# touched without a bump")
expect "unbumped manifest is rejected" fail "version (0.27.15) is not greater than" -- \
  --base-ref "$HEAD_SHA" --head-ref "$UNBUMPED" --main-ref "$HEAD_SHA"

# 2. Bumped to a version nobody uploaded. Leg 1 passes, leg 2 must fail.
UNUPLOADED=$(synth "$HEAD_SHA" model-finetune 0.99.0)
expect "bumped-but-unuploaded version is rejected" fail \
  "0.99.0 is not published to bioimage-io/model-finetune" -- \
  --base-ref "$HEAD_SHA" --head-ref "$UNUPLOADED" --main-ref "$HEAD_SHA"

# 3. The repo's real pre-existing condition: apps/model-runner/manifest.yaml
#    says 2.11.6 but the live artifact's newest is 2.11.5, so the first
#    model-runner PR fails leg 2 until someone uploads 2.11.6.
PRE_EXISTING=$(synth "$HEAD_SHA" model-runner 2.11.5)
expect "real apps/model-runner 2.11.6 fails the existence leg" fail \
  "2.11.6 is not published to bioimage-io/model-runner" -- \
  --base-ref "$PRE_EXISTING" --head-ref "$HEAD_SHA" --main-ref "$PRE_EXISTING"

# 4. Discriminator: a correctly prepared PR must PASS, or the two failures
#    above prove nothing. 0.27.15 is bumped over 0.27.14 and is published.
PREPARED=$(synth "$HEAD_SHA" model-finetune 0.27.14)
expect "correctly prepared PR passes both legs" pass \
  "0.27.15 is published to bioimage-io/model-finetune" -- \
  --base-ref "$PREPARED" --head-ref "$HEAD_SHA" --main-ref "$PREPARED"

# 5. A PR that touches no gated app is not gated at all.
UNRELATED=$(synth_unrelated "$HEAD_SHA")
expect "PR touching no gated app is skipped" pass "No gated app changed" -- \
  --base-ref "$HEAD_SHA" --head-ref "$UNRELATED" --main-ref "$HEAD_SHA"

echo
echo "Stubbed server (the existence leg must not fail toward green):"

stub 404
expect "404 artifact fails the gate" fail "does not exist" -- \
  --base-ref "$PREPARED" --head-ref "$HEAD_SHA" --main-ref "$PREPARED"
stub_stop

stub 500
expect "HTTP 500 fails the gate" fail "returned HTTP 500" -- \
  --base-ref "$PREPARED" --head-ref "$HEAD_SHA" --main-ref "$PREPARED"
stub_stop

stub garbage
expect "unreadable 200 payload fails the gate" fail "unreadable payload" -- \
  --base-ref "$PREPARED" --head-ref "$HEAD_SHA" --main-ref "$PREPARED"
stub_stop

STUB_URL="http://127.0.0.1:1"
expect "unreachable server fails the gate" fail "failed:" -- \
  --base-ref "$PREPARED" --head-ref "$HEAD_SHA" --main-ref "$PREPARED"
STUB_URL=

echo
echo "$PASSED passed, $FAILED failed"
[ "$FAILED" -eq 0 ]
