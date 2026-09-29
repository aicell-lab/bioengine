#!/usr/bin/env bash
# Foreground entry point for a daemon-supervised mount (svamp serve apply).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
if [ -f .demo-token ]; then
  export OMEZARR_VIEW_DEMO_TOKEN="$(cat .demo-token)"
fi
exec "${OMEZARR_VIEW_PYTHON:-/data/nmechtel/bioengine/.dev/omezarr-probe/.venv/bin/python}" \
  -m omezarr_view.server
