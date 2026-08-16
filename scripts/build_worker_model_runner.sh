#!/bin/bash
#
# Build (and optionally publish) the model-runner worker image by hand.
#
# This is the worker image with the model-runner app's pinned
# dependencies preinstalled (docker/worker-model-runner.Dockerfile), so
# a worker launched from it starts the model-runner as a startup app
# without any deploy-time package installation. It is deliberately NOT
# built by docker-publish-worker.yml — build it on demand when the
# worker or the model-runner requirements change.
#
# It is published as a SEPARATE GHCR package that shares the worker
# version, so the pairing is obvious but the two never pollute each
# other's tag list or scan findings. The tag is read from
# pyproject.toml in this checkout — the same version that gets baked
# into the image's bioengine package.
#
# Usage:
#   scripts/build_worker_model_runner.sh [--push]
#
# Environment overrides:
#   IMAGE        image name (default ghcr.io/aicell-lab/bioengine-worker-model-runner)
#   TAG          image tag  (default: version from pyproject.toml)
#   RAY_VERSION  Ray to bake (default: the Dockerfile's pinned version)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

VERSION="$(grep -E '^version\s*=' "$PROJECT_ROOT/pyproject.toml" \
    | sed -E 's/version\s*=\s*"(.*)"/\1/' | head -1)"

IMAGE="${IMAGE:-ghcr.io/aicell-lab/bioengine-worker-model-runner}"
TAG="${TAG:-$VERSION}"
REF="${IMAGE}:${TAG}"

BUILD_ARGS=()
if [[ -n "${RAY_VERSION:-}" ]]; then
    BUILD_ARGS+=(--build-arg "RAY_VERSION=${RAY_VERSION}")
fi

echo "Building ${REF} from ${PROJECT_ROOT}"
docker build \
    -f "$PROJECT_ROOT/docker/worker-model-runner.Dockerfile" \
    -t "$REF" \
    "${BUILD_ARGS[@]}" \
    "$PROJECT_ROOT"

echo "Built ${REF}"

if [[ "${1:-}" == "--push" ]]; then
    echo "Pushing ${REF}"
    docker push "$REF"
else
    echo "Not pushed. Re-run with --push to publish."
fi
