#!/bin/bash
#
# Build (and optionally publish) the model-runner image by hand.
#
# This is the BioEngine worker image with the model-runner app's pinned
# dependencies preinstalled (docker/model-runner.Dockerfile), so a
# worker launched from it starts the model-runner as a startup app
# without any deploy-time package installation. It is deliberately NOT
# built by docker-publish-worker.yml — build it on demand when the
# model-runner requirements change.
#
# It is published as a SEPARATE GHCR package so it never pollutes the
# worker image's tag list or scan findings. The tag is the MODEL-RUNNER
# APP version read from apps/model-runner/manifest.yaml: the app's pins
# are what the image preinstalls, so the app version is what a rebuild
# tracks. The BioEngine version that each tag is built against is read
# from pyproject.toml and baked in as the io.bioengine.version label —
# one image version, one pinned BioEngine version.
#
# Usage:
#   scripts/build_model_runner.sh [--push]
#
# Environment overrides:
#   IMAGE        image name (default ghcr.io/aicell-lab/model-runner)
#   TAG          image tag  (default: version from apps/model-runner/manifest.yaml)
#   RAY_VERSION  Ray to bake (default: the Dockerfile's pinned version)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_RUNNER_VERSION="$(grep -E '^version\s*:' "$PROJECT_ROOT/apps/model-runner/manifest.yaml" \
    | sed -E 's/version\s*:\s*"?([^"]*)"?/\1/' | head -1)"
BIOENGINE_VERSION="$(grep -E '^version\s*=' "$PROJECT_ROOT/pyproject.toml" \
    | sed -E 's/version\s*=\s*"(.*)"/\1/' | head -1)"

IMAGE="${IMAGE:-ghcr.io/aicell-lab/model-runner}"
TAG="${TAG:-$MODEL_RUNNER_VERSION}"
REF="${IMAGE}:${TAG}"

BUILD_ARGS=(
    --build-arg "MODEL_RUNNER_VERSION=${MODEL_RUNNER_VERSION}"
    --build-arg "BIOENGINE_VERSION=${BIOENGINE_VERSION}"
)
if [[ -n "${RAY_VERSION:-}" ]]; then
    BUILD_ARGS+=(--build-arg "RAY_VERSION=${RAY_VERSION}")
fi

echo "Building ${REF} from ${PROJECT_ROOT} (BioEngine ${BIOENGINE_VERSION})"
docker build \
    -f "$PROJECT_ROOT/docker/model-runner.Dockerfile" \
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
