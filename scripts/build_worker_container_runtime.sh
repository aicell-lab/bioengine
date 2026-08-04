#!/bin/bash
#
# Build the container-runtime worker image by hand.
#
# This image is the rootful, single-machine target for opt-in
# container-as-runtime GPU apps (@bioengine.app(container_image=…) +
# worker --enable-container-runtime). It is deliberately NOT built by
# docker-publish-worker.yml on every release — it is a niche, opt-in
# variant with a heavier base (ubuntu:24.04 for podman 4.9.3) whose CVE
# surface should stay out of the default worker's scan report. Build it
# on demand with this script when you actually need it.
#
# It is published as a SEPARATE GHCR package that shares the worker
# version, so the pairing is obvious but the two never pollute each
# other's tag list or scan findings.
#
# Usage:
#   scripts/build_worker_container_runtime.sh [--push]
#
# Environment overrides:
#   IMAGE        image name (default ghcr.io/aicell-lab/bioengine-worker-container-runtime)
#   TAG          image tag  (default: version from pyproject.toml)
#   RAY_VERSION  Ray to bake (default: the Dockerfile's pinned version)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

VERSION="$(grep -E '^version\s*=' "$PROJECT_ROOT/pyproject.toml" \
    | sed -E 's/version\s*=\s*"(.*)"/\1/' | head -1)"

IMAGE="${IMAGE:-ghcr.io/aicell-lab/bioengine-worker-container-runtime}"
TAG="${TAG:-$VERSION}"
REF="${IMAGE}:${TAG}"

BUILD_ARGS=()
if [[ -n "${RAY_VERSION:-}" ]]; then
    BUILD_ARGS+=(--build-arg "RAY_VERSION=${RAY_VERSION}")
fi

echo "Building ${REF} from ${PROJECT_ROOT}"
docker build \
    -f "$PROJECT_ROOT/docker/worker-container-runtime.Dockerfile" \
    -t "$REF" \
    "${BUILD_ARGS[@]}" \
    "$PROJECT_ROOT"

echo "Built ${REF}"

if [[ "${1:-}" == "--push" ]]; then
    echo "Pushing ${REF}"
    docker push "$REF"
else
    echo "Not pushed. Re-run with --push to publish, or for a local"
    echo "single-machine worker load it into the rootful runtime directly."
fi
