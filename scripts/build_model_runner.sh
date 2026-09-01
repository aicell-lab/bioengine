#!/bin/bash
#
# Build (and optionally publish) the model-runner image by hand.
#
# This is the BioEngine worker image with the model-runner app's pinned
# dependencies preinstalled (docker/model-runner.Dockerfile), so a
# worker launched from it starts the model-runner as a startup app
# without any deploy-time package installation. It is deliberately NOT
# built by docker-publish-worker.yml — build it on demand.
#
# Rebuild when either half of what is baked in changes:
#   * the app's pins — apps/model-runner/requirements-{entry,runtime}.txt
#   * the BioEngine code the app runs on — the bioengine/ package,
#     requirements-worker.txt, or the Ray pin
# A BioEngine-only change still needs a new model-runner app version:
# the tag is the app version, so there is no other way to publish it.
# The push guard below enforces that.
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
#   FORCE        set to 1 to overwrite an already-published tag
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

PUSH=""
[[ "${1:-}" == "--push" ]] && PUSH=1

# Refuse to overwrite a published tag. A tag is one immutable (app pins,
# BioEngine code) pair — silently replacing it means a cluster pinned to
# that tag gets different code on its next pull, with nothing in the
# version to show for it.
if [[ -n "$PUSH" && "${FORCE:-}" != "1" ]]; then
    if docker manifest inspect "$REF" >/dev/null 2>&1; then
        cat >&2 <<EOF
${REF} is already published.

Bump 'version' in apps/model-runner/manifest.yaml and re-run. This applies
even when only BioEngine changed: the tag is the app version, so a new
BioEngine build has no other way to be published.

FORCE=1 overwrites the tag — only for a build known to be byte-identical.
EOF
        exit 1
    fi
fi

echo "Building ${REF} from ${PROJECT_ROOT} (BioEngine ${BIOENGINE_VERSION})"
docker build \
    -f "$PROJECT_ROOT/docker/model-runner.Dockerfile" \
    -t "$REF" \
    "${BUILD_ARGS[@]}" \
    "$PROJECT_ROOT"

echo "Built ${REF}"

if [[ -n "$PUSH" ]]; then
    echo "Pushing ${REF}"
    docker push "$REF"
else
    echo "Not pushed. Re-run with --push to publish."
fi
