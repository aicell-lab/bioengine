#!/bin/bash
#
# Build (and optionally publish) a model-finetune prebuilt worker image.
#
# This is the BioEngine worker image with the model-finetune app's dependencies
# preinstalled, so a worker launched from it starts the app with no deploy-time
# package install. It is deliberately NOT built by docker-publish-worker.yml —
# build it on demand, like the model-runner image.
#
# ONE general GHCR package (ghcr.io/aicell-lab/model-finetune) with a per-backend
# tag suffix: <app-version>-<backend>, e.g. 0.21.0-cellpose. Only the CELLPOSE
# variant is buildable today (docker/model-finetune.Dockerfile, single-GPU,
# micro-sam disabled). A micro-sam variant needs a different base (numpy>=2 vs
# the worker's 1.26.4 pin — see the Dockerfile header) and is not yet wired here.
#
# Rebuild when either half of what is baked in changes:
#   * the app's pins — apps/model-finetune/requirements-{entry,runtime-cellpose}.txt
#   * the BioEngine code the app runs on — the bioengine/ package,
#     requirements-worker.txt, or the Ray pin
# A BioEngine-only change still needs a new model-finetune app version: the tag
# is the app version, so there is no other way to publish it. The push guard
# below enforces that.
#
# Usage:
#   scripts/build_model_finetune.sh [--push]
#
# Environment overrides:
#   BACKEND      backend variant (default cellpose; only cellpose supported today)
#   IMAGE        image name (default ghcr.io/aicell-lab/model-finetune)
#   TAG          image tag  (default: <manifest-version>-<backend>)
#   RAY_VERSION  Ray to bake (default: the Dockerfile's pinned version)
#   FORCE        set to 1 to overwrite an already-published tag
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BACKEND="${BACKEND:-cellpose}"
if [[ "$BACKEND" != "cellpose" ]]; then
    echo "Unsupported BACKEND='$BACKEND'. Only 'cellpose' is buildable today; a" >&2
    echo "micro-sam variant needs a different base (see docker/model-finetune.Dockerfile)." >&2
    exit 2
fi
DOCKERFILE="$PROJECT_ROOT/docker/model-finetune.Dockerfile"

MODEL_FINETUNE_VERSION="$(grep -E '^version\s*:' "$PROJECT_ROOT/apps/model-finetune/manifest.yaml" \
    | sed -E 's/version\s*:\s*"?([^"]*)"?/\1/' | head -1)"
BIOENGINE_VERSION="$(grep -E '^version\s*=' "$PROJECT_ROOT/pyproject.toml" \
    | sed -E 's/version\s*=\s*"(.*)"/\1/' | head -1)"

IMAGE="${IMAGE:-ghcr.io/aicell-lab/model-finetune}"
TAG="${TAG:-${MODEL_FINETUNE_VERSION}-${BACKEND}}"
REF="${IMAGE}:${TAG}"

BUILD_ARGS=(
    --build-arg "MODEL_FINETUNE_VERSION=${MODEL_FINETUNE_VERSION}"
    --build-arg "BIOENGINE_VERSION=${BIOENGINE_VERSION}"
)
if [[ -n "${RAY_VERSION:-}" ]]; then
    BUILD_ARGS+=(--build-arg "RAY_VERSION=${RAY_VERSION}")
fi

PUSH=""
[[ "${1:-}" == "--push" ]] && PUSH=1

# Refuse to overwrite a published tag — a tag is one immutable (app pins,
# BioEngine code) pair; silently replacing it hands a cluster pinned to that tag
# different code on its next pull, with nothing in the version to show it.
if [[ -n "$PUSH" && "${FORCE:-}" != "1" ]]; then
    if docker manifest inspect "$REF" >/dev/null 2>&1; then
        cat >&2 <<EOF
${REF} is already published.

Bump 'version' in apps/model-finetune/manifest.yaml and re-run. This applies
even when only BioEngine changed: the tag is the app version, so a new BioEngine
build has no other way to be published.

FORCE=1 overwrites the tag — only for a build known to be byte-identical.
EOF
        exit 1
    fi
fi

echo "Building ${REF} from ${PROJECT_ROOT} (BioEngine ${BIOENGINE_VERSION}, backend ${BACKEND})"
docker build \
    -f "$DOCKERFILE" \
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
