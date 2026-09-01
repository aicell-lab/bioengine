# BioEngine worker image with the model-runner app's pinned dependencies
# preinstalled. Ray's runtime_env venv inherits the image's site-packages,
# so at deploy time every pin resolves as already satisfied and the app
# skips the multi-GB download/install that otherwise runs on first startup.
#
# Mirrors docker/worker.Dockerfile, but installs the model-runner
# requirements FIRST: they are by far the largest layer and change the
# least, so worker-requirement bumps and bioengine code updates rebuild
# only the cheap layers on top. Keep the two files in sync when the
# worker build changes.
#
# The image is versioned by the MODEL-RUNNER APP version (from
# apps/model-runner/manifest.yaml), not by the BioEngine version — the
# app's pins are what this image exists to preinstall, so they are what
# a rebuild is triggered by. Each published tag is built against one
# specific BioEngine version, recorded in the io.bioengine.version label
# and the BIOENGINE_VERSION env var since the tag no longer carries it.
#
# Build (from the repo root) via scripts/build_model_runner.sh, which
# fills both version args from the checkout. By hand:
#   docker build \
#       -f docker/model-runner.Dockerfile \
#       --build-arg MODEL_RUNNER_VERSION=<app-version> \
#       --build-arg BIOENGINE_VERSION=<bioengine-version> \
#       -t model-runner:<app-version> .
#
# Rebuild whenever apps/model-runner/requirements-*.txt change — the
# preinstall only short-circuits the deploy-time install while the pins
# match the deployed app version.

# Rolling tag — each build picks up current Debian-slim security patches.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

# Model-runner app dependencies — installed before everything else so
# this multi-GB layer survives worker-requirement and bioengine updates.
# requirements-worker.txt is installed after and wins on any conflict.
COPY apps/model-runner/requirements-entry.txt \
     apps/model-runner/requirements-runtime.txt \
     /app/model-runner/
RUN pip install -U pip && \
    pip install -r model-runner/requirements-entry.txt \
                -r model-runner/requirements-runtime.txt

# Worker requirements — intentionally does NOT pin Ray. Ray is installed
# as the very last step, controlled by the RAY_VERSION build arg, so
# changing the Ray version doesn't invalidate this layer (or any of the
# layers between here and the final Ray install) on rebuild.
COPY requirements-worker.txt /app/
RUN pip install -r requirements-worker.txt

COPY bioengine/ /app/bioengine/
COPY pyproject.toml README.md LICENSE /app/

# Install the bioengine package without dependencies — all runtime deps
# (including ray's transitive deps that survive a version bump) are
# already in requirements-worker.txt.
RUN pip install --no-deps .

# Ray install — kept as the final step so RAY_VERSION can be overridden
# at build time without invalidating any prior layer cache. protobuf is
# re-constrained here: tensorflow (model-runner) needs <5, and without
# the pin Ray's opentelemetry deps upgrade protobuf to 7, which breaks
# both tensorflow and Ray Serve 2.55.
ARG RAY_VERSION=2.55.1
RUN pip install "ray[client,serve]==${RAY_VERSION}" "protobuf>=4,<5"

# Surface the active Ray version inside the image for diagnostics
ENV BIOENGINE_RAY_VERSION=${RAY_VERSION}

# Version metadata last, so bumping either version rebuilds nothing but
# this layer. org.opencontainers.image.source links the GHCR package to
# this repo (CI injects it for the published worker image; this image is
# built manually, so it's baked in here).
ARG MODEL_RUNNER_VERSION=unknown
ARG BIOENGINE_VERSION=unknown
ENV BIOENGINE_MODEL_RUNNER_VERSION=${MODEL_RUNNER_VERSION} \
    BIOENGINE_VERSION=${BIOENGINE_VERSION}
LABEL org.opencontainers.image.source=https://github.com/aicell-lab/bioengine \
      org.opencontainers.image.version=${MODEL_RUNNER_VERSION} \
      io.bioengine.version=${BIOENGINE_VERSION}

CMD [ "/bin/bash" ]
