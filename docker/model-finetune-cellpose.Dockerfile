# BioEngine worker image for a Cellpose-ONLY model-finetune deployment:
# the model-finetune app's CPU-entry and Cellpose-runtime pins are
# preinstalled, and MODEL_FINETUNE_BACKENDS=cellpose is baked in so the
# micro-sam runtime is never composed. Ray's runtime_env venv inherits the
# image's site-packages, so at deploy time every Cellpose pin resolves as
# already satisfied and the app skips the multi-GB download/install that
# otherwise runs on first startup — letting a Cellpose-only site start fast
# and fit on a single GPU (no second GPU held for a micro-sam runtime).
#
# Mirrors docker/model-runner.Dockerfile: installs the app requirements
# FIRST (the largest, least-changing layer), then worker requirements,
# the bioengine package, and Ray last. Keep in sync with worker.Dockerfile
# when the worker build changes.
#
# Cellpose-specific note: the Cellpose runtime pins numpy==1.26.4 (cellpose
# requires numpy 1.x); the entry declares an unpinned numpy. Installing both
# requirement files together resolves numpy to 1.26.4, which satisfies both
# and gives the entry and the Cellpose runtime a consistent numpy for
# array passing. (This is exactly why Cellpose lives in its own runtime env
# in the both-backends deploy; a Cellpose-only image can bake one numpy.)
#
# The image is versioned by the MODEL-FINETUNE APP version (from
# apps/model-finetune/manifest.yaml) — the app's pins are what it exists to
# preinstall — and published as a SEPARATE GHCR package
# (ghcr.io/aicell-lab/model-finetune-cellpose) so it never pollutes the
# worker image's tag list. Each tag is built against one BioEngine version,
# recorded in the io.bioengine.version label / BIOENGINE_VERSION env.
#
# Build (from the repo root) via scripts/build_model_finetune_cellpose.sh,
# which fills the version args from the checkout. Rebuild whenever
# apps/model-finetune/requirements-{entry,runtime-cellpose}.txt change, or
# the baked-in BioEngine code (bioengine/ package, requirements-worker.txt,
# or the Ray pin) changes; a BioEngine-only change still needs a new
# model-finetune app version to be publishable (the build script's push
# guard enforces it).

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

# Model-finetune Cellpose app dependencies — installed before everything
# else so this multi-GB layer survives worker-requirement and bioengine
# updates. The entry + Cellpose-runtime requirements are installed together
# so numpy resolves once to the Cellpose pin (1.26.4). requirements-worker.txt
# is installed after and wins on any conflict.
COPY apps/model-finetune/requirements-entry.txt \
     apps/model-finetune/requirements-runtime-cellpose.txt \
     /app/model-finetune/
RUN pip install -U pip && \
    pip install -r model-finetune/requirements-entry.txt \
                -r model-finetune/requirements-runtime-cellpose.txt

# Worker requirements — intentionally does NOT pin Ray. Ray is installed as
# the very last step, controlled by the RAY_VERSION build arg, so changing
# the Ray version doesn't invalidate this layer on rebuild.
COPY requirements-worker.txt /app/
RUN pip install -r requirements-worker.txt

COPY bioengine/ /app/bioengine/
COPY pyproject.toml README.md LICENSE /app/

# Install the bioengine package without dependencies — all runtime deps are
# already in requirements-worker.txt.
RUN pip install --no-deps .

# Ray install — kept as the final step so RAY_VERSION can be overridden at
# build time without invalidating any prior layer cache. protobuf is
# re-constrained here because Ray's opentelemetry deps otherwise upgrade
# protobuf past what Ray Serve 2.55 supports.
ARG RAY_VERSION=2.55.1
RUN pip install "ray[client,serve]==${RAY_VERSION}" "protobuf>=4,<5"

ENV BIOENGINE_RAY_VERSION=${RAY_VERSION}

# Compose the Cellpose runtime only — the micro-sam runtime is never
# imported (entry.py reads this at import time), so no second GPU is held.
ENV MODEL_FINETUNE_BACKENDS=cellpose

# Version metadata last, so bumping either version rebuilds nothing but
# this layer.
ARG MODEL_FINETUNE_VERSION=unknown
ARG BIOENGINE_VERSION=unknown
ENV BIOENGINE_MODEL_FINETUNE_VERSION=${MODEL_FINETUNE_VERSION} \
    BIOENGINE_VERSION=${BIOENGINE_VERSION}
LABEL org.opencontainers.image.source=https://github.com/aicell-lab/bioengine \
      org.opencontainers.image.version=${MODEL_FINETUNE_VERSION} \
      io.bioengine.version=${BIOENGINE_VERSION}

CMD [ "/bin/bash" ]
