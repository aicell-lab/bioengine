# BioEngine worker image with the model-finetune app's dependencies baked in.
# Ray's runtime_env venv inherits the image's site-packages, so at deploy time
# every pin resolves as already satisfied and the app skips the multi-GB
# download/install that otherwise runs on first startup.
#
# Published as ONE general GHCR package, ghcr.io/aicell-lab/model-finetune, with
# a per-backend tag suffix (e.g. 0.21.0-cellpose). This file builds the CELLPOSE
# variant: the CPU-entry + Cellpose-runtime pins are preinstalled and
# MODEL_FINETUNE_BACKENDS=cellpose is baked in, so the micro-sam runtime is never
# composed and the app fits a single GPU.
#
# The micro-sam variant is a SEPARATE Dockerfile (docker/model-finetune-microsam.Dockerfile),
# not a parameterised flag off this one, because the two backends need different
# base pins: cellpose runs the whole image at numpy==1.26.4 / protobuf<5 (this
# file), while micro-sam needs numpy>=2 (its python-elf AIS decoder) and
# protobuf<6. One Dockerfile cannot bake both numpy majors, so there are two.
#
# Mirrors docker/model-runner.Dockerfile: installs the app requirements FIRST
# (the largest, least-changing layer), then worker requirements, the bioengine
# package, and Ray last. Keep in sync with worker.Dockerfile when the worker
# build changes.
#
# numpy: the Cellpose runtime pins numpy==1.26.4 (cellpose needs numpy 1.x); the
# entry declares an unpinned numpy. Installing both requirement files together
# resolves numpy to 1.26.4, which satisfies both and gives the entry and the
# Cellpose runtime a consistent numpy for array passing. requirements-worker.txt
# installs after and also pins 1.26.4, so there is no conflict.
#
# The image is versioned by the MODEL-FINETUNE APP version (from
# apps/model-finetune/manifest.yaml) — the app's pins are what it exists to
# preinstall. Each tag is built against one BioEngine version, recorded in the
# io.bioengine.version label / BIOENGINE_VERSION env.
#
# Build (from the repo root) via scripts/build_model_finetune.sh, which fills the
# version args from the checkout. Rebuild whenever
# apps/model-finetune/requirements-{entry,runtime-cellpose}.txt change, or the
# baked-in BioEngine code (bioengine/ package, requirements-worker.txt, or the
# Ray pin) changes; a BioEngine-only change still needs a new model-finetune app
# version to be publishable (the build script's push guard enforces it).

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

# Model-finetune Cellpose app dependencies — installed before everything else so
# this multi-GB layer survives worker-requirement and bioengine updates. The
# entry + Cellpose-runtime requirements are installed together so numpy resolves
# once to the Cellpose pin (1.26.4). requirements-worker.txt installs after and
# wins on any conflict.
COPY apps/model-finetune/requirements-entry.txt \
     apps/model-finetune/requirements-runtime-cellpose.txt \
     /app/model-finetune/
RUN pip install -U pip && \
    pip install -r model-finetune/requirements-entry.txt \
                -r model-finetune/requirements-runtime-cellpose.txt

# Worker requirements — intentionally does NOT pin Ray. Ray is installed as the
# very last step, controlled by the RAY_VERSION build arg, so changing the Ray
# version doesn't invalidate this layer on rebuild.
COPY requirements-worker.txt /app/
RUN pip install -r requirements-worker.txt

COPY bioengine/ /app/bioengine/
COPY pyproject.toml README.md LICENSE /app/

# Install the bioengine package without dependencies — all runtime deps are
# already in requirements-worker.txt.
RUN pip install --no-deps .

# Ray install — kept as the final step so RAY_VERSION can be overridden at build
# time without invalidating any prior layer cache. protobuf is re-constrained
# here because Ray's opentelemetry deps otherwise upgrade protobuf past what Ray
# Serve 2.55 supports.
ARG RAY_VERSION=2.55.1
RUN pip install "ray[client,serve]==${RAY_VERSION}" "protobuf>=4,<5"

ENV BIOENGINE_RAY_VERSION=${RAY_VERSION}

# Compose the Cellpose runtime only — the micro-sam runtime is never imported
# (entry.py reads this at import time), so no second GPU is held.
ENV MODEL_FINETUNE_BACKENDS=cellpose

# Version metadata last, so bumping either version rebuilds nothing but this
# layer.
ARG MODEL_FINETUNE_VERSION=unknown
ARG BIOENGINE_VERSION=unknown
ENV BIOENGINE_MODEL_FINETUNE_VERSION=${MODEL_FINETUNE_VERSION} \
    BIOENGINE_VERSION=${BIOENGINE_VERSION}
LABEL org.opencontainers.image.source=https://github.com/aicell-lab/bioengine \
      org.opencontainers.image.version=${MODEL_FINETUNE_VERSION} \
      io.bioengine.version=${BIOENGINE_VERSION}

CMD [ "/bin/bash" ]
