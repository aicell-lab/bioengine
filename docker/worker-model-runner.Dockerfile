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
# Build (from the repo root):
#   docker build \
#       -f docker/worker-model-runner.Dockerfile \
#       -t bioengine-worker-model-runner:<bioengine-version> .
#
# Rebuild whenever apps/model-runner/requirements-*.txt change — the
# preinstall only short-circuits the deploy-time install while the pins
# match the deployed app version.

# Rolling tag — each build picks up current Debian-slim security patches.
FROM python:3.11-slim

# Links the GHCR package to this repo (CI injects this for the published
# worker image; this image is built manually, so it's baked in here).
LABEL org.opencontainers.image.source=https://github.com/aicell-lab/bioengine

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

# micro-sam is an inference-time dependency of BioImage.IO packages exported
# with an AIS decoder: their vendored predictor_adaptor.py imports
# micro_sam.instance_segmentation whenever the checkpoint carries a
# decoder_state. It cannot live in requirements-runtime.txt — python-elf
# declares numpy>=2.0 against the tensorflow-mandated numpy<2.0, so a normal
# resolve is impossible, and pip rejects --no-deps inside a requirements file.
# That numpy floor is not real for the code paths used here; the imports work
# on the pinned 1.26.4.
RUN pip install --no-deps \
        micro-sam==1.8.11 \
        torch-em==0.10.3 \
        python-elf==0.9.2 && \
    pip install bioimage_cpp kornia pooch xxhash

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

CMD [ "/bin/bash" ]
