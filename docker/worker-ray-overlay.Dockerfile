# Lightweight overlay over a published BioEngine worker image: swaps
# only the Ray pin without rebuilding system packages, Python, or the
# rest of the dependency tree. Pulls one image layer set, runs one pip
# install, sets one env var — done.
#
# Build:
#   docker build \
#       --build-arg BIOENGINE_IMAGE=ghcr.io/aicell-lab/bioengine-worker:<bioengine-version> \
#       --build-arg RAY_VERSION=<ray-version> \
#       -f docker/worker-ray-overlay.Dockerfile \
#       -t bioengine-worker:<bioengine-version>-ray<ray-version> .
#
# BIOENGINE_IMAGE: the published image to use as the base. Pin to a specific
#   tag for reproducible builds; `latest` is fine for ad-hoc work.
# RAY_VERSION:     the exact Ray release to swap in. Must satisfy the
#   range BioEngine supports (>=2.33.0, <3.0.0) — see pyproject.toml.

ARG BIOENGINE_IMAGE=ghcr.io/aicell-lab/bioengine-worker:latest
FROM ${BIOENGINE_IMAGE}

ARG RAY_VERSION=2.55.1
RUN pip install --no-cache-dir "ray[client,serve]==${RAY_VERSION}"

ENV BIOENGINE_RAY_VERSION=${RAY_VERSION}
