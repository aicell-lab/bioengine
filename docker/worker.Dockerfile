# Rolling tag — each build picks up current Debian-slim security patches.
FROM python:3.11-slim

# Set environment variables for installation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Hypha
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Set up working directory
WORKDIR /app

# Copy requirements first — note: this file intentionally does NOT pin
# Ray. Ray is installed as the very last step, controlled by the
# RAY_VERSION build arg, so changing the Ray version doesn't invalidate
# this layer (or any of the layers between here and the final Ray
# install) on rebuild.
COPY requirements-worker.txt /app/
RUN pip install -U pip && \
    pip install -r requirements-worker.txt

# Copy the rest of the application code
COPY bioengine/ /app/bioengine/
COPY pyproject.toml README.md LICENSE /app/

# Install the bioengine package without dependencies — all runtime deps
# (including ray's transitive deps that survive a version bump) are
# already in requirements-worker.txt.
RUN pip install --no-deps .

# ---------------------------------------------------------------------------
# Ray install — kept as the final step so RAY_VERSION can be overridden
# at build time without invalidating any prior layer cache. The default
# tracks the latest stable Ray release within the BioEngine-supported
# range (>=2.33.0, <3.0.0). To build against a different Ray:
#
#   docker build --build-arg RAY_VERSION=2.54.1 \
#                -f docker/worker.Dockerfile -t bioengine-worker:dev .
# ---------------------------------------------------------------------------
ARG RAY_VERSION=2.55.1
RUN pip install "ray[client,serve]==${RAY_VERSION}"

# Surface the active Ray version inside the image for diagnostics
ENV BIOENGINE_RAY_VERSION=${RAY_VERSION}

CMD [ "/bin/bash" ]
