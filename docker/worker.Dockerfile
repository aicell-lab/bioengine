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
# CONSTRAINED, because this layer does not see requirements-worker.txt's pins. Ray pulls in
# virtualenv, virtualenv declares a platformdirs FLOOR, and pip is free to satisfy that floor
# by upgrading straight over the pin installed above -- silently, with the build still green,
# and the passwd-less-uid bug back on the next rebuild. A constraint makes pip ERROR instead.
#
# The constraint is EXTRACTED from the requirements file rather than restated, so the two
# cannot drift; `grep` exits non-zero and fails the build if the pin is ever removed, which is
# the case where a restated copy would have gone quietly out of date. The whole file cannot be
# used as a constraints file: it carries an extra (`httpx[http2]`), which pip rejects there.
RUN grep -E '^platformdirs' requirements-worker.txt > /tmp/platformdirs-constraint.txt && \
    pip install -c /tmp/platformdirs-constraint.txt "ray[client,serve]==${RAY_VERSION}" && \
    rm -f /tmp/platformdirs-constraint.txt

# THE REGRESSION GATE. The constraint above stops the one path we know about; this catches any
# other. It builds a venv the way Ray does -- empty environment, uid with NO passwd entry -- and
# a non-zero exit fails the BUILD rather than shipping an image whose first runtime_env deploy
# fails on a cluster. 61234 is arbitrary and deliberately absent from /etc/passwd; /tmp is 1777
# so it is writable. `env -i` clears PATH, hence the absolute interpreter.
USER 61234:61234
RUN env -i /usr/local/bin/python -m virtualenv /tmp/venv-passwdless-gate > /tmp/gate.log 2>&1 \
    || { echo "REGRESSION: a uid with no passwd entry can no longer build a venv -- the" >&2; \
         echo "platformdirs pin has been defeated, and no new runtime_env venv will build." >&2; \
         tail -20 /tmp/gate.log >&2; exit 1; }
USER root
RUN rm -rf /tmp/venv-passwdless-gate /tmp/gate.log

# Surface the active Ray version inside the image for diagnostics
ENV BIOENGINE_RAY_VERSION=${RAY_VERSION}

CMD [ "/bin/bash" ]
