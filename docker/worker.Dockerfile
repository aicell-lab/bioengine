# Rolling tag — picks up current Debian-slim security patches each build.
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements-worker.txt /app/
RUN pip install -U pip && pip install -r requirements-worker.txt

COPY bioengine/ /app/bioengine/
COPY pyproject.toml README.md LICENSE /app/
RUN pip install --no-deps .

# Ray installed last so RAY_VERSION overrides don't invalidate earlier layers.
ARG RAY_VERSION=2.55.1
RUN pip install "ray[client,serve]==${RAY_VERSION}"

# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    PATH="/opt/venv/bin:$PATH"

# curl: helm chart startup/liveness probe. git: runtime_env git_url installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ARG RAY_VERSION=2.55.1
ENV BIOENGINE_RAY_VERSION=${RAY_VERSION}

WORKDIR /app

CMD [ "/bin/bash" ]
