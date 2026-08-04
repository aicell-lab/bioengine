# PoC-only worker image for container-as-runtime GPU apps (single-machine,
# ROOTFUL). Differs from worker.Dockerfile in three ways:
#   1. Ubuntu 24.04 base — ships podman 4.9.3 via apt (podman 5.x rejects
#      --pid=host on cgroup-v1 hosts, which Ray's container plugin requires).
#   2. Bundles nvidia-container-toolkit (nvidia-ctk + CDI hooks) so the worker
#      can emit a CDI spec at startup for nested-GPU passthrough.
#   3. Patches the installed Ray to drop the hardcoded `--userns=keep-id`
#      podman flag, which breaks /proc remount in a nested rootful container.
#
# This image is NOT a drop-in replacement for the production worker; it is the
# rootful PoC target for @bioengine.app(container_image=…). Pip runtime stays
# the default and is unaffected.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# System deps: build toolchain + git (framework), podman 4.9.3 + fuse-overlayfs
# + uidmap (nested container runtime). Python 3.11 via deadsnakes to match the
# app image's interpreter — the deployment class is cloudpickled from the worker
# (head) into the app-image replica, so the two Python versions must agree.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential curl gnupg ca-certificates software-properties-common \
        podman fuse-overlayfs uidmap \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# nvidia-container-toolkit: provides nvidia-ctk (CDI spec generation) and the
# nvidia-cdi-hook binaries (create-symlinks etc.).
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
 && curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list \
 && apt-get update && apt-get install -y --no-install-recommends nvidia-container-toolkit \
 && rm -rf /var/lib/apt/lists/*

# Isolated venv (Ubuntu 24.04 marks the system Python externally-managed).
RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app

# See worker.Dockerfile: requirements first (does NOT pin Ray), so the
# RAY_VERSION build arg can change without invalidating this layer.
COPY requirements-worker.txt /app/
RUN pip install -U pip && \
    pip install -r requirements-worker.txt

COPY bioengine/ /app/bioengine/
COPY pyproject.toml README.md LICENSE /app/
RUN pip install --no-deps .

ARG RAY_VERSION=2.55.1
RUN pip install "ray[client,serve]==${RAY_VERSION}"
ENV BIOENGINE_RAY_VERSION=${RAY_VERSION}

# Drop Ray's hardcoded `--userns=keep-id` podman flag. keep-id is a rootless
# volume-permission remap; combined with --pid=host in a nested ROOTFUL
# container it fails to remount /proc ("mount proc to proc: Operation not
# permitted"). The grep guard fails the build loudly if a Ray bump renames or
# removes the anchor line, so we never silently ship an unpatched image.
# The bare string `userns=keep-id` also appears in a doc comment (the example
# command), so the post-patch guard must check the QUOTED code anchor is gone,
# not the bare string.
RUN f="$(python -c 'import ray._private.runtime_env.image_uri as m; print(m.__file__)')" \
 && grep -q '"--userns=keep-id",' "$f" \
 && sed -i '/"--userns=keep-id",/d' "$f" \
 && ! grep -q '"--userns=keep-id",' "$f" \
 && echo "Patched out --userns=keep-id in $f"

CMD [ "/bin/bash" ]
