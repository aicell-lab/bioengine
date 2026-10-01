# BioEngine worker image for a micro-sam-ONLY model-finetune deployment — the
# micro-sam sibling of docker/model-finetune-cellpose.Dockerfile. Bakes the
# CPU-entry + micro-sam-runtime pins and sets MODEL_FINETUNE_BACKENDS=microsam,
# so a worker from it serves/fine-tunes only the micro-sam backend (all vit_*
# model types + AIS decoder + ONNX prompt decoder) on a single GPU with no
# deploy-time install. Published to the same general GHCR package as cellpose,
# ghcr.io/aicell-lab/model-finetune, with a -microsam tag suffix (e.g.
# 0.21.0-microsam).
#
# WHY A SEPARATE DOCKERFILE (not a flag off the cellpose one): the two backends
# need different numpy/protobuf majors in the base image, and one image can bake
# only one of each. Cellpose runs the whole image at numpy==1.26.4 / protobuf<5.
# micro-sam's python-elf (the AIS decoder) hard-requires numpy>=2, and onnx pulls
# protobuf which must stay <6 (Ray Serve 2.55 reads FieldDescriptor.label, dropped
# in protobuf 7) — so THIS image runs the whole thing at numpy 2.x / protobuf<6.
#
# OPEN VALIDATION QUESTION (confirm on the deNBI worker before relying on a tag):
# the STANDARD bioengine worker pins numpy==1.26.4 (requirements-worker.txt), and
# the micro-sam runtime returns arrays as hypha wire-dict bytes precisely because
# in the both-backends deploy the worker's ProxyDeployment runs numpy 1.x while
# the micro-sam runtime venv runs numpy 2.x (see apps/model-finetune
# requirements-runtime.txt + RuntimeApp._nd). This image removes that split by
# running EVERYTHING at numpy 2.x — proxy included — so the mismatch the wire-dict
# works around no longer exists. That is only valid if the bioengine worker + Ray
# 2.55 run correctly on numpy 2.x. If they do not, a micro-sam prebuilt image
# needs a different mechanism (an isolated runtime_env venv fed by a baked
# wheelhouse) rather than baking micro-sam into the base site-packages.
#
# Mirrors docker/model-runner.Dockerfile layer order, with one deliberate
# difference: worker requirements install FIRST here, then the micro-sam app
# layer, so python-elf's numpy>=2 cleanly upgrades the worker's 1.26.4 (two
# separate pip installs — the later >=2 is not blocked by the earlier ==). Ray is
# last so RAY_VERSION can change without invalidating the big micro-sam layer.
#
# Versioned by the MODEL-FINETUNE APP version (apps/model-finetune/manifest.yaml);
# the BioEngine version each tag is built against is baked as io.bioengine.version.
# Build via scripts/build_model_finetune.sh BACKEND=microsam. Rebuild whenever
# apps/model-finetune/requirements-{entry,runtime}.txt change, or the baked-in
# BioEngine code (bioengine/ package, requirements-worker.txt, or the Ray pin).

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

# Worker requirements first. Its numpy==1.26.4 is intentionally overridden by the
# micro-sam layer below (python-elf needs numpy>=2); the two installs are
# separate, so the later numpy>=2 upgrades cleanly rather than conflicting.
COPY requirements-worker.txt /app/
RUN pip install -U pip && pip install -r requirements-worker.txt

# Model-finetune micro-sam app dependencies — the CPU entry + micro-sam runtime.
# python-elf (AIS decoder) pulls numpy>=2, upgrading the worker's 1.26.4; protobuf
# resolves to <6. This is the largest layer.
COPY apps/model-finetune/requirements-entry.txt \
     apps/model-finetune/requirements-runtime.txt \
     /app/model-finetune/
RUN pip install -r model-finetune/requirements-entry.txt \
                -r model-finetune/requirements-runtime.txt

COPY bioengine/ /app/bioengine/
COPY pyproject.toml README.md LICENSE /app/

# Install the bioengine package without dependencies — all runtime deps are
# already satisfied above.
RUN pip install --no-deps .

# Ray install — kept last so RAY_VERSION can be overridden without invalidating
# the micro-sam layer. protobuf re-constrained to <6: onnx pulls protobuf
# transitively and protobuf 6's upb FieldDescriptor drops `.label`, which Ray
# 2.55.1's serve config _proto_to_dict still reads.
ARG RAY_VERSION=2.55.1
RUN pip install "ray[client,serve]==${RAY_VERSION}" "protobuf>=4,<6"

ENV BIOENGINE_RAY_VERSION=${RAY_VERSION}

# Compose the micro-sam runtime only — the Cellpose runtime is never imported
# (entry.py reads this at import time), so no second GPU is held.
ENV MODEL_FINETUNE_BACKENDS=microsam

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
