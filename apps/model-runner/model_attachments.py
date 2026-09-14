"""Compatibility shim for models that misuse ``attachments`` as a weights channel.

DEPRECATED. Scheduled for removal once ``stupendous-sheep`` is republished; it
is the only model in the zoo that needs this (276 scanned, 27 declare both a
file-sourced architecture and attachments, 26 of those never open the
attachment at import).

``attachments`` is not a weights channel and passing a weights path to an
architecture's ``__init__`` is not a supported pattern — a conforming model
declares its weights under ``weights.<format>.source`` and lets
``bioimageio.core`` load them, which needs no filesystem access at all. See
bioimage-io/core-bioimage-io-python#503.

MitoNet 2D (``stupendous-sheep``) does neither: it declares its TorchScript
graph under ``attachments``, passes the filename as ``kwargs: {model:
MitoNet_v1.pth}``, and ``torch.jit.load``s it from a path built off the
architecture's own ``__file__``. Core imports a file-sourced architecture into
a fresh directory holding only that ``.py``, so the join cannot resolve and the
model fails to load.

This module works around that by importing the architecture ahead of the
pipeline, which populates ``sys.modules`` under the sha-keyed name core will
look up, so core reuses the module — and the directory it lives in, which by
then holds the attachments. Do not extend it to new models: fix the model.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)

from bioimageio.core.digest_spec import import_callable
from bioimageio.spec.utils import get_reader


def _file_architecture(model_description: Any) -> Any:
    """The architecture node of the pytorch weights, if it is file-sourced.

    A library architecture already lives on ``sys.path`` next to whatever it
    needs, and must never be written into.
    """
    weights = getattr(model_description.weights, "pytorch_state_dict", None)
    architecture = getattr(weights, "architecture", None)
    # ``source_file`` on v0.4's CallableFromFile, ``source`` on v0.5's
    # ArchitectureFromFileDescr.
    if hasattr(architecture, "source_file") or hasattr(architecture, "source"):
        return architecture
    return None


def _attachment_sources(model_description: Any) -> List[Any]:
    attachments = getattr(model_description, "attachments", None)
    if attachments is None:
        return []
    files = getattr(attachments, "files", None)  # v0.4
    if files is not None:
        return list(files)
    return [getattr(a, "source", a) for a in attachments]  # v0.5


def stage_architecture_attachments(model_description: Any) -> List[str]:
    """Import the architecture and link every attachment next to its module.

    Best-effort: a model served from non-pytorch weights may not import at all,
    and failing to stage only leaves the historical behaviour in place.
    Returns the names staged.
    """
    architecture = _file_architecture(model_description)
    sources = _attachment_sources(model_description)
    if architecture is None or not sources:
        return []

    module = sys.modules.get(import_callable(architecture).__module__)
    module_file = Path(getattr(module, "__file__", "") or "")
    if not module_file.exists():
        return []
    module_dir = module_file if module_file.is_dir() else module_file.parent

    staged = []
    for source in sources:
        reader = get_reader(source)
        destination = module_dir / reader.original_file_name
        if destination.exists():
            continue
        root = getattr(reader, "original_root", None)
        local = Path(root, reader.original_file_name) if isinstance(root, Path) else None
        # Symlink only. Copying would spend a full download on every model that
        # declares an attachment its architecture never opens — 26 of the 27 in
        # the zoo, including six micro-SAM models attaching 100+ MB decoders.
        if local is None or not local.is_file():
            continue
        os.symlink(local, destination)
        staged.append(reader.original_file_name)

    if staged:
        logger.warning(
            "DEPRECATED: staged %s beside the architecture of %r. This model "
            "reads a declared attachment at import time, which the spec does "
            "not support (core-bioimage-io-python#503); it should declare the "
            "file under weights.<format>.source instead. This shim will be "
            "removed — the model needs fixing, not the runner.",
            ", ".join(staged),
            getattr(model_description, "id", None) or getattr(model_description, "name", "?"),
        )
    return staged
