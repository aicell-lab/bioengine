"""Place a model's declared attachment files beside its imported architecture.

``bioimageio.core`` imports a file-sourced architecture by writing *only* that
one source file into a fresh temporary directory, so a file the architecture
opens relative to its own ``__file__`` is never there — no matter that the
package ships it. MitoNet 2D (``stupendous-sheep``) passes its TorchScript
graph that way (``kwargs: {model: MitoNet_v1.pth}``, declared under
``attachments``) and dies with "The provided filename ... does not exist".

Importing the architecture ahead of the pipeline populates ``sys.modules``
under the same sha-keyed module name core will look up, so core reuses the
module — and the directory it lives in, which by then holds the attachments.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Any, List

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
        if local is not None and local.is_file():
            os.symlink(local, destination)
        else:
            with open(destination, "wb") as f:
                shutil.copyfileobj(reader, f)
        staged.append(reader.original_file_name)
    return staged
