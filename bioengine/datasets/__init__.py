"""BioEngine datasets — class API and the lazy module-level accessor.

This module wears two hats:

1. It still exports the ``BioEngineDatasets`` class for direct use and type
   annotations: ``from bioengine.datasets import BioEngineDatasets``.

2. It also acts as the **process-local dataset accessor**: user code in
   ``@bioengine.app`` deployments calls ``bioengine.datasets.list_datasets()``
   etc., and the module-level ``__getattr__`` below transparently delegates
   to a lazily-constructed singleton built from ``BIOENGINE_DATA_SERVER_URL``.

Standard attribute lookup wins: ``bioengine.datasets.BioEngineDatasets`` and
``bioengine.datasets.utils`` (the submodule) resolve by normal Python
machinery, so the delegation only fires for ``BioEngineDatasets`` instance
methods that aren't already module attributes.
"""

from bioengine.datasets.datasets import BioEngineDatasets


def __getattr__(name: str):
    """Delegate attribute access to the lazy process-local singleton.

    Anything that's a regular module attribute (the class, submodules
    that have been imported) is resolved before this fires.
    """
    from bioengine._app.accessors import _get_datasets

    instance = _get_datasets()
    try:
        return getattr(instance, name)
    except AttributeError as exc:
        raise AttributeError(
            f"module 'bioengine.datasets' has no attribute {name!r}"
        ) from exc


__all__ = ["BioEngineDatasets"]
