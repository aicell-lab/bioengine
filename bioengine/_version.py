"""Static version string for the ``bioengine`` package.

Ray Serve replicas import ``bioengine`` via ``runtime_env.py_modules``,
which ships only the module source directory — not the sibling
``bioengine-*.dist-info/METADATA`` file. That leaves
``importlib.metadata.metadata("bioengine")`` failing on the replica
side and ``__version__`` silently becoming ``None`` (test reports,
``get_version()`` responses, and cache-invalidation keys all lose
signal). Baking the version into a real source file sidesteps the
transport limitation — ``py_modules`` ships this file, so replicas
see the same string that ``pip`` sees.

Must stay in lock-step with ``pyproject.toml``'s ``version`` field.
The ``version-check.yml`` CI workflow enforces the match.
"""
__version__ = "0.11.23"
