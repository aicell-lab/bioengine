"""Numpy-backed helpers for RuntimeB.

This module is *not* a ``@bioengine.app`` host and is *not* imported at
top level from any decorator-bearing module, so it can freely
top-of-file-import heavy dependencies. RuntimeB pulls these helpers in
lazily inside method bodies — at that point we're already running on a
replica whose runtime_env installed ``numpy`` from RuntimeB's ``pip=``
declaration.
"""

import numpy as np


def numpy_version() -> str:
    return np.__version__


def stats(values: list) -> dict:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
        "count": len(arr),
        "sorted": sorted(values),
    }
