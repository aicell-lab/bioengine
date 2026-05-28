"""Shared helpers — demonstrates that cross-file imports now Just Work.

This module is imported from runtime_a, runtime_b, runtime_c, and entry
via relative imports (``from .utils import ...``). That was impossible in
the v0.5 ``exec()`` model where each file ran in an isolated globals dict.
"""

import datetime


def base_status(name: str, **extras) -> dict:
    """Compose a uniform status dict shared by all runtimes."""
    return {
        "name": name,
        "status": "ok",
        "now": datetime.datetime.now().isoformat(),
        **extras,
    }


def assert_pong(value: str) -> None:
    """Tiny assertion helper used by the entry's smoke test."""
    assert value == "pong", f"expected 'pong', got {value!r}"
