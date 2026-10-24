"""An exported-but-empty environment variable means unset, everywhere.

The convention holds in four independent places already: click's own
``resolve_envvar_value`` returns a value only if it is truthy, the worker
launcher's ``_passthrough_env`` filters with ``if value``, ``resolve_token``
reads ``HYPHA_TOKEN`` then tests it for truth, and the CLI test fixtures pass
empty strings precisely to mean "not set".

``get_server_url`` was the one place that disagreed, because
``os.environ.get(name, default)`` only reaches its default when the key is
*absent*: an empty value is a present value, so the caller got ``""``. Masked
in practice by the click ``envvar=`` in front of it, which never forwards an
empty value — so this was latent rather than live, and the test is here to keep
it that way.

Found by the independent review of PR #229.
"""

import os

import pytest

from bioengine.cli.utils import DEFAULT_SERVER_URL, get_server_url, get_token

FLAG_URL = "https://flag.example.invalid"
ENV_URL = "https://env.example.invalid"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in ("BIOENGINE_SERVER_URL", "HYPHA_TOKEN", "BIOENGINE_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_an_empty_variable_falls_through_to_the_default(monkeypatch) -> None:
    """The regression. ``os.environ.get(name, default)`` cannot express this."""
    monkeypatch.setenv("BIOENGINE_SERVER_URL", "")
    assert get_server_url(None) == DEFAULT_SERVER_URL


def test_an_absent_variable_still_falls_through() -> None:
    assert get_server_url(None) == DEFAULT_SERVER_URL


def test_a_set_variable_is_used(monkeypatch) -> None:
    monkeypatch.setenv("BIOENGINE_SERVER_URL", ENV_URL)
    assert get_server_url(None) == ENV_URL


def test_the_flag_outranks_both(monkeypatch) -> None:
    monkeypatch.setenv("BIOENGINE_SERVER_URL", ENV_URL)
    assert get_server_url(FLAG_URL) == FLAG_URL


def test_an_empty_token_source_falls_through_to_the_next(monkeypatch) -> None:
    """``get_token``'s ``or`` chain already skipped empty values correctly."""
    monkeypatch.setenv("HYPHA_TOKEN", "")
    monkeypatch.setenv("BIOENGINE_TOKEN", "FALLBACK-NOT-A-CREDENTIAL")
    assert get_token(None) == "FALLBACK-NOT-A-CREDENTIAL"


def test_all_token_sources_empty_returns_none_not_empty_string(monkeypatch) -> None:
    """The same defect, one function over, found by writing this file's control.

    An ``or`` chain ends on its *last* operand, so with every source exported
    but empty ``get_token`` returned ``""`` while declaring ``Optional[str]``.
    A caller testing ``is None`` would have treated an absent token as present.
    """
    monkeypatch.setenv("HYPHA_TOKEN", "")
    monkeypatch.setenv("BIOENGINE_TOKEN", "")
    assert get_token(None) is None
