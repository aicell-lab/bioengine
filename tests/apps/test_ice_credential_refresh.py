"""Pin the on-demand ICE credential rotation for long-running deployments.

Coturn on hypha.aicell.io hands out short-lived TURN credentials (default
TTL 1h, verified live). The pre-refresh code fetched them once at
_register_services time and never touched them again — so a
ProxyDeployment running for more than ~1h silently lost TURN relay:
existing WebRTC connections stayed up on the old credentials, but any
new peer request past the expiry hit the RTC offer path with expired
username/credential fields and negotiation failed silently.

The refresh flow:

1. ``_register_services`` stores the shared ``rtc_config`` dict on
   ``self._rtc_config`` and parses the expiry from the coturn response
   into ``self._ice_expires_at``.
2. Every health-check tick, ``check_health`` calls
   ``_refresh_ice_if_expiring``. If we're within
   ``_ICE_REFRESH_MARGIN_SECONDS`` (15 min) of expiry, it fetches a
   fresh list and mutates ``self._rtc_config["ice_servers"]`` in place.
3. hypha_rpc's ``register_rtc_service`` captured the same dict by
   reference inside its offer callback, so every subsequent WebRTC
   connection request reads the rotated credentials without any
   re-registration.

These tests pin the parser, the margin constant, the mutation semantics
(in place — replacing the dict would break the closure), and the
integration between ``check_health`` and ``_refresh_ice_if_expiring``.
"""
from __future__ import annotations

import inspect

from bioengine.apps import proxy_deployment as pd_module

# Aliases to make the test-reference formulas readable.
_ProxyDeployment = pd_module.ProxyDeployment.func_or_class


def test_refresh_margin_matches_coturn_typical_ttl_headroom() -> None:
    """15 min is enough headroom to survive a couple of failed fetches at
    the health-check cadence (10 s) even if coturn drops TTL to something
    like 30 min. Guard the number so a future contributor can't shorten
    it below the retry budget."""
    assert _ProxyDeployment._ICE_REFRESH_MARGIN_SECONDS >= 300


def test_parse_expiry_reads_coturn_username_prefix() -> None:
    """coturn returns entries with ``username`` = ``"<unix_ts>:<user>"``.
    The parser must extract the timestamp head, ignore the tail, and
    accept only decimal digits (Hypha's HMAC scheme never uses hex)."""
    servers = [
        {
            "urls": ["turns:turn.hypha.aicell.io:5349?transport=tcp"],
            "username": "1783046021:a-nils-mech-gmail-com",
            "credential": "0P1nU+FDW1RbFJFe1Qvlirfi0jo=",
        },
        {
            "urls": ["turn:turn.hypha.aicell.io:3478"],
            "username": "1783046021:a-nils-mech-gmail-com",
            "credential": "0P1nU+FDW1RbFJFe1Qvlirfi0jo=",
        },
    ]
    assert _ProxyDeployment._parse_ice_expiry(servers) == 1783046021.0


def test_parse_expiry_returns_none_for_custom_static_list() -> None:
    """When a deploy pins ``ice_servers=[...]`` via the constructor and
    they don't have a coturn-style timestamp username, the parser must
    return None so refresh becomes a no-op — nothing to refresh against."""
    servers = [
        {"urls": ["stun:stun.example.org:19302"]},
        {
            "urls": ["turn:turn.example.org:3478"],
            "username": "static-user",
            "credential": "static-pass",
        },
    ]
    assert _ProxyDeployment._parse_ice_expiry(servers) is None


def test_parse_expiry_returns_none_on_empty_list() -> None:
    assert _ProxyDeployment._parse_ice_expiry([]) is None
    assert _ProxyDeployment._parse_ice_expiry(None) is None


def test_register_services_stores_rtc_config_and_expiry() -> None:
    """_register_services must store self._rtc_config (for later
    mutation) and self._ice_expires_at (so refresh knows when to fire).
    Regressing either field silently disables refresh."""
    src = inspect.getsource(_ProxyDeployment._register_services)
    assert "self._rtc_config = rtc_config" in src, (
        "expected _register_services to hold a reference to rtc_config so "
        "_refresh_ice_if_expiring can mutate it in place."
    )
    assert "self._ice_expires_at" in src, (
        "expected _register_services to capture the parsed expiry timestamp."
    )


def test_check_health_calls_refresh() -> None:
    """The refresh hook must live inside check_health, at the end (after
    all other checks pass). Placing it elsewhere would either miss ticks
    (if it's under a conditional) or delay real health failures."""
    src = inspect.getsource(_ProxyDeployment.check_health)
    assert "_refresh_ice_if_expiring" in src, (
        "check_health must call _refresh_ice_if_expiring so long-running "
        "deployments rotate TURN credentials before they expire."
    )


def test_refresh_mutates_in_place_not_replaces() -> None:
    """Replacing self._rtc_config with a fresh dict would break the
    closure reference hypha_rpc captured at registration — clients would
    keep seeing the pre-refresh list forever. The refresh MUST mutate
    the existing dict's ``ice_servers`` key in place."""
    src = inspect.getsource(_ProxyDeployment._refresh_ice_if_expiring)
    assert 'self._rtc_config["ice_servers"] = new_ice' in src, (
        "refresh must mutate self._rtc_config['ice_servers'] in place — "
        "replacing the whole dict breaks the closure reference held by "
        "hypha_rpc.register_rtc_service."
    )
    # And must not reassign self._rtc_config itself.
    assert "self._rtc_config = " not in src.replace(
        "self._rtc_config = None", ""  # ignore any None-init line if present
    ), (
        "refresh must not reassign self._rtc_config; only mutate its "
        "['ice_servers'] entry."
    )


def test_refresh_is_noop_when_no_rtc_config() -> None:
    """WebRTC-disabled deploys (rtc_config never populated) must skip
    the refresh cleanly, not raise. Guard against a KeyError regression."""
    src = inspect.getsource(_ProxyDeployment._refresh_ice_if_expiring)
    # Match the guard shape — must return early on None.
    assert "self._rtc_config is None" in src


def test_deregister_clears_refresh_state() -> None:
    """After _deregister_services, the next registration must repopulate
    both rtc_config and expiry from scratch — leaving them set would
    make refresh operate on a dead RTC service and log confusing errors."""
    src = inspect.getsource(_ProxyDeployment._deregister_services)
    assert "self._rtc_config = None" in src
    assert "self._ice_expires_at = None" in src
