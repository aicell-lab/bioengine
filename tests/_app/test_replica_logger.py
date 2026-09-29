"""A logger bound at module scope must still reach the replica log.

``bioengine.logger`` resolves to ``ray.serve`` inside a replica and to
``bioengine.app`` everywhere else, on ``BIOENGINE_REPLICA``. Setting that env
var in the replica's runtime_env is necessary but not sufficient: Ray ships a
deployment class to its replica by value, so the user's module is imported in
the *build task* — where the var is deliberately absent — and the replica never
re-imports it. Measured on a live worker: import pid 3588, call pid 3649. So a
module-scope ``logger = bioengine.logger`` froze the build task's logger and
every record it was given vanished. Hence the proxy: the branch is decided per
call, wherever the assignment happens to live.
"""

from __future__ import annotations

import logging
import pickle

import pytest

import bioengine
from bioengine._app import accessors


@pytest.fixture(autouse=True)
def _restore_app_logger():
    """Undo the process-global configuration the fallback branch installs."""
    logger = logging.getLogger("bioengine.app")
    handlers = list(logger.handlers)
    level, propagate = logger.level, logger.propagate
    yield
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_replica_env_var_selects_the_ray_serve_logger(monkeypatch) -> None:
    monkeypatch.setenv("BIOENGINE_REPLICA", "1")
    assert accessors._get_logger() is logging.getLogger("ray.serve")


def test_falls_back_outside_a_replica(monkeypatch) -> None:
    monkeypatch.delenv("BIOENGINE_REPLICA", raising=False)
    assert accessors._get_logger() is logging.getLogger("bioengine.app")


def test_the_branch_is_re_evaluated_not_cached(monkeypatch) -> None:
    """``_setup_replica`` sets the env var late; a cached fallback outlives it."""
    monkeypatch.delenv("BIOENGINE_REPLICA", raising=False)
    assert accessors._get_logger().name == "bioengine.app"

    monkeypatch.setenv("BIOENGINE_REPLICA", "1")
    assert accessors._get_logger().name == "ray.serve"


def test_module_scope_access_in_a_replica_reaches_ray_serve(monkeypatch) -> None:
    monkeypatch.setenv("BIOENGINE_REPLICA", "1")
    assert bioengine.logger.name == "ray.serve"


def test_a_binding_made_outside_a_replica_still_logs_as_one(monkeypatch) -> None:
    """The reported bug: bound in the build task, used in the replica."""
    monkeypatch.delenv("BIOENGINE_REPLICA", raising=False)
    logger = bioengine.logger  # what `logger = bioengine.logger` captures
    assert logger.name == "bioengine.app"

    monkeypatch.setenv("BIOENGINE_REPLICA", "1")
    assert logger.name == "ray.serve"

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    serve_logger = logging.getLogger("ray.serve")
    serve_logger.addHandler(handler)
    try:
        logger.info("from a module-scope binding")
    finally:
        serve_logger.removeHandler(handler)

    assert [r.getMessage() for r in records] == ["from a module-scope binding"]


def test_the_proxy_survives_pickling_as_a_proxy(monkeypatch) -> None:
    """Ray pickles the user module's globals; resolving on the way out would
    reintroduce exactly the freeze this proxy exists to prevent."""
    monkeypatch.delenv("BIOENGINE_REPLICA", raising=False)
    revived = pickle.loads(pickle.dumps(bioengine.logger))

    monkeypatch.setenv("BIOENGINE_REPLICA", "1")
    assert revived.name == "ray.serve"


def test_the_proxy_forwards_the_whole_public_logger_api(monkeypatch) -> None:
    """``bioengine.logger`` is documented as a logger, so every public name a
    ``logging.Logger`` offers has to survive the indirection. Narrowing the
    proxy to a hand-picked set of methods would break app code silently."""
    monkeypatch.setenv("BIOENGINE_REPLICA", "1")
    proxy = bioengine.logger
    concrete = logging.getLogger("ray.serve")

    missing = [
        name
        for name in dir(concrete)
        if not name.startswith("_") and not hasattr(proxy, name)
    ]

    assert missing == []


def test_the_proxy_is_not_a_logging_logger_instance(monkeypatch) -> None:
    """A deliberate, caller-visible limitation: the proxy is not a subclass, so
    ``isinstance(bioengine.logger, logging.Logger)`` is false. No consumer in
    this repo does that check; this pins the fact so a future one is not
    surprised by it."""
    monkeypatch.setenv("BIOENGINE_REPLICA", "1")

    assert not isinstance(bioengine.logger, logging.Logger)


def test_the_fallback_logger_actually_emits_info(monkeypatch) -> None:
    """Unconfigured, ``bioengine.app`` inherits the root level of WARNING and
    has no handler — an ``INFO`` call is discarded outright."""
    monkeypatch.delenv("BIOENGINE_REPLICA", raising=False)
    logger = logging.getLogger("bioengine.app")
    logger.handlers = []
    logger.setLevel(logging.NOTSET)

    logger = accessors._get_logger()

    assert logger.isEnabledFor(logging.INFO)
    assert logger.handlers


def test_the_fallback_logger_is_configured_once(monkeypatch) -> None:
    monkeypatch.delenv("BIOENGINE_REPLICA", raising=False)
    logging.getLogger("bioengine.app").handlers = []

    first = accessors._get_logger()
    accessors._get_logger()

    assert len(first.handlers) == 1
