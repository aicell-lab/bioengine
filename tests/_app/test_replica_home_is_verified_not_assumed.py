"""A replica must prove its HOME was applied, not report what it intended to set.

A fresh `model-finetune` replica served with HOME still at the image's read-only `/home/ray`,
refused all 12 model types, and could not start any training. The only evidence the replica
emitted was this, from the same replica id:

    BioEngine: replica environment ready (source=..., HOME=.../home, TMPDIR=.../tmp)

That line was printed from the LOCAL variables the function had just assigned. It reported
intent. So the log agreed with the fix while the replica disagreed with both, and the first real
symptom was `OSError: [Errno 30] Read-only file system` from the app's first write — two steps
from the cause, in a different component, with a path the log said was not in use.

These arms pin the two properties that make that impossible: the applied environment is READ BACK
and checked, and HOME is proven WRITABLE rather than merely correctly named.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bioengine._app import replica_init


@pytest.fixture
def hook_env(monkeypatch, tmp_path):
    """Minimal env for ``setup_replica_environment``; the sync is stubbed per test."""
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    monkeypatch.setenv("BIOENGINE_APP_DIR", str(app_dir))
    monkeypatch.setenv("BIOENGINE_ARTIFACT_VERSION", "1.2.3")
    monkeypatch.setenv("BIOENGINE_ARTIFACT_ID", "ws-test/my-app")
    monkeypatch.setenv("HOME", "/home/ray")  # the read-only default being escaped
    monkeypatch.setattr(
        replica_init, "_ensure_source", lambda app_dir, version, logger: app_dir / "source"
    )
    return app_dir


def test_a_correctly_applied_HOME_succeeds(hook_env) -> None:
    """The normal path still works, so these arms are not simply refusing everything."""
    replica_init.setup_replica_environment()
    assert os.environ["HOME"] == str(hook_env / "home")
    assert Path(os.path.expanduser("~")).resolve() == (hook_env / "home").resolve()


def test_the_ready_LINE_is_logged_and_names_the_resolved_path(hook_env, caplog) -> None:
    """Behavioural half: the line exists and carries the resolved path."""
    import logging

    caplog.set_level(logging.INFO, logger="ray.serve")
    replica_init.setup_replica_environment()
    ready = [r.message for r in caplog.records if "replica environment ready" in r.message]
    assert ready, "no ready line was logged at all"
    assert "home_resolves_to=" in ready[-1], (
        "the ready line does not report what Path.home() resolves to, which is the fact that "
        "was missing when a replica served on /home/ray while logging the right path"
    )


def test_the_ready_LINE_is_BUILT_from_the_read_back_not_the_local(hook_env) -> None:
    """Structural, and it has to be.

    I first wrote this behaviourally — log the line, assert it contains `os.environ['HOME']`.
    A mutant that reverted the f-string to the local `home` SURVIVED it: on the happy path the
    two are the same string, so the assertion passed either way. And there is no behavioural
    state where they differ *and* the line is still logged, because a mismatch now raises
    before the log. So the only way to pin where the value comes from is to read the source.
    """
    import inspect

    src = inspect.getsource(replica_init.setup_replica_environment)
    # rindex, not index: the phrase also appears in the explanatory comment ABOVE the checks,
    # and anchoring there swept in the RuntimeError message — which legitimately contains
    # `HOME={home}` — so this arm failed on the clean tree. The LAST occurrence is the log call.
    ready = src[src.rindex("replica environment ready") :]
    assert "os.environ.get('HOME')" in ready or 'os.environ.get("HOME")' in ready, (
        "the ready line interpolates the local variable rather than the applied environment; "
        "that is exactly what let a replica log the correct HOME while serving on /home/ray"
    )
    assert "HOME={home}" not in ready, (
        "the ready line still prints the intended local `home`"
    )


def test_an_unapplied_HOME_RAISES_at_startup_instead_of_serving(
    monkeypatch, hook_env
) -> None:
    """The reported failure mode: the assignment does not take effect.

    Simulated at the boundary the code trusts — `expanduser` — because that is what
    `Path.home()` consults and what the app's first write would resolve through. A replica in
    this state must not reach the point of serving.
    """
    monkeypatch.setattr(os.path, "expanduser", lambda p: "/home/ray")
    with pytest.raises(RuntimeError) as caught:
        replica_init.setup_replica_environment()
    msg = str(caught.value)
    assert "HOME was not applied" in msg
    assert "/home/ray" in msg, "the refusal does not name the path actually in use"


def test_a_correctly_named_but_UNWRITABLE_HOME_also_raises(monkeypatch, hook_env) -> None:
    """Resolving correctly is not the property the app needs.

    An NFS mount can resolve exactly right and still refuse writes, which is the `Errno 30`
    the user saw. So the probe writes, and a failure is refused here rather than surfacing from
    the app's first status write.
    """
    real_write = Path.write_text

    def _refuse(self, *a, **k):
        if ".bioengine-home-probe-" in self.name:
            raise OSError(30, "Read-only file system")
        return real_write(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", _refuse)
    with pytest.raises(RuntimeError) as caught:
        replica_init.setup_replica_environment()
    assert "not writable" in str(caught.value)
    assert "Errno 30" in str(caught.value) or "Read-only" in str(caught.value)


def test_the_write_probe_leaves_nothing_behind(hook_env) -> None:
    """A probe that accumulated one file per replica start would be a slow leak on a shared
    mount, so it is removed even though the happy path is the common one."""
    replica_init.setup_replica_environment()
    leftovers = list((hook_env / "home").glob(".bioengine-home-probe-*"))
    assert leftovers == [], f"probe files left behind: {leftovers}"


def test_the_stale_flock_diagnosis_is_no_longer_stated_as_current() -> None:
    """The row that reported this bug was filed by reading a docstring.

    `_acquire_claim` described a lock failure as leaving HOME unset — true before the ordering
    was inverted, false in the same commit that inverted it. It was quoted verbatim as "the
    mechanism, named rather than guessed". A comment that contradicts its own file is a defect
    with a real cost, so this asserts the claim is marked historical.
    """
    doc = replica_init._acquire_claim.__doc__ or ""
    assert "No locks available" in doc, "the lock hazard should still be documented"
    assert "HISTORICAL" in doc, (
        "the paragraph about HOME being unset must be marked historical, or it will be read "
        "as a live diagnosis again"
    )
    assert "sets ``HOME`` and" in doc or "sets `HOME`" in doc, (
        "the docstring should say what the code does now, not only that the old claim was wrong"
    )
