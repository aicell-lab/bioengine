"""A replica that cannot write its run store must say so, not look healthy.

Measured 2026-11-25 on de.NBI: a `model-finetune` replica whose `Path.home()`
resolved to `/home/ray` on a read-only filesystem reported RUNNING with every
deployment HEALTHY, while `start_training`, `list_training_sessions` and
`get_training_status` all failed with an identical bare `OSError` out of the
`mkdir` in `sessions_root`. `get_training_capabilities` still answered, which is
why nothing looked wrong from outside.

Two things are pinned here:

* the failure is NAMED and says what it means for the app, instead of being a
  bare filesystem error repeated at a dozen call sites with nothing tying them
  together;
* the health check FAILS on it, so Serve takes the replica out of the routing
  table instead of leaving it to answer every run call with an exception.

The second is the one that matters. A clearer error on a replica nobody removes
is a better-labelled outage, not a fix.

HOW THE UNWRITABLE STORE IS MODELLED, AND WHY IT IS NOT A CHMOD. The first
version of this file made a directory mode 0o555 and asserted the write failed.
That works for an ordinary user and does NOT work for root, which ignores the
permission bits -- and CI runs as root. Measured: all seven tests resting on it
failed in CI with `DID NOT RAISE`, while passing locally under an unprivileged
uid. The file's own fixture-blocks-a-write controls are what reported it, which
is exactly their job, but the failure mode is worth naming: a permission-based
fixture is silently inert in the one environment that gates the merge.

So the refusal is injected at the syscall boundary instead, with the errno the
production fault actually raised -- EROFS, a read-only filesystem -- and scoped
to paths under the fake home. That is environment-independent, and it models the
measured fault more precisely than a mode change does. The permission-based
arms are KEPT as additional evidence that a real filesystem refusal reaches the
same code path, and they skip where they cannot bite rather than passing
vacuously there.
"""

import errno
import os
import stat
import time
from pathlib import Path

import pytest

import training

# CI runs as root, where a chmod cannot make a directory unwritable. The
# permission-based arms below are skipped there rather than passing for the
# wrong reason; the injected arms carry the behaviour in every environment.
root_cannot_be_denied = pytest.mark.skipif(
    os.geteuid() == 0,
    reason=(
        "running as root, which ignores permission bits: a chmod-based "
        "fixture cannot deny a write here, so this arm would pass vacuously. "
        "The EROFS-injecting arms cover the behaviour."
    ),
)


def _refuse_under(root: Path, method_name: str, monkeypatch):
    """Make one `Path` method raise EROFS for anything under *root*.

    Injected at the boundary the code under test actually calls, with the errno
    the production fault raised, so the error the app sees is the error it saw
    on the cluster. Scoped to *root* so the rest of the process -- pytest's own
    tmp handling included -- is untouched.
    """
    real = getattr(Path, method_name)

    def refusing(self, *args, **kwargs):
        if self == root or root in self.parents:
            raise OSError(errno.EROFS, os.strerror(errno.EROFS), str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, method_name, refusing)


@pytest.fixture()
def unwritable_home(tmp_path, monkeypatch):
    """A home directory that exists, can be listed, and refuses a `mkdir`.

    That is the real shape of the failure -- a readable path the store cannot be
    created under -- rather than a path that simply does not exist. The refusal
    is EROFS, injected at `Path.mkdir`: the same errno the production replica's
    read-only home raised, and the only way to get it in an environment that may
    be running as root. See the module docstring.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    _refuse_under(home, "mkdir", monkeypatch)
    return home


@pytest.fixture()
def unwritable_home_by_permission(tmp_path, monkeypatch):
    """The same condition produced by a real permission bit, not an injection.

    Kept so that something asserts a genuine filesystem refusal reaches the same
    code path. Only meaningful for an unprivileged uid, hence
    `root_cannot_be_denied` on every test that uses it.
    """
    home = tmp_path / "home"
    home.mkdir()
    os.chmod(home, stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    yield home
    # Restored so pytest can clean the tmp tree up.
    os.chmod(home, stat.S_IRWXU)


@pytest.fixture()
def writable_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    return home


def test_the_fixture_really_blocks_a_write(unwritable_home):
    """THE CONTROL FOR THE FILE. If the store can be created after all, every
    assertion below passes for the wrong reason, so the refusal is asserted
    directly first -- and by errno, not merely as "some OSError", so that the
    fault being modelled is the measured one and not an unrelated failure."""
    with pytest.raises(OSError) as exc:
        (unwritable_home / "probe").mkdir()
    assert exc.value.errno == errno.EROFS, exc.value


@root_cannot_be_denied
def test_a_REAL_permission_bit_also_blocks_the_store(unwritable_home_by_permission):
    """And the injection is not the only thing that produces this.

    This is the arm that would be vacuous as root; it is skipped there instead.
    Its errno is EACCES rather than EROFS -- a different refusal reaching the
    same handler, which is the point of keeping it.
    """
    with pytest.raises(OSError) as exc:
        (unwritable_home_by_permission / "probe").mkdir()
    assert exc.value.errno == errno.EACCES, exc.value
    with pytest.raises(training.SessionsRootUnavailable):
        training.sessions_root()


def test_an_unwritable_root_raises_a_NAMED_error(unwritable_home):
    with pytest.raises(training.SessionsRootUnavailable):
        training.sessions_root()


def test_the_message_names_the_PATH_and_the_consequence(unwritable_home):
    """A bare `OSError: [Errno 30] Read-only file system` does not say that the
    app cannot serve runs, and that is the fact an operator needs."""
    with pytest.raises(training.SessionsRootUnavailable) as exc:
        training.sessions_root()
    message = str(exc.value)
    assert "micro_sam_sessions" in message, message
    assert str(unwritable_home) in message, message
    assert "cannot serve any training-run operation" in message, message


def test_a_writable_root_still_just_works(writable_home):
    """THE CONTROL. Both assertions above would pass if `sessions_root` raised
    unconditionally."""
    root = training.sessions_root()
    assert root.is_dir()
    assert root == writable_home / ".bioengine" / "micro_sam_sessions"


def test_it_is_still_an_OSError_so_existing_handlers_keep_working(unwritable_home):
    """Nothing in this app catches `OSError` around the run store today, but
    callers outside it might, and a named error that silently stops matching
    `except OSError` would turn a handled failure into a crash.

    This documents the choice: SessionsRootUnavailable is a RuntimeError, NOT an
    OSError, so if a caller ever does rely on that it has to be found rather
    than discovered in production.
    """
    assert issubclass(training.SessionsRootUnavailable, RuntimeError)
    assert not issubclass(training.SessionsRootUnavailable, OSError)


# ── the part that makes it visible, not just legible ─────────────────────────


def entry_module():
    import entry

    return entry


def _entry_class():
    import entry

    for obj in vars(entry).values():
        if isinstance(obj, type) and obj.__name__ == "EntryApp":
            return obj
    for obj in vars(entry).values():
        cls = getattr(obj, "__wrapped_class__", None) or getattr(
            obj, "func_or_class", None
        )
        if isinstance(cls, type) and cls.__name__ == "EntryApp":
            return cls
    raise AssertionError("EntryApp not found")


class _FakeHypha:
    async def echo(self, value):
        return value


@pytest.mark.asyncio
async def test_the_HEALTH_CHECK_fails_when_the_run_store_is_unwritable(
    unwritable_home,
):
    """CLAUSE 3. The failure must NOT surface only as a per-call exception.

    Without this the replica answers `get_training_capabilities` happily, passes
    its health check, stays in the routing table, and raises on every run call
    forever. Raising is what makes Serve replace it.

    Replacement is NOT established as the cure, and an earlier version of this
    docstring said it was, on the grounds that a replacement might come up
    somewhere its home directory could be written. The measurement says the
    opposite: the broken replica and
    a working one had the same read-only `/home/ray`, and what differed was
    whether the per-application HOME override took effect inside the process. So
    replacement helps only if that failure was transient. If it is not, the
    replica keeps failing this check and the app reports UNHEALTHY -- which is
    the honest outcome, and is what the clause asks for either way.
    """
    cls = _entry_class()
    inst = cls.__new__(cls)
    inst._s3 = object()
    inst._hypha = _FakeHypha()
    with pytest.raises(RuntimeError, match="training-run store unusable"):
        await cls._health(inst)


@pytest.mark.asyncio
async def test_the_HEALTH_CHECK_passes_when_the_run_store_is_writable(writable_home):
    """THE CONTROL for the test above: it would pass if `_health` raised for any
    reason at all, including a broken fake."""
    cls = _entry_class()
    inst = cls.__new__(cls)
    inst._s3 = object()
    inst._hypha = _FakeHypha()
    await cls._health(inst)  # must not raise


# ── a store that EXISTS and refuses writes ───────────────────────────────────
#
# The fixture above makes the store un-CREATABLE. That is not the only shape of
# the failure, and it is not the shape `mkdir(exist_ok=True)` is blind to.


@pytest.fixture()
def read_only_existing_store(tmp_path, monkeypatch):
    """The run store already exists, and nothing can be written into it.

    This is a read-only NFS export, an image with `~/.bioengine` baked in, or a
    remount after the store was created. `Path.mkdir(parents=True,
    exist_ok=True)` SUCCEEDS on it, because Linux reports EEXIST before EROFS
    or EACCES and `exist_ok` swallows the EEXIST -- so a check that only creates
    the directory passes while every run write fails.
    """
    home = tmp_path / "home"
    store = home / ".bioengine" / "micro_sam_sessions"
    store.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    # `mkdir` is deliberately NOT refused: the whole point of this fixture is a
    # store that CAN be "created" -- `mkdir(exist_ok=True)` succeeds on it --
    # and refuses every write. Refusing mkdir too would collapse it into the
    # fixture above and the gap would stop being modelled.
    _refuse_under(store, "write_text", monkeypatch)
    _refuse_under(store, "open", monkeypatch)
    return store


@pytest.fixture()
def read_only_store_by_permission(tmp_path, monkeypatch):
    """The write-refusing store produced by a real permission bit.

    The counterpart of `unwritable_home_by_permission`, and skipped as root for
    the same reason.
    """
    home = tmp_path / "home"
    store = home / ".bioengine" / "micro_sam_sessions"
    store.mkdir(parents=True)
    os.chmod(store, stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    yield store
    os.chmod(store, stat.S_IRWXU)


def test_the_read_only_store_fixture_really_blocks_a_write(read_only_existing_store):
    """THE CONTROL. If the store accepts a write after all, everything below
    passes for the wrong reason, so the refusal is asserted directly first."""
    with pytest.raises(OSError) as exc:
        (read_only_existing_store / "probe").write_text("x")
    assert exc.value.errno == errno.EROFS, exc.value


@root_cannot_be_denied
def test_a_REAL_read_only_store_is_also_refused_by_the_probe(
    read_only_store_by_permission,
):
    """A genuine filesystem refusal reaching the writability probe.

    Vacuous as root, so skipped there; the EROFS-injecting arm above carries the
    behaviour everywhere.
    """
    with pytest.raises(OSError) as exc:
        (read_only_store_by_permission / "probe").write_text("x")
    assert exc.value.errno == errno.EACCES, exc.value
    # Creating it still succeeds -- that is the gap -- and the probe still refuses.
    assert training.sessions_root() == read_only_store_by_permission
    with pytest.raises(training.SessionsRootUnavailable, match="refuses writes"):
        training.assert_sessions_root_writable()


def test_CREATING_the_store_SUCCEEDS_even_though_it_refuses_writes(
    read_only_existing_store,
):
    """The gap itself, asserted rather than described.

    `sessions_root()` is not wrong to return here -- creating the directory is
    all it claims to do -- but it is why the health check needed a second
    function. If this ever starts raising, the writability probe below is no
    longer load-bearing and this file should say so.
    """
    assert training.sessions_root() == read_only_existing_store


def test_the_WRITABILITY_probe_refuses_a_store_that_only_EXISTS(
    read_only_existing_store,
):
    with pytest.raises(training.SessionsRootUnavailable, match="refuses writes"):
        training.assert_sessions_root_writable()


def test_the_probe_leaves_NOTHING_behind_on_a_store_it_accepts(writable_home):
    before = sorted(p.name for p in training.sessions_root().iterdir())
    training.assert_sessions_root_writable()
    after = sorted(p.name for p in training.sessions_root().iterdir())
    assert before == after, f"the probe left a file behind: {set(after) - set(before)}"


@pytest.mark.asyncio
async def test_the_HEALTH_CHECK_fails_on_a_store_that_EXISTS_but_refuses_writes(
    read_only_existing_store,
):
    """CLAUSE 2, in the shape that actually provoked this issue.

    The first version of this check called `sessions_root()` only, so it passed
    on this store while every run operation failed -- the condition the issue
    reports, with the check that was supposed to catch it reporting healthy.
    """
    cls = _entry_class()
    inst = cls.__new__(cls)
    inst._s3 = object()
    inst._hypha = _FakeHypha()
    with pytest.raises(RuntimeError, match="training-run store unusable"):
        await cls._health(inst)


@pytest.mark.asyncio
async def test_a_HUNG_store_fails_the_health_check_rather_than_blocking_the_loop(
    writable_home, monkeypatch
):
    """The probe is a synchronous round trip to a network filesystem.

    Run on the event loop without a limit, a hung mount blocks the whole
    replica -- every other method included -- and surfaces only through Serve's
    own timeout, as the less specific failure the echo probe below it exists to
    avoid.
    """
    monkeypatch.setattr(entry_module(), "_HEALTH_CALL_TIMEOUT_S", 0.2)

    def _hang():
        time.sleep(5.0)

    monkeypatch.setattr(training, "assert_sessions_root_writable", _hang)
    cls = _entry_class()
    inst = cls.__new__(cls)
    inst._s3 = object()
    inst._hypha = _FakeHypha()
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="did not answer a write"):
        await cls._health(inst)
    elapsed = time.monotonic() - started
    assert elapsed < 3.0, f"the check waited {elapsed:.1f}s on a hung store"
