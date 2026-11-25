"""A session_id from a caller cannot name a path outside the sessions root.

`stop_training` took `session_id` straight from the client, `write_status` does
`mkdir(parents=True)`, and `session_dir` was `sessions_root() / session_id` with
no validation. Measured before the fix, in a container with a throwaway HOME:

    'never-existed-123'   -> created a session directory and a status.json
    '../escaped'          -> wrote outside the sessions root
    '/tmp/<dir>/escape'   -> wrote outside HOME entirely

The third works because ``Path(a) / b`` DISCARDS ``a`` when ``b`` is absolute.
The method carried no permission check and the app is deployed with
``authorized_users`` of ``{"*": ["*"]}``, so this was reachable without signing
in.

Two independent layers are tested separately, because the point of having two is
that softening either one alone still refuses:

* an allow-list on the id's shape, checked before any path is built;
* a containment assertion on the resolved path.

A third layer — only the run's owner or a workspace admin may stop a run — needs
the recorded owner, which is a separate change, and is NOT covered here. What IS
covered is the other half of that rule: a mutating call for a session that does
not exist creates nothing.
"""

import json

import pytest

import training

VALID = "2026-11-25-150730-abcdef12"


# The listing is per-caller now: a run with no recorded owner is returned only to
# a caller holding rw or admin on `bioimage-io`. The runs planted below carry no
# owner, so these tests list as an admin. Listing as anyone else would return {}
# and every junk-tolerance assertion would pass while measuring nothing.
ADMIN_CTX = {
    "user": {
        "id": "github|777000",
        "is_anonymous": False,
        "scope": {"workspaces": {"bioimage-io": "a"}},
    }
}

# Every one of these must be refused. Each is a real shape an attacker or a
# confused client can send, not a decorative variation.
BAD_IDS = [
    ("parent escape", ".."),
    ("parent escape with child", "../escaped"),
    ("nested parent escape", "a/../../escaped"),
    ("absolute path", "/tmp/escape-target"),
    ("absolute root", "/"),
    ("empty", ""),
    ("dot", "."),
    ("slash inside", "2026-11-25-150730-abcdef12/sub"),
    ("backslash", "2026-11-25-150730-abcdef12\\sub"),
    ("NUL byte", "2026-11-25-150730-abcdef12\x00"),
    ("trailing newline", VALID + "\n"),
    ("leading newline", "\n" + VALID),
    ("leading space", " " + VALID),
    ("wrong separator", "2026_11_25_150730_abcdef12"),
    ("uppercase hex", "2026-11-25-150730-ABCDEF12"),
    ("short hex", "2026-11-25-150730-abcdef1"),
    ("long hex", "2026-11-25-150730-abcdef123"),
    ("non-hex suffix", "2026-11-25-150730-zzzzzzzz"),
    ("tilde", "~"),
    ("home expansion", "~/escaped"),
    ("url-ish", "http://example.invalid/x"),
]


@pytest.fixture()
def root(tmp_path, monkeypatch):
    """Point the sessions root at a temp dir, so nothing touches a real worker."""
    r = tmp_path / "sessions"
    r.mkdir()
    monkeypatch.setattr(training, "sessions_root", lambda: r)
    return r


# ── the control: the fix must not break a legitimate id ──────────────────────


def test_a_REAL_session_id_still_works(root):
    """THE CONTROL. Every refusal below is worthless if the allow-list also
    refuses what the server itself generates — the failure mode of an allow-list
    is locking out legitimate traffic, not letting something through."""
    assert training.validate_session_id(VALID) == VALID
    d = training.session_dir(VALID)
    assert d == root / VALID
    assert str(d.resolve()).startswith(str(root.resolve()))


def test_the_ids_THE_SERVER_GENERATES_all_pass():
    """The allow-list is defined as the generator's own shape, so generated ids
    must pass. Asserted against freshly generated ones rather than a literal, so
    a change to `new_session_id` that breaks the pair is caught here."""
    for _ in range(25):
        sid = training.new_session_id()
        assert training.validate_session_id(sid) == sid, sid


def test_every_live_production_id_shape_is_accepted():
    """Measured against production before enforcing this: all 261 live session
    ids match the generator's shape. These are real ids from that listing, so
    enforcing the allow-list makes no existing run unreachable."""
    for sid in (
        "2026-08-17-185709-febd6afd",
        "2026-08-24-221729-7e201991",
        "2026-11-25-043426-d18b01a9",
        "2026-11-25-140104-e9cc366d",
    ):
        assert training.validate_session_id(sid) == sid, sid


# ── layer 1: the allow-list ──────────────────────────────────────────────────


@pytest.mark.parametrize("label,bad", BAD_IDS, ids=[b[0] for b in BAD_IDS])
def test_validate_session_id_REFUSES(label, bad):
    with pytest.raises(training.InvalidSessionId):
        training.validate_session_id(bad)


@pytest.mark.parametrize("label,bad", BAD_IDS, ids=[b[0] for b in BAD_IDS])
def test_session_dir_REFUSES_and_WRITES_NOTHING(root, label, bad):
    before = sorted(p.name for p in root.iterdir())
    with pytest.raises(training.InvalidSessionId):
        training.session_dir(bad)
    assert (
        sorted(p.name for p in root.iterdir()) == before
    ), f"{label}: the refusal still touched the sessions root"


@pytest.mark.parametrize("bad", [None, 7, 1.5, b"bytes", ["list"], {"d": 1}])
def test_a_non_string_session_id_is_refused_by_TYPE(bad):
    """A bytes id is the interesting one: `Path() / b"x"` raises TypeError
    somewhere deeper, which is a refusal by accident rather than by rule."""
    with pytest.raises(training.InvalidSessionId):
        training.validate_session_id(bad)


def test_the_pattern_is_anchored_with_Z_NOT_dollar():
    """`$` also matches just before a trailing newline, so `^...$` ACCEPTS
    "<valid-id>\\n" — which would then be used as a directory name. This asserts
    the bug is absent and shows the anchor choice is load-bearing rather than
    stylistic."""
    import re

    lax = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{6}-[0-9a-f]{8}$")
    assert lax.match(VALID + "\n"), "the lax pattern is supposed to be wrong here"
    assert not training.SESSION_ID_PATTERN.match(VALID + "\n")


def test_the_error_message_does_not_echo_the_whole_value():
    """The value is attacker-controlled and the message reaches logs. A length
    and a short prefix are enough to debug a real mistake."""
    nasty = "../" * 40 + "payload-marker"
    with pytest.raises(training.InvalidSessionId) as exc:
        training.validate_session_id(nasty)
    assert "payload-marker" not in str(exc.value), str(exc.value)
    assert str(len(nasty)) in str(exc.value)


# ── layer 2: containment of the resolved path ────────────────────────────────


def test_containment_still_refuses_when_the_ALLOW_LIST_IS_REMOVED(root, monkeypatch):
    """The two layers are independent, and this is what proves it: with layer 1
    disabled, layer 2 must still refuse an escaping id. Without this test,
    deleting the containment check would leave the whole suite green and the
    'belt and braces' claim would be decoration."""
    monkeypatch.setattr(training, "validate_session_id", lambda sid: sid)
    for bad in ("../escaped", "/tmp/escape-target", "a/../../escaped"):
        with pytest.raises(training.InvalidSessionId):
            training.session_dir(bad)


def test_containment_allows_a_legitimate_id_when_the_allow_list_is_removed(
    root, monkeypatch
):
    """The paired control for the test above: layer 2 alone must not refuse
    everything, or the previous test would pass for the wrong reason."""
    monkeypatch.setattr(training, "validate_session_id", lambda sid: sid)
    assert training.session_dir(VALID) == root / VALID


def test_a_symlink_out_of_the_root_is_refused(root, monkeypatch):
    """`resolve()` follows symlinks, so a session directory that is a link to
    somewhere else is caught by layer 2 even though its NAME is valid. This is
    the case the allow-list cannot see."""
    outside = root.parent / "outside"
    outside.mkdir()
    (root / VALID).symlink_to(outside, target_is_directory=True)
    with pytest.raises(training.InvalidSessionId):
        training.session_dir(VALID)


# ── every path helper is covered, not just one ───────────────────────────────


@pytest.mark.parametrize(
    "helper",
    ["_status_path", "_stop_path", "_params_path", "checkpoint_path"],
)
def test_every_path_helper_inherits_the_guard(root, helper):
    """These all build on `session_dir`, so guarding the chokepoint covers every
    method taking a session_id. Asserted per helper, because 'they all call it'
    is the kind of claim that stops being true silently."""
    fn = getattr(training, helper)
    with pytest.raises(training.InvalidSessionId):
        fn("../escaped")


def test_no_path_helper_bypasses_the_chokepoint():
    """A structural guard. If a new helper builds on `sessions_root()` directly
    it will not be validated, and that is invisible in a behavioural test."""
    import pathlib

    src = pathlib.Path(training.__file__).read_text()
    uses = [
        line.strip()
        for line in src.splitlines()
        if "sessions_root()" in line and not line.strip().startswith("#")
        # the DEFINITION is not a use -- counting it is how this test came to
        # allow 3 when only 2 call sites exist, which tolerated exactly the
        # extra call site it is supposed to catch
        and not line.strip().startswith("def sessions_root")
    ]
    # EXACT, not an upper bound, for that same reason.
    assert uses == ["root = sessions_root()", "root = sessions_root()"], (
        f"expected exactly 2 call sites for sessions_root() -- one in session_dir, "
        f"one in the listing that iterates the root -- and found {len(uses)}: "
        f"{uses}. A new caller may be building a session path without going "
        "through session_dir. If it is legitimate, change this assertion "
        "deliberately rather than relaxing it."
    )


def test_that_chokepoint_guard_CAN_fail(tmp_path, monkeypatch):
    """The control for the structural test above, which reads a file and so can
    silently measure the wrong thing. Point it at a module that really does have
    a third call site and it must complain."""
    import pathlib

    fake = tmp_path / "fake_training.py"
    fake.write_text(
        "def sessions_root():\n    pass\n"
        "root = sessions_root()\n"
        "root = sessions_root()\n"
        "sneaky = sessions_root() / 'x'\n"
    )
    monkeypatch.setattr(training, "__file__", str(fake))
    with pytest.raises(AssertionError):
        test_no_path_helper_bypasses_the_chokepoint()


# ── reads refuse too, and the status reader does not swallow it ──────────────


def test_read_status_REFUSES_an_escaping_id_rather_than_returning_empty(root):
    """`read_status` returns {} for a missing session, which would quietly
    swallow an escape attempt as 'no such session'. It must raise instead."""
    with pytest.raises(training.InvalidSessionId):
        training.read_status("../escaped")
    # paired control: a valid-but-absent session still reports absence quietly
    assert training.read_status(VALID) == {}


# ── a mutating call for a session that does not exist creates NOTHING ────────
#
# `write_status` calls `mkdir(parents=True)`, which is how a real session's
# directory comes into being on its first write. So stopping a run that never
# existed MANUFACTURED it, complete with a status record that then appeared in
# the run listing. Measured before the fix: stop_training('<valid-but-unknown>')
# returned STOPPED and left a status.json behind.


def _entry_instance():
    import entry

    cls = getattr(entry.EntryApp, "func_or_class", entry.EntryApp)
    inst = cls.__new__(cls)
    inst._training_tasks = {}
    return cls, inst


@pytest.mark.asyncio
async def test_stopping_an_UNKNOWN_session_refuses_and_creates_nothing(root):
    cls, inst = _entry_instance()
    unknown = "2026-01-01-000000-deadbeef"
    with pytest.raises(training.SessionNotFound):
        await cls.stop_training(inst, session_id=unknown, context=ADMIN_CTX)
    assert (
        sorted(p.name for p in root.iterdir()) == []
    ), "stopping an unknown session created something in the sessions root"
    assert not (root / unknown).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["../escaped", ""])
async def test_stopping_with_an_ESCAPING_id_refuses_and_creates_nothing(root, bad):
    cls, inst = _entry_instance()
    with pytest.raises(training.InvalidSessionId):
        await cls.stop_training(inst, session_id=bad, context=ADMIN_CTX)
    assert sorted(p.name for p in root.iterdir()) == []
    # and nothing landed next to the root either
    assert sorted(p.name for p in root.parent.iterdir()) == ["sessions"]


@pytest.mark.asyncio
async def test_stopping_with_an_ABSOLUTE_id_leaves_THAT_PATH_untouched(root, tmp_path):
    """Asserts on the ESCAPED path itself, which is the clause's exact wording.

    The parametrised test above only checks the sessions root and its parent.
    For an absolute id the escape lands somewhere else entirely — `Path(a) / b`
    discards `a` when `b` is absolute — so checking the root proves nothing
    about where the write would have gone. This names the target and asserts the
    file is absent there.
    """
    target = tmp_path / "escape-target" / "abs-escape"
    cls, inst = _entry_instance()
    with pytest.raises(training.InvalidSessionId):
        await cls.stop_training(inst, session_id=str(target), context=ADMIN_CTX)
    assert not (target / "status.json").exists(), f"wrote into {target}"
    assert not target.exists(), f"created {target}"
    assert not target.parent.exists(), f"created {target.parent}"
    assert sorted(p.name for p in root.iterdir()) == []


def test_the_absolute_escape_control_WOULD_have_landed_there(root, tmp_path):
    """The paired control for the test above. It asserts an absence, and an
    absence is only evidence if the write would otherwise have happened THERE.
    This reproduces the path arithmetic the old code performed."""
    target = tmp_path / "escape-target" / "abs-escape"
    # What the unguarded `sessions_root() / session_id` produced:
    assert root / str(target) == target, (
        "Path join no longer discards the left side for an absolute right side, "
        "so this control no longer reproduces the original escape"
    )


@pytest.mark.asyncio
async def test_a_REAL_session_can_still_be_stopped(root):
    """THE CONTROL for both tests above. Refusing everything is not a fix, it is
    an outage, and these refusals would read identically either way."""
    cls, inst = _entry_instance()
    sid = training.new_session_id()
    training.write_status(sid, status="TRAINING", message="control")
    out = await cls.stop_training(inst, session_id=sid, context=ADMIN_CTX)
    assert out["status"] == "STOPPED", out
    assert training.read_status(sid)["terminated_by"] == "user_stop"


# ── the listing must TOLERATE junk on disk, not die on it ───────────────────
#
# THE REGRESSION THIS FIX INTRODUCED, caught in review. `get_status` validates
# the id, and `list_sessions` called it for every directory under the root — so
# ONE directory whose name is not a session id made `list_training_sessions`
# raise for every caller, turning a stray directory into a total outage of the
# run history.
#
# It is not hypothetical: until this fix is deployed the running version lets
# any unauthenticated caller create exactly such a directory, so the fixed
# version can come up with junk already on disk. The "all 261 live ids conform"
# measurement was taken BEFORE the rollout and cannot speak for after it.
#
# The rule: validate what a CALLER hands us, tolerate what we FIND.


def _plant(root, name):
    d = root / name
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"status": "COMPLETED"}))
    return name


@pytest.mark.parametrize(
    "junk",
    ["never-existed-123", "escaped", "s-cold", "2026-13-99-999999-zzzzzzzz", "x"],
)
def test_a_NON_CONFORMING_directory_is_SKIPPED_not_fatal(root, junk):
    """One junk directory must not hide the real runs."""
    real = _plant(root, VALID)
    _plant(root, junk)
    listed = training.list_sessions(ADMIN_CTX)
    assert real in listed, (
        f"a directory named {junk!r} beside a real session made the whole "
        f"listing fail or drop the real run: {sorted(listed)}"
    )
    assert junk not in listed, f"{junk!r} was listed as a session"
    assert sorted(listed) == [real]


def test_the_listing_survives_MANY_junk_directories(root):
    """The shape an abuser would actually leave behind."""
    real = _plant(root, VALID)
    for i in range(20):
        _plant(root, f"junk-{i}")
    listed = training.list_sessions(ADMIN_CTX)
    assert sorted(listed) == [real], sorted(listed)


def test_the_listing_control_lists_what_it_SHOULD(root):
    """THE CONTROL. Every assertion above would also pass if list_sessions
    returned {} for everything."""
    a = _plant(root, "2026-11-25-150730-abcdef12")
    b = _plant(root, "2026-11-25-150731-abcdef13")
    assert sorted(training.list_sessions(ADMIN_CTX)) == sorted([a, b])


def test_a_SYMLINKED_session_directory_is_skipped_not_fatal(root):
    """The residual the review found: a directory whose NAME is a valid session
    id but which points OUT of the sessions root.

    The name check cannot see that — only the containment check inside
    `session_dir` can — so without catching its refusal here, one bad link takes
    the entire run history down. That is the same outage the name check exists to
    prevent, arriving through the other layer.
    """
    real = _plant(root, VALID)
    outside = root.parent / "outside"
    outside.mkdir()
    (outside / "status.json").write_text(json.dumps({"status": "COMPLETED"}))
    (root / "2026-11-25-150731-abcdef13").symlink_to(outside, target_is_directory=True)

    listed = training.list_sessions(ADMIN_CTX)
    assert (
        real in listed
    ), f"a symlinked session directory took the whole listing down: {sorted(listed)}"
    assert "2026-11-25-150731-abcdef13" not in listed, "the escaping link was listed"
    assert sorted(listed) == [real]


def test_the_symlink_skip_does_not_swallow_a_LEGITIMATE_inner_symlink(root):
    """THE CONTROL. A link that stays INSIDE the root is not an escape and must
    still be listed, or the test above would pass by skipping every symlink."""
    target = root / VALID
    _plant(root, VALID)
    link = root / "2026-11-25-150731-abcdef13"
    link.symlink_to(target, target_is_directory=True)
    listed = training.list_sessions(ADMIN_CTX)
    assert sorted(listed) == sorted([VALID, "2026-11-25-150731-abcdef13"]), sorted(
        listed
    )
