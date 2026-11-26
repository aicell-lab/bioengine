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


MODULE_SCOPE = "<module or class body>"


def _sessions_root_calls(module_file: str) -> dict:
    """How many times each scope CALLS `sessions_root`, from the syntax tree.

    Three properties, and round 4 of the review established that my first
    version had only the first:

    1. AST rather than text, which is a repair: the line-matching version
       counted a DOCSTRING mentioning `sessions_root()` in prose as a call
       site, so it reported four where three existed and would have misfired
       on any new docstring naming the function.

    2. A COUNT PER SCOPE, not a set of names. A set cannot see a SECOND call
       inside an already-allowed function -- a `leak = sessions_root() / ctx`
       added inside `list_sessions` was MISSED by the set and caught by main's
       exact line count. Multiplicity is part of the claim.

    3. Calls OUTSIDE any function get a scope of their own, so a module-level
       or class-body `root = sessions_root() / x` is visible. The set version
       missed both -- and main's own control used exactly the module-level
       shape, which my rewrite moved inside a function, so the control stopped
       testing the case it existed for.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(module_file).read_text())
    counts: dict = {}

    def walk(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # A nested function is attributed to ITSELF, not to its
                # parent: a closure that reaches the root is its own caller.
                walk(child, child.name)
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "sessions_root"
            ):
                counts[scope] = counts.get(scope, 0) + 1
            walk(child, scope)

    walk(tree, MODULE_SCOPE)
    return counts


#: WHICH scopes may reach the sessions root, and HOW MANY TIMES each. Named and
#: counted rather than either alone: a set of names tolerates a second,
#: unvalidated call inside an allowed function, and a bare total tolerates any
#: three calls anywhere.
SESSIONS_ROOT_CALLS = {
    # builds every session path, and validates the id before it does
    "session_dir": 1,
    # iterates the root to list what is there. CHANGED DELIBERATELY: this used
    # to be `list_sessions`, which walked the root itself. The co-manager
    # listing needs the records and the dataset ids from ONE walk -- asking
    # separately read every record three times over, 783 reads for the 261 runs
    # on the production store -- so the walk moved into this single helper and
    # `list_sessions` now delegates to it. The guard is unchanged in strength:
    # still three scopes, still one call each, and `list_sessions` reaching the
    # root again would now be a FOURTH entry and go red.
    "run_names_on_disk": 1,
    # probes that the root accepts a WRITE. It has to read the root to write
    # in it, and it deliberately does not build a session path: a health check
    # that created a session directory would be a side effect on every probe.
    "assert_sessions_root_writable": 1,
}


def test_no_path_helper_bypasses_the_chokepoint():
    """A structural guard. If a new helper builds on `sessions_root()` directly
    it will not be validated, and that is invisible in a behavioural test."""
    calls = _sessions_root_calls(training.__file__)
    assert calls == SESSIONS_ROOT_CALLS, (
        f"the calls to sessions_root() changed: found {calls}, expected "
        f"{SESSIONS_ROOT_CALLS}. A new caller may be building a session path "
        f"without going through session_dir, or an allowed function may have "
        f"gained a SECOND, unvalidated use. A call attributed to "
        f"'{MODULE_SCOPE}' is at module or class scope and belongs in neither. "
        "If it is legitimate, change SESSIONS_ROOT_CALLS deliberately, with a "
        "line saying why that scope may reach the root."
    )


#: The body of each permitted function in the fake module. Parameterised, so a
#: control can put an extra call INSIDE one of them -- which is the case round 4
#: was about and round 5 found untested: my earlier controls added a NEW
#: function, and a new function is caught by a set of names just as well, so
#: they could not tell a per-function COUNT from a set.
#: The synthetic module the controls build. It has to name the SAME scopes as
#: `SESSIONS_ROOT_CALLS`, or "a module with exactly the allowed callers passes"
#: is a claim about a module the guard does not actually allow. `list_sessions`
#: became `run_names_on_disk` when the root walk moved into that single helper;
#: see the constant for why.
_ALLOWED_BODIES = {
    "session_dir": "    root = sessions_root()\n",
    "run_names_on_disk": "    root = sessions_root()\n",
    "assert_sessions_root_writable": "    root = sessions_root()\n",
}


def _module_source(bodies=None, extra=""):
    merged = {**_ALLOWED_BODIES, **(bodies or {})}
    src = "def sessions_root():\n    pass\n"
    for name, body in merged.items():
        src += f"def {name}(*a, **k):\n{body}"
    return src + extra


def _point_guard_at(tmp_path, monkeypatch, name, bodies=None, extra=""):
    """Write a fake module with the permitted functions, optionally altered."""
    fake = tmp_path / name
    fake.write_text(_module_source(bodies, extra))
    monkeypatch.setattr(training, "__file__", str(fake))
    return fake


@pytest.mark.parametrize(
    "label, bodies, extra",
    [
        # THE case a set of names cannot see, and now it really is inside an
        # allowed function: `list_sessions` reaching the root TWICE. Round 5
        # proved the old version of this control was satisfied by a set,
        # because it added a function called `list_sessions_extra` instead.
        (
            "a SECOND call inside run_names_on_disk itself",
            {
                "run_names_on_disk": (
                    "    root = sessions_root()\n"
                    "    leak = sessions_root() / 'x'\n"
                    "    return root, leak\n"
                )
            },
            "",
        ),
        # A closure nested inside an allowed function. Attributed to ITSELF,
        # so this tests the nesting rule rather than the count -- labelled for
        # what it measures, which the previous label overstated.
        (
            "a closure nested inside session_dir, attributed to itself",
            {
                "session_dir": (
                    "    def inner():\n"
                    "        return sessions_root()\n"
                    "    return inner()\n"
                )
            },
            "",
        ),
        # main's own control used exactly this shape, at MODULE level.
        ("a module-level call", None, 'sneaky = sessions_root() / "x"\n'),
        ("a class-body call", None, "class C:\n    ROOT = sessions_root()\n"),
        (
            "an unlisted function",
            None,
            "def sneaky(sid):\n    return sessions_root() / sid\n",
        ),
    ],
)
def test_that_chokepoint_guard_CAN_fail(tmp_path, monkeypatch, label, bodies, extra):
    """The controls for the structural test above, which reads a file and so
    can silently measure the wrong thing -- as two of these did until round 5
    of the review mutated the assertion back to a set of names and watched
    every one of them still pass."""
    _point_guard_at(tmp_path, monkeypatch, "fake_training.py", bodies, extra)
    with pytest.raises(AssertionError):
        test_no_path_helper_bypasses_the_chokepoint()


def test_the_guard_PASSES_on_exactly_the_allowed_set(tmp_path, monkeypatch):
    """The other half of the control, and the one main's version lacked: a
    module with exactly the allowed callers must PASS. Without it, a guard that
    refused everything would satisfy every test above and look like coverage."""
    _point_guard_at(tmp_path, monkeypatch, "ok_training.py")
    test_no_path_helper_bypasses_the_chokepoint()


def test_a_DOCSTRING_mentioning_the_function_is_not_a_call_site(tmp_path, monkeypatch):
    """The defect the line-matching version had, pinned so it cannot return.

    It counted prose. A module whose docstring names `sessions_root()` and
    whose callers are exactly the allowed set must pass.
    """
    fake = tmp_path / "doc_training.py"
    fake.write_text(
        '"""A module whose prose mentions sessions_root() twice.\n\n'
        'Calling sessions_root() is the subject of this sentence.\n"""\n'
        "def sessions_root():\n    pass\n"
        "def session_dir(sid):\n    root = sessions_root()\n"
        "def run_names_on_disk():\n    root = sessions_root()\n"
        "def assert_sessions_root_writable():\n"
        '    """Mentions sessions_root() in its own docstring too."""\n'
        "    root = sessions_root()\n"
    )
    monkeypatch.setattr(training, "__file__", str(fake))
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
