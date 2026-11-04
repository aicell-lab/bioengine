# Git hooks

## `pre-push` — the per-path push policy, as a check

CLAUDE.md's per-path table says which paths require a pull request and which may be
pushed straight to `main`. This hook enforces the PR-only half of it: a push whose
**remote ref is `refs/heads/main`** is refused when it touches `bioengine/**`,
`docker/**`, `apps/model-runner/**`, `apps/model-finetune/**`, `pyproject.toml`,
`.dockerignore` or `requirements*.txt`. Pushing those same paths to a feature branch
is the supported workflow and is left alone.

It exists because the rule cannot be enforced on the server. The repository has one
active ruleset, `Require PR`, and it is scoped by **ref**, not by path — its
conditions are `{"ref_name": {"include": ["~DEFAULT_BRANCH"]}}` with no path
component, so it forbids direct pushes to `main` for *every* path rather than for the
PR-only ones. It also carries a bypass for the `admin` repository role, which is why
direct pushes succeed at all and why only a `remote: Bypassed rule violations` line
reports them.

## Install (opt-in)

```bash
git config core.hooksPath .github/hooks
```

**Read this before running that command on a shared clone.** `core.hooksPath` is
stored in `.git/config`, which every worktree of a clone shares, so installing the
hook arms it for every checkout and every agent session using that clone at once. A
hook that refuses a push is, from inside another session's turn, hard to tell apart
from a credential or network failure. Install it in your own clone; do not arm it in
a clone other people are working in without telling them.

`git push --no-verify` skips it, as it skips every pre-push hook. That is deliberate:
this is a guard against the rule being misread, not an access control. The access
control is the ruleset, and it is bypassed for admins.

## Test

```bash
bash .github/hooks/tests/control.sh
```

Eight arms: four PR-only paths refused on `main`, two allowed paths permitted on
`main`, a PR-only path permitted on a feature branch, and — the one that makes the
other seven mean anything — the same forbidden push **with the hook removed**, which
must succeed. Without that arm, a harness that refused everything for an unrelated
reason would score 7/7 and prove nothing. Each arm builds a fresh bare origin and
clone, so arms cannot contaminate each other.
