# BioEngine Worker Testing

This directory contains the complete test suite for BioEngine Worker, including unit tests, integration tests, and end-to-end tests.

## Environment Setup

### 1. Create and Activate Environment
```bash
conda create -n bioengine-worker python=3.11.9
conda activate bioengine-worker
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### 3. Environment Configuration
The `.env` file in the project root contains required environment variables including `HYPHA_TOKEN`. This is automatically loaded by the test configuration.

### 4. Tests that act on a live cluster

Some tests deploy applications to, and call services on, the production workspaces at `https://hypha.aicell.io`. They are deselected unless you pass `--live`:

```bash
pytest tests/                 # offline tests only
pytest tests/ --live          # also runs tests that act on the live cluster
```

A test counts as live if it requests `hypha_client`, `hypha_token` or `model_runner`, or is marked `@pytest.mark.live`. Anything that reaches a cluster by some other route — a browser driven at the deployed app, a client built inline — has to carry the marker. Liveness has nothing to do with the `--ignore` lists in circulation: `tests/apps/model-runner/` and `tests/test_artifact_version.py` are live and neither lives under `tests/end_to_end/`.

**Do not rely on your working directory to keep a credential away from the suite.** There are at least three ways one arrives, and only the last is under your control:

- `load_dotenv()` above searches **upward** from `tests/`, not just the directory you started in. A worktree under `<repo>/.claude/worktrees/` has no `.env` of its own but still reaches the repo root's, so it is *not* protected. A worktree outside the repo tree (say under `/tmp`) does stop the walk.
- `tests/test_artifact_version.py` loads the repo-root `.env` by explicit path, wherever you run it from.
- `source .env`, `export HYPHA_TOKEN=…`, or a token already in your shell defeats all of the above regardless of location.

Running inside the worker image with only the repo bind-mounted also stops the upward walk, because the search cannot escape the mount.

Because none of that is reliable, the gate does not depend on it. Whenever a credential resolves, the run prints — before the first test, and again next to the wall clock at the end — which server it is, which workspace each token is scoped to, and how many live-reaching tests are enabled. That last number is printed **with or without `--live`**: the flag only prevents the accidental case, so once someone has opted in deliberately the count in the log is what makes an after-the-fact audit possible. Runtime is the corroborating signal; the same scope takes roughly 30s offline and several minutes against a cluster.

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### Specific Test Categories
```bash
# End-to-end tests that verify Hypha service API interaction with BioEngine worker
pytest tests/end_to_end/ -v
```

### Test Options
```bash
# Stop on first failure
pytest tests/ --maxfail=1

# Generate coverage report
pytest tests/ --cov=bioengine --cov-report=html
```

## Test Structure

- `tests/end_to_end/` - Full system tests that verify Hypha service API interaction with BioEngine worker, including Ray cluster integration
- `tests/conftest.py` - Shared fixtures and configuration

## Environment Requirements

- **Python 3.11.9** - Tested and verified version
- **HYPHA_TOKEN** - Authentication for Hypha server access
- **Network Access** - Required for end-to-end tests
- **System Resources** - 8GB RAM minimum for Ray cluster tests
