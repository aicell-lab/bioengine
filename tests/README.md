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

Because that load is automatic, whether a credential resolves depends on which checkout you start pytest from — a git worktree has no `.env`, the main checkout does. Tests that act on a real Hypha cluster are therefore deselected unless you pass `--live`, and whenever a credential does resolve the run prints the server and workspace it could reach before the first test. Both are independent of the `--ignore` list; `tests/apps/model-runner/` is live even though it is not under `tests/end_to_end/`.

```bash
pytest tests/                 # offline tests only
pytest tests/ --live          # also runs tests that deploy to and call the live cluster
```

A test counts as live if it requests `hypha_client`, `hypha_token` or `model_runner`, or is marked `@pytest.mark.live`. Anything new that reaches a cluster by some other route has to carry the marker.

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
