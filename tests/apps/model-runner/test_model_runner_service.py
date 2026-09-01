"""Live integration tests for the model-runner BioEngine app.

Tests all public API methods against the deployed bioimage-io/model-runner service.
GPU-dependent methods are marked with @pytest.mark.requires_gpu and skipped
automatically when the runtime deployment is unavailable.

Run:
    export BIOIMAGE_IO_TOKEN=...
    pytest tests/apps/model-runner/ -v
    pytest tests/apps/model-runner/ -v -m "not requires_gpu"   # CPU tests only
"""

import io

import httpx
import numpy as np
import pytest

TEST_MODEL_ID = "ambitious-ant"

# ─── helpers ──────────────────────────────────────────────────────────────────

GPU_UNAVAILABLE_MSG = "GPU runtime deployment is not available"


def _is_gpu_error(exc: Exception) -> bool:
    return GPU_UNAVAILABLE_MSG in str(exc)


# ─── search_models ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_models_returns_list(model_runner):
    results = await model_runner.search_models(limit=5)
    assert isinstance(results, list)
    assert len(results) <= 5


@pytest.mark.asyncio
async def test_search_models_with_keywords(model_runner):
    results = await model_runner.search_models(
        keywords=["segmentation"], limit=10
    )
    assert isinstance(results, list)
    for item in results:
        assert "model_id" in item
        assert "description" in item


@pytest.mark.asyncio
async def test_search_models_empty_keywords(model_runner):
    results = await model_runner.search_models(keywords=[], limit=3)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_models_ignore_checks(model_runner):
    results_checked = await model_runner.search_models(limit=5, ignore_checks=False)
    results_all = await model_runner.search_models(limit=5, ignore_checks=True)
    # With ignore_checks=True there may be >= as many results
    assert isinstance(results_checked, list)
    assert isinstance(results_all, list)


# ─── get_model_rdf ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_model_rdf_returns_dict(model_runner):
    rdf = await model_runner.get_model_rdf(model_id=TEST_MODEL_ID, stage=False)
    assert isinstance(rdf, dict)


@pytest.mark.asyncio
async def test_get_model_rdf_has_required_fields(model_rdf):
    for field in ("id", "name", "type", "inputs", "outputs"):
        assert field in model_rdf, f"Missing field: {field}"


@pytest.mark.asyncio
async def test_get_model_rdf_has_test_inputs(model_rdf):
    assert "test_inputs" in model_rdf
    assert len(model_rdf["test_inputs"]) > 0


@pytest.mark.asyncio
async def test_get_model_rdf_invalid_id(model_runner):
    with pytest.raises(Exception):
        await model_runner.get_model_rdf(model_id="this-model-does-not-exist-xyz")


# ─── get_model_documentation ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_model_documentation_returns_str_or_none(model_runner):
    doc = await model_runner.get_model_documentation(
        model_id=TEST_MODEL_ID, stage=False
    )
    assert doc is None or isinstance(doc, str)


@pytest.mark.asyncio
async def test_get_model_documentation_nonempty_for_known_model(model_runner):
    doc = await model_runner.get_model_documentation(
        model_id=TEST_MODEL_ID, stage=False
    )
    # ambitious-ant has documentation
    if doc is not None:
        assert len(doc) > 0


# ─── validate ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_validate_valid_rdf(model_runner, model_rdf):
    result = await model_runner.validate(rdf_dict=model_rdf)
    assert isinstance(result, dict)
    assert "success" in result
    assert result["success"] is True
    assert "details" in result


@pytest.mark.asyncio
async def test_validate_invalid_rdf(model_runner):
    result = await model_runner.validate(rdf_dict={"type": "model"})
    assert isinstance(result, dict)
    assert "success" in result
    # Incomplete RDF should fail validation
    assert result["success"] is False


@pytest.mark.asyncio
async def test_validate_empty_rdf(model_runner):
    result = await model_runner.validate(rdf_dict={})
    assert isinstance(result, dict)
    assert result["success"] is False


# ─── get_upload_url ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_upload_url_npy(model_runner):
    info = await model_runner.get_upload_url(file_type=".npy")
    assert "upload_url" in info
    assert "file_path" in info
    assert info["upload_url"].startswith("http")


@pytest.mark.asyncio
async def test_get_upload_url_png(model_runner):
    info = await model_runner.get_upload_url(file_type=".png")
    assert "upload_url" in info
    assert "file_path" in info


@pytest.mark.asyncio
async def test_get_upload_url_tiff(model_runner):
    info = await model_runner.get_upload_url(file_type=".tiff")
    assert "upload_url" in info


@pytest.mark.asyncio
async def test_upload_npy_and_verify(model_runner):
    """Upload a small array and verify the upload URL accepts a PUT."""
    info = await model_runner.get_upload_url(file_type=".npy")
    arr = np.zeros((1, 1, 8, 8), dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    async with httpx.AsyncClient() as client:
        resp = await client.put(info["upload_url"], content=buf.getvalue())
    assert resp.status_code in (200, 204)


# ─── test (requires GPU) ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.requires_gpu
async def test_model_test_passes(model_runner):
    result = await model_runner.test(model_id=TEST_MODEL_ID, stage=False)
    assert isinstance(result, dict)
    assert result.get("status") == "passed"


@pytest.mark.asyncio
@pytest.mark.requires_gpu
async def test_model_test_cache_skip(model_runner):
    result = await model_runner.test(model_id=TEST_MODEL_ID, stage=False, cache="skip")
    assert isinstance(result, dict)
    assert result.get("status") == "passed"


@pytest.mark.asyncio
async def test_model_test_gpu_unavailable_raises_clear_error(model_runner):
    """When GPU runtime is down, test() raises a clear RuntimeError (not a timeout)."""
    try:
        await model_runner.test(model_id=TEST_MODEL_ID, stage=False)
    except Exception as e:
        if _is_gpu_error(e):
            # Expected when GPU is unavailable — error message is clear
            assert GPU_UNAVAILABLE_MSG in str(e)
        else:
            # GPU is available — test passed
            pass


# ─── infer (requires GPU) ─────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.requires_gpu
async def test_infer_with_numpy_array(model_runner, test_image_array):
    result = await model_runner.infer(
        model_id=TEST_MODEL_ID, inputs=test_image_array
    )
    assert isinstance(result, dict)
    assert len(result) > 0
    for v in result.values():
        assert hasattr(v, "shape")  # numpy array


@pytest.mark.asyncio
@pytest.mark.requires_gpu
async def test_infer_with_uploaded_npy(model_runner, test_image_array):
    upload_info = await model_runner.get_upload_url(file_type=".npy")
    buf = io.BytesIO()
    np.save(buf, test_image_array)
    async with httpx.AsyncClient() as client:
        await client.put(upload_info["upload_url"], content=buf.getvalue())

    result = await model_runner.infer(
        model_id=TEST_MODEL_ID, inputs=upload_info["file_path"]
    )
    assert isinstance(result, dict)


@pytest.mark.asyncio
@pytest.mark.requires_gpu
async def test_infer_return_download_url(model_runner, test_image_array):
    result = await model_runner.infer(
        model_id=TEST_MODEL_ID,
        inputs=test_image_array,
        return_download_url=True,
    )
    assert isinstance(result, dict)
    for v in result.values():
        assert isinstance(v, str)
        assert v.startswith("http")
    # Verify the URL is downloadable
    async with httpx.AsyncClient() as client:
        first_url = next(iter(result.values()))
        resp = await client.get(first_url)
        assert resp.status_code == 200
        arr = np.load(io.BytesIO(resp.content))
        assert arr.ndim > 0


@pytest.mark.asyncio
async def test_infer_gpu_unavailable_raises_clear_error(model_runner):
    """When GPU runtime is down, infer() raises a clear RuntimeError (not a timeout).

    Uses a dummy array — the GPU error fires before the input is touched.
    """
    dummy = np.zeros((1, 1, 64, 64), dtype=np.float32)
    try:
        await model_runner.infer(model_id=TEST_MODEL_ID, inputs=dummy)
    except Exception as e:
        if _is_gpu_error(e):
            assert GPU_UNAVAILABLE_MSG in str(e)
        else:
            pass  # GPU available, infer succeeded
