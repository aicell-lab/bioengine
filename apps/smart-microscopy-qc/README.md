# Smart Microscopy QC

Real-time visual quality-control inspector for live microscopy. A microscope (or any client) submits an acquired frame together with a free-text instruction describing the QC metrics to check, and a vision-language model returns a textual description of what it sees relative to that instruction.

## Method

| | |
|---|---|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| License | Apache 2.0 |
| Engine | HuggingFace `transformers` 4.51.3 + `torch==2.5.1` |
| Precision | FP16 |
| Hardware | 1× NVIDIA A40-16C vGPU slice per replica (Ampere sm_86, 16 GB framebuffer, time-shared with co-tenants on the host A40) |
| Image budget | server-side downscale to ≤ `1280 × 28 × 28` pixels and ≤ 2048 longest side, with a hard reject above 200 MP |

### Why Qwen2.5-VL-3B-Instruct (FP16)

- Apache 2.0 license — usable in any deployment.
- 3B FP16 weights (~6 GB) + Qwen's vision encoder + KV cache fit comfortably in one A40-16C vGPU slice with substantial headroom for activations.
- Loaded directly via `transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(...)` — no quantisation kernel in the path.
- Dynamic input resolution via Qwen's processor — works with arbitrary microscopy frame sizes once the server-side downscale step has bounded them.
- Returns coherent multi-bullet QC reports (focus, illumination uniformity, object count, contamination, etc.) on real fluorescence-microscopy frames; see Operating characteristics below for measured behaviour.

### Why not the 7B AWQ variant (initial target)

7B-AWQ would be the higher-quality choice but no AWQ kernel stack currently serves Qwen2.5-VL on this cluster:

- **vLLM 0.10.x** — V0 multimodal input-prep raises `InputProcessingError: list index out of range` on every prompt shape; V1 engine refuses to initialise from a Ray Serve actor thread.
- **vLLM 0.9.x** — model-registry subprocess fails to inspect `Qwen2_5_VLForConditionalGeneration` and swallows the underlying error.
- **vLLM 0.7.x** — transitively pins an older Ray than the host pod, which Ray refuses to load.
- **autoawq Triton kernel** — bundled `awq_gemm_triton` doesn't compile against the Triton shipped with current torch.
- **autoawq-kernels CUDA path** — `awq_ext.gemm_forward_cuda` raises `expected scalar type Int but found Half` on the `lm_head` linear for `Qwen2.5-VL-7B-Instruct-AWQ` (known upstream issue, fix sits in autoawq 0.2.8 which itself caps `transformers <= 4.47.1` — older than the 4.49 minimum Qwen2.5-VL needs).

3B FP16 lands on this cluster as-is. Revisit 7B-AWQ when any of the upstream stacks above unblocks.

## Image and instruction limits

| Limit | Value | Where enforced |
|---|---|---|
| Image file size | 25 MB | Streamed download; raises if exceeded mid-stream |
| Image pixel count | ≤ 1280 × 28 × 28 (≈ 1.0 MP) AND longest side ≤ 2048 | Server-side downscale in `_download_image` (PIL `Image.resize` with LANCZOS) |
| Image hard reject | 200 MP (≈ 14000 × 14000) | Raises `ValueError` before downscale |
| Instruction length | 4000 characters | Server-side `ValueError` (also mirrored in the browser UI which clips the textarea) |
| Generation timeout | 120 s per `_run_vlm` call | `asyncio.wait_for` |

If the server downscales, the response carries `downscaled_from: [W, H]` and a `downscale_note` so callers can see whether the QC verdict was rendered on the original or a resized version.

## API

### `inspect(image_ref, instruction, max_new_tokens=512) -> dict`

| Parameter | Type | Description |
|---|---|---|
| `image_ref` | `str` | Either an `https://...` URL (public or presigned) **or** a Hypha artifact reference `<workspace>/<alias>:<file_path>` (e.g. `ws-user-github\|49943582/qc-samples:images/frame_001.tif`). |
| `instruction` | `str` | Free-text QC question. Max 4000 chars. |
| `max_new_tokens` | `int` | Response token budget. Default 512, range 1–1024. |

**Returns:**

```json
{
  "description": "- Focus: in focus, clear outlines …",
  "image_size": [1024, 1024],
  "source_url": "https://hypha.aicell.io/s3/…",
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "tokens_generated": 66,
  "generation_time_s": 2.55,
  "tokens_per_second": 25.9,
  "processing_time_s": 3.0,
  "downscaled_from": [4096, 4096],
  "downscale_note": "Image downscaled from 4096x4096 to 980x980 before VLM. The QC verdict applies to the resized image."
}
```

`downscaled_from` and `downscale_note` are only present when the server resized the image.

### `ping() -> dict`

Liveness probe returning `{status, model, uptime_s}`.

### `get_model_info() -> dict`

Describes the served model and the input/output contract:

```json
{
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "task": "vision-language",
  "engine": "huggingface-transformers",
  "dtype": "float16",
  "device": "cuda:0",
  "max_image_bytes": 26214400,
  "max_instruction_chars": 4000,
  "max_pixels": 1003520,
  "max_long_side": 2048,
  "hard_reject_pixels": 209715200,
  "license": "Qwen2.5-VL Apache 2.0 weights"
}
```

## Operating characteristics (measured on KTH A40-16C vGPU)

10 back-to-back `inspect()` calls, same 512×512 HPA RGB image, identical 200-char QC instruction, `max_new_tokens=192`, 66 generated tokens each:

| Metric | min | median | mean | max | std | spread |
|---|---:|---:|---:|---:|---:|---:|
| tok/s | 20.4 | 24.5 | 24.1 | 26.3 | 1.8 | 5.9 |
| e2e seconds | 2.79 | 3.13 | 3.30 | 4.26 | — | 1.47 |

The vGPU profile time-shares the underlying A40 with other tenants; the tight ~2 tok/s standard deviation indicates contention is currently mild.

VRAM is not exposed directly via the Hypha service. The model load reports ~6 GB for weights at FP16; with KV cache, activations, and the vision encoder, the steady-state working set is well within the 16 GB framebuffer.

## Browser UI

`frontend/index.html` ships as the artifact's `frontend_entry`, so once the artifact is uploaded it is reachable at:

```
https://hypha.aicell.io/{workspace}/view/{artifact-id}/
```

The page:

1. Prompts the user to sign in to Hypha (the user's own token, never the worker's).
2. Accepts a drag-dropped or file-picked image.
3. Takes a QC instruction (UI mirrors the 4000-char server cap with a live counter).
4. Uploads the image to a `smart-microscopy-qc-scratch-<random>` artifact in the user's workspace.
5. Calls `inspect(...)`.
6. Displays the response with token/throughput metadata.
7. **Always** deletes the scratch artifact in a `finally` block, success or failure.

The page expects the QC service ID via `?ws_service_id=<full-id>&server=<hypha-url>` URL params; without them it falls back to the short artifact form `bioimage-io/smart-microscopy-qc`.

## Usage example (Python)

```python
from hypha_rpc import connect_to_server

server = await connect_to_server({
    "server_url": "https://hypha.aicell.io",
    "token": HYPHA_TOKEN,
})
worker  = await server.get_service("bioimage-io/bioengine-worker-kth-...:bioengine-worker")
status  = await worker.get_app_status(["smart-microscopy-qc"])
ws_sid  = status["smart-microscopy-qc"]["service_ids"]["websocket_service_id"]
qc      = await server.get_service(ws_sid)

result = await qc.inspect(
    image_ref="https://example.org/scan.tif",
    instruction=(
        "Assess focus quality, illumination uniformity, and the approximate "
        "number of cells. Flag any dark spots or contamination."
    ),
)
print(result["description"])
```
