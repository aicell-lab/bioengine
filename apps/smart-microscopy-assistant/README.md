# Smart Microscopy Assistant

VLM-backed microscopy analyst. Define re-usable visual tests with a few positive and negative reference images, then run them against new images to get a **PASSED / FAILED / UNSURE** verdict. Or describe an image with a free-text instruction.

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
| Generation timeout | 180 s per `_generate_with_images` call | `asyncio.wait_for` |

If the server downscales, the response carries `downscaled_from: [W, H]` and a `downscale_note` so callers can see whether the verdict was rendered on the original or a resized version.

## Two modes

The app has two operating modes that share the same `inspect()` entry point:

1. **Describe mode** (`instruction` only) — single image + free-text question, get a natural-language description.
2. **Few-shot verdict mode** (`visual_test_name` set) — a previously-defined "visual test" supplies positive/negative reference images and a criterion. The model is prompted with the references first, then the new image, and asked to return one of `passed`, `failed`, or `unsure` with a short reason.

Define a visual test once with `create_visual_test(...)`, then call `inspect(image_ref, visual_test_name=...)` as many times as you like — references stay on the replica's disk.

## API

### `inspect(image_ref, instruction=None, visual_test_name=None, max_new_tokens=512) -> dict`

| Parameter | Type | Description |
|---|---|---|
| `image_ref` | `str` | Either an `https://...` URL (public or presigned) **or** a Hypha artifact reference `<workspace>/<alias>:<file_path>` (e.g. `ws-user-github\|49943582/qc-samples:images/frame_001.tif`). |
| `instruction` | `str?` | Free-text instruction. Required when `visual_test_name` is not given. Optional when it is — then it overrides the visual test's stored description. Max 4000 chars. |
| `visual_test_name` | `str?` | Name of a visual test created via `create_visual_test(...)`. Switches into few-shot verdict mode. |
| `max_new_tokens` | `int` | Response token budget. Default 512, range 1–1024. |

**Returns (describe mode):**

```json
{
  "mode": "describe",
  "description": "- Focus: in focus, clear outlines …",
  "image_size": [1024, 1024],
  "source_url": "https://hypha.aicell.io/s3/…",
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "tokens_generated": 66,
  "generation_time_s": 2.55,
  "tokens_per_second": 25.9,
  "processing_time_s": 3.0
}
```

**Returns (few-shot verdict mode):**

```json
{
  "mode": "few-shot",
  "visual_test_name": "focus-quality",
  "visual_test_description": "Sharp cell outlines, no motion blur, distinct staining patterns.",
  "verdict": "passed",
  "reason": "Cell outlines are crisp and the staining is well-resolved.",
  "description": "VERDICT: passed\nREASON: Cell outlines are crisp …",
  "n_positive_examples": 3,
  "n_negative_examples": 3,
  "image_size": [1024, 1024],
  "source_url": "https://hypha.aicell.io/s3/…",
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "tokens_generated": 24,
  "generation_time_s": 0.92,
  "tokens_per_second": 26.1,
  "processing_time_s": 1.3
}
```

`verdict` is one of `"passed"`, `"failed"`, or `"unsure"`. The model returns `unsure` when the visible evidence is genuinely ambiguous or insufficient; the parser also defaults to `unsure` when the output doesn't follow the `VERDICT: …` schema (the raw text is always in `description`).

`downscaled_from` and `downscale_note` may be present in either mode when the server resized the inspected image.

### Few-shot quality notes

The 3B model handles **specific, visually-grounded criteria** ("at least 5 distinct cells", "any saturated pixels", "vertical motion blur") considerably better than **coarse class differences** ("good vs. bad image"). Two patterns observed on the live deployment:

- A criterion phrased as a measurable property (cell count, focus sharpness on a defined region, presence of a specific artefact) generally returns a verdict aligned with the actual content.
- A criterion phrased as broad quality vs. anti-quality, with references that span very different visual styles, can occasionally produce verdicts that echo the positive-class reason regardless of the new image. The 3B model isn't large enough to discriminate sharply by visual gestalt alone.

If a visual test isn't discriminating well: tighten the `description` (it goes into the prompt verbatim) to spell out *what to look for*; consider asking a more specific question via the `instruction` override at inspect time.

### Visual-test management

| Method | Description |
|---|---|
| `create_visual_test(name, description, positive_image_refs, negative_image_refs)` | Define or replace a visual test. References can be HTTPS URLs (public or presigned) or Hypha artifact refs. Images are downloaded, downscaled (capped at ~512×512), and persisted to `$HOME/visual_tests/<name>/`. |
| `list_visual_tests()` | List all visual tests on this replica. |
| `get_visual_test(name)` | Return one visual test's full record. |
| `delete_visual_test(name)` | Remove a visual test and its cached reference images. |

Limits enforced by `create_visual_test`:

- `1 ≤ N_positive ≤ 5`, `1 ≤ N_negative ≤ 5`. More examples eat the model's context budget without improving few-shot quality.
- `name` must match `^[a-z0-9][a-z0-9-]{0,49}$`.
- `description` ≤ 800 characters.
- Each reference image is fetched once and stored at ≤ 512×512 to keep the prompt's image-token cost bounded.

Persistence: visual tests live under `$HOME/visual_tests/` on the replica's filesystem. On the KTH BioEngine worker (and any worker whose `apps_workdir` resolves to PVC-backed storage), that directory is mounted from a persistent volume — the per-app working directory is the same path across actor restarts, pod rolls, and full stop+deploy cycles. Empirically verified by creating a visual test, performing `stop_app → deploy_app` (fresh deploy, `recovered_app=False`), and seeing the test still present on the new actor.

A visual test is therefore **persistent within a worker** but **not portable across workers** (each worker pod has its own PVC). To share a library across deployments, re-call `create_visual_test()` against the source image refs.

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
  "min_examples_per_class": 1,
  "max_examples_per_class": 5,
  "max_visual_test_name_chars": 50,
  "max_visual_test_desc_chars": 800,
  "verdicts": ["passed", "failed", "unsure"],
  "license": "Qwen2.5-VL Apache 2.0 weights"
}
```

## Operating characteristics (measured on KTH A40-16C vGPU)

10 back-to-back `inspect()` calls, same 512×512 HPA RGB image, identical 200-char instruction, `max_new_tokens=192`, 66 generated tokens each:

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

The page has two modes:

- **Analyze** — drag in one or more images, pick a saved visual test (or type a free-text instruction), and hit *Run analysis*. Each image is uploaded and inspected sequentially; the result row shows a colored verdict chip (Passed / Failed / Unsure / Described / Error), the reason, and an expandable details panel with tok/s, timing, and the raw model output.
- **Define visual test** — name, criterion, plus positive + negative example galleries (1–5 each). On save, examples are uploaded to a scratch artifact, presigned with the caller's session, and handed to `create_visual_test()`; the worker downloads them once, downscales, and caches under `$HOME/visual_tests/<name>/`.

An info button in the top bar opens a popover with the served model details (pulled from `get_model_info`). The activity log is hidden behind an expandable "Activity log" panel at the bottom.

The page expects the service ID via `?ws_service_id=<full-id>&server=<hypha-url>` URL params; without them it falls back to the short artifact form `bioimage-io/smart-microscopy-assistant`.

## Usage example (Python)

```python
from hypha_rpc import connect_to_server

server = await connect_to_server({
    "server_url": "https://hypha.aicell.io",
    "token": HYPHA_TOKEN,
})
worker  = await server.get_service("bioimage-io/bioengine-worker-kth-...:bioengine-worker")
status  = await worker.get_app_status(["smart-microscopy-assistant"])
ws_sid  = status["smart-microscopy-assistant"]["service_ids"]["websocket_service_id"]
qc      = await server.get_service(ws_sid)

# Define a visual test once
await qc.create_visual_test(
    name="has-cells",
    description="PASS: visible cellular structures with nuclei. FAIL: flat, empty, or uniform regions.",
    positive_image_refs=["https://example.org/cells_1.tif", "https://example.org/cells_2.tif"],
    negative_image_refs=["https://example.org/flat_1.png",  "https://example.org/flat_2.png"],
)

# Run it against any number of new images
result = await qc.inspect(
    image_ref="https://example.org/scan.tif",
    visual_test_name="has-cells",
)
print(result["verdict"], "—", result["reason"])
```
