"""VLM backends this app can serve, and how much GPU each one needs.

One app, one model per deployment. The backend is chosen at deploy time with
the ``SMA_MODEL`` environment variable::

    await worker.deploy_app(
        artifact_id="bioimage-io/smart-microscopy-assistant",
        application_id="sma-32b",
        application_env_vars={"SmartMicroscopyAssistant": {"SMA_MODEL": "qwen3-vl-32b"}},
    )

``gpu_memory_mb`` is read off the selected backend by the ``@bioengine.app``
decorator, so each deployment books only the VRAM its model actually needs.
Where the cluster advertises the ``VRAM_MB`` custom resource that makes two
backends share one card: a 4B screener and an 8B can co-reside on a 32 GB GPU,
while ``qwen3-vl-32b`` claims most of it alone. Running several backends at
once is deploying this artifact several times, each with its own
``application_id`` and ``SMA_MODEL``; Ray then packs them by real memory and
leaves a replica pending rather than letting it OOM a neighbour.

This module is imported during introspection, in a baseline environment with
no torch and no transformers — keep it to the standard library.
"""

from typing import Any, Dict

#: ``compute_dtype`` is the dtype the 4-bit kernels accumulate in, and it is
#: also what the unquantised modules are held in. Qwen3 was trained in bfloat16
#: and overflows float16 on long prompts; Qwen2.5-VL-7B has served in float16
#: since 0.14.0, so it stays there rather than changing a verified backend.
#:
#: ``gpu_memory_mb`` is a worst-case peak plus headroom, not a weights-only
#: figure: it has to cover a few-shot prompt carrying 10 reference images and a
#: full-size query. Re-measure when a backend's quantisation changes — the VRAM
#: table in README.md records which figures were measured and on what hardware.
#:
#: Host RAM is deliberately NOT per-backend: quantisation runs at load time off
#: the full bf16 checkpoint, but ``low_cpu_mem_usage`` streams it one shard at a
#: time and every backend here shards at ~5 GB, so the app's flat ``memory_mb``
#: covers the 32B as well as the 4B.
MODELS: Dict[str, Dict[str, Any]] = {
    "qwen2.5-vl-7b": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "arch": "Qwen2_5_VLForConditionalGeneration",
        "params": "8.3B",
        "quantization": "nf4",
        "compute_dtype": "float16",
        "gpu_memory_mb": 12 * 1024,
        "license": "apache-2.0 (Qwen2.5-VL-7B-Instruct weights)",
    },
    "qwen3-vl-4b": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "arch": "Qwen3VLForConditionalGeneration",
        "params": "4.4B",
        "quantization": "nf4",
        "compute_dtype": "bfloat16",
        "gpu_memory_mb": 8 * 1024,
        "license": "apache-2.0 (Qwen3-VL-4B-Instruct weights)",
    },
    "qwen3-vl-8b": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "arch": "Qwen3VLForConditionalGeneration",
        "params": "8.8B",
        "quantization": "nf4",
        "compute_dtype": "bfloat16",
        "gpu_memory_mb": 12 * 1024,
        "license": "apache-2.0 (Qwen3-VL-8B-Instruct weights)",
    },
    "qwen3-vl-32b": {
        "model_id": "Qwen/Qwen3-VL-32B-Instruct",
        "arch": "Qwen3VLForConditionalGeneration",
        "params": "33B",
        "quantization": "nf4",
        "compute_dtype": "bfloat16",
        "gpu_memory_mb": 26 * 1024,
        "license": "apache-2.0 (Qwen3-VL-32B-Instruct weights)",
    },
}

#: Changing this changes which model every deployment that does not set
#: SMA_MODEL will load, including the production one. It stays on the backend
#: the app has served since 0.14.0.
DEFAULT_MODEL = "qwen2.5-vl-7b"


def resolve(name: str = "") -> tuple[str, Dict[str, Any]]:
    """Return ``(backend_name, backend)`` for ``name``, or the default.

    Raises on an unknown name rather than falling back: a typo in
    ``SMA_MODEL`` that silently served a different model would be invisible
    in every reply, since the verdict format does not change with the model.
    """
    picked = (name or "").strip() or DEFAULT_MODEL
    if picked not in MODELS:
        raise ValueError(
            f"SMA_MODEL={picked!r} is not a known backend; "
            f"valid values are {sorted(MODELS)}."
        )
    return picked, MODELS[picked]
