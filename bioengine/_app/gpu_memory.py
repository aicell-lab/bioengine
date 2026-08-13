"""Device-wide VRAM sampling via the CUDA driver API.

On NVIDIA vGPU C-series time-slicing the guest's NVML ``memoryUsed`` is
unreliable — it latches stale values (a fully idle GPU can report ~14 GiB
used) and carries a fixed offset — but the CUDA driver's ``cuMemGetInfo``
reads the real device-wide free/total inside the guest. This wraps
``libcuda.so.1`` directly with ctypes so sampling needs no torch / pynvml /
cudart dependency; the driver lib is always injected by the NVIDIA
container runtime.
"""

from __future__ import annotations

import ctypes
from typing import Tuple

_CUDA_SUCCESS = 0


class CudaMemorySampler:
    """Holds a CUDA context on device 0 and samples device-wide VRAM.

    Constructed once per GPU replica. ``__init__`` retains the device's
    *primary* context — reusing torch's context when one already exists
    (zero extra VRAM) or creating a thin one otherwise — and raises
    ``RuntimeError`` if the driver cannot be initialised. That failure IS
    the fail-hard gate: a replica Ray assigned a GPU to but which cannot
    create a CUDA context is unhealthy, not silently CPU-bound.
    """

    def __init__(self) -> None:
        try:
            lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise RuntimeError(f"libcuda.so.1 not loadable: {e}") from e
        self._lib = lib

        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        lib.cuCtxPushCurrent_v2.argtypes = [ctypes.c_void_p]
        lib.cuCtxPopCurrent_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cuMemGetInfo_v2.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]

        # Device 0 is the replica's assigned GPU — Ray scopes CUDA_VISIBLE_DEVICES.
        self._check(lib.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self._check(lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        self._check(
            lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
            "cuDevicePrimaryCtxRetain",
        )
        self._ctx = ctx

    def _check(self, rc: int, call: str) -> None:
        if rc != _CUDA_SUCCESS:
            raise RuntimeError(f"{call} failed with CUDA error {rc}")

    def sample(self) -> Tuple[int, int]:
        """Return ``(used_bytes, total_bytes)`` device-wide.

        Called from a ``to_thread`` worker: push the primary context onto
        this thread, query, pop. Primary contexts are multi-thread
        shareable and ``cuMemGetInfo`` is a read-only query, so this is safe
        even while another thread runs torch on the same context.
        """
        lib = self._lib
        self._check(lib.cuCtxPushCurrent_v2(self._ctx), "cuCtxPushCurrent")
        try:
            free = ctypes.c_size_t()
            total = ctypes.c_size_t()
            self._check(
                lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)),
                "cuMemGetInfo",
            )
        finally:
            popped = ctypes.c_void_p()
            lib.cuCtxPopCurrent_v2(ctypes.byref(popped))
        return total.value - free.value, total.value
