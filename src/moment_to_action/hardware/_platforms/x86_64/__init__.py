"""x86_64 platform backend and resource monitor."""

from __future__ import annotations

from ._cpu_backend import X86_64CPUBackend
from ._gpu_backend import X86_64GPUBackend
from ._resources import X86_64ResourceMonitor

__all__ = [
    "X86_64CPUBackend",
    "X86_64GPUBackend",
    "X86_64ResourceMonitor",
]
