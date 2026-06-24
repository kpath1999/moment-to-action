"""QCS6490 platform backends and resource monitor."""

from __future__ import annotations

from ._cpu_backend import QCS6490CPUBackend
from ._gpu_backend import QCS6490GPUBackend
from ._htp_backend import QCS6490HTPBackend
from ._resources import QCS6490ResourceMonitor

__all__ = [
    "QCS6490CPUBackend",
    "QCS6490GPUBackend",
    "QCS6490HTPBackend",
    "QCS6490ResourceMonitor",
]
