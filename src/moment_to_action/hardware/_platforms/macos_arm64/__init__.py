"""macOS arm64 (Apple Silicon) platform backend and resource monitor."""

from __future__ import annotations

from ._cpu_backend import MacOSARM64CPUBackend
from ._resources import MacOSARM64ResourceMonitor

__all__ = [
    "MacOSARM64CPUBackend",
    "MacOSARM64ResourceMonitor",
]
