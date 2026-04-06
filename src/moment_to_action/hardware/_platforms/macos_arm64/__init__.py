"""macOS arm64 (Apple Silicon) platform package.

Public API:
    ``MacOSARM64Backend``          — unified inference backend (routes .tflite / .onnx)
    ``MacOSARM64ResourceMonitor``  — resource monitoring via psutil estimates
"""

from __future__ import annotations

from ._backend import MacOSARM64Backend
from ._resources import MacOSARM64ResourceMonitor

__all__ = [
    "MacOSARM64Backend",
    "MacOSARM64ResourceMonitor",
]
