"""QCS6490 platform package.

Public API:
    ``QCS6490Backend``       — unified inference backend (routes .tflite / .onnx)
    ``QCS6490PowerMonitor``  — power monitoring via sysfs

The format-specific sub-backends (``LiteRTBackend``, ``ONNXBackend``) are
internal implementation details — import them directly from their modules
if needed for testing.
"""

from __future__ import annotations

from ._backend import QCS6490Backend
from ._power import QCS6490PowerMonitor

__all__ = [
    "QCS6490Backend",
    "QCS6490PowerMonitor",
]
