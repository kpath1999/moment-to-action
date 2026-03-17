"""QCS6490 platform package.

Public API:
    ``QCS6490Backend``       — unified inference backend (routes .tflite / .onnx)
    ``QCS6490PowerMonitor``  — power monitoring via sysfs

The individual format-specific backends (``LiteRTBackend``, ``ONNXBackend``)
are implementation details of ``QCS6490Backend`` and are exported only for
testing or direct low-level use.  CPU inference is handled by
``LiteRTBackend(ComputeUnit.CPU)`` — no separate class exists.
"""

from __future__ import annotations

from ._backend import QCS6490Backend
from ._litert import LiteRTBackend
from ._onnx import ONNXBackend
from ._power import QCS6490PowerMonitor

__all__ = [
    "LiteRTBackend",
    "ONNXBackend",
    "QCS6490Backend",
    "QCS6490PowerMonitor",
]
