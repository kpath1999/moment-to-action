"""ONNX Runtime backend for the QCS6490 platform.

Currently identical to the base implementation (CPU-only). Provided as a
separate subclass so platform-specific ONNX configuration (e.g. QNN ONNX EP)
can be added in the future without modifying shared code.
"""

from __future__ import annotations

from moment_to_action.hardware._platforms._runtimes import ONNXBackend


class QCS6490ONNXBackend(ONNXBackend):
    """QCS6490-specific ONNX backend (currently CPU-only, identical to base)."""
