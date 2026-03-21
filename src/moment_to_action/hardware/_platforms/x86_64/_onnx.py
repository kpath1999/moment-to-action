"""ONNX Runtime backend for the x86_64 platform.

Currently identical to the shared base implementation (CPU-only). Provided as a
separate subclass so platform-specific execution providers can be added in the
future (e.g. CUDA, ROCm) without modifying shared code.
"""

from __future__ import annotations

from moment_to_action.hardware._platforms._runtimes import ONNXBackend


class X86_64ONNXBackend(ONNXBackend):  # noqa: N801
    """x86_64-specific ONNX backend (CPU-only, identical to base)."""
