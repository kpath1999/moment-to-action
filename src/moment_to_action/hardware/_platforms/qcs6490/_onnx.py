"""ONNX Runtime backend for the QCS6490 platform.

Uses the QNN ONNX Execution Provider to route inference to the GPU or NPU
when an accelerated compute unit is requested.  Raises ``RuntimeError`` at
session-creation time if the QNN EP is not available, so the caller can fall
back to the CPU backend.
"""

from __future__ import annotations

import logging

import onnxruntime as ort

from moment_to_action.hardware._platforms._runtimes import ONNXBackend
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)

# ONNX Runtime execution-provider name for the Qualcomm QNN plugin.
_QNN_EP_NAME = "QNNExecutionProvider"

# backend_path values accepted by the QNN ONNX EP.
_QNN_NPU_BACKEND = "HTP"  # Hexagon Tensor Processor (NPU)
_QNN_GPU_BACKEND = "GPU"  # Adreno GPU


class QCS6490ONNXBackend(ONNXBackend):
    """QCS6490-specific ONNX backend with QNN Execution Provider support.

    When *compute_unit* is ``NPU`` or ``GPU``, inference is routed to the
    Qualcomm ``QNNExecutionProvider``.  Raises ``RuntimeError`` if the EP
    is not registered in the current onnxruntime installation so that the
    caller can fall back to the CPU backend.

    Args:
        compute_unit: Target compute unit.  Defaults to ``CPU``.
    """

    def __init__(self, compute_unit: ComputeUnit = ComputeUnit.CPU) -> None:
        super().__init__()
        self._unit = compute_unit

    def _get_providers(self) -> list[str | tuple[str, dict]]:
        """Return the ONNX EP list for the configured compute unit.

        For ``CPU``, returns only ``["CPUExecutionProvider"]``.

        For ``NPU`` or ``GPU``, verifies that ``QNNExecutionProvider`` is
        registered in the current onnxruntime build and raises
        ``RuntimeError`` if it is not — preventing a silent fall-through to
        CPU inside onnxruntime.

        Returns:
            Provider list accepted by ``ort.InferenceSession``.

        Raises:
            RuntimeError: If the QNN ONNX EP is unavailable for NPU/GPU.
        """
        if self._unit == ComputeUnit.CPU:
            return ["CPUExecutionProvider"]

        if self._unit in (ComputeUnit.NPU, ComputeUnit.GPU):
            available = ort.get_available_providers()
            if _QNN_EP_NAME not in available:
                msg = (
                    f"{self._unit.name} QNN ONNX EP unavailable (available providers: {available})"
                )
                raise RuntimeError(msg)
            backend_path = _QNN_NPU_BACKEND if self._unit == ComputeUnit.NPU else _QNN_GPU_BACKEND
            logger.info("QNN ONNX EP → %s (%s)", self._unit.name, backend_path)
            return [(_QNN_EP_NAME, {"backend_path": backend_path})]

        # DSP and any other unit are not yet handled — fall through to CPU.
        return ["CPUExecutionProvider"]

    def get_supported_unit(self) -> ComputeUnit:
        """Return the compute unit this backend was configured for."""
        return self._unit
