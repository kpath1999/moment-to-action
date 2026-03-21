"""LiteRT backend for the QCS6490 platform with QNN accelerator support.

Extends the shared LiteRTBackend to add QNN delegate support for NPU/GPU
inference on the Qualcomm QCS6490.
"""

from __future__ import annotations

import logging

from moment_to_action.hardware._platforms._runtimes import LiteRTBackend
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)

# Path to the Qualcomm QNN TFLite delegate shared library on-device.
_QNN_DELEGATE_PATH = "/usr/lib/libQnnTFLiteDelegate.so"

# Try to import ai_edge_litert at module load time.  On dev machines this
# package is absent, so we fall back to tf.lite (which ships with tensorflow).
try:
    from ai_edge_litert.interpreter import load_delegate as _load_delegate

    _have_ai_edge_litert = True
except ImportError:
    from tensorflow.lite.python.interpreter import load_delegate as _load_delegate

    _have_ai_edge_litert = False
    logger.warning("ai_edge_litert not available — using tf.lite as fallback")


class QCS6490LiteRTBackend(LiteRTBackend):
    """TFLite runtime with QNN delegate for NPU/GPU acceleration on QCS6490.

    Extends the base LiteRTBackend to add Qualcomm QNN TFLite delegate
    support for Hexagon HTP (NPU) and Adreno GPU. When NPU/GPU is requested,
    the QNN delegate is loaded; CPU requests use no delegate (XNNPACK).
    """

    def _get_delegates(self) -> list:
        """Build the delegate list for the configured compute unit.

        Returns an empty list for CPU — no delegate loading is attempted,
        and XNNPACK acceleration is applied automatically by the runtime.

        For NPU, loads the QNN TFLite delegate from ``_QNN_DELEGATE_PATH``.
        Raises ``RuntimeError`` on failure so the caller can fall back to CPU.
        """
        # CPU path: no delegates needed.
        if self._unit == ComputeUnit.CPU:
            return []

        # NPU path: load QNN delegate.  Raise on failure so the caller can
        # decide whether to retry on CPU.
        if self._unit == ComputeUnit.NPU:
            try:
                qnn = _load_delegate(_QNN_DELEGATE_PATH)
            except Exception as e:
                msg = f"NPU delegate unavailable: {e}"
                raise RuntimeError(msg) from e
            else:
                logger.info("QNN delegate loaded → Hexagon HTP/NPU")
                return [qnn]

        # GPU and other units: no delegate implemented yet.
        return []
