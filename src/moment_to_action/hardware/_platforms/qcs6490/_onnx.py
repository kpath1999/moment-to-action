"""ONNX Runtime backend for the QCS6490 platform.

Uses the ``onnxruntime-qnn`` Plugin Execution Provider (v2.x) to route
inference to the GPU or NPU when an accelerated compute unit is requested.
Raises ``RuntimeError`` at session-creation time if the QNN plugin is not
installed or no QNN devices are found, so the caller can fall back to the
CPU backend.
"""

from __future__ import annotations

import logging

import onnxruntime as ort

from moment_to_action.hardware._platforms._runtimes import ONNXBackend
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)

# ONNX Runtime execution-provider name for the Qualcomm QNN plugin.
_QNN_EP_NAME = "QNNExecutionProvider"

# Module-level sentinel: True once the QNN EP library has been registered
# with this ORT process.  Avoids repeated ``register_execution_provider_library``
# calls across multiple ``QCS6490ONNXBackend`` instances.
_QNN_EP_REGISTERED: bool = False


def _ensure_qnn_ep_registered() -> None:
    """Register the QNN plugin EP library with this ORT process (no-op if already done).

    Raises:
        RuntimeError: If ``onnxruntime-qnn`` is not installed.
    """
    global _QNN_EP_REGISTERED  # noqa: PLW0603
    if _QNN_EP_REGISTERED:
        return
    try:
        import onnxruntime_qnn as _qnn
    except ImportError as exc:
        msg = "onnxruntime-qnn is not installed; cannot use QNN ONNX EP"
        raise RuntimeError(msg) from exc
    ort.register_execution_provider_library(_QNN_EP_NAME, _qnn.get_library_path())
    _QNN_EP_REGISTERED = True
    logger.debug("Registered QNN plugin EP: %s", _qnn.get_library_path())


def _qnn_backend_path(unit: ComputeUnit) -> str:
    """Return the QNN backend shared-library path for *unit* (HTP or GPU)."""
    import onnxruntime_qnn as _qnn

    return _qnn.get_qnn_htp_path() if unit == ComputeUnit.NPU else _qnn.get_qnn_gpu_path()


class QCS6490ONNXBackend(ONNXBackend):
    """QCS6490-specific ONNX backend with QNN Plugin Execution Provider support.

    When *compute_unit* is ``NPU`` or ``GPU``, inference is routed to the
    Qualcomm ``QNNExecutionProvider`` via the ``onnxruntime-qnn`` 2.x plugin
    API (``SessionOptions.add_provider_for_devices``).  Raises ``RuntimeError``
    at session-creation time if the plugin is unavailable, so the caller can
    fall back to the CPU backend.

    Args:
        compute_unit: Target compute unit.  Defaults to ``CPU``.
    """

    def __init__(self, compute_unit: ComputeUnit = ComputeUnit.CPU) -> None:
        super().__init__()
        self._unit = compute_unit

    def _get_providers(self) -> list[str | tuple[str, dict]]:
        """Return the CPU provider list used by the base-class session creation."""
        return ["CPUExecutionProvider"]

    def _make_inference_session(self, path: str) -> ort.InferenceSession:
        """Create an ONNX Runtime session, routing NPU/GPU through the QNN plugin EP.

        For ``CPU`` (and any unhandled unit), delegates to the base-class
        implementation which uses ``CPUExecutionProvider``.

        For ``NPU`` or ``GPU``, registers the ``onnxruntime-qnn`` plugin,
        discovers available QNN devices, configures ``SessionOptions`` with
        the appropriate backend library path, and creates the session.

        Args:
            path: Filesystem path to the ``.onnx`` model file.

        Returns:
            A freshly created ``ort.InferenceSession``.

        Raises:
            RuntimeError: If ``onnxruntime-qnn`` is not installed or no QNN
                devices are available on this system.
        """
        if self._unit not in (ComputeUnit.NPU, ComputeUnit.GPU):
            logger.debug("[QCS6490ONNXBackend] CPU/DSP unit, delegating to base class")
            return super()._make_inference_session(path)

        logger.info(
            "[QCS6490ONNXBackend] GPU/NPU path for unit=%s, model=%s",
            self._unit.name,
            path,
        )

        logger.info("[QCS6490ONNXBackend] Calling _ensure_qnn_ep_registered()...")
        _ensure_qnn_ep_registered()
        logger.info("[QCS6490ONNXBackend] QNN EP registration complete")

        devices = [
            d for d in ort.get_ep_devices() if d.ep_name == _QNN_EP_NAME
        ]
        logger.info(
            "[QCS6490ONNXBackend] Discovered %d QNN device(s): %s",
            len(devices),
            [d.device for d in devices],
        )
        if not devices:
            msg = f"{self._unit.name} QNN plugin EP unavailable (no devices discovered)"
            raise RuntimeError(msg)

        backend_path = _qnn_backend_path(self._unit)
        logger.info(
            "[QCS6490ONNXBackend] Backend path for %s: %s",
            self._unit.name,
            backend_path,
        )

        logger.info(
            "[QCS6490ONNXBackend] Creating SessionOptions and adding QNN provider..."
        )
        so = ort.SessionOptions()
        so.add_provider_for_devices(devices, {"backend_path": backend_path})
        logger.info(
            "[QCS6490ONNXBackend] SessionOptions configured with"
            " add_provider_for_devices()"
        )

        logger.info(
            "[QCS6490ONNXBackend] Creating InferenceSession with QNN provider..."
        )
        session = ort.InferenceSession(path, sess_opts=so)
        logger.info(
            "[QCS6490ONNXBackend] ✓ InferenceSession created successfully with"
            " QNN provider"
        )
        return session

    def get_supported_unit(self) -> ComputeUnit:
        """Return the compute unit this backend was configured for."""
        return self._unit
