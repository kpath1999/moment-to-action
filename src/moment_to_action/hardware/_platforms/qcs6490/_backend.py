"""Unified inference backend for the QCS6490 platform.

``QCS6490Backend`` is the single :class:`InferenceBackend` that the platform
package exposes.  It owns:

- **Format routing** — ``.tflite`` files go to LiteRT, ``.onnx`` files go to
  ONNX Runtime.  All sub-backends are created eagerly at construction time.
- **Accelerator fallback** — if the NPU/GPU backend raises at load time, the
  model is retried on the always-present CPU backend.
- **Model handle tracking** — each loaded model is paired (via
  ``_ModelHandle``) with the sub-backend that created it, so ``run()``
  needs no isinstance checks.

All the complexity lives here so that :class:`ComputeBackend` (the public
orchestrator) can be a three-field thin wrapper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from moment_to_action.hardware._platforms._base import InferenceBackend, ModelInput
from moment_to_action.hardware._platforms.qcs6490._litert import LiteRTBackend
from moment_to_action.hardware._platforms.qcs6490._onnx import ONNXBackend
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)

# Supported model file suffixes.
_TFLITE_SUFFIX = ".tflite"
_ONNX_SUFFIX = ".onnx"


# ---------------------------------------------------------------------------
# Internal handle type
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ModelHandle:
    """Opaque model handle that pairs a raw runtime object with its backend.

    Callers treat this as completely opaque — they just pass it back to
    ``QCS6490Backend.run()`` and friends.
    """

    raw: Any = field(repr=False)
    backend: InferenceBackend = field(repr=False)


# ---------------------------------------------------------------------------
# Unified backend
# ---------------------------------------------------------------------------


class QCS6490Backend(InferenceBackend):
    """Unified inference backend for the Qualcomm QCS6490.

    Internally delegates to format-specific sub-backends, all created eagerly:

    - ``.tflite`` → ``_accel_backend`` (NPU/GPU, optional) with automatic
      fallback to ``_cpu_backend`` (always present).
    - ``.onnx``   → ``_onnx_backend`` (CPU via ONNX Runtime).

    Args:
        preferred_unit: The compute unit to attempt first for TFLite models.

    Usage::

        backend = QCS6490Backend(preferred_unit=ComputeUnit.NPU)
        handle  = backend.load_model("mobileclip.tflite")
        outputs = backend.run(handle, image_tensor)
    """

    def __init__(self, preferred_unit: ComputeUnit = ComputeUnit.NPU) -> None:
        self._preferred_unit = preferred_unit

        # CPU backend is always available — the unconditional fallback.
        self._litert_cpu_backend: LiteRTBackend = LiteRTBackend(compute_unit=ComputeUnit.CPU)

        # Accelerator backend is optional — None if the delegate is missing.
        self._litert_accel_backend: LiteRTBackend | None = self._try_make_accel_backend(
            preferred_unit
        )

        # ONNX backend is always available (falls back to ImportError at load
        # time if onnxruntime is not installed).
        self._onnx_backend: ONNXBackend = ONNXBackend()

        logger.info(
            "QCS6490Backend: preferred=%s accel=%s",
            preferred_unit.name,
            self._litert_accel_backend.get_supported_unit().name
            if self._litert_accel_backend
            else "unavailable",
        )

    # ------------------------------------------------------------------
    # Sub-backend factories
    # ------------------------------------------------------------------

    @staticmethod
    def _try_make_accel_backend(unit: ComputeUnit) -> LiteRTBackend | None:
        """Try to create an NPU/GPU LiteRT backend; return ``None`` on failure.

        CPU is not an accelerator — if *unit* is ``CPU``, returns ``None``
        immediately so that ``_cpu_backend`` is used directly.
        """
        if unit not in (ComputeUnit.NPU, ComputeUnit.GPU):
            return None
        try:
            return LiteRTBackend(compute_unit=unit)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "%s delegate unavailable (%s) — TFLite will run on CPU",
                unit.name,
                e,
            )
            return None

    # ------------------------------------------------------------------
    # InferenceBackend interface
    # ------------------------------------------------------------------

    def load_model(self, path: str) -> Any:
        """Load a model, routing by file extension.

        ``.tflite`` files are tried on the accelerator first, then CPU.
        ``.onnx`` files go directly to the ONNX sub-backend.

        Args:
            path: Filesystem path to the model file.

        Returns:
            A :class:`_ModelHandle` — pass it back to :meth:`run`.

        Raises:
            ValueError: If the file extension is unrecognised.
            RuntimeError: If loading fails even on the CPU fallback path.
        """
        if path.endswith(_TFLITE_SUFFIX):
            return self._load_tflite(path)
        if path.endswith(_ONNX_SUFFIX):
            return self._load_onnx(path)

        msg = (
            f"Unsupported model format: {path!r}. Expected {_TFLITE_SUFFIX!r} or {_ONNX_SUFFIX!r}."
        )
        raise ValueError(msg)

    def run(self, handle: Any, inputs: ModelInput) -> list[np.ndarray]:
        """Run inference via the sub-backend that loaded the model.

        O(1) dispatch — no isinstance checks.  The handle already knows which
        sub-backend owns it.

        Args:
            handle: Handle returned by :meth:`load_model`.
            inputs: Single ndarray or name→tensor dict.

        Returns:
            List of output tensors, one per model output slot.
        """
        h: _ModelHandle = handle
        return h.backend.run(h.raw, inputs)

    def get_supported_unit(self) -> ComputeUnit:
        """Return the best compute unit available.

        Returns the accelerator unit if the delegate loaded successfully,
        otherwise ``CPU``.
        """
        if self._litert_accel_backend is not None:
            return self._litert_accel_backend.get_supported_unit()
        return self._litert_cpu_backend.get_supported_unit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_tflite(self, path: str) -> _ModelHandle:
        """Load a .tflite model, trying the accelerator then falling back to CPU."""
        if self._litert_accel_backend is not None:
            try:
                raw = self._litert_accel_backend.load_model(path)
                return _ModelHandle(raw=raw, backend=self._litert_accel_backend)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Accel load failed for %r (%s) — retrying on CPU",
                    path,
                    e,
                )

        raw = self._litert_cpu_backend.load_model(path)
        return _ModelHandle(raw=raw, backend=self._litert_cpu_backend)

    def _load_onnx(self, path: str) -> _ModelHandle:
        """Load an .onnx model via the ONNX sub-backend."""
        raw = self._onnx_backend.load_model(path)
        return _ModelHandle(raw=raw, backend=self._onnx_backend)
