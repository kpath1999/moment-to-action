"""Unified inference backend for the QCS6490 platform.

``QCS6490Backend`` is the single :class:`InferenceBackend` that the platform
package exposes.  It owns:

- **Format routing** — ``.tflite`` files go to LiteRT (or CPU fallback),
  ``.onnx`` files go to ONNX Runtime.  Sub-backends are created lazily.
- **Accelerator fallback** — if the NPU/GPU backend raises, the model is
  retried on the CPU sub-backend.
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

    Internally delegates to format-specific sub-backends:

    - ``.tflite`` → :class:`LiteRTBackend` (NPU/GPU or CPU)
    - ``.onnx``   → :class:`ONNXBackend` (created on first ``.onnx`` load)

    If the accelerated sub-backend fails at load time, the model is retried
    on CPU transparently.

    Args:
        preferred_unit: The compute unit to attempt first for TFLite models.

    Usage::

        backend = QCS6490Backend(preferred_unit=ComputeUnit.NPU)
        handle  = backend.load_model("mobileclip.tflite")
        outputs = backend.run(handle, image_tensor)
    """

    def __init__(self, preferred_unit: ComputeUnit = ComputeUnit.NPU) -> None:
        self._preferred_unit = preferred_unit

        # Sub-backend for .tflite models — set up eagerly.
        self._litert_backend: InferenceBackend = self._make_litert_backend(preferred_unit)

        # Sub-backend for .onnx models — created lazily on first use.
        self._onnx_backend: InferenceBackend | None = None

        logger.info(
            "QCS6490Backend: preferred=%s, litert_unit=%s",
            preferred_unit.name,
            self._litert_backend.get_supported_unit().name,
        )

    # ------------------------------------------------------------------
    # Sub-backend factories
    # ------------------------------------------------------------------

    @staticmethod
    def _make_litert_backend(unit: ComputeUnit) -> InferenceBackend:
        """Create the best TFLite sub-backend for *unit*.

        Tries NPU/GPU first; silently falls back to ``LiteRTBackend(CPU)``
        if the hardware delegate is unavailable (e.g. dev machine, CI).
        ``LiteRTBackend`` handles CPU natively — no separate class needed.
        """
        from moment_to_action.hardware._platforms.qcs6490._litert import LiteRTBackend

        try:
            if unit in (ComputeUnit.NPU, ComputeUnit.GPU):
                return LiteRTBackend(compute_unit=unit)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "%s delegate unavailable (%s) — falling back to CPU",
                unit.name,
                e,
            )
        return LiteRTBackend(compute_unit=ComputeUnit.CPU)

    def _get_onnx_backend(self) -> InferenceBackend:
        """Return the ONNX sub-backend, creating it on first call."""
        if self._onnx_backend is None:
            from moment_to_action.hardware._platforms.qcs6490._onnx import ONNXBackend

            self._onnx_backend = ONNXBackend()
        return self._onnx_backend

    # ------------------------------------------------------------------
    # InferenceBackend interface
    # ------------------------------------------------------------------

    def load_model(self, path: str) -> Any:
        """Load a model, routing by file extension.

        ``.tflite`` files are loaded by the LiteRT/CPU sub-backend.  If the
        accelerated backend fails, loading is retried on CPU.  ``.onnx`` files
        are loaded by the ONNX sub-backend (created on first use).

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
        """Return the compute unit of the LiteRT sub-backend.

        This reflects the *actual* unit in use (may be CPU even if NPU was
        requested, due to fallback).
        """
        return self._litert_backend.get_supported_unit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_tflite(self, path: str) -> _ModelHandle:
        """Load a .tflite model, falling back to CPU on accelerator failure."""
        from moment_to_action.hardware._platforms.qcs6490._litert import LiteRTBackend

        try:
            raw = self._litert_backend.load_model(path)
            return _ModelHandle(raw=raw, backend=self._litert_backend)
        except Exception as e:
            if self._litert_backend.get_supported_unit() != ComputeUnit.CPU:
                logger.warning(
                    "Model load failed on %s (%s) — retrying on CPU",
                    self._litert_backend.get_supported_unit().name,
                    e,
                )
                self._litert_backend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
                raw = self._litert_backend.load_model(path)
                return _ModelHandle(raw=raw, backend=self._litert_backend)
            raise

    def _load_onnx(self, path: str) -> _ModelHandle:
        """Load an .onnx model via the lazily-created ONNX sub-backend."""
        onnx = self._get_onnx_backend()
        raw = onnx.load_model(path)
        return _ModelHandle(raw=raw, backend=onnx)
