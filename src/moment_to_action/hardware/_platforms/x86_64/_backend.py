"""Unified inference backend for the x86_64 platform.

``X86_64Backend`` is the single :class:`InferenceBackend` that the platform
package exposes.  It owns:

- **Format routing** — ``.tflite`` files go to LiteRT, ``.onnx`` files go to
  ONNX Runtime.  Both sub-backends are created eagerly at construction time.
- **Model handle tracking** — each loaded model is paired (via ``_ModelHandle``)
  with the sub-backend that created it, so ``run()`` needs no isinstance checks.

All the complexity lives here so that :class:`ComputeBackend` (the public
orchestrator) can be a three-field thin wrapper.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, cast

import attrs

if TYPE_CHECKING:
    import numpy as np

from moment_to_action.hardware._platforms._base import InferenceBackend, ModelInput
from moment_to_action.hardware._platforms.x86_64._litert import X86_64LiteRTBackend
from moment_to_action.hardware._platforms.x86_64._onnx import X86_64ONNXBackend
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)

# Supported model file suffixes.
_TFLITE_SUFFIX = ".tflite"
_ONNX_SUFFIX = ".onnx"


# ---------------------------------------------------------------------------
# Internal handle type
# ---------------------------------------------------------------------------


@attrs.define(slots=True)
class _ModelHandle:
    """Opaque model handle that pairs a raw runtime object with its backend.

    Callers treat this as completely opaque — they just pass it back to
    ``X86_64Backend.run()`` and friends.
    """

    raw: Any = attrs.field(repr=False)
    backend: InferenceBackend = attrs.field(repr=False)


# ---------------------------------------------------------------------------
# Unified backend
# ---------------------------------------------------------------------------


class X86_64Backend(InferenceBackend):  # noqa: N801
    """Unified inference backend for x86_64 (CPU-only).

    Internally delegates to format-specific sub-backends, all created eagerly:

    - ``.tflite`` → ``_litert_backend`` (CPU via LiteRT/XNNPACK)
    - ``.onnx``   → ``_onnx_backend`` (CPU via ONNX Runtime)

    Both CPU-only; GPU support (CUDA/ROCm) can be added in the future via
    sub-backend overrides.

    Usage::

        backend = X86_64Backend()
        handle  = backend.load_model("mobileclip.tflite")
        outputs = backend.run(handle, image_tensor)
    """

    def __init__(self) -> None:
        # CPU-only backends — always available.
        self._litert_backend: X86_64LiteRTBackend = X86_64LiteRTBackend(
            compute_unit=ComputeUnit.CPU
        )
        self._onnx_backend: X86_64ONNXBackend = X86_64ONNXBackend()

        logger.info("X86_64Backend: CPU-only (LiteRT + ONNX Runtime)")

    # ------------------------------------------------------------------
    # InferenceBackend interface
    # ------------------------------------------------------------------

    def load_model(self, path: str | os.PathLike[str]) -> object:
        """Load a model, routing by file extension.

        ``.tflite`` files go to the LiteRT sub-backend.
        ``.onnx`` files go directly to the ONNX sub-backend.

        Args:
            path: Filesystem path to the model file.

        Returns:
            A :class:`_ModelHandle` — pass it back to :meth:`run`.

        Raises:
            ValueError: If the file extension is unrecognised.
            RuntimeError: If loading fails.
        """
        path = os.fspath(path)  # normalise to str for extension checks
        if path.endswith(_TFLITE_SUFFIX):
            return self._load_tflite(path)
        if path.endswith(_ONNX_SUFFIX):
            return self._load_onnx(path)

        msg = (
            f"Unsupported model format: {path!r}. Expected {_TFLITE_SUFFIX!r} or {_ONNX_SUFFIX!r}."
        )
        raise ValueError(msg)

    def run(self, handle: object, inputs: ModelInput) -> list[np.ndarray]:
        """Run inference via the sub-backend that loaded the model.

        O(1) dispatch — no isinstance checks.  The handle already knows which
        sub-backend owns it.

        Args:
            handle: Handle returned by :meth:`load_model`.
            inputs: Single ndarray or name→tensor dict.

        Returns:
            List of output tensors, one per model output slot.
        """
        h = cast("_ModelHandle", handle)
        return h.backend.run(h.raw, inputs)

    def get_input_details(self, handle: object) -> list[dict]:
        """Return input tensor metadata, delegating to the owning sub-backend.

        Args:
            handle: Handle returned by :meth:`load_model`.
        """
        h = cast("_ModelHandle", handle)
        return h.backend.get_input_details(h.raw)

    def get_output_details(self, handle: object) -> list[dict]:
        """Return output tensor metadata, delegating to the owning sub-backend.

        Args:
            handle: Handle returned by :meth:`load_model`.
        """
        h = cast("_ModelHandle", handle)
        return h.backend.get_output_details(h.raw)

    def get_supported_unit(self) -> ComputeUnit:
        """Return ``ComputeUnit.CPU`` (x86_64 is CPU-only)."""
        return ComputeUnit.CPU

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_tflite(self, path: str) -> _ModelHandle:
        """Load a .tflite model via the LiteRT sub-backend."""
        raw = self._litert_backend.load_model(path)
        return _ModelHandle(raw=raw, backend=self._litert_backend)

    def _load_onnx(self, path: str) -> _ModelHandle:
        """Load an .onnx model via the ONNX sub-backend."""
        raw = self._onnx_backend.load_model(path)
        return _ModelHandle(raw=raw, backend=self._onnx_backend)
