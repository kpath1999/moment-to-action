"""Unified inference backend for macOS arm64 (Apple Silicon).

``MacOSARM64Backend`` provides the same CPU-only LiteRT + ONNX Runtime
inference path as the x86_64 backend.  It is a dedicated class so that
future x86_64-specific changes (e.g. XNNPACK tuning, AVX flags) do not
accidentally break the macOS development path.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, cast

import attrs

if TYPE_CHECKING:
    import numpy as np

from moment_to_action.hardware._platforms._base import InferenceBackend, ModelInput
from moment_to_action.hardware._platforms._runtimes._litert import LiteRTBackend
from moment_to_action.hardware._platforms._runtimes._onnx import ONNXBackend
from moment_to_action.hardware._platforms._runtimes._torch_policy import (
    resolve_torch_execution_policy,
)
from moment_to_action.hardware._types import ComputeUnit

if TYPE_CHECKING:
    from moment_to_action.hardware._types import TorchExecutionPolicy

logger = logging.getLogger(__name__)

_TFLITE_SUFFIX = ".tflite"
_ONNX_SUFFIX = ".onnx"


# ---------------------------------------------------------------------------
# Internal handle type
# ---------------------------------------------------------------------------


@attrs.define(slots=True)
class _ModelHandle:
    """Opaque model handle that pairs a raw runtime object with its backend."""

    raw: Any = attrs.field(repr=False)
    backend: InferenceBackend = attrs.field(repr=False)


# ---------------------------------------------------------------------------
# Unified backend
# ---------------------------------------------------------------------------


class MacOSARM64Backend(InferenceBackend):
    """Unified inference backend for macOS arm64 (CPU-only).

    Internally delegates to format-specific sub-backends:

    - ``.tflite`` → LiteRT (CPU)
    - ``.onnx``   → ONNX Runtime (CPU)

    Usage::

        backend = MacOSARM64Backend()
        handle  = backend.load_model("mobileclip.tflite")
        outputs = backend.run(handle, image_tensor)
    """

    def __init__(self) -> None:
        self._litert_backend: LiteRTBackend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
        self._onnx_backend: ONNXBackend = ONNXBackend()

        logger.info("MacOSARM64Backend: CPU-only (LiteRT + ONNX Runtime)")

    # ------------------------------------------------------------------
    # InferenceBackend interface
    # ------------------------------------------------------------------

    def load_model(self, path: str | os.PathLike[str]) -> object:
        """Load a model, routing by file extension.

        Args:
            path: Filesystem path to the model file.

        Returns:
            A :class:`_ModelHandle` — pass it back to :meth:`run`.

        Raises:
            ValueError: If the file extension is unrecognised.
            RuntimeError: If loading fails.
        """
        path = os.fspath(path)
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
        """Return ``ComputeUnit.CPU`` (macOS arm64 is CPU-only)."""
        return ComputeUnit.CPU

    def resolve_torch_policy(self, requested: str = "auto") -> TorchExecutionPolicy:
        """Resolve torch execution policy for this platform."""
        return resolve_torch_execution_policy(requested)

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
