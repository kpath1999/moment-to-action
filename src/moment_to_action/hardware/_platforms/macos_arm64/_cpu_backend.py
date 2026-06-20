"""macOS arm64 (Apple Silicon) CPU backend — TFLite + ONNX Runtime."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import onnxruntime as ort

from moment_to_action.hardware._backend import ComputeBackend
from moment_to_action.hardware._platforms.macos_arm64._models import (
    MacOSARM64ONNXModel,
    MacOSARM64TfliteModel,
)
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    from moment_to_action.hardware._loaded_model import LoadedModel

logger = logging.getLogger(__name__)


def _load_litert_interpreter(path: str) -> object:
    """Load and allocate a LiteRT interpreter for CPU inference.

    Args:
        path: Filesystem path to the ``.tflite`` model file.

    Returns:
        An allocated LiteRT interpreter.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from tensorflow.lite.python.interpreter import Interpreter  # noqa: PLC0415

    interp = Interpreter(model_path=path, experimental_delegates=[])
    interp.allocate_tensors()
    return interp


class MacOSARM64CPUBackend(ComputeBackend):
    """CPU inference backend for macOS arm64 (Apple Silicon).

    Handles TFLite models via LiteRT and ONNX models via ONNX Runtime.
    CPU-only — no NPU or GPU inference on macOS arm64.
    """

    _SUPPORTED_DTYPES: frozenset[DataType] = frozenset({DataType.FP32})
    _SUPPORTED_FORMATS: frozenset[ModelType] = frozenset({ModelType.TFLITE, ModelType.ONNX})

    def __init__(self) -> None:
        """Initialize the macOS arm64 CPU backend."""
        self._interp_cache: dict[str, object] = {}
        self._session_cache: dict[str, object] = {}
        logger.info("MacOSARM64CPUBackend: initialized (LiteRT + ONNX Runtime)")

    @property
    def unit(self) -> ComputeUnit:
        """The compute unit — always CPU."""
        return ComputeUnit.CPU

    @property
    def supported_dtypes(self) -> set[DataType]:
        """Supported data types: FP32."""
        return set(self._SUPPORTED_DTYPES)

    @property
    def supported_formats(self) -> set[ModelType]:
        """Supported formats: TFLITE and ONNX."""
        return set(self._SUPPORTED_FORMATS)

    def load_tflite(self, path: str | os.PathLike[str]) -> LoadedModel:
        """Load a TFLite model on CPU via LiteRT.

        Args:
            path: Path to the ``.tflite`` model file.

        Returns:
            A :class:`~_models.MacOSARM64TfliteModel` backed by LiteRT.
        """
        p = os.fspath(path)
        if p not in self._interp_cache:
            self._interp_cache[p] = _load_litert_interpreter(p)
            logger.info("MacOSARM64CPUBackend: loaded %s on CPU", p)
        return MacOSARM64TfliteModel(interp=self._interp_cache[p])

    def load_onnx(self, path: str | os.PathLike[str]) -> LoadedModel:
        """Load an ONNX model on CPU via ONNX Runtime.

        Args:
            path: Path to the ``.onnx`` model file.

        Returns:
            A :class:`~_models.MacOSARM64ONNXModel` backed by CPU EP.
        """
        p = os.fspath(path)
        if p not in self._session_cache:
            session = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
            self._session_cache[p] = session
            logger.info("MacOSARM64CPUBackend: loaded %s via onnxruntime", p)
        return MacOSARM64ONNXModel(session=self._session_cache[p])
