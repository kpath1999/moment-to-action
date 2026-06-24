"""QCS6490 CPU backend — TFLite via LiteRT/XNNPACK + ONNX Runtime."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import onnxruntime as ort

from moment_to_action.hardware._backend import ComputeBackend
from moment_to_action.hardware._platforms._shared import _load_litert_interpreter
from moment_to_action.hardware._platforms.qcs6490._models import (
    QCS6490ONNXModel,
    QCS6490TfliteModel,
)
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    from moment_to_action.hardware._loaded_model import LoadedModel

logger = logging.getLogger(__name__)


class QCS6490CPUBackend(ComputeBackend):
    """CPU inference backend for the QCS6490.

    Handles TFLite models via LiteRT/XNNPACK and ONNX models via ONNX Runtime.
    This backend is always available — it is the unconditional fallback when
    the NPU or GPU backend cannot be initialized.
    """

    _SUPPORTED_DTYPES: frozenset[DataType] = frozenset({DataType.FP32})
    _SUPPORTED_FORMATS: frozenset[ModelType] = frozenset({ModelType.TFLITE, ModelType.ONNX})

    def __init__(self) -> None:
        """Initialize the QCS6490 CPU backend."""
        logger.info("QCS6490CPUBackend: initialized (LiteRT + ONNX Runtime)")

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
        """Load a TFLite model on CPU via LiteRT/XNNPACK.

        Args:
            path: Path to the ``.tflite`` model file.

        Returns:
            A :class:`~_models.QCS6490TfliteModel` backed by XNNPACK.
        """
        p = os.fspath(path)
        interp = _load_litert_interpreter(p)
        logger.info("QCS6490CPUBackend: loaded %s on CPU", p)
        return QCS6490TfliteModel(unit=ComputeUnit.CPU, interp=interp)

    def load_onnx(self, path: str | os.PathLike[str]) -> LoadedModel:
        """Load an ONNX model on CPU via ONNX Runtime.

        Args:
            path: Path to the ``.onnx`` model file.

        Returns:
            A :class:`~_models.QCS6490ONNXModel` backed by CPU EP.
        """
        p = os.fspath(path)
        session = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        logger.info("QCS6490CPUBackend: loaded %s via onnxruntime", p)
        return QCS6490ONNXModel(session=session)
