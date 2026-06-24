"""x86_64 CPU backend — TFLite via LiteRT/XNNPACK + ONNX Runtime + DLC debug."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import onnxruntime as ort

from moment_to_action.hardware._backend import ComputeBackend
from moment_to_action.hardware._platforms._shared import _load_litert_interpreter
from moment_to_action.hardware._platforms.x86_64._models import (
    X86_64DLCModel,
    X86_64ONNXModel,
    X86_64TfliteModel,
)
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    from moment_to_action.hardware._loaded_model import LoadedModel

logger = logging.getLogger(__name__)

# QAIRT CPU backend string.
_QAIRT_CPU_BACKEND = "CPU"


class X86_64CPUBackend(ComputeBackend):  # noqa: N801
    """CPU inference backend for x86_64.

    Handles TFLite via LiteRT/XNNPACK, ONNX via ONNX Runtime, and DLC via
    QAIRT CPU backend (for local debugging of QCS6490 models without a device).

    DLC support requires the QAIRT SDK to be installed (``m2a qairt install``).
    """

    _SUPPORTED_DTYPES: frozenset[DataType] = frozenset({DataType.FP32})
    _SUPPORTED_FORMATS: frozenset[ModelType] = frozenset(
        {ModelType.TFLITE, ModelType.ONNX, ModelType.DLC}
    )

    def __init__(self) -> None:
        """Initialize the x86_64 CPU backend."""
        logger.info("X86_64CPUBackend: initialized (LiteRT + ONNX Runtime + DLC debug)")

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
        """Supported formats: TFLITE, ONNX, and DLC (debug)."""
        return set(self._SUPPORTED_FORMATS)

    def load_tflite(self, path: str | os.PathLike[str]) -> LoadedModel:
        """Load a TFLite model on CPU via LiteRT/XNNPACK.

        Args:
            path: Path to the ``.tflite`` model file.

        Returns:
            An :class:`~_models.X86_64TfliteModel` backed by XNNPACK.
        """
        p = os.fspath(path)
        interp = _load_litert_interpreter(p)
        logger.info("X86_64CPUBackend: loaded %s on CPU", p)
        return X86_64TfliteModel(interp=interp)

    def load_onnx(self, path: str | os.PathLike[str]) -> LoadedModel:
        """Load an ONNX model on CPU via ONNX Runtime.

        Args:
            path: Path to the ``.onnx`` model file.

        Returns:
            An :class:`~_models.X86_64ONNXModel` backed by CPU EP.
        """
        p = os.fspath(path)
        session = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        logger.info("X86_64CPUBackend: loaded %s via onnxruntime", p)
        return X86_64ONNXModel(session=session)

    def load_dlc(self, path: str | os.PathLike[str]) -> LoadedModel:
        """Load a DLC model via QAIRT CPU backend (debug path).

        Requires the QAIRT SDK to be installed (``m2a qairt install``).

        Args:
            path: Path to the ``.dlc`` model file.

        Returns:
            An :class:`~_models.X86_64DLCModel` initialized on the CPU backend.

        Raises:
            RuntimeError: If the QAIRT SDK is not installed.
        """
        try:
            import qairt  # noqa: PLC0415
        except Exception as exc:
            msg = "QAIRT SDK is not available; install it with 'm2a qairt install'"
            raise RuntimeError(msg) from exc
        raw = qairt.load(os.fspath(path))
        raw.initialize(backend=_QAIRT_CPU_BACKEND)
        logger.info("X86_64CPUBackend: loaded DLC %s on CPU backend", path)
        return X86_64DLCModel(raw=raw)
