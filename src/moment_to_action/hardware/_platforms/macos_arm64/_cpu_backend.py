"""macOS arm64 (Apple Silicon) CPU backend — TFLite + ONNX Runtime + Torch + llama.cpp."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import onnxruntime as ort

from moment_to_action.hardware._backend import ComputeBackend
from moment_to_action.hardware._loaded_models._onnx import OnnxModel
from moment_to_action.hardware._loaded_models._tflite import TfliteModel
from moment_to_action.hardware._platforms._shared import _load_litert_interpreter
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    from moment_to_action.hardware._loaded_model import LoadedModel

logger = logging.getLogger(__name__)


class MacOSARM64CPUBackend(ComputeBackend):
    """CPU inference backend for macOS arm64 (Apple Silicon).

    Handles TFLite models via LiteRT, ONNX models via ONNX Runtime,
    PyTorch models on CPU, and GGUF models via llama-server on CPU (``--ngl 0``).
    GPU inference (MPS/Metal) is handled by the GPU backend.
    """

    _SUPPORTED_DTYPES: frozenset[DataType] = frozenset({DataType.FP32})
    _SUPPORTED_FORMATS: frozenset[ModelType] = frozenset(
        {ModelType.TFLITE, ModelType.ONNX, ModelType.TORCH, ModelType.LLAMA_CPP}
    )

    def __init__(self) -> None:
        """Initialize the macOS arm64 CPU backend."""
        logger.info("MacOSARM64CPUBackend: initialized (LiteRT + ONNX Runtime + Torch + llama.cpp)")

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
        """Supported formats: TFLITE, ONNX, TORCH, and LLAMA_CPP."""
        return set(self._SUPPORTED_FORMATS)

    def load_tflite(self, path: str | os.PathLike[str], *, dtype: DataType) -> LoadedModel:
        """Load a TFLite model on CPU via LiteRT.

        Args:
            path: Path to the ``.tflite`` model file.
            dtype: Data type of the model (e.g. ``DataType.FP32``).

        Returns:
            A :class:`~moment_to_action.hardware._loaded_models.TfliteModel`
            backed by LiteRT.
        """
        self._check_dtype(dtype)
        p = os.fspath(path)
        interp = _load_litert_interpreter(p)
        logger.info("MacOSARM64CPUBackend: loaded %s on CPU", p)
        return TfliteModel(unit=ComputeUnit.CPU, interp=interp, dtype=dtype)

    def load_onnx(self, path: str | os.PathLike[str], *, dtype: DataType) -> LoadedModel:
        """Load an ONNX model on CPU via ONNX Runtime.

        Args:
            path: Path to the ``.onnx`` model file.
            dtype: Data type of the model (e.g. ``DataType.FP32``).

        Returns:
            An :class:`~moment_to_action.hardware._loaded_models.OnnxModel`
            backed by CPU EP.
        """
        self._check_dtype(dtype)
        p = os.fspath(path)
        session = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        logger.info("MacOSARM64CPUBackend: loaded %s via onnxruntime", p)
        return OnnxModel(unit=ComputeUnit.CPU, session=session, dtype=dtype)

    def load_torch(self, path: str | os.PathLike[str], *, dtype: DataType) -> LoadedModel:
        """Load a PyTorch model on CPU.

        Args:
            path: Path to the saved model file.
            dtype: Data type of the model (e.g. ``DataType.FP32``).

        Returns:
            A :class:`~moment_to_action.hardware._loaded_models.TorchModel`
            running on CPU.
        """
        self._check_dtype(dtype)
        import torch  # noqa: PLC0415

        from moment_to_action.hardware._loaded_models._torch import TorchModel  # noqa: PLC0415

        p = os.fspath(path)
        model = torch.load(p, map_location="cpu", weights_only=False)
        logger.info("MacOSARM64CPUBackend: loaded %s via PyTorch on CPU", p)
        return TorchModel(unit=ComputeUnit.CPU, model=model, dtype=dtype)

    def load_llama_cpp(
        self,
        path: str | os.PathLike[str],
        *,
        mmproj: str | os.PathLike[str] | None = None,
        server_path: str | os.PathLike[str] | None = None,
        port: int | None = None,
        dtype: DataType,
    ) -> LoadedModel:
        """Load a GGUF model via llama-server on CPU (``--ngl 0``).

        Args:
            path: Path to the ``.gguf`` model file.
            mmproj: Optional path to the multimodal projector file.
            server_path: Path to the ``llama-server`` binary.
            port: Port for llama-server. If ``None``, a free port is assigned.
            dtype: Data type of the model (e.g. ``DataType.FP32``).

        Returns:
            A :class:`~moment_to_action.hardware._loaded_models.LlamaModel`
            running on CPU.
        """
        self._check_dtype(dtype)
        from moment_to_action.hardware._loaded_models._llama import (  # noqa: PLC0415
            _start_llama_model,
        )

        p = os.fspath(path)
        mp = os.fspath(mmproj) if mmproj is not None else None
        sp = os.fspath(server_path) if server_path is not None else None
        logger.info("MacOSARM64CPUBackend: loading %s via llama-server (CPU)", p)
        return _start_llama_model(
            path=p,
            mmproj=mp,
            server_path=sp,
            port=port,
            unit=ComputeUnit.CPU,
            cpu_only=True,
            dtype=dtype,
        )
