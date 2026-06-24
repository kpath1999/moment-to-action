"""x86_64 GPU backend — CUDA via PyTorch + llama.cpp."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch

from moment_to_action.hardware._backend import ComputeBackend
from moment_to_action.hardware._loaded_models._llama import _start_llama_model
from moment_to_action.hardware._loaded_models._torch import TorchModel
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    from moment_to_action.hardware._loaded_model import LoadedModel

logger = logging.getLogger(__name__)


class X86_64GPUBackend(ComputeBackend):  # noqa: N801
    """GPU (CUDA) inference backend for x86_64.

    Handles PyTorch models on CUDA and GGUF models via llama-server with
    CUDA GPU layers.

    Raises:
        RuntimeError: At construction time if CUDA is not available.
    """

    _SUPPORTED_DTYPES: frozenset[DataType] = frozenset({DataType.FP16, DataType.FP32})
    _SUPPORTED_FORMATS: frozenset[ModelType] = frozenset({ModelType.TORCH, ModelType.LLAMA_CPP})

    def __init__(self) -> None:
        """Initialize the x86_64 GPU backend.

        Raises:
            RuntimeError: If CUDA is not available.
        """
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            msg = "CUDA not available; x86_64 GPU backend requires a CUDA-capable GPU"
            raise RuntimeError(msg)
        logger.info("X86_64GPUBackend: initialized (CUDA)")

    @property
    def unit(self) -> ComputeUnit:
        """The compute unit — GPU."""
        return ComputeUnit.GPU

    @property
    def supported_dtypes(self) -> set[DataType]:
        """Supported data types."""
        return set(self._SUPPORTED_DTYPES)

    @property
    def supported_formats(self) -> set[ModelType]:
        """Supported formats."""
        return set(self._SUPPORTED_FORMATS)

    def load_torch(self, path: str | os.PathLike[str], *, dtype: DataType) -> LoadedModel:
        """Load a PyTorch model on CUDA.

        Args:
            path: Path to the saved model file.
            dtype: Data type of the model (e.g. ``DataType.FP32``).

        Returns:
            A :class:`~moment_to_action.hardware._loaded_models.TorchModel`
            running on CUDA.
        """
        self._check_dtype(dtype)

        p = os.fspath(path)
        model = torch.load(p, map_location="cuda", weights_only=False)
        logger.info("X86_64GPUBackend: loaded %s via PyTorch on CUDA", p)
        return TorchModel(unit=ComputeUnit.GPU, model=model, dtype=dtype)

    def load_llama_cpp(
        self,
        path: str | os.PathLike[str],
        *,
        mmproj: str | os.PathLike[str] | None = None,
        server_path: str | os.PathLike[str] | None = None,
        port: int | None = None,
        dtype: DataType,
    ) -> LoadedModel:
        """Load a GGUF model via llama-server on GPU (CUDA).

        Args:
            path: Path to the ``.gguf`` model file.
            mmproj: Optional path to the multimodal projector file.
            server_path: Path to the ``llama-server`` binary.
            port: Port for llama-server. If ``None``, a free port is assigned.
            dtype: Data type of the model (e.g. ``DataType.FP32``).

        Returns:
            A :class:`~moment_to_action.hardware._loaded_models.LlamaModel`
            running on GPU.
        """
        self._check_dtype(dtype)

        p = os.fspath(path)
        mp = os.fspath(mmproj) if mmproj is not None else None
        sp = os.fspath(server_path) if server_path is not None else None
        logger.info("X86_64GPUBackend: loading %s via llama-server (CUDA)", p)
        return _start_llama_model(
            path=p,
            mmproj=mp,
            server_path=sp,
            port=port,
            unit=ComputeUnit.GPU,
            cpu_only=False,
            dtype=dtype,
        )
