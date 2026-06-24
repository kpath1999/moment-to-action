"""macOS arm64 GPU backend — MPS (Metal) via PyTorch + llama.cpp."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from moment_to_action.hardware._backend import ComputeBackend
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    from moment_to_action.hardware._loaded_model import LoadedModel

logger = logging.getLogger(__name__)


class MacOSARM64GPUBackend(ComputeBackend):
    """GPU (MPS/Metal) inference backend for macOS arm64 (Apple Silicon).

    Handles PyTorch models on MPS and GGUF models via llama-server with
    Metal GPU layers.

    Raises:
        RuntimeError: At construction time if MPS is not available.
    """

    _SUPPORTED_DTYPES: frozenset[DataType] = frozenset({DataType.FP16, DataType.FP32})
    _SUPPORTED_FORMATS: frozenset[ModelType] = frozenset({ModelType.TORCH, ModelType.LLAMA_CPP})

    def __init__(self) -> None:
        """Initialize the macOS arm64 GPU backend.

        Raises:
            RuntimeError: If MPS is not available.
        """
        import torch  # noqa: PLC0415

        if not torch.backends.mps.is_available():
            msg = "MPS not available; macOS arm64 GPU backend requires Apple Silicon with Metal"
            raise RuntimeError(msg)
        logger.info("MacOSARM64GPUBackend: initialized (MPS/Metal)")

    @property
    def unit(self) -> ComputeUnit:
        """The compute unit — GPU."""
        return ComputeUnit.GPU

    @property
    def supported_dtypes(self) -> set[DataType]:
        """Supported data types: FP16 and FP32."""
        return set(self._SUPPORTED_DTYPES)

    @property
    def supported_formats(self) -> set[ModelType]:
        """Supported formats: TORCH and LLAMA_CPP."""
        return set(self._SUPPORTED_FORMATS)

    def load_torch(self, path: str | os.PathLike[str]) -> LoadedModel:
        """Load a PyTorch model on MPS (Metal).

        Args:
            path: Path to the saved model file.

        Returns:
            A :class:`~moment_to_action.hardware._loaded_models.TorchModel`
            running on MPS.
        """
        import torch  # noqa: PLC0415

        from moment_to_action.hardware._loaded_models._torch import TorchModel  # noqa: PLC0415

        p = os.fspath(path)
        model = torch.load(p, map_location="mps", weights_only=False)
        logger.info("MacOSARM64GPUBackend: loaded %s via PyTorch on MPS", p)
        return TorchModel(unit=ComputeUnit.GPU, model=model)

    def load_llama_cpp(
        self,
        path: str | os.PathLike[str],
        *,
        mmproj: str | os.PathLike[str] | None = None,
        server_path: str | os.PathLike[str] | None = None,
        port: int | None = None,
    ) -> LoadedModel:
        """Load a GGUF model via llama-server on GPU (Metal).

        Args:
            path: Path to the ``.gguf`` model file.
            mmproj: Optional path to the multimodal projector file.
            server_path: Path to the ``llama-server`` binary.
            port: Port for llama-server. If ``None``, a free port is assigned.

        Returns:
            A :class:`~moment_to_action.hardware._loaded_models.LlamaModel`
            running on GPU.
        """
        from moment_to_action.hardware._loaded_models._llama import (  # noqa: PLC0415
            _start_llama_model,
        )

        p = os.fspath(path)
        mp = os.fspath(mmproj) if mmproj is not None else None
        sp = os.fspath(server_path) if server_path is not None else None
        logger.info("MacOSARM64GPUBackend: loading %s via llama-server (Metal)", p)
        return _start_llama_model(
            path=p, mmproj=mp, server_path=sp, port=port, unit=ComputeUnit.GPU, cpu_only=False
        )
