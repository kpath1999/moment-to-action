"""QCS6490 Adreno GPU backend — llama.cpp via Vulkan."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from moment_to_action.hardware._backend import ComputeBackend
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    from moment_to_action.hardware._loaded_model import LoadedModel

logger = logging.getLogger(__name__)


class QCS6490GPUBackend(ComputeBackend):
    """Adreno GPU inference backend for the QCS6490.

    Supports GGUF models via llama-server using the Vulkan backend on the
    Adreno GPU.  TFLite GPU execution is not supported (no Adreno TFLite
    delegate is available on this platform).
    """

    _SUPPORTED_DTYPES: frozenset[DataType] = frozenset({DataType.FP16, DataType.FP32})
    _SUPPORTED_FORMATS: frozenset[ModelType] = frozenset({ModelType.LLAMA_CPP})

    def __init__(self) -> None:
        """Initialize the QCS6490 GPU backend."""
        logger.info("QCS6490GPUBackend: initialized (Vulkan / Adreno GPU)")

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
        """Supported formats: LLAMA_CPP."""
        return set(self._SUPPORTED_FORMATS)

    def load_llama_cpp(
        self,
        path: str | os.PathLike[str],
        *,
        mmproj: str | os.PathLike[str] | None = None,
        server_path: str | os.PathLike[str] | None = None,
        port: int | None = None,
        dtype: DataType,
    ) -> LoadedModel:
        """Load a GGUF model via llama-server on the Adreno GPU (Vulkan).

        Args:
            path: Path to the ``.gguf`` model file.
            mmproj: Optional path to the multimodal projector file.
            server_path: Path to the ``llama-server`` binary. If ``None``,
                resolved by the caller from AppConfig or PATH.
            port: Port for llama-server. If ``None``, a free port is assigned.
            dtype: Data type of the model (e.g. ``DataType.FP32``).

        Returns:
            A :class:`~moment_to_action.hardware._loaded_models.LlamaModel`
            running on the Adreno GPU via Vulkan.
        """
        self._check_dtype(dtype)
        from moment_to_action.hardware._loaded_models._llama import (  # noqa: PLC0415
            _start_llama_model,
        )

        p = os.fspath(path)
        mp = os.fspath(mmproj) if mmproj is not None else None
        sp = os.fspath(server_path) if server_path is not None else None
        logger.info("QCS6490GPUBackend: loading %s via llama-server (GPU/Vulkan)", p)
        return _start_llama_model(
            path=p,
            mmproj=mp,
            server_path=sp,
            port=port,
            unit=ComputeUnit.GPU,
            cpu_only=False,
            dtype=dtype,
        )
