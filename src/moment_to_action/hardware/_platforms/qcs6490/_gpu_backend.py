"""QCS6490 Adreno GPU backend — TFLite via GPU delegate (placeholder)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from moment_to_action.hardware._backend import ComputeBackend
from moment_to_action.hardware._platforms._shared import _load_litert_interpreter
from moment_to_action.hardware._platforms.qcs6490._models import QCS6490TfliteModel
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    from moment_to_action.hardware._loaded_model import LoadedModel

logger = logging.getLogger(__name__)


class QCS6490GPUBackend(ComputeBackend):
    """Adreno GPU inference backend for the QCS6490.

    Currently runs TFLite models without a GPU delegate (no Adreno TFLite
    delegate is available on this platform).  Models execute on CPU via
    XNNPACK as a fallback.  A future update will wire in the Adreno GPU EP
    when the delegate becomes available.
    """

    _SUPPORTED_DTYPES: frozenset[DataType] = frozenset({DataType.FP16, DataType.FP32})
    _SUPPORTED_FORMATS: frozenset[ModelType] = frozenset({ModelType.TFLITE})

    def __init__(self) -> None:
        """Initialize the GPU backend (no delegate loaded currently)."""
        logger.info(
            "QCS6490GPUBackend: initialized (GPU delegate not yet available, using CPU fallback)"
        )

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
        """Supported formats: TFLITE."""
        return set(self._SUPPORTED_FORMATS)

    def load_tflite(self, path: str | os.PathLike[str]) -> LoadedModel:
        """Load a TFLite model targeting the Adreno GPU.

        Currently falls through to CPU (XNNPACK) because no Adreno TFLite
        delegate is bundled on this platform.

        Args:
            path: Path to the ``.tflite`` model file.

        Returns:
            A :class:`~_models.QCS6490TfliteModel` running on GPU (currently CPU).
        """
        p = os.fspath(path)
        interp = _load_litert_interpreter(p)
        logger.info("QCS6490GPUBackend: loaded %s (CPU fallback)", p)
        return QCS6490TfliteModel(unit=ComputeUnit.GPU, interp=interp)
