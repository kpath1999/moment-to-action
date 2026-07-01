"""Hardware Abstraction Layer for moment-to-action."""

from __future__ import annotations

from ._loaded_model import LoadedModel, LoadedStreamableModel
from ._metrics import InferenceMetrics, LlamaCppInferenceMetrics
from ._platform import Platform, detect_platform
from ._types import (
    ComputeUnit,
    ComputeUnitUsageSample,
    DataType,
    ModelType,
    PlatformType,
)

__all__ = [
    "ComputeUnit",
    "ComputeUnitUsageSample",
    "DataType",
    "InferenceMetrics",
    "LlamaCppInferenceMetrics",
    "LoadedModel",
    "LoadedStreamableModel",
    "ModelType",
    "Platform",
    "PlatformType",
    "detect_platform",
]
