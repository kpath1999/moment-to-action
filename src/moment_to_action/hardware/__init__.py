"""Hardware Abstraction Layer for moment-to-action.

Public API::

    from moment_to_action.hardware import (
        BenchmarkResult,
        ComputeUnit,
        DataType,
        LoadedModel,
        ModelType,
        Platform,
        PlatformType,
    )
"""

from __future__ import annotations

from ._loaded_model import LoadedModel
from ._platform import Platform
from ._types import (
    BenchmarkResult,
    ComputeUnit,
    ComputeUnitUsageSample,
    DataType,
    ModelType,
    PlatformType,
)

__all__ = [
    "BenchmarkResult",
    "ComputeUnit",
    "ComputeUnitUsageSample",
    "DataType",
    "LoadedModel",
    "ModelType",
    "Platform",
    "PlatformType",
]
