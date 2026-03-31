"""Hardware Abstraction Layer for moment-to-action.

Public API::

    from moment_to_action.hardware import (
        BenchmarkResult,
        ComputeBackend,
        ComputeUnit,
        PowerSample,
        TorchExecutionPolicy,
    )
"""

from __future__ import annotations

from ._backend import BenchmarkResult, ComputeBackend
from ._types import ComputeUnit, PowerSample, TorchExecutionPolicy

__all__ = [
    "BenchmarkResult",
    "ComputeBackend",
    "ComputeUnit",
    "PowerSample",
    "TorchExecutionPolicy",
]
