"""Hardware type definitions — enums and data models.

Kept separate to avoid circular imports: stages and preprocessors
import ``ComputeUnit`` from here, not from the backend module.
"""

from __future__ import annotations

import typing as t
from enum import Enum
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from datetime import datetime

    import torch


class ComputeUnit(str, Enum):
    """Available compute units on a hardware accelerator platform."""

    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"


class DataType(str, Enum):
    """Model data / quantization types."""

    W8A8 = "W8A8"
    """8-bit weights, 8-bit activations (full integer quantization)."""

    W8A16 = "W8A16"
    """8-bit weights, 16-bit activations (hybrid quantization)."""

    FP16 = "FP16"
    """16-bit floating point."""

    FP32 = "FP32"
    """32-bit floating point."""


class ModelType(str, Enum):
    """Supported model file formats."""

    ONNX = "ONNX"
    DLC = "DLC"
    TFLITE = "TFLITE"
    TORCH = "TORCH"
    LLAMA_CPP = "LLAMA_CPP"


class PlatformType(str, Enum):
    """Known hardware platforms supported by this codebase."""

    QCS6490 = "QCS6490"
    """Qualcomm QCS6490 (Snapdragon 778G) — Hexagon HTP NPU, Adreno 642L GPU."""

    X86_64 = "X86_64"
    """Standard x86_64 laptop/desktop CPU (Intel/AMD)."""

    MACOS_ARM64 = "MACOS_ARM64"
    """Apple Silicon macOS host for local development/testing."""


@attrs.frozen
class BenchmarkResult:
    """Latency statistics from a :meth:`~moment_to_action.hardware.Platform.benchmark` run.

    All times are in milliseconds.

    Attributes:
        mean_ms: Mean inference latency across all runs.
        p50_ms: Median (50th percentile) latency.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        min_ms: Minimum observed latency.
        max_ms: Maximum observed latency.
        compute_unit: Name of the compute unit used (e.g. ``"CPU"``).
        n_runs: Number of inference runs performed.
    """

    mean_ms: float
    """Mean inference latency across all runs."""

    p50_ms: float
    """Median (50th percentile) latency."""

    p95_ms: float
    """95th percentile latency."""

    p99_ms: float
    """99th percentile latency."""

    min_ms: float
    """Minimum observed latency."""

    max_ms: float
    """Maximum observed latency."""

    compute_unit: str
    """Name of the compute unit used (e.g. ``"CPU"``, ``"NPU"``)."""

    n_runs: int
    """Number of inference runs performed."""


@attrs.frozen
class ComputeUnitUsageSample:
    """A single compute unit usage measurement snapshot for one compute unit."""

    timestamp: datetime
    """Time when the sample was taken."""

    device: ComputeUnit
    """The device that was active during sampling."""

    usage_pct: float
    """Utilisation percentage (0-100)."""

    frequency_mhz: float
    """Operating frequency in MHz."""

    memory_mb: float
    """Memory usage in megabytes (across ALL processes)."""

    power_mw: float
    """Power draw in milliwatts."""

    def json(self) -> dict[str, t.Any]:
        """Generate a JSON-serializable dictionary representation of this sample."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "device": self.device,
            "usage_pct": self.usage_pct,
            "frequency_mhz": self.frequency_mhz,
            "memory_mb": self.memory_mb,
            "power_mw": self.power_mw,
        }


@attrs.frozen
class TorchExecutionPolicy:
    """Resolved torch execution configuration for model loading and inference.

    Attributes:
        device: Resolved ``torch.device`` instance.
        dtype: Resolved ``torch.dtype`` instance.
    """

    device: torch.device
    """Resolved torch device."""

    dtype: torch.dtype
    """Resolved torch dtype."""
