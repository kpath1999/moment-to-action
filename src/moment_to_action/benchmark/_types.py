from __future__ import annotations

from datetime import datetime

import attrs

from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models import ModelID


def _default_batch_sizes() -> list[int]:
    """Default benchmark batch sizes."""
    return [1]


@attrs.frozen
class VariantID:
    """Composite key that uniquely identifies a model variant."""

    model_id: ModelID
    compute_unit: ComputeUnit


@attrs.frozen
class CostProfile:
    """Optional power and energy cost metrics for a profiled variant."""

    power_mw: float | None = None
    energy_per_inference_mj: float | None = None


@attrs.frozen
class VariantProfile:
    """INFaaS-style profile record for a specific model variant."""

    variant_id: VariantID
    accuracy: float | None
    load_latency_ms: float
    inference_mean_ms: float
    inference_p50_ms: float
    inference_p95_ms: float
    inference_p99_ms: float
    peak_memory_mb: float
    max_batch_size: int
    hardware_target: str
    cost: CostProfile
    model_size_bytes: int
    n_runs: int
    profiled_at: datetime

    def json(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this profile."""
        return {
            "variant_id": {
                "model_id": self.variant_id.model_id.value,
                "compute_unit": self.variant_id.compute_unit.value,
            },
            "accuracy": self.accuracy,
            "load_latency_ms": self.load_latency_ms,
            "inference_mean_ms": self.inference_mean_ms,
            "inference_p50_ms": self.inference_p50_ms,
            "inference_p95_ms": self.inference_p95_ms,
            "inference_p99_ms": self.inference_p99_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "max_batch_size": self.max_batch_size,
            "hardware_target": self.hardware_target,
            "cost": {
                "power_mw": self.cost.power_mw,
                "energy_per_inference_mj": self.cost.energy_per_inference_mj,
            },
            "model_size_bytes": self.model_size_bytes,
            "n_runs": self.n_runs,
            "profiled_at": self.profiled_at.isoformat(),
        }


@attrs.frozen
class BenchmarkConfig:
    """Configuration for a model benchmark session."""

    n_warmup: int = 5
    n_runs: int = 20
    batch_sizes: list[int] = attrs.Factory(_default_batch_sizes)
