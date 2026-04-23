from __future__ import annotations

from datetime import UTC, datetime

import pytest

from moment_to_action.benchmark import BenchmarkConfig, CostProfile, VariantID, VariantProfile
from moment_to_action.hardware import ComputeUnit
from moment_to_action.models import ModelID


@pytest.mark.unit
class TestBenchmarkTypes:
    """Tests for benchmark type objects."""

    def test_variant_id_is_hashable(self) -> None:
        """VariantID is usable as a dict key."""
        variant_id = VariantID(model_id=ModelID.YOLO_V12_N, compute_unit=ComputeUnit.CPU)
        data = {variant_id: "ok"}
        assert data[variant_id] == "ok"

    def test_cost_profile_defaults(self) -> None:
        """CostProfile defaults all optional fields to None."""
        cost = CostProfile()
        assert cost.power_mw is None
        assert cost.energy_per_inference_mj is None

    def test_variant_profile_json(self) -> None:
        """VariantProfile.json() serialises variant_id and profiled_at correctly."""
        profiled_at = datetime(2026, 1, 1, tzinfo=UTC)
        profile = VariantProfile(
            variant_id=VariantID(model_id=ModelID.MOBILECLIP_S2, compute_unit=ComputeUnit.NPU),
            accuracy=0.77,
            load_latency_ms=12.0,
            inference_mean_ms=6.5,
            inference_p50_ms=6.0,
            inference_p95_ms=7.0,
            inference_p99_ms=7.5,
            peak_memory_mb=123.0,
            max_batch_size=4,
            hardware_target="x86_64",
            cost=CostProfile(power_mw=1500.0, energy_per_inference_mj=9.75),
            model_size_bytes=1024,
            n_runs=20,
            profiled_at=profiled_at,
        )

        payload = profile.json()
        assert payload["variant_id"] == {
            "model_id": "mobileclip_s2",
            "compute_unit": "NPU",
        }
        assert payload["profiled_at"] == profiled_at.isoformat()

    def test_benchmark_config_defaults(self) -> None:
        """BenchmarkConfig initialises with expected default values."""
        config = BenchmarkConfig()
        assert config.n_warmup == 5
        assert config.n_runs == 20
        assert config.batch_sizes == [1]

    def test_types_are_frozen(self) -> None:
        """Frozen attrs classes raise AttributeError on mutation attempts."""
        profile = CostProfile(power_mw=1.0, energy_per_inference_mj=2.0)
        with pytest.raises(AttributeError):
            profile.power_mw = 2.0  # type: ignore[misc]
