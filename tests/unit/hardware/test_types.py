"""Unit tests for hardware types."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from moment_to_action.hardware._types import ComputeUnit, ComputeUnitUsageSample


def _make_sample(**kwargs: object) -> ComputeUnitUsageSample:
    """Helper to build a ComputeUnitUsageSample with sensible defaults."""
    defaults: dict[str, object] = {
        "timestamp": datetime.now(tz=UTC),
        "device": ComputeUnit.CPU,
        "usage_pct": 0.0,
        "frequency_mhz": 0.0,
        "memory_mb": 0.0,
        "power_mw": 0.0,
    }
    return ComputeUnitUsageSample(**{**defaults, **kwargs})  # type: ignore[arg-type]


@pytest.mark.unit
class TestComputeUnit:
    """Tests for ComputeUnit enum."""

    def test_computeunit_has_all_members(self) -> None:
        """Test that ComputeUnit has all expected members."""
        members = [member.name for member in ComputeUnit]
        assert "CPU" in members
        assert "NPU" in members
        assert "GPU" in members
        assert "DSP" in members
        assert len(members) == 4

    def test_computeunit_cpu_member(self) -> None:
        """Test that CPU member exists and is accessible."""
        assert hasattr(ComputeUnit, "CPU")
        assert isinstance(ComputeUnit.CPU, ComputeUnit)

    def test_computeunit_npu_member(self) -> None:
        """Test that NPU member exists and is accessible."""
        assert hasattr(ComputeUnit, "NPU")
        assert isinstance(ComputeUnit.NPU, ComputeUnit)

    def test_computeunit_gpu_member(self) -> None:
        """Test that GPU member exists and is accessible."""
        assert hasattr(ComputeUnit, "GPU")
        assert isinstance(ComputeUnit.GPU, ComputeUnit)

    def test_computeunit_dsp_member(self) -> None:
        """Test that DSP member exists and is accessible."""
        assert hasattr(ComputeUnit, "DSP")
        assert isinstance(ComputeUnit.DSP, ComputeUnit)


@pytest.mark.unit
class TestComputeUnitUsageSample:
    """Tests for ComputeUnitUsageSample data class."""

    def test_sample_construction_basic(self) -> None:
        """Test basic construction with valid values."""
        ts = datetime.now(tz=UTC)
        sample = ComputeUnitUsageSample(
            timestamp=ts,
            device=ComputeUnit.CPU,
            usage_pct=75.5,
            frequency_mhz=2400.0,
            memory_mb=4096.0,
            power_mw=150.0,
        )
        assert sample.timestamp == ts
        assert sample.device == ComputeUnit.CPU
        assert sample.usage_pct == 75.5
        assert sample.frequency_mhz == 2400.0
        assert sample.memory_mb == 4096.0
        assert sample.power_mw == 150.0

    def test_sample_construction_npu(self) -> None:
        """Test construction with NPU compute unit."""
        sample = _make_sample(device=ComputeUnit.NPU, power_mw=200.0, usage_pct=90.0)
        assert sample.device == ComputeUnit.NPU
        assert sample.power_mw == 200.0

    def test_sample_construction_gpu(self) -> None:
        """Test construction with GPU compute unit."""
        sample = _make_sample(device=ComputeUnit.GPU, power_mw=500.0, usage_pct=95.0)
        assert sample.device == ComputeUnit.GPU
        assert sample.power_mw == 500.0

    def test_sample_construction_dsp(self) -> None:
        """Test construction with DSP compute unit."""
        sample = _make_sample(device=ComputeUnit.DSP, power_mw=100.0, usage_pct=50.0)
        assert sample.device == ComputeUnit.DSP
        assert sample.power_mw == 100.0

    def test_sample_with_zero_values(self) -> None:
        """Test construction with zero values."""
        sample = _make_sample(usage_pct=0.0, frequency_mhz=0.0, memory_mb=0.0, power_mw=0.0)
        assert sample.usage_pct == 0.0
        assert sample.power_mw == 0.0

    def test_sample_with_high_values(self) -> None:
        """Test construction with high values."""
        sample = _make_sample(
            device=ComputeUnit.GPU,
            power_mw=10000.0,
            usage_pct=100.0,
            frequency_mhz=3500.0,
            memory_mb=65536.0,
        )
        assert sample.power_mw == 10000.0
        assert sample.usage_pct == 100.0

    def test_sample_field_access(self) -> None:
        """Test that all fields of ComputeUnitUsageSample are accessible."""
        sample = _make_sample(usage_pct=85.25, power_mw=250.5)
        assert hasattr(sample, "timestamp")
        assert hasattr(sample, "device")
        assert hasattr(sample, "usage_pct")
        assert hasattr(sample, "frequency_mhz")
        assert hasattr(sample, "memory_mb")
        assert hasattr(sample, "power_mw")
