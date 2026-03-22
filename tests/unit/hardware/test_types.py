"""Unit tests for hardware types."""

from __future__ import annotations

import pytest

from moment_to_action.hardware._types import ComputeUnit, PowerSample


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
class TestPowerSample:
    """Tests for PowerSample data class."""

    def test_powersample_construction_basic(self) -> None:
        """Test basic PowerSample construction with valid values."""
        sample = PowerSample(
            timestamp=1234567890.5,
            compute_unit=ComputeUnit.CPU,
            power_mw=150.0,
            utilization_pct=75.5,
        )
        assert sample.timestamp == 1234567890.5
        assert sample.compute_unit == ComputeUnit.CPU
        assert sample.power_mw == 150.0
        assert sample.utilization_pct == 75.5

    def test_powersample_construction_npu(self) -> None:
        """Test PowerSample construction with NPU compute unit."""
        sample = PowerSample(
            timestamp=1234567890.0,
            compute_unit=ComputeUnit.NPU,
            power_mw=200.0,
            utilization_pct=90.0,
        )
        assert sample.compute_unit == ComputeUnit.NPU
        assert sample.power_mw == 200.0

    def test_powersample_construction_gpu(self) -> None:
        """Test PowerSample construction with GPU compute unit."""
        sample = PowerSample(
            timestamp=1234567890.0,
            compute_unit=ComputeUnit.GPU,
            power_mw=500.0,
            utilization_pct=95.0,
        )
        assert sample.compute_unit == ComputeUnit.GPU
        assert sample.power_mw == 500.0

    def test_powersample_construction_dsp(self) -> None:
        """Test PowerSample construction with DSP compute unit."""
        sample = PowerSample(
            timestamp=1234567890.0,
            compute_unit=ComputeUnit.DSP,
            power_mw=100.0,
            utilization_pct=50.0,
        )
        assert sample.compute_unit == ComputeUnit.DSP
        assert sample.power_mw == 100.0

    def test_powersample_with_zero_values(self) -> None:
        """Test PowerSample construction with zero values."""
        sample = PowerSample(
            timestamp=0.0,
            compute_unit=ComputeUnit.CPU,
            power_mw=0.0,
            utilization_pct=0.0,
        )
        assert sample.timestamp == 0.0
        assert sample.power_mw == 0.0
        assert sample.utilization_pct == 0.0

    def test_powersample_with_high_values(self) -> None:
        """Test PowerSample construction with high values."""
        sample = PowerSample(
            timestamp=9999999999.99,
            compute_unit=ComputeUnit.GPU,
            power_mw=10000.0,
            utilization_pct=100.0,
        )
        assert sample.timestamp == 9999999999.99
        assert sample.power_mw == 10000.0
        assert sample.utilization_pct == 100.0

    def test_powersample_field_access(self) -> None:
        """Test that all fields of PowerSample are accessible."""
        sample = PowerSample(
            timestamp=123.45,
            compute_unit=ComputeUnit.NPU,
            power_mw=250.5,
            utilization_pct=85.25,
        )
        assert hasattr(sample, "timestamp")
        assert hasattr(sample, "compute_unit")
        assert hasattr(sample, "power_mw")
        assert hasattr(sample, "utilization_pct")

    def test_powersample_model_validate(self) -> None:
        """Test PowerSample validation with dict."""
        data = {
            "timestamp": 456.78,
            "compute_unit": ComputeUnit.DSP,
            "power_mw": 120.0,
            "utilization_pct": 60.0,
        }
        sample = PowerSample(**data)
        assert sample.timestamp == 456.78
        assert sample.compute_unit == ComputeUnit.DSP
