"""Unit tests for the ComputeBackend main abstraction layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moment_to_action.hardware._backend import (
    BenchmarkResult,
    ComputeBackend,
    _make_backend,
    _make_power_monitor,
)
from moment_to_action.hardware._platforms._detection import Platform
from moment_to_action.hardware._platforms.qcs6490 import QCS6490Backend, QCS6490PowerMonitor
from moment_to_action.hardware._platforms.x86_64 import X86_64Backend, X86_64PowerMonitor
from moment_to_action.hardware._types import ComputeUnit


@pytest.mark.unit
class TestBenchmarkResult:
    """Test BenchmarkResult data class."""

    def test_benchmark_result_construction(self) -> None:
        """Test BenchmarkResult can be constructed with required fields."""
        result = BenchmarkResult(
            mean_ms=10.5,
            p50_ms=10.0,
            p95_ms=12.0,
            p99_ms=13.0,
            min_ms=9.0,
            max_ms=15.0,
            compute_unit="CPU",
            n_runs=20,
        )

        assert result.mean_ms == 10.5
        assert result.p50_ms == 10.0
        assert result.p95_ms == 12.0
        assert result.p99_ms == 13.0
        assert result.min_ms == 9.0
        assert result.max_ms == 15.0
        assert result.compute_unit == "CPU"
        assert result.n_runs == 20

    def test_benchmark_result_is_frozen(self) -> None:
        """Test BenchmarkResult is immutable (frozen)."""
        result = BenchmarkResult(
            mean_ms=10.5,
            p50_ms=10.0,
            p95_ms=12.0,
            p99_ms=13.0,
            min_ms=9.0,
            max_ms=15.0,
            compute_unit="CPU",
            n_runs=20,
        )

        with pytest.raises(AttributeError):
            result.mean_ms = 11.0  # type: ignore[misc]


@pytest.mark.unit
class TestComputeBackendConstruction:
    """Test ComputeBackend construction and initialization."""

    def test_compute_backend_construction_default(self) -> None:
        """Test ComputeBackend construction with default unit (NPU)."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_be.return_value = MagicMock()
                    mock_be.return_value.get_supported_unit.return_value = ComputeUnit.NPU

                    backend = ComputeBackend()

                    assert backend.preferred_unit == ComputeUnit.NPU

    def test_compute_backend_construction_cpu(self) -> None:
        """Test ComputeBackend construction with CPU preference."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_be.return_value = MagicMock()
                    mock_be.return_value.get_supported_unit.return_value = ComputeUnit.CPU

                    backend = ComputeBackend(preferred_unit=ComputeUnit.CPU)

                    assert backend.preferred_unit == ComputeUnit.CPU
                    mock_be.assert_called_once_with(ComputeUnit.CPU)

    def test_compute_backend_active_unit_property(self) -> None:
        """Test ComputeBackend.active_unit returns backend's supported unit."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_backend = MagicMock()
                    mock_be.return_value = mock_backend
                    mock_backend.get_supported_unit.return_value = ComputeUnit.CPU

                    backend = ComputeBackend(preferred_unit=ComputeUnit.NPU)

                    assert backend.active_unit == ComputeUnit.CPU

    def test_compute_backend_power_monitor_property(self) -> None:
        """Test ComputeBackend.power_monitor property."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_power_monitor = MagicMock()
                    mock_pm.return_value = mock_power_monitor
                    mock_be.return_value = MagicMock()
                    mock_be.return_value.get_supported_unit.return_value = ComputeUnit.CPU

                    backend = ComputeBackend()

                    assert backend.power_monitor is mock_power_monitor

    def test_make_power_monitor_qcs6490(self) -> None:
        """Test that _make_power_monitor returns QCS6490PowerMonitor for QCS6490 platform."""
        with patch("moment_to_action.hardware._backend.detect_platform") as mock_detect:
            mock_detect.return_value = Platform.QCS6490

            power_monitor = _make_power_monitor()

            assert isinstance(power_monitor, QCS6490PowerMonitor)

    def test_make_backend_qcs6490(self) -> None:
        """Test that _make_backend returns QCS6490Backend for QCS6490 platform."""
        with patch("moment_to_action.hardware._backend.detect_platform") as mock_detect:
            mock_detect.return_value = Platform.QCS6490

            backend = _make_backend(ComputeUnit.NPU)

            assert isinstance(backend, QCS6490Backend)

    def test_make_power_monitor_macos_arm64(self) -> None:
        """Test that _make_power_monitor returns x86_64 monitor for macOS arm64."""
        with patch("moment_to_action.hardware._backend.detect_platform") as mock_detect:
            mock_detect.return_value = Platform.MACOS_ARM64

            power_monitor = _make_power_monitor()

            assert isinstance(power_monitor, X86_64PowerMonitor)

    def test_make_backend_macos_arm64(self) -> None:
        """Test that _make_backend returns x86_64 runtime path for macOS arm64."""
        with patch("moment_to_action.hardware._backend.detect_platform") as mock_detect:
            mock_detect.return_value = Platform.MACOS_ARM64

            backend = _make_backend(ComputeUnit.CPU)

            assert isinstance(backend, X86_64Backend)

    def test_make_power_monitor_x86_64(self) -> None:
        """Test that _make_power_monitor returns X86_64PowerMonitor for X86_64 platform."""
        with patch("moment_to_action.hardware._backend.detect_platform") as mock_detect:
            mock_detect.return_value = Platform.X86_64

            power_monitor = _make_power_monitor()

            assert isinstance(power_monitor, X86_64PowerMonitor)

    def test_make_backend_x86_64(self) -> None:
        """Test that _make_backend returns X86_64Backend for X86_64 platform."""
        with patch("moment_to_action.hardware._backend.detect_platform") as mock_detect:
            mock_detect.return_value = Platform.X86_64

            backend = _make_backend(ComputeUnit.CPU)

            assert isinstance(backend, X86_64Backend)


@pytest.mark.unit
class TestInferenceBackendDefaultTorchPolicy:
    """Tests for InferenceBackend.resolve_torch_policy default."""

    def test_resolve_torch_policy_raises_not_implemented(self) -> None:
        """Default resolve_torch_policy raises NotImplementedError."""
        from moment_to_action.hardware._platforms._runtimes._litert import LiteRTBackend
        from moment_to_action.hardware._types import ComputeUnit

        backend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
        with pytest.raises(NotImplementedError, match="does not implement torch policy"):
            backend.resolve_torch_policy("auto")


@pytest.mark.unit
class TestComputeBackendDelegation:
    """Test ComputeBackend delegation to platform backend."""

    def test_compute_backend_load_model_delegates(self) -> None:
        """Test ComputeBackend.load_model delegates to backend."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_backend = MagicMock()
                    mock_be.return_value = mock_backend
                    mock_backend.get_supported_unit.return_value = ComputeUnit.CPU
                    mock_backend.load_model.return_value = "model_handle"

                    backend = ComputeBackend()
                    handle = backend.load_model("/tmp/model.tflite")

                    mock_backend.load_model.assert_called_once_with("/tmp/model.tflite")
                    assert handle == "model_handle"

    def test_compute_backend_run_delegates(self) -> None:
        """Test ComputeBackend.run delegates to backend."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_backend = MagicMock()
                    mock_be.return_value = mock_backend
                    mock_backend.get_supported_unit.return_value = ComputeUnit.CPU
                    output = np.array([1.0, 2.0])
                    mock_backend.run.return_value = [output]

                    backend = ComputeBackend()
                    input_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
                    outputs = backend.run("handle", input_tensor)

                    mock_backend.run.assert_called_once_with("handle", input_tensor)
                    assert len(outputs) == 1

    def test_compute_backend_get_input_details_delegates(self) -> None:
        """Test ComputeBackend.get_input_details delegates to backend."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_backend = MagicMock()
                    mock_be.return_value = mock_backend
                    mock_backend.get_supported_unit.return_value = ComputeUnit.CPU
                    input_details = [{"name": "input", "shape": (1, 224, 224, 3)}]
                    mock_backend.get_input_details.return_value = input_details

                    backend = ComputeBackend()
                    details = backend.get_input_details("handle")

                    mock_backend.get_input_details.assert_called_once_with("handle")
                    assert details == input_details

    def test_compute_backend_get_output_details_delegates(self) -> None:
        """Test ComputeBackend.get_output_details delegates to backend."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_backend = MagicMock()
                    mock_be.return_value = mock_backend
                    mock_backend.get_supported_unit.return_value = ComputeUnit.CPU
                    output_details = [{"name": "output", "shape": (1, 1000)}]
                    mock_backend.get_output_details.return_value = output_details

                    backend = ComputeBackend()
                    details = backend.get_output_details("handle")

                    mock_backend.get_output_details.assert_called_once_with("handle")
                    assert details == output_details

    def test_compute_backend_resolve_torch_policy_delegates(self) -> None:
        """Test ComputeBackend.resolve_torch_policy delegates to backend."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_backend = MagicMock()
                    mock_be.return_value = mock_backend
                    mock_backend.get_supported_unit.return_value = ComputeUnit.CPU
                    mock_backend.resolve_torch_policy.return_value.device = "cpu"
                    mock_backend.resolve_torch_policy.return_value.dtype = "float32"

                    backend = ComputeBackend()
                    policy = backend.resolve_torch_policy("auto")

                    mock_backend.resolve_torch_policy.assert_called_once_with("auto")
                    assert policy.device == "cpu"
                    assert policy.dtype == "float32"


@pytest.mark.unit
class TestComputeBackendBenchmarking:
    """Test ComputeBackend.benchmark method."""

    def test_benchmark_returns_benchmark_result(self) -> None:
        """Test ComputeBackend.benchmark returns BenchmarkResult with correct structure."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_backend = MagicMock()
                    mock_be.return_value = mock_backend
                    mock_backend.get_supported_unit.return_value = ComputeUnit.CPU
                    output = np.array([1.0, 2.0])
                    mock_backend.run.return_value = [output]

                    backend = ComputeBackend()
                    input_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
                    result = backend.benchmark("handle", input_tensor, n_runs=10)

                    assert isinstance(result, BenchmarkResult)
                    assert result.n_runs == 10
                    assert result.compute_unit == "CPU"

    def test_benchmark_result_structure(self) -> None:
        """Test benchmark result contains all required percentile fields."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_backend = MagicMock()
                    mock_be.return_value = mock_backend
                    mock_backend.get_supported_unit.return_value = ComputeUnit.CPU
                    output = np.array([1.0, 2.0])
                    mock_backend.run.return_value = [output]

                    backend = ComputeBackend()
                    input_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
                    result = backend.benchmark("handle", input_tensor, n_runs=20)

                    # Check all percentile fields exist
                    assert hasattr(result, "mean_ms")
                    assert hasattr(result, "p50_ms")
                    assert hasattr(result, "p95_ms")
                    assert hasattr(result, "p99_ms")
                    assert hasattr(result, "min_ms")
                    assert hasattr(result, "max_ms")

    def test_benchmark_latencies_are_positive(self) -> None:
        """Test benchmark latencies are all positive values."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_backend = MagicMock()
                    mock_be.return_value = mock_backend
                    mock_backend.get_supported_unit.return_value = ComputeUnit.CPU
                    output = np.array([1.0, 2.0])
                    mock_backend.run.return_value = [output]

                    backend = ComputeBackend()
                    input_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
                    result = backend.benchmark("handle", input_tensor, n_runs=20)

                    # All latencies should be >= 0
                    assert result.min_ms >= 0.0
                    assert result.max_ms >= 0.0
                    assert result.mean_ms >= 0.0
                    assert result.p50_ms >= 0.0
                    assert result.p95_ms >= 0.0
                    assert result.p99_ms >= 0.0

    def test_benchmark_percentiles_ordering(self) -> None:
        """Test benchmark percentiles are in correct order."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_backend = MagicMock()
                    mock_be.return_value = mock_backend
                    mock_backend.get_supported_unit.return_value = ComputeUnit.CPU
                    output = np.array([1.0, 2.0])
                    mock_backend.run.return_value = [output]

                    backend = ComputeBackend()
                    input_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
                    result = backend.benchmark("handle", input_tensor, n_runs=20)

                    assert result.min_ms <= result.p50_ms
                    assert result.p50_ms <= result.p95_ms
                    assert result.p95_ms <= result.p99_ms
                    assert result.p99_ms <= result.max_ms

    def test_benchmark_default_n_runs(self) -> None:
        """Test benchmark uses default n_runs=20."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_backend = MagicMock()
                    mock_be.return_value = mock_backend
                    mock_backend.get_supported_unit.return_value = ComputeUnit.CPU
                    output = np.array([1.0, 2.0])
                    mock_backend.run.return_value = [output]

                    backend = ComputeBackend()
                    input_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
                    result = backend.benchmark("handle", input_tensor)

                    assert result.n_runs == 20
                    assert mock_backend.run.call_count == 20

    def test_benchmark_custom_n_runs(self) -> None:
        """Test benchmark accepts custom n_runs parameter."""
        with patch("moment_to_action.hardware._backend.detect_platform"):
            with patch("moment_to_action.hardware._backend._make_power_monitor") as mock_pm:
                with patch("moment_to_action.hardware._backend._make_backend") as mock_be:
                    mock_pm.return_value = MagicMock()
                    mock_backend = MagicMock()
                    mock_be.return_value = mock_backend
                    mock_backend.get_supported_unit.return_value = ComputeUnit.CPU
                    output = np.array([1.0, 2.0])
                    mock_backend.run.return_value = [output]

                    backend = ComputeBackend()
                    input_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
                    result = backend.benchmark("handle", input_tensor, n_runs=50)

                    assert result.n_runs == 50
                    assert mock_backend.run.call_count == 50
