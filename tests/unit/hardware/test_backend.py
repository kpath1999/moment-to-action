"""Unit tests for Platform hardware entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moment_to_action.hardware import BenchmarkResult, Platform
from moment_to_action.hardware._types import ComputeUnit, PlatformType


@pytest.mark.unit
class TestBenchmarkResult:
    """Tests for BenchmarkResult data class."""

    def test_construction(self) -> None:
        """BenchmarkResult stores all fields correctly."""
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

    def test_is_frozen(self) -> None:
        """BenchmarkResult is immutable."""
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
class TestPlatformProperties:
    """Tests for Platform properties."""

    def _make_platform(self) -> Platform:
        """Build a Platform with mocked x86_64 backends."""
        with (
            patch("moment_to_action.hardware._platform._detect_platform") as mock_detect,
            patch(
                "moment_to_action.hardware._platform.Platform._init_x86_64",
                autospec=True,
            ) as mock_init,
        ):
            mock_detect.return_value = PlatformType.X86_64

            def _setup(self: Platform) -> None:
                from moment_to_action.hardware._resource_monitor import ResourceMonitor

                self._resource_monitor = MagicMock(spec=ResourceMonitor)
                self._backends = {ComputeUnit.CPU: MagicMock()}

            mock_init.side_effect = _setup
            return Platform()

    def test_platform_type_property(self) -> None:
        """platform_type returns the detected PlatformType."""
        p = self._make_platform()
        assert p.platform_type == PlatformType.X86_64

    def test_supported_units_property(self) -> None:
        """supported_units returns set of registered backend keys."""
        p = self._make_platform()
        assert p.supported_units == {ComputeUnit.CPU}

    def test_resource_monitor_property(self) -> None:
        """resource_monitor returns the monitor built by _init_*."""
        p = self._make_platform()
        assert p.resource_monitor is not None


@pytest.mark.unit
class TestPlatformDetection:
    """Tests for _detect_platform() routing in Platform.__init__."""

    @staticmethod
    def _stub_init(platform_instance: Platform) -> None:
        """Stub _init_* to avoid importing platform-specific modules."""
        platform_instance._resource_monitor = MagicMock()  # type: ignore[assignment]
        platform_instance._backends = {ComputeUnit.CPU: MagicMock()}  # type: ignore[assignment]

    def test_init_x86_64_called_on_x86(self) -> None:
        """Platform calls _init_x86_64 when _detect_platform returns X86_64."""
        with patch("moment_to_action.hardware._platform._detect_platform") as mock_detect:
            mock_detect.return_value = PlatformType.X86_64
            with patch.object(
                Platform, "_init_x86_64", autospec=True, side_effect=self._stub_init
            ) as mock_init:
                Platform()
                mock_init.assert_called_once()

    def test_init_qcs6490_called_on_qcs6490(self) -> None:
        """Platform calls _init_qcs6490 when _detect_platform returns QCS6490."""
        with patch("moment_to_action.hardware._platform._detect_platform") as mock_detect:
            mock_detect.return_value = PlatformType.QCS6490
            with patch.object(
                Platform, "_init_qcs6490", autospec=True, side_effect=self._stub_init
            ) as mock_init:
                Platform()
                mock_init.assert_called_once()

    def test_init_macos_arm64_called_on_macos(self) -> None:
        """Platform calls _init_macos_arm64 when _detect_platform returns MACOS_ARM64."""
        with patch("moment_to_action.hardware._platform._detect_platform") as mock_detect:
            mock_detect.return_value = PlatformType.MACOS_ARM64
            with patch.object(
                Platform, "_init_macos_arm64", autospec=True, side_effect=self._stub_init
            ) as mock_init:
                Platform()
                mock_init.assert_called_once()


@pytest.mark.unit
class TestPlatformLoadDelegation:
    """Tests for Platform.load_* delegation to ComputeBackend instances."""

    def _make_platform_with_mock_cpu(self) -> tuple[Platform, MagicMock]:
        """Return (Platform, mock_cpu_backend) with a single mocked CPU backend."""
        with (
            patch("moment_to_action.hardware._platform._detect_platform") as mock_detect,
            patch.object(Platform, "_init_x86_64", autospec=True) as mock_init,
        ):
            mock_detect.return_value = PlatformType.X86_64
            mock_cpu = MagicMock()

            def _setup(self: Platform) -> None:
                self._resource_monitor = MagicMock()
                self._backends = {ComputeUnit.CPU: mock_cpu}

            mock_init.side_effect = _setup
            p = Platform()
        return p, mock_cpu

    def test_load_tflite_delegates(self) -> None:
        """load_tflite calls backend.load_tflite with the path."""
        p, mock_cpu = self._make_platform_with_mock_cpu()
        path = Path("/fake/model.tflite")
        mock_cpu.load_tflite.return_value = MagicMock()

        result = p.load_tflite(ComputeUnit.CPU, path)

        mock_cpu.load_tflite.assert_called_once_with(path)
        assert result is mock_cpu.load_tflite.return_value

    def test_load_onnx_delegates(self) -> None:
        """load_onnx calls backend.load_onnx with the path."""
        p, mock_cpu = self._make_platform_with_mock_cpu()
        path = Path("/fake/model.onnx")
        mock_cpu.load_onnx.return_value = MagicMock()

        result = p.load_onnx(ComputeUnit.CPU, path)

        mock_cpu.load_onnx.assert_called_once_with(path)
        assert result is mock_cpu.load_onnx.return_value

    def test_load_dlc_delegates(self) -> None:
        """load_dlc calls backend.load_dlc with the path."""
        p, mock_cpu = self._make_platform_with_mock_cpu()
        path = Path("/fake/model.dlc")
        mock_cpu.load_dlc.return_value = MagicMock()

        result = p.load_dlc(ComputeUnit.CPU, path)

        mock_cpu.load_dlc.assert_called_once_with(path)
        assert result is mock_cpu.load_dlc.return_value

    def test_load_torch_delegates(self) -> None:
        """load_torch calls backend.load_torch with the path."""
        p, mock_cpu = self._make_platform_with_mock_cpu()
        path = Path("/fake/model.pt")
        mock_cpu.load_torch.return_value = MagicMock()

        result = p.load_torch(ComputeUnit.CPU, path)

        mock_cpu.load_torch.assert_called_once_with(path)
        assert result is mock_cpu.load_torch.return_value

    def test_load_llama_cpp_delegates(self) -> None:
        """load_llama_cpp calls backend.load_llama_cpp with path and mmproj."""
        p, mock_cpu = self._make_platform_with_mock_cpu()
        path = Path("/fake/model.gguf")
        mmproj = Path("/fake/mmproj.gguf")
        mock_cpu.load_llama_cpp.return_value = MagicMock()

        result = p.load_llama_cpp(ComputeUnit.CPU, path, mmproj=mmproj)

        mock_cpu.load_llama_cpp.assert_called_once_with(path, _mmproj=mmproj)
        assert result is mock_cpu.load_llama_cpp.return_value

    def test_unknown_unit_raises_value_error(self) -> None:
        """Requesting a unit with no registered backend raises ValueError."""
        p, _ = self._make_platform_with_mock_cpu()

        with pytest.raises(ValueError, match="NPU"):
            p.load_tflite(ComputeUnit.NPU, Path("/fake/model.tflite"))


@pytest.mark.unit
class TestPlatformBenchmark:
    """Tests for Platform.benchmark."""

    def _make_platform(self) -> Platform:
        """Build a Platform with mocked x86_64 backends."""
        with (
            patch("moment_to_action.hardware._platform._detect_platform") as mock_detect,
            patch.object(Platform, "_init_x86_64", autospec=True) as mock_init,
        ):
            mock_detect.return_value = PlatformType.X86_64

            def _setup(self: Platform) -> None:
                self._resource_monitor = MagicMock()
                self._backends = {ComputeUnit.CPU: MagicMock()}

            mock_init.side_effect = _setup
            return Platform()

    def test_benchmark_returns_benchmark_result(self) -> None:
        """benchmark() returns a BenchmarkResult."""
        p = self._make_platform()
        mock_model = MagicMock()
        mock_model.unit = ComputeUnit.CPU
        mock_model.run.return_value = np.array([1.0, 2.0])

        result = p.benchmark(mock_model, np.zeros((1, 224, 224, 3), dtype=np.float32), n_runs=10)

        assert isinstance(result, BenchmarkResult)
        assert result.n_runs == 10
        assert result.compute_unit == "CPU"

    def test_benchmark_runs_n_times(self) -> None:
        """benchmark() calls model.run exactly n_runs times."""
        p = self._make_platform()
        mock_model = MagicMock()
        mock_model.unit = ComputeUnit.CPU
        inputs = np.zeros((1, 224, 224, 3), dtype=np.float32)

        p.benchmark(mock_model, inputs, n_runs=7)

        assert mock_model.run.call_count == 7

    def test_benchmark_latencies_are_nonnegative(self) -> None:
        """All latency fields in BenchmarkResult are >= 0."""
        p = self._make_platform()
        mock_model = MagicMock()
        mock_model.unit = ComputeUnit.CPU

        result = p.benchmark(mock_model, {}, n_runs=5)

        assert result.min_ms >= 0.0
        assert result.p50_ms >= 0.0
        assert result.p95_ms >= 0.0
        assert result.p99_ms >= 0.0
        assert result.max_ms >= 0.0
        assert result.mean_ms >= 0.0

    def test_benchmark_percentile_ordering(self) -> None:
        """Percentile ordering: min <= p50 <= p95 <= p99 <= max."""
        p = self._make_platform()
        mock_model = MagicMock()
        mock_model.unit = ComputeUnit.CPU

        result = p.benchmark(mock_model, {}, n_runs=20)

        assert result.min_ms <= result.p50_ms
        assert result.p50_ms <= result.p95_ms
        assert result.p95_ms <= result.p99_ms
        assert result.p99_ms <= result.max_ms

    def test_benchmark_default_n_runs_is_20(self) -> None:
        """Default n_runs is 20."""
        p = self._make_platform()
        mock_model = MagicMock()
        mock_model.unit = ComputeUnit.CPU

        result = p.benchmark(mock_model, {})

        assert result.n_runs == 20
        assert mock_model.run.call_count == 20
