"""Unit tests for x86_64 platform backend and power monitoring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moment_to_action.hardware._platforms.x86_64 import X86_64Backend, X86_64PowerMonitor
from moment_to_action.hardware._types import ComputeUnit, PowerSample


@pytest.mark.unit
class TestX86_64Backend:  # noqa: N801
    """Test X86_64Backend construction and routing."""

    def test_x86_64_backend_construction(self) -> None:
        """Test X86_64Backend construction."""
        mock_litert = MagicMock()
        mock_onnx = MagicMock()
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64LiteRTBackend",
                return_value=mock_litert,
            ),
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = X86_64Backend()
            assert backend is not None

    def test_x86_64_get_supported_unit_returns_cpu(self) -> None:
        """Test X86_64Backend.get_supported_unit returns CPU."""
        mock_litert = MagicMock()
        mock_onnx = MagicMock()
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64LiteRTBackend",
                return_value=mock_litert,
            ),
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = X86_64Backend()
            assert backend.get_supported_unit() == ComputeUnit.CPU

    def test_x86_64_load_tflite_routes_correctly(self) -> None:
        """Test X86_64Backend.load_model routes .tflite to LiteRT."""
        mock_litert = MagicMock()
        mock_litert.load_model.return_value = "mock_litert_handle"
        mock_onnx = MagicMock()
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64LiteRTBackend",
                return_value=mock_litert,
            ),
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = X86_64Backend()
            handle = backend.load_model("/tmp/model.tflite")

            mock_litert.load_model.assert_called_once_with("/tmp/model.tflite")
            assert handle is not None

    def test_x86_64_load_onnx_routes_correctly(self) -> None:
        """Test X86_64Backend.load_model routes .onnx to ONNX."""
        mock_litert = MagicMock()
        mock_onnx = MagicMock()
        mock_onnx.load_model.return_value = "mock_onnx_handle"
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64LiteRTBackend",
                return_value=mock_litert,
            ),
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = X86_64Backend()
            handle = backend.load_model("/tmp/model.onnx")

            mock_onnx.load_model.assert_called_once_with("/tmp/model.onnx")
            assert handle is not None

    def test_x86_64_load_unsupported_format_raises(self) -> None:
        """Test X86_64Backend.load_model raises ValueError for unsupported format."""
        mock_litert = MagicMock()
        mock_onnx = MagicMock()
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64LiteRTBackend",
                return_value=mock_litert,
            ),
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = X86_64Backend()
            with pytest.raises(ValueError, match="Unsupported model format"):
                backend.load_model("/tmp/model.pt")

    def test_x86_64_run_delegates_to_backend(self) -> None:
        """Test X86_64Backend.run delegates to the appropriate sub-backend."""
        mock_litert = MagicMock()
        mock_litert.load_model.return_value = "mock_handle"
        output_tensor = np.array([1.0, 2.0])
        mock_litert.run.return_value = [output_tensor]
        mock_onnx = MagicMock()
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64LiteRTBackend",
                return_value=mock_litert,
            ),
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = X86_64Backend()
            handle = backend.load_model("/tmp/model.tflite")

            input_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
            outputs = backend.run(handle, input_tensor)

            mock_litert.run.assert_called_once()
            assert len(outputs) == 1

    def test_x86_64_get_input_details_delegates(self) -> None:
        """Test X86_64Backend.get_input_details delegates correctly."""
        mock_litert = MagicMock()
        mock_litert.load_model.return_value = "mock_handle"
        input_details = [{"name": "input", "shape": (1, 224, 224, 3)}]
        mock_litert.get_input_details.return_value = input_details
        mock_onnx = MagicMock()
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64LiteRTBackend",
                return_value=mock_litert,
            ),
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = X86_64Backend()
            handle = backend.load_model("/tmp/model.tflite")
            details = backend.get_input_details(handle)

            assert details == input_details

    def test_x86_64_get_output_details_delegates(self) -> None:
        """Test X86_64Backend.get_output_details delegates correctly."""
        mock_litert = MagicMock()
        mock_litert.load_model.return_value = "mock_handle"
        output_details = [{"name": "output", "shape": (1, 1000)}]
        mock_litert.get_output_details.return_value = output_details
        mock_onnx = MagicMock()
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64LiteRTBackend",
                return_value=mock_litert,
            ),
            patch(
                "moment_to_action.hardware._platforms.x86_64._backend.X86_64ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = X86_64Backend()
            handle = backend.load_model("/tmp/model.tflite")
            details = backend.get_output_details(handle)

            assert details == output_details


@pytest.mark.unit
class TestX86_64PowerMonitor:  # noqa: N801
    """Test X86_64PowerMonitor power sampling."""

    def test_x86_64_power_monitor_sample_cpu_rapl_available(self) -> None:
        """Test X86_64PowerMonitor.sample returns PowerSample for CPU when RAPL available."""
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = True
        mock_rapl_path.read_text.return_value = "1000000\n"
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._power._RAPL_ENERGY_PATH",
                mock_rapl_path,
            ),
            patch("psutil.cpu_percent", return_value=50.0),
        ):
            monitor = X86_64PowerMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, PowerSample)
            assert sample.compute_unit == ComputeUnit.CPU
            assert sample.power_mw >= 0.0
            assert 0.0 <= sample.utilization_pct <= 100.0
            assert sample.timestamp > 0.0

    def test_x86_64_power_monitor_sample_cpu_rapl_fallback(self) -> None:
        """Test X86_64PowerMonitor.sample falls back to estimate when RAPL unavailable."""
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = False
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._power._RAPL_ENERGY_PATH",
                mock_rapl_path,
            ),
            patch("psutil.cpu_percent", return_value=25.0),
            patch("psutil.cpu_freq") as mock_freq,
        ):
            mock_freq.return_value.current = 2400  # 2.4 GHz
            monitor = X86_64PowerMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, PowerSample)
            assert sample.compute_unit == ComputeUnit.CPU
            assert sample.power_mw >= 0.0
            assert sample.utilization_pct == 25.0

    def test_x86_64_power_monitor_non_cpu_unit_returns_zero(self) -> None:
        """Test X86_64PowerMonitor returns zero power for non-CPU units."""
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = True
        with patch(
            "moment_to_action.hardware._platforms.x86_64._power._RAPL_ENERGY_PATH",
            mock_rapl_path,
        ):
            monitor = X86_64PowerMonitor()
            sample = monitor.sample(ComputeUnit.NPU)

            assert sample.compute_unit == ComputeUnit.NPU
            assert sample.power_mw == 0.0
            assert sample.utilization_pct == 0.0

    def test_x86_64_power_monitor_rapl_read_failure_fallback(self) -> None:
        """Test X86_64PowerMonitor falls back to estimate on RAPL read failure."""
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = True
        mock_rapl_path.read_text.side_effect = FileNotFoundError()
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._power._RAPL_ENERGY_PATH",
                mock_rapl_path,
            ),
            patch("psutil.cpu_percent", return_value=30.0),
            patch("psutil.cpu_freq") as mock_freq,
        ):
            mock_freq.return_value.current = 2000
            monitor = X86_64PowerMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, PowerSample)
            assert sample.power_mw >= 0.0

    def test_x86_64_power_monitor_multiple_samples(self) -> None:
        """Test X86_64PowerMonitor returns valid samples over multiple calls."""
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = False
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._power._RAPL_ENERGY_PATH",
                mock_rapl_path,
            ),
            patch("psutil.cpu_percent", return_value=50.0),
            patch("psutil.cpu_freq") as mock_freq,
        ):
            mock_freq.return_value.current = 2000

            monitor = X86_64PowerMonitor()
            sample1 = monitor.sample(ComputeUnit.CPU)
            sample2 = monitor.sample(ComputeUnit.CPU)

            assert sample1.power_mw >= 0.0
            assert sample2.power_mw >= 0.0
