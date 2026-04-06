"""Unit tests for the macOS arm64 platform backend and power monitor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moment_to_action.hardware._platforms.macos_arm64 import (
    MacOSARM64Backend,
    MacOSARM64PowerMonitor,
)
from moment_to_action.hardware._types import ComputeUnit, PowerSample


@pytest.mark.unit
class TestMacOSARM64Backend:
    """Tests for MacOSARM64Backend construction and routing."""

    def test_construction_succeeds(self) -> None:
        """MacOSARM64Backend can be constructed without error."""
        with (
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.LiteRTBackend"),
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.ONNXBackend"),
        ):
            backend = MacOSARM64Backend()
            assert backend is not None

    def test_get_supported_unit_returns_cpu(self) -> None:
        """MacOSARM64Backend reports CPU as the supported compute unit."""
        with (
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.LiteRTBackend"),
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.ONNXBackend"),
        ):
            backend = MacOSARM64Backend()
            assert backend.get_supported_unit() == ComputeUnit.CPU

    def test_load_tflite_routes_to_litert(self) -> None:
        """load_model routes .tflite files to LiteRTBackend."""
        mock_litert = MagicMock()
        mock_litert.load_model.return_value = "raw_handle"
        with (
            patch(
                "moment_to_action.hardware._platforms.macos_arm64._backend.LiteRTBackend",
                return_value=mock_litert,
            ),
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.ONNXBackend"),
        ):
            backend = MacOSARM64Backend()
            backend.load_model("/tmp/model.tflite")
            mock_litert.load_model.assert_called_once_with("/tmp/model.tflite")

    def test_load_onnx_routes_to_onnx(self) -> None:
        """load_model routes .onnx files to ONNXBackend."""
        mock_onnx = MagicMock()
        mock_onnx.load_model.return_value = "raw_handle"
        with (
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.LiteRTBackend"),
            patch(
                "moment_to_action.hardware._platforms.macos_arm64._backend.ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = MacOSARM64Backend()
            backend.load_model("/tmp/model.onnx")
            mock_onnx.load_model.assert_called_once_with("/tmp/model.onnx")

    def test_load_unsupported_format_raises_value_error(self) -> None:
        """load_model raises ValueError for unrecognised file extensions."""
        with (
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.LiteRTBackend"),
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.ONNXBackend"),
        ):
            backend = MacOSARM64Backend()
            with pytest.raises(ValueError, match="Unsupported model format"):
                backend.load_model("/tmp/model.pt")

    def test_run_delegates_to_sub_backend(self) -> None:
        """run() delegates inference to the sub-backend that loaded the model."""
        expected_output = [np.array([1.0, 2.0])]
        mock_litert = MagicMock()
        mock_litert.load_model.return_value = "raw"
        mock_litert.run.return_value = expected_output
        with (
            patch(
                "moment_to_action.hardware._platforms.macos_arm64._backend.LiteRTBackend",
                return_value=mock_litert,
            ),
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.ONNXBackend"),
        ):
            backend = MacOSARM64Backend()
            handle = backend.load_model("/tmp/model.tflite")
            inputs = np.zeros((1, 224, 224, 3), dtype=np.float32)
            outputs = backend.run(handle, inputs)
            mock_litert.run.assert_called_once()
            assert outputs is expected_output

    def test_get_input_details_delegates(self) -> None:
        """get_input_details() delegates to the sub-backend."""
        details = [{"name": "input", "shape": (1, 224, 224, 3)}]
        mock_litert = MagicMock()
        mock_litert.load_model.return_value = "raw"
        mock_litert.get_input_details.return_value = details
        with (
            patch(
                "moment_to_action.hardware._platforms.macos_arm64._backend.LiteRTBackend",
                return_value=mock_litert,
            ),
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.ONNXBackend"),
        ):
            backend = MacOSARM64Backend()
            handle = backend.load_model("/tmp/model.tflite")
            assert backend.get_input_details(handle) == details

    def test_get_output_details_delegates(self) -> None:
        """get_output_details() delegates to the sub-backend."""
        details = [{"name": "output", "shape": (1, 1000)}]
        mock_litert = MagicMock()
        mock_litert.load_model.return_value = "raw"
        mock_litert.get_output_details.return_value = details
        with (
            patch(
                "moment_to_action.hardware._platforms.macos_arm64._backend.LiteRTBackend",
                return_value=mock_litert,
            ),
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.ONNXBackend"),
        ):
            backend = MacOSARM64Backend()
            handle = backend.load_model("/tmp/model.tflite")
            assert backend.get_output_details(handle) == details

    def test_resolve_torch_policy_delegates_to_helper(self) -> None:
        """resolve_torch_policy() delegates to resolve_torch_execution_policy."""
        import torch

        mock_policy = MagicMock(device=torch.device("cpu"), dtype=torch.float32)
        with (
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.LiteRTBackend"),
            patch("moment_to_action.hardware._platforms.macos_arm64._backend.ONNXBackend"),
            patch(
                "moment_to_action.hardware._platforms.macos_arm64._backend.resolve_torch_execution_policy",
                return_value=mock_policy,
            ) as mock_resolve,
        ):
            backend = MacOSARM64Backend()
            policy = backend.resolve_torch_policy("auto")
            mock_resolve.assert_called_once_with("auto")
            assert policy is mock_policy


@pytest.mark.unit
class TestMacOSARM64PowerMonitor:
    """Tests for MacOSARM64PowerMonitor."""

    def test_sample_cpu_returns_power_sample(self) -> None:
        """sample(CPU) returns a valid PowerSample."""
        with patch("psutil.cpu_percent", return_value=50.0):
            monitor = MacOSARM64PowerMonitor()
            sample = monitor.sample(ComputeUnit.CPU)
        assert isinstance(sample, PowerSample)
        assert sample.device == ComputeUnit.CPU
        assert sample.power_mw >= 0.0
        assert 0.0 <= sample.utilization_pct <= 100.0
        assert sample.timestamp > 0.0

    def test_sample_non_cpu_returns_zero_power(self) -> None:
        """sample(NPU) returns zero power (no NPU on macOS arm64)."""
        monitor = MacOSARM64PowerMonitor()
        sample = monitor.sample(ComputeUnit.NPU)
        assert sample.device == ComputeUnit.NPU
        assert sample.power_mw == 0.0
        assert sample.utilization_pct == 0.0

    def test_power_increases_with_load(self) -> None:
        """Higher CPU utilization produces higher estimated power."""
        monitor = MacOSARM64PowerMonitor()
        with (
            patch("psutil.cpu_percent", return_value=0.0),
            patch("psutil.cpu_freq") as mock_freq,
        ):
            mock_freq.return_value.current = 3000
            sample_idle = monitor.sample(ComputeUnit.CPU)

        with (
            patch("psutil.cpu_percent", return_value=100.0),
            patch("psutil.cpu_freq") as mock_freq,
        ):
            mock_freq.return_value.current = 3000
            sample_loaded = monitor.sample(ComputeUnit.CPU)

        assert sample_loaded.power_mw > sample_idle.power_mw

    def test_fallback_freq_on_os_error(self) -> None:
        """Falls back to default freq when psutil.cpu_freq raises OSError."""
        with (
            patch("psutil.cpu_percent", return_value=100.0),
            patch("psutil.cpu_freq", side_effect=OSError("cpu_freq unavailable")),
        ):
            monitor = MacOSARM64PowerMonitor()
            sample = monitor.sample(ComputeUnit.CPU)
        # base=50 + 3.0 GHz x 100% x 0.6 = 50 + 180 = 230
        assert sample.power_mw == pytest.approx(230.0)

    def test_fallback_freq_on_attribute_error(self) -> None:
        """Falls back to default freq when psutil.cpu_freq raises AttributeError."""
        with (
            patch("psutil.cpu_percent", return_value=0.0),
            patch("psutil.cpu_freq", side_effect=AttributeError),
        ):
            monitor = MacOSARM64PowerMonitor()
            sample = monitor.sample(ComputeUnit.CPU)
        assert sample.power_mw == pytest.approx(50.0)  # base only, 0% load

    def test_utilization_pct_matches_cpu_percent(self) -> None:
        """utilization_pct field reflects psutil.cpu_percent value."""
        with patch("psutil.cpu_percent", return_value=42.0):
            monitor = MacOSARM64PowerMonitor()
            sample = monitor.sample(ComputeUnit.CPU)
        assert sample.utilization_pct == 42.0
