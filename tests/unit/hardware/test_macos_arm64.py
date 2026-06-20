"""Unit tests for the macOS arm64 platform backend and power monitor."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from moment_to_action.hardware._platforms.macos_arm64 import (
    MacOSARM64CPUBackend,
    MacOSARM64ResourceMonitor,
)
from moment_to_action.hardware._platforms.macos_arm64._models import (
    MacOSARM64ONNXModel,
    MacOSARM64TfliteModel,
)
from moment_to_action.hardware._types import (
    ComputeUnit,
    ComputeUnitUsageSample,
    DataType,
    ModelType,
)


@pytest.mark.unit
class TestMacOSARM64CPUBackend:
    """Tests for MacOSARM64CPUBackend properties and load methods."""

    def test_construction(self) -> None:
        """MacOSARM64CPUBackend constructs without error."""
        assert MacOSARM64CPUBackend() is not None

    def test_unit_property(self) -> None:
        """Unit property returns CPU."""
        assert MacOSARM64CPUBackend().unit == ComputeUnit.CPU

    def test_supported_dtypes(self) -> None:
        """supported_dtypes returns {FP32}."""
        assert MacOSARM64CPUBackend().supported_dtypes == {DataType.FP32}

    def test_supported_formats(self) -> None:
        """supported_formats includes TFLITE and ONNX."""
        fmts = MacOSARM64CPUBackend().supported_formats
        assert ModelType.TFLITE in fmts
        assert ModelType.ONNX in fmts
        assert ModelType.DLC not in fmts

    def test_load_tflite_returns_tflite_model(self) -> None:
        """load_tflite returns a MacOSARM64TfliteModel."""
        mock_interp = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.macos_arm64._cpu_backend._load_litert_interpreter",
            return_value=mock_interp,
        ):
            model = MacOSARM64CPUBackend().load_tflite("/tmp/model.tflite")

        assert isinstance(model, MacOSARM64TfliteModel)

    def test_load_tflite_caches_interpreter(self) -> None:
        """load_tflite with the same path reuses the cached interpreter."""
        mock_interp = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.macos_arm64._cpu_backend._load_litert_interpreter",
            return_value=mock_interp,
        ) as mock_load:
            backend = MacOSARM64CPUBackend()
            backend.load_tflite("/tmp/model.tflite")
            backend.load_tflite("/tmp/model.tflite")

        assert mock_load.call_count == 1

    def test_load_onnx_returns_onnx_model(self) -> None:
        """load_onnx returns a MacOSARM64ONNXModel."""
        mock_session = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.macos_arm64._cpu_backend.ort.InferenceSession",
            return_value=mock_session,
        ):
            model = MacOSARM64CPUBackend().load_onnx("/tmp/model.onnx")

        assert isinstance(model, MacOSARM64ONNXModel)

    def test_load_onnx_caches_session(self) -> None:
        """load_onnx with the same path reuses the cached session."""
        mock_session = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.macos_arm64._cpu_backend.ort.InferenceSession",
            return_value=mock_session,
        ) as mock_cls:
            backend = MacOSARM64CPUBackend()
            backend.load_onnx("/tmp/model.onnx")
            backend.load_onnx("/tmp/model.onnx")

        assert mock_cls.call_count == 1

    def test_load_dlc_raises_not_implemented(self) -> None:
        """load_dlc raises NotImplementedError (not in supported_formats)."""
        with pytest.raises(NotImplementedError):
            MacOSARM64CPUBackend().load_dlc("/tmp/model.dlc")

    def test_load_torch_raises_not_implemented(self) -> None:
        """load_torch raises NotImplementedError (not in supported_formats)."""
        with pytest.raises(NotImplementedError):
            MacOSARM64CPUBackend().load_torch("/tmp/model.pt")


@pytest.mark.unit
class TestMacOSARM64ResourceMonitor:
    """Tests for MacOSARM64ResourceMonitor."""

    def test_sample_cpu_returns_power_sample(self) -> None:
        """sample(CPU) returns a valid PowerSample."""
        with patch("psutil.cpu_percent", return_value=50.0):
            monitor = MacOSARM64ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)
        assert isinstance(sample, ComputeUnitUsageSample)
        assert sample.device == ComputeUnit.CPU
        assert sample.power_mw >= 0.0
        assert 0.0 <= sample.usage_pct <= 100.0
        assert isinstance(sample.timestamp, datetime)

    def test_sample_non_cpu_returns_zero_power(self) -> None:
        """sample(NPU) returns zero power (no NPU on macOS arm64)."""
        monitor = MacOSARM64ResourceMonitor()
        sample = monitor.sample(ComputeUnit.NPU)
        assert sample.device == ComputeUnit.NPU
        assert sample.power_mw == 0.0
        assert sample.usage_pct == 0.0

    def test_power_increases_with_load(self) -> None:
        """Higher CPU utilization produces higher estimated power."""
        monitor = MacOSARM64ResourceMonitor()
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
            monitor = MacOSARM64ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)
        # base=50 + 3.0 GHz x 100% x 0.6 = 50 + 180 = 230
        assert sample.power_mw == pytest.approx(230.0)

    def test_fallback_freq_on_attribute_error(self) -> None:
        """Falls back to default freq when psutil.cpu_freq raises AttributeError."""
        with (
            patch("psutil.cpu_percent", return_value=0.0),
            patch("psutil.cpu_freq", side_effect=AttributeError),
        ):
            monitor = MacOSARM64ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)
        assert sample.power_mw == pytest.approx(50.0)  # base only, 0% load

    def test_utilization_pct_matches_cpu_percent(self) -> None:
        """utilization_pct field reflects psutil.cpu_percent value."""
        with patch("psutil.cpu_percent", return_value=42.0):
            monitor = MacOSARM64ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)
        assert sample.usage_pct == 42.0
