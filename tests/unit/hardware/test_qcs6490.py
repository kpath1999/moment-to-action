"""Unit tests for QCS6490 platform backend and power monitoring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moment_to_action.hardware._platforms.qcs6490._backend import (
    QCS6490Backend,
    _ModelHandle,
)
from moment_to_action.hardware._platforms.qcs6490._litert import QCS6490LiteRTBackend
from moment_to_action.hardware._platforms.qcs6490._power import QCS6490PowerMonitor
from moment_to_action.hardware._types import ComputeUnit, PowerSample


@pytest.mark.unit
class TestQCS6490Backend:
    """Test QCS6490Backend construction and routing."""

    def test_qcs6490_backend_construction_with_npu_preferred(self) -> None:
        """Test QCS6490Backend construction with NPU preferred unit."""
        mock_litert_cpu = MagicMock()
        mock_litert_accel = MagicMock()
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                side_effect=[mock_litert_cpu, mock_litert_accel],
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = QCS6490Backend(preferred_unit=ComputeUnit.NPU)
            assert backend is not None

    def test_qcs6490_backend_construction_with_gpu_preferred(self) -> None:
        """Test QCS6490Backend construction with GPU preferred unit."""
        mock_litert_cpu = MagicMock()
        mock_litert_accel = MagicMock()
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                side_effect=[mock_litert_cpu, mock_litert_accel],
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = QCS6490Backend(preferred_unit=ComputeUnit.GPU)
            assert backend is not None

    def test_qcs6490_backend_construction_with_cpu_preferred(self) -> None:
        """Test QCS6490Backend construction with CPU preferred unit."""
        mock_litert_cpu = MagicMock()
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                return_value=mock_litert_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = QCS6490Backend(preferred_unit=ComputeUnit.CPU)
            assert backend is not None

    def test_try_make_accel_backend_cpu_unit_returns_none(self) -> None:
        """Test _try_make_accel_backend returns None for CPU unit."""
        result = QCS6490Backend._try_make_accel_backend(ComputeUnit.CPU)
        assert result is None

    def test_try_make_accel_backend_npu_success(self) -> None:
        """Test _try_make_accel_backend creates backend for NPU unit."""
        mock_backend = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
            return_value=mock_backend,
        ):
            result = QCS6490Backend._try_make_accel_backend(ComputeUnit.NPU)
            assert result is not None
            assert result == mock_backend

    def test_try_make_accel_backend_gpu_success(self) -> None:
        """Test _try_make_accel_backend creates backend for GPU unit."""
        mock_backend = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
            return_value=mock_backend,
        ):
            result = QCS6490Backend._try_make_accel_backend(ComputeUnit.GPU)
            assert result is not None
            assert result == mock_backend

    def test_try_make_accel_backend_npu_failure_returns_none(self) -> None:
        """Test _try_make_accel_backend returns None when delegate fails."""
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
            side_effect=RuntimeError("QNN delegate not found"),
        ):
            result = QCS6490Backend._try_make_accel_backend(ComputeUnit.NPU)
            assert result is None

    def test_qcs6490_load_tflite_routes_to_litert(self) -> None:
        """Test QCS6490Backend.load_model routes .tflite to LiteRT."""
        mock_litert_cpu = MagicMock()
        mock_litert_cpu.load_model.return_value = "mock_handle"
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                return_value=mock_litert_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = QCS6490Backend(preferred_unit=ComputeUnit.CPU)
            handle = backend.load_model("/tmp/model.tflite")

            assert isinstance(handle, _ModelHandle)
            mock_litert_cpu.load_model.assert_called_once_with("/tmp/model.tflite")

    def test_qcs6490_load_tflite_accel_fallback_to_cpu(self) -> None:
        """Test .tflite loading falls back to CPU when accel fails."""
        mock_litert_accel = MagicMock()
        mock_litert_accel.load_model.side_effect = RuntimeError("Accel unavailable")
        mock_litert_accel.get_supported_unit.return_value = ComputeUnit.NPU
        mock_litert_cpu = MagicMock()
        mock_litert_cpu.load_model.return_value = "cpu_handle"
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                side_effect=[mock_litert_cpu, mock_litert_accel],
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = QCS6490Backend(preferred_unit=ComputeUnit.NPU)
            handle = backend.load_model("/tmp/model.tflite")

            assert isinstance(handle, _ModelHandle)
            assert handle.backend == mock_litert_cpu

    def test_qcs6490_load_onnx_routes_correctly(self) -> None:
        """Test QCS6490Backend.load_model routes .onnx to ONNX."""
        mock_litert_cpu = MagicMock()
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU
        mock_onnx = MagicMock()
        mock_onnx.load_model.return_value = "onnx_handle"

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                return_value=mock_litert_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = QCS6490Backend()
            handle = backend.load_model("/tmp/model.onnx")

            assert isinstance(handle, _ModelHandle)
            mock_onnx.load_model.assert_called_once_with("/tmp/model.onnx")
            assert handle.backend == mock_onnx

    def test_qcs6490_load_unsupported_format_raises(self) -> None:
        """Test QCS6490Backend.load_model raises ValueError for unsupported format."""
        mock_litert_cpu = MagicMock()
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                return_value=mock_litert_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = QCS6490Backend()
            with pytest.raises(ValueError, match="Unsupported model format"):
                backend.load_model("/tmp/model.pt")

    def test_qcs6490_run_delegates_to_backend(self) -> None:
        """Test QCS6490Backend.run delegates to the appropriate sub-backend."""
        mock_litert_cpu = MagicMock()
        output_tensor = np.array([1.0, 2.0])
        mock_litert_cpu.run.return_value = [output_tensor]
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                return_value=mock_litert_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = QCS6490Backend()
            handle = _ModelHandle(raw="mock_raw", backend=mock_litert_cpu)
            input_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
            outputs = backend.run(handle, input_tensor)

            mock_litert_cpu.run.assert_called_once()
            assert len(outputs) == 1

    def test_qcs6490_get_input_details_delegates(self) -> None:
        """Test QCS6490Backend.get_input_details delegates correctly."""
        mock_litert_cpu = MagicMock()
        input_details = [{"name": "input", "shape": (1, 224, 224, 3)}]
        mock_litert_cpu.get_input_details.return_value = input_details
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                return_value=mock_litert_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = QCS6490Backend()
            handle = _ModelHandle(raw="mock_raw", backend=mock_litert_cpu)
            details = backend.get_input_details(handle)

            assert details == input_details

    def test_qcs6490_get_output_details_delegates(self) -> None:
        """Test QCS6490Backend.get_output_details delegates correctly."""
        mock_litert_cpu = MagicMock()
        output_details = [{"name": "output", "shape": (1, 1000)}]
        mock_litert_cpu.get_output_details.return_value = output_details
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                return_value=mock_litert_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = QCS6490Backend()
            handle = _ModelHandle(raw="mock_raw", backend=mock_litert_cpu)
            details = backend.get_output_details(handle)

            assert details == output_details

    def test_qcs6490_resolve_torch_policy_delegates_to_helper(self) -> None:
        """Test QCS6490 torch policy is resolved by shared helper."""
        mock_litert_cpu = MagicMock()
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                return_value=mock_litert_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.resolve_torch_execution_policy"
            ) as mock_resolve,
        ):
            mock_resolve.return_value.device = "cpu"
            mock_resolve.return_value.dtype = "float32"

            backend = QCS6490Backend()
            policy = backend.resolve_torch_policy("auto")

            mock_resolve.assert_called_once_with("auto")
            assert policy.device == "cpu"
            assert policy.dtype == "float32"

    def test_qcs6490_get_supported_unit_with_accel(self) -> None:
        """Test QCS6490Backend.get_supported_unit returns accel unit when available."""
        mock_litert_cpu = MagicMock()
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU
        mock_litert_accel = MagicMock()
        mock_litert_accel.get_supported_unit.return_value = ComputeUnit.NPU
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                side_effect=[mock_litert_cpu, mock_litert_accel],
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = QCS6490Backend(preferred_unit=ComputeUnit.NPU)
            unit = backend.get_supported_unit()

            assert unit == ComputeUnit.NPU

    def test_qcs6490_get_supported_unit_without_accel(self) -> None:
        """Test QCS6490Backend.get_supported_unit returns CPU when accel unavailable."""
        mock_litert_cpu = MagicMock()
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                return_value=mock_litert_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490Backend._try_make_accel_backend",
                return_value=None,
            ),
        ):
            backend = QCS6490Backend(preferred_unit=ComputeUnit.NPU)
            unit = backend.get_supported_unit()

            assert unit == ComputeUnit.CPU


@pytest.mark.unit
class TestQCS6490LiteRTBackend:
    """Test QCS6490LiteRTBackend delegate loading."""

    def test_get_delegates_cpu_unit_returns_empty_list(self) -> None:
        """Test _get_delegates returns empty list for CPU unit."""
        backend = QCS6490LiteRTBackend(compute_unit=ComputeUnit.CPU)
        delegates = backend._get_delegates()

        assert delegates == []

    def test_get_delegates_npu_unit_loads_qnn_delegate(self) -> None:
        """Test _get_delegates loads QNN delegate for NPU unit."""
        mock_delegate = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._litert._load_delegate",
            return_value=mock_delegate,
        ):
            backend = QCS6490LiteRTBackend(compute_unit=ComputeUnit.NPU)
            delegates = backend._get_delegates()

            assert len(delegates) == 1
            assert delegates[0] == mock_delegate

    def test_get_delegates_npu_unit_raises_on_missing_delegate(self) -> None:
        """Test _get_delegates raises RuntimeError if QNN delegate missing."""
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._litert._load_delegate",
            side_effect=FileNotFoundError("Delegate not found"),
        ):
            with pytest.raises(RuntimeError, match="NPU delegate unavailable"):
                QCS6490LiteRTBackend(compute_unit=ComputeUnit.NPU)._get_delegates()

    def test_get_delegates_gpu_unit_returns_empty_list(self) -> None:
        """Test _get_delegates returns empty list for GPU unit (not yet implemented)."""
        backend = QCS6490LiteRTBackend(compute_unit=ComputeUnit.GPU)
        delegates = backend._get_delegates()

        assert delegates == []

    def test_get_delegates_npu_with_load_delegate_exception(self) -> None:
        """Test _get_delegates raises RuntimeError on any delegate load exception."""
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._litert._load_delegate",
            side_effect=RuntimeError("Delegate load failed"),
        ):
            with pytest.raises(RuntimeError, match="NPU delegate unavailable"):
                QCS6490LiteRTBackend(compute_unit=ComputeUnit.NPU)._get_delegates()


@pytest.mark.unit
class TestQCS6490PowerMonitor:
    """Test QCS6490PowerMonitor power sampling and utilization reading."""

    def test_qcs6490_power_monitor_hw_available_reads_sensor(self) -> None:
        """Test PowerMonitor reads hw sensor when sysfs path exists."""
        mock_sysfs = MagicMock()
        mock_sysfs.exists.return_value = True
        mock_power_path = MagicMock()
        mock_power_path.read_text.return_value = "5000000\n"

        with (
            patch("moment_to_action.hardware._platforms.qcs6490._power.Path") as mock_path,
            patch("psutil.cpu_percent", return_value=50.0),
        ):
            mock_path.return_value = mock_sysfs
            monitor = QCS6490PowerMonitor()
            assert monitor._hw_available is True

    def test_qcs6490_power_monitor_hw_unavailable_estimates(self) -> None:
        """Test PowerMonitor uses estimates when sysfs unavailable."""
        mock_sysfs = MagicMock()
        mock_sysfs.exists.return_value = False

        with patch("moment_to_action.hardware._platforms.qcs6490._power.Path") as mock_path:
            mock_path.return_value = mock_sysfs
            monitor = QCS6490PowerMonitor()
            assert monitor._hw_available is False

    def test_qcs6490_power_monitor_sample_hw_available(self) -> None:
        """Test sample returns PowerSample from hardware sensor."""
        mock_sysfs_root = MagicMock()
        mock_sysfs_root.exists.return_value = True
        mock_power_path = MagicMock()
        mock_power_path.read_text.return_value = "5000000\n"

        with (
            patch("moment_to_action.hardware._platforms.qcs6490._power.Path") as mock_path_cls,
            patch("psutil.cpu_percent", return_value=50.0),
        ):

            def path_side_effect(path_str: str) -> MagicMock:
                if "battery/power_now" in path_str:
                    return mock_power_path
                # Default for sysfs check in __init__
                return mock_sysfs_root

            mock_path_cls.side_effect = path_side_effect

            monitor = QCS6490PowerMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, PowerSample)
            assert sample.device == ComputeUnit.CPU
            assert sample.power_mw == 5000.0
            assert sample.utilization_pct == 50.0

    def test_qcs6490_power_monitor_sample_hw_unavailable_fallback(self) -> None:
        """Test sample falls back to estimate when sysfs unavailable."""
        mock_sysfs_root = MagicMock()
        mock_sysfs_root.exists.return_value = False

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._power.Path",
                return_value=mock_sysfs_root,
            ),
            patch("psutil.cpu_percent", return_value=25.0),
        ):
            monitor = QCS6490PowerMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, PowerSample)
            assert sample.device == ComputeUnit.CPU
            assert sample.power_mw == 300.0
            assert sample.utilization_pct == 25.0

    def test_qcs6490_power_monitor_sample_hw_read_error_fallback(self) -> None:
        """Test sample falls back to estimate on hardware read error."""
        mock_sysfs_root = MagicMock()
        mock_sysfs_root.exists.return_value = True
        mock_power_path = MagicMock()
        mock_power_path.read_text.side_effect = FileNotFoundError()

        with (
            patch("moment_to_action.hardware._platforms.qcs6490._power.Path") as mock_path_cls,
            patch("psutil.cpu_percent", return_value=30.0),
        ):

            def path_side_effect(path_str: str) -> MagicMock:
                if "battery/power_now" in path_str:
                    return mock_power_path
                # Default for sysfs check in __init__
                return mock_sysfs_root

            mock_path_cls.side_effect = path_side_effect

            monitor = QCS6490PowerMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, PowerSample)
            assert sample.device == ComputeUnit.CPU
            assert sample.power_mw == 300.0
            assert sample.utilization_pct == 30.0

    def test_qcs6490_power_monitor_read_utilization_cpu(self) -> None:
        """Test _read_utilization returns CPU percent for CPU unit."""
        with patch("psutil.cpu_percent", return_value=45.0):
            util = QCS6490PowerMonitor._read_utilization(ComputeUnit.CPU)
            assert util == 45.0

    def test_qcs6490_power_monitor_read_utilization_gpu_available(self) -> None:
        """Test _read_utilization reads GPU busy percentage when available."""
        mock_gpu_path = MagicMock()
        mock_gpu_path.exists.return_value = True
        mock_gpu_path.read_text.return_value = "75\n"

        with patch(
            "moment_to_action.hardware._platforms.qcs6490._power._KGSL_GPU_BUSY_PATH",
            mock_gpu_path,
        ):
            util = QCS6490PowerMonitor._read_utilization(ComputeUnit.GPU)
            assert util == 75.0

    def test_qcs6490_power_monitor_read_utilization_gpu_unavailable(self) -> None:
        """Test _read_utilization returns 0.0 for GPU when path unavailable."""
        mock_gpu_path = MagicMock()
        mock_gpu_path.exists.return_value = False

        with patch(
            "moment_to_action.hardware._platforms.qcs6490._power._KGSL_GPU_BUSY_PATH",
            mock_gpu_path,
        ):
            util = QCS6490PowerMonitor._read_utilization(ComputeUnit.GPU)
            assert util == 0.0

    def test_qcs6490_power_monitor_read_utilization_gpu_read_error(self) -> None:
        """Test _read_utilization returns 0.0 on GPU read error."""
        mock_gpu_path = MagicMock()
        mock_gpu_path.exists.return_value = True
        mock_gpu_path.read_text.side_effect = ValueError("Invalid value")

        with patch(
            "moment_to_action.hardware._platforms.qcs6490._power._KGSL_GPU_BUSY_PATH",
            mock_gpu_path,
        ):
            util = QCS6490PowerMonitor._read_utilization(ComputeUnit.GPU)
            assert util == 0.0

    def test_qcs6490_power_monitor_read_utilization_npu_returns_zero(self) -> None:
        """Test _read_utilization returns 0.0 for NPU (no sysfs interface)."""
        util = QCS6490PowerMonitor._read_utilization(ComputeUnit.NPU)
        assert util == 0.0

    def test_qcs6490_power_monitor_read_utilization_dsp_returns_zero(self) -> None:
        """Test _read_utilization returns 0.0 for DSP (no sysfs interface)."""
        util = QCS6490PowerMonitor._read_utilization(ComputeUnit.DSP)
        assert util == 0.0

    def test_qcs6490_power_monitor_multiple_samples_npu(self) -> None:
        """Test PowerMonitor returns consistent samples for NPU."""
        mock_sysfs_root = MagicMock()
        mock_sysfs_root.exists.return_value = False

        with patch(
            "moment_to_action.hardware._platforms.qcs6490._power.Path",
            return_value=mock_sysfs_root,
        ):
            monitor = QCS6490PowerMonitor()
            sample1 = monitor.sample(ComputeUnit.NPU)
            sample2 = monitor.sample(ComputeUnit.NPU)

            assert sample1.power_mw == 500.0
            assert sample2.power_mw == 500.0
            assert sample1.device == ComputeUnit.NPU
            assert sample2.device == ComputeUnit.NPU

    def test_qcs6490_power_monitor_multiple_samples_gpu(self) -> None:
        """Test PowerMonitor returns consistent samples for GPU."""
        mock_sysfs_root = MagicMock()
        mock_sysfs_root.exists.return_value = False

        with patch(
            "moment_to_action.hardware._platforms.qcs6490._power.Path",
            return_value=mock_sysfs_root,
        ):
            monitor = QCS6490PowerMonitor()
            sample1 = monitor.sample(ComputeUnit.GPU)
            sample2 = monitor.sample(ComputeUnit.GPU)

            assert sample1.power_mw == 800.0
            assert sample2.power_mw == 800.0
            assert sample1.device == ComputeUnit.GPU
            assert sample2.device == ComputeUnit.GPU

    def test_qcs6490_load_tflite_accel_success_path(self) -> None:
        """Test .tflite loading uses accel backend when it succeeds.

        Covers the return path through the accelerator handle (line 212 in
        _backend.py) — the happy path when NPU/GPU delegate loads cleanly.
        """
        mock_litert_cpu = MagicMock()
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU
        mock_litert_accel = MagicMock()
        # Accel succeeds — no exception raised.
        mock_litert_accel.load_model.return_value = "accel_raw_handle"
        mock_litert_accel.get_supported_unit.return_value = ComputeUnit.NPU
        mock_onnx = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                side_effect=[mock_litert_cpu, mock_litert_accel],
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                return_value=mock_onnx,
            ),
        ):
            backend = QCS6490Backend(preferred_unit=ComputeUnit.NPU)
            handle = backend.load_model("/tmp/model.tflite")

            # The handle's backing sub-backend should be the accel backend.
            assert isinstance(handle, _ModelHandle)
            assert handle.backend is mock_litert_accel
            assert handle.raw == "accel_raw_handle"
            mock_litert_accel.load_model.assert_called_once_with("/tmp/model.tflite")
            # CPU backend should NOT be called — accel succeeded.
            mock_litert_cpu.load_model.assert_not_called()
