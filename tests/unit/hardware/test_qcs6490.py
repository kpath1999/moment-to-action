"""Unit tests for QCS6490 platform backend and resource monitoring."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moment_to_action.hardware._platforms.qcs6490._backend import (
    QCS6490Backend,
    _ModelHandle,
)
from moment_to_action.hardware._platforms.qcs6490._litert import (
    QCS6490LiteRTBackend,
    _collect_htp_diagnostics,
    _ensure_fastrpc_permissions,
    _is_in_fastrpc_group,
    _parse_delegate_options,
    _probe_delegate_load,
)
from moment_to_action.hardware._platforms.qcs6490._onnx import QCS6490ONNXBackend
from moment_to_action.hardware._platforms.qcs6490._qnn_net_run import QCS6490QNNNetRunBackend
from moment_to_action.hardware._platforms.qcs6490._resources import QCS6490ResourceMonitor
from moment_to_action.hardware._types import ComputeUnit, ComputeUnitUsageSample


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
            assert backend.get_supported_unit() == ComputeUnit.NPU

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

    def test_qcs6490_load_so_routes_to_qnn_net_run(self) -> None:
        """Test QCS6490Backend.load_model routes .so to qnn-net-run backend."""
        mock_litert_cpu = MagicMock()
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU
        mock_onnx = MagicMock()
        mock_qnn = MagicMock()
        mock_qnn.load_model.return_value = "qnn_handle"

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
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490QNNNetRunBackend",
                return_value=mock_qnn,
            ),
        ):
            backend = QCS6490Backend(preferred_unit=ComputeUnit.GPU)
            handle = backend.load_model("/tmp/model.so")

            assert isinstance(handle, _ModelHandle)
            mock_qnn.load_model.assert_called_once_with("/tmp/model.so")
            assert handle.backend == mock_qnn

    def test_try_make_onnx_accel_backend_cpu_unit_returns_none(self) -> None:
        """Test _try_make_onnx_accel_backend returns None for CPU unit."""
        result = QCS6490Backend._try_make_onnx_accel_backend(ComputeUnit.CPU)
        assert result is None

    def test_try_make_onnx_accel_backend_npu_success(self) -> None:
        """Test _try_make_onnx_accel_backend creates backend for NPU unit."""
        mock_backend = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
            return_value=mock_backend,
        ):
            result = QCS6490Backend._try_make_onnx_accel_backend(ComputeUnit.NPU)
            assert result is mock_backend

    def test_try_make_onnx_accel_backend_gpu_success(self) -> None:
        """Test _try_make_onnx_accel_backend creates backend for GPU unit."""
        mock_backend = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
            return_value=mock_backend,
        ):
            result = QCS6490Backend._try_make_onnx_accel_backend(ComputeUnit.GPU)
            assert result is mock_backend

    def test_try_make_onnx_accel_backend_failure_returns_none(self) -> None:
        """Test _try_make_onnx_accel_backend returns None when backend raises."""
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
            side_effect=RuntimeError("QNN ONNX EP not found"),
        ):
            result = QCS6490Backend._try_make_onnx_accel_backend(ComputeUnit.NPU)
            assert result is None

    def test_qcs6490_load_onnx_falls_back_to_cpu_on_accel_failure(self) -> None:
        """Test .onnx loading falls back to CPU when accel backend raises."""
        mock_litert_cpu = MagicMock()
        mock_litert_cpu.get_supported_unit.return_value = ComputeUnit.CPU

        mock_onnx_cpu = MagicMock()
        mock_onnx_cpu.load_model.return_value = "cpu_onnx_handle"
        mock_onnx_accel = MagicMock()
        mock_onnx_accel.load_model.side_effect = RuntimeError("QNN EP unavailable")

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490LiteRTBackend",
                return_value=mock_litert_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490ONNXBackend",
                side_effect=[mock_onnx_cpu, mock_onnx_accel],
            ),
        ):
            backend = QCS6490Backend(preferred_unit=ComputeUnit.NPU)
            handle = backend.load_model("/tmp/model.onnx")

        assert isinstance(handle, _ModelHandle)
        assert handle.backend is mock_onnx_cpu

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
            patch(
                "moment_to_action.hardware._platforms.qcs6490._backend.QCS6490Backend._try_make_onnx_accel_backend",
                return_value=None,
            ),
        ):
            backend = QCS6490Backend(preferred_unit=ComputeUnit.NPU)
            unit = backend.get_supported_unit()

            assert unit == ComputeUnit.CPU

    @pytest.mark.unit
    class TestQCS6490QNNNetRunBackend:
        """Tests for the qnn-net-run subprocess backend."""

        def test_qnn_net_run_backend_run_reads_outputs(self) -> None:
            """run() loads raw outputs and reshapes YOLO-style tensors."""
            backend = QCS6490QNNNetRunBackend(compute_unit=ComputeUnit.GPU)
            handle = backend.load_model("/tmp/model.so")
            input_tensor = np.zeros((1, 640, 640, 3), dtype=np.float32)

            def _fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                output_dir = Path(cmd[cmd.index("--output_dir") + 1])
                result_dir = output_dir / "Result_1"
                result_dir.mkdir(parents=True, exist_ok=True)
                data = np.arange(84 * 2, dtype=np.float32)
                (result_dir / "0.raw").write_bytes(data.tobytes())
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with patch(
                "moment_to_action.hardware._platforms.qcs6490._qnn_net_run.subprocess.run",
                side_effect=_fake_run,
            ):
                outputs = backend.run(handle, input_tensor)

            assert len(outputs) == 1
            assert outputs[0].shape == (1, 84, 2)


@pytest.mark.unit
class TestQCS6490LiteRTBackend:
    """Test QCS6490LiteRTBackend delegate loading."""

    def test_get_delegates_cpu_unit_returns_empty_list(self) -> None:
        """Test _get_delegates returns empty list for CPU unit."""
        backend = QCS6490LiteRTBackend(compute_unit=ComputeUnit.CPU)
        delegates = backend._get_delegates()

        assert delegates == []

    def test_ensure_fastrpc_permissions_raises_when_group_missing(self) -> None:
        """Test that NPU permission checks fail explicitly without re-execing."""
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._litert._is_in_fastrpc_group",
            return_value=False,
        ):
            with pytest.raises(RuntimeError, match="fastrpc"):
                _ensure_fastrpc_permissions()

    def test_get_delegates_npu_unit_loads_qnn_delegate(self) -> None:
        """Test _get_delegates loads QNN delegate for NPU unit."""
        mock_delegate = MagicMock()
        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._ensure_fastrpc_permissions",
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._probe_delegate_load",
                return_value=None,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._load_delegate",
                return_value=mock_delegate,
            ),
        ):
            backend = QCS6490LiteRTBackend(compute_unit=ComputeUnit.NPU)
            delegates = backend._get_delegates()

            assert len(delegates) == 1
            assert delegates[0] == mock_delegate

    def test_get_delegates_npu_unit_raises_on_missing_delegate(self) -> None:
        """Test _get_delegates raises RuntimeError if QNN delegate missing."""
        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._ensure_fastrpc_permissions",
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._probe_delegate_load",
                return_value="native crash — SIGSEGV (exit -11)",
            ),
        ):
            with pytest.raises(RuntimeError, match="NPU delegate unavailable"):
                QCS6490LiteRTBackend(compute_unit=ComputeUnit.NPU)._get_delegates()

    def test_get_delegates_gpu_unit_loads_qnn_delegate(self) -> None:
        """Test _get_delegates loads QNN delegate for GPU unit."""
        mock_delegate = MagicMock()
        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._probe_delegate_load",
                return_value=None,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._load_delegate",
                return_value=mock_delegate,
            ),
        ):
            backend = QCS6490LiteRTBackend(compute_unit=ComputeUnit.GPU)
            delegates = backend._get_delegates()

            assert len(delegates) == 1
            assert delegates[0] == mock_delegate

    def test_parse_delegate_options_parses_key_value_pairs(self) -> None:
        """GPU delegate options should be configurable from env for diagnostics."""
        with patch.dict(
            "os.environ",
            {"MOMENT_TO_ACTION_QNN_GPU_DELEGATE_OPTIONS": "precision_loss_allowed=0,foo=bar"},
            clear=False,
        ):
            options = _parse_delegate_options("MOMENT_TO_ACTION_QNN_GPU_DELEGATE_OPTIONS")

        assert options == {"precision_loss_allowed": "0", "foo": "bar"}

    def test_get_delegates_gpu_unit_merges_env_options(self) -> None:
        """GPU delegate load should include env-provided options."""
        mock_delegate = MagicMock()
        with (
            patch.dict(
                "os.environ",
                {"MOMENT_TO_ACTION_QNN_GPU_DELEGATE_OPTIONS": "precision_loss_allowed=0"},
                clear=False,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._probe_delegate_load",
                return_value=None,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._load_delegate",
                return_value=mock_delegate,
            ) as load_delegate,
        ):
            delegates = QCS6490LiteRTBackend(compute_unit=ComputeUnit.GPU)._get_delegates()

        assert delegates == [mock_delegate]
        load_delegate.assert_called_once_with(
            "/usr/lib/libQnnTFLiteDelegate.so",
            {"backend_type": "gpu", "precision_loss_allowed": "0"},
        )

    def test_get_delegates_gpu_unit_raises_on_missing_delegate(self) -> None:
        """Test _get_delegates raises RuntimeError if GPU delegate load fails."""
        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._probe_delegate_load",
                return_value="native crash — SIGSEGV (exit -11)",
            ),
        ):
            with pytest.raises(RuntimeError, match="GPU delegate unavailable"):
                QCS6490LiteRTBackend(compute_unit=ComputeUnit.GPU)._get_delegates()

    def test_get_delegates_npu_with_load_delegate_exception(self) -> None:
        """Test _get_delegates raises RuntimeError on any delegate load exception."""
        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._ensure_fastrpc_permissions",
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._probe_delegate_load",
                return_value=None,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._load_delegate",
                side_effect=RuntimeError("Delegate load failed"),
            ),
        ):
            with pytest.raises(RuntimeError, match="NPU delegate unavailable"):
                QCS6490LiteRTBackend(compute_unit=ComputeUnit.NPU)._get_delegates()

    def test_is_in_fastrpc_group_true_for_primary_group(self) -> None:
        """Test group check returns True when fastrpc is the primary gid."""
        with (
            patch("grp.getgrnam", return_value=MagicMock(gr_gid=123)),
            patch("os.getgid", return_value=123),
        ):
            assert _is_in_fastrpc_group() is True

    def test_is_in_fastrpc_group_true_for_supplementary_group(self) -> None:
        """Group check returns True when fastrpc appears in supplementary gids."""
        with (
            patch("grp.getgrnam", return_value=MagicMock(gr_gid=123)),
            patch("os.getgid", return_value=999),
            patch("os.getgroups", return_value=[12, 123, 456]),
        ):
            assert _is_in_fastrpc_group() is True

    def test_is_in_fastrpc_group_false_when_group_missing(self) -> None:
        """Test group check returns False when fastrpc group does not exist."""
        with patch("grp.getgrnam", side_effect=KeyError("fastrpc")):
            assert _is_in_fastrpc_group() is False

    def test_probe_delegate_load_timeout_and_failure_paths(self) -> None:
        """Test delegate probe timeout, segfault, and generic failure formatting."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1),
        ):
            timeout_result = _probe_delegate_load("lib.so", {"backend_type": "htp"})
        assert timeout_result is not None
        assert "timed out" in timeout_result

        proc = MagicMock(returncode=139, stderr=b"segv", stdout=b"")
        with patch("subprocess.run", return_value=proc):
            crash_result = _probe_delegate_load("lib.so", {"backend_type": "htp"})
        assert crash_result is not None
        assert "SIGSEGV" in crash_result

        proc2 = MagicMock(returncode=5, stderr=b"oops", stdout=b"")
        with patch("subprocess.run", return_value=proc2):
            fail_result = _probe_delegate_load("lib.so", {"backend_type": "gpu"})
        assert fail_result is not None
        assert "delegate probe exited 5" in fail_result

    def test_collect_htp_diagnostics_includes_env(self) -> None:
        """Test HTP diagnostics include the ADSP library path from environment."""
        with patch.dict("os.environ", {"ADSP_LIBRARY_PATH": "/tmp/adsp"}, clear=False):
            diagnostics = _collect_htp_diagnostics()
        assert "ADSP_LIBRARY_PATH=/tmp/adsp" in diagnostics

    def test_collect_htp_diagnostics_reads_remoteproc_entries(self, tmp_path: Path) -> None:
        """Diagnostics should include remoteproc entries with missing firmware fallback."""
        remoteproc = tmp_path / "remoteproc0"
        remoteproc.mkdir(parents=True)
        (remoteproc / "name").write_text("cdsp", encoding="utf-8")
        (remoteproc / "state").write_text("running", encoding="utf-8")

        with patch(
            "pathlib.Path.glob",
            side_effect=lambda self, _pattern: (
                [remoteproc] if str(self) == "/sys/class/remoteproc" else []
            ),
            autospec=True,
        ):
            diagnostics = _collect_htp_diagnostics()

        assert "cdsp:running:?" in diagnostics

    def test_collect_htp_diagnostics_skips_incomplete_remoteproc(self, tmp_path: Path) -> None:
        """Diagnostics should skip remoteproc entries without both name/state files."""
        remoteproc = tmp_path / "remoteproc1"
        remoteproc.mkdir(parents=True)
        (remoteproc / "name").write_text("cdsp", encoding="utf-8")

        with patch(
            "pathlib.Path.glob",
            side_effect=lambda self, _pattern: (
                [remoteproc] if str(self) == "/sys/class/remoteproc" else []
            ),
            autospec=True,
        ):
            diagnostics = _collect_htp_diagnostics()

        assert "cdsp:" not in diagnostics

    def test_ensure_fastrpc_permissions_noop_when_group_present(self) -> None:
        """Permission guard should not raise when membership check succeeds."""
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._litert._is_in_fastrpc_group",
            return_value=True,
        ):
            _ensure_fastrpc_permissions()

    def test_is_in_fastrpc_group_outer_exception_returns_false(self) -> None:
        """Unexpected errors in group checks should degrade to False."""
        with patch("grp.getgrnam", side_effect=RuntimeError("unexpected")):
            assert _is_in_fastrpc_group() is False

    def test_probe_delegate_load_success_and_start_failure_paths(self) -> None:
        """Probe should return None on success and message when subprocess cannot start."""
        ok_proc = MagicMock(returncode=0, stderr=b"", stdout=b"")
        with patch("subprocess.run", return_value=ok_proc):
            assert _probe_delegate_load("lib.so", {"backend_type": "htp"}) is None

        with patch("subprocess.run", side_effect=OSError("boom")):
            err = _probe_delegate_load("lib.so", {"backend_type": "gpu"})
        assert err is not None
        assert "could not be started" in err

    def test_get_delegates_gpu_unit_raises_on_load_delegate_exception(self) -> None:
        """GPU path should wrap delegate load exceptions in RuntimeError."""
        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._probe_delegate_load",
                return_value=None,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._litert._load_delegate",
                side_effect=RuntimeError("gpu delegate load failed"),
            ),
        ):
            with pytest.raises(RuntimeError, match="GPU delegate unavailable"):
                QCS6490LiteRTBackend(compute_unit=ComputeUnit.GPU)._get_delegates()

    def test_get_delegates_unhandled_unit_returns_empty(self) -> None:
        """Unhandled compute units should return no delegates."""
        backend = QCS6490LiteRTBackend(compute_unit=ComputeUnit.DSP)
        assert backend._get_delegates() == []


@pytest.mark.unit
class TestQCS6490ONNXBackend:
    """Test QCS6490ONNXBackend QNN Execution Provider routing."""

    def test_construction_with_cpu_unit(self) -> None:
        """Test QCS6490ONNXBackend constructs with CPU unit."""
        backend = QCS6490ONNXBackend(compute_unit=ComputeUnit.CPU)
        assert backend.get_supported_unit() == ComputeUnit.CPU

    def test_construction_with_npu_unit(self) -> None:
        """Test QCS6490ONNXBackend constructs with NPU unit."""
        backend = QCS6490ONNXBackend(compute_unit=ComputeUnit.NPU)
        assert backend.get_supported_unit() == ComputeUnit.NPU

    def test_construction_defaults_to_cpu(self) -> None:
        """Test QCS6490ONNXBackend defaults to CPU when no unit given."""
        backend = QCS6490ONNXBackend()
        assert backend.get_supported_unit() == ComputeUnit.CPU

    def test_get_providers_cpu_returns_cpu_ep(self) -> None:
        """Test _get_providers returns CPUExecutionProvider for CPU unit."""
        backend = QCS6490ONNXBackend(compute_unit=ComputeUnit.CPU)
        providers = backend._get_providers()
        assert providers == ["CPUExecutionProvider"]

    def test_get_providers_npu_returns_cpu_ep(self) -> None:
        """Test _get_providers always returns CPUExecutionProvider (NPU handled elsewhere)."""
        backend = QCS6490ONNXBackend(compute_unit=ComputeUnit.NPU)
        providers = backend._get_providers()
        assert providers == ["CPUExecutionProvider"]

    def test_get_providers_dsp_falls_back_to_cpu_ep(self) -> None:
        """Test _get_providers returns CPUExecutionProvider for unhandled units."""
        backend = QCS6490ONNXBackend(compute_unit=ComputeUnit.DSP)
        providers = backend._get_providers()
        assert providers == ["CPUExecutionProvider"]

    def test_make_inference_session_cpu_delegates_to_base(self) -> None:
        """Test _make_inference_session for CPU delegates to base class."""
        backend = QCS6490ONNXBackend(compute_unit=ComputeUnit.CPU)
        mock_session = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._onnx.ort.InferenceSession",
            return_value=mock_session,
        ) as mock_cls:
            result = backend._make_inference_session("/tmp/model.onnx")
        assert result is mock_session
        _, kwargs = mock_cls.call_args
        assert kwargs.get("providers") == ["CPUExecutionProvider"]

    def test_make_inference_session_dsp_delegates_to_base(self) -> None:
        """Test _make_inference_session for DSP falls through to base class (CPU)."""
        backend = QCS6490ONNXBackend(compute_unit=ComputeUnit.DSP)
        mock_session = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._onnx.ort.InferenceSession",
            return_value=mock_session,
        ):
            result = backend._make_inference_session("/tmp/model.onnx")
        assert result is mock_session

    def test_make_inference_session_npu_creates_qnn_session(self) -> None:
        """Test _make_inference_session creates QNN plugin EP session for NPU."""
        backend = QCS6490ONNXBackend(compute_unit=ComputeUnit.NPU)
        mock_device = MagicMock()
        mock_device.ep_name = "QNNExecutionProvider"
        mock_session = MagicMock()
        mock_so = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx._ensure_qnn_ep_registered",
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx.ort.get_ep_devices",
                return_value=[mock_device],
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx._qnn_backend_path",
                return_value="/path/to/libQnnHtp.so",
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx.ort.SessionOptions",
                return_value=mock_so,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx.ort.InferenceSession",
                return_value=mock_session,
            ),
        ):
            result = backend._make_inference_session("/tmp/model.onnx")

        assert result is mock_session
        mock_so.add_provider_for_devices.assert_called_once_with(
            [mock_device], {"backend_path": "/path/to/libQnnHtp.so"}
        )

    def test_make_inference_session_gpu_creates_qnn_session(self) -> None:
        """Test _make_inference_session creates QNN plugin EP session for GPU."""
        backend = QCS6490ONNXBackend(compute_unit=ComputeUnit.GPU)
        mock_device = MagicMock()
        mock_device.ep_name = "QNNExecutionProvider"
        mock_session = MagicMock()
        mock_so = MagicMock()

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx._ensure_qnn_ep_registered",
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx.ort.get_ep_devices",
                return_value=[mock_device],
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx._qnn_backend_path",
                return_value="/path/to/libQnnGpu.so",
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx.ort.SessionOptions",
                return_value=mock_so,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx.ort.InferenceSession",
                return_value=mock_session,
            ),
        ):
            result = backend._make_inference_session("/tmp/model.onnx")

        assert result is mock_session
        mock_so.add_provider_for_devices.assert_called_once_with(
            [mock_device], {"backend_path": "/path/to/libQnnGpu.so"}
        )

    def test_make_inference_session_npu_raises_when_no_devices(self) -> None:
        """Test _make_inference_session raises RuntimeError when no QNN devices found."""
        backend = QCS6490ONNXBackend(compute_unit=ComputeUnit.NPU)

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx._ensure_qnn_ep_registered",
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx.ort.get_ep_devices",
                return_value=[],
            ),
        ):
            with pytest.raises(RuntimeError, match="QNN plugin EP unavailable"):
                backend._make_inference_session("/tmp/model.onnx")

    def test_make_inference_session_gpu_raises_when_no_devices(self) -> None:
        """Test _make_inference_session raises RuntimeError for GPU when no devices."""
        backend = QCS6490ONNXBackend(compute_unit=ComputeUnit.GPU)

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx._ensure_qnn_ep_registered",
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx.ort.get_ep_devices",
                return_value=[],
            ),
        ):
            with pytest.raises(RuntimeError, match="QNN plugin EP unavailable"):
                backend._make_inference_session("/tmp/model.onnx")

    def test_ensure_qnn_ep_registered_no_op_when_already_registered(self) -> None:
        """Test _ensure_qnn_ep_registered is a no-op when already registered."""
        import moment_to_action.hardware._platforms.qcs6490._onnx as onnx_mod
        from moment_to_action.hardware._platforms.qcs6490._onnx import _ensure_qnn_ep_registered

        original = onnx_mod._QNN_EP_REGISTERED
        onnx_mod._QNN_EP_REGISTERED = True
        try:
            with patch(
                "moment_to_action.hardware._platforms.qcs6490._onnx.ort"
                ".register_execution_provider_library"
            ) as mock_reg:
                _ensure_qnn_ep_registered()
            mock_reg.assert_not_called()
        finally:
            onnx_mod._QNN_EP_REGISTERED = original

    def test_ensure_qnn_ep_registered_registers_library(self) -> None:
        """Test _ensure_qnn_ep_registered calls register_execution_provider_library."""
        import moment_to_action.hardware._platforms.qcs6490._onnx as onnx_mod
        from moment_to_action.hardware._platforms.qcs6490._onnx import _ensure_qnn_ep_registered

        original = onnx_mod._QNN_EP_REGISTERED
        onnx_mod._QNN_EP_REGISTERED = False
        mock_qnn = MagicMock()
        mock_qnn.get_library_path.return_value = "/fake/libonnxruntime_providers_qnn.so"
        try:
            with (
                patch.dict(sys.modules, {"onnxruntime_qnn": mock_qnn}),
                patch(
                    "moment_to_action.hardware._platforms.qcs6490._onnx.ort"
                    ".register_execution_provider_library"
                ) as mock_reg,
            ):
                _ensure_qnn_ep_registered()
            mock_reg.assert_called_once_with(
                "QNNExecutionProvider", "/fake/libonnxruntime_providers_qnn.so"
            )
            assert onnx_mod._QNN_EP_REGISTERED is True
        finally:
            onnx_mod._QNN_EP_REGISTERED = original

    def test_ensure_qnn_ep_registered_raises_when_not_installed(self) -> None:
        """Test _ensure_qnn_ep_registered raises RuntimeError when package absent."""
        import moment_to_action.hardware._platforms.qcs6490._onnx as onnx_mod
        from moment_to_action.hardware._platforms.qcs6490._onnx import _ensure_qnn_ep_registered

        original = onnx_mod._QNN_EP_REGISTERED
        onnx_mod._QNN_EP_REGISTERED = False
        try:
            with patch.dict(sys.modules, {"onnxruntime_qnn": None}):
                with pytest.raises(RuntimeError, match="onnxruntime-qnn is not installed"):
                    _ensure_qnn_ep_registered()
        finally:
            onnx_mod._QNN_EP_REGISTERED = original

    def test_qnn_backend_path_npu_returns_htp_path(self) -> None:
        """Test _qnn_backend_path returns HTP library path for NPU."""
        from moment_to_action.hardware._platforms.qcs6490._onnx import _qnn_backend_path

        mock_qnn = MagicMock()
        mock_qnn.get_qnn_htp_path.return_value = "/path/to/libQnnHtp.so"
        with patch.dict(sys.modules, {"onnxruntime_qnn": mock_qnn}):
            result = _qnn_backend_path(ComputeUnit.NPU)
        assert result == "/path/to/libQnnHtp.so"

    def test_qnn_backend_path_gpu_returns_gpu_path(self) -> None:
        """Test _qnn_backend_path returns GPU library path for GPU."""
        from moment_to_action.hardware._platforms.qcs6490._onnx import _qnn_backend_path

        mock_qnn = MagicMock()
        mock_qnn.get_qnn_gpu_path.return_value = "/path/to/libQnnGpu.so"
        with patch.dict(sys.modules, {"onnxruntime_qnn": mock_qnn}):
            result = _qnn_backend_path(ComputeUnit.GPU)
        assert result == "/path/to/libQnnGpu.so"


@pytest.mark.unit
class TestQCS6490ResourceMonitor:
    """Test QCS6490ResourceMonitor power sampling and utilization reading."""

    def test_qcs6490_power_monitor_hw_available_reads_sensor(self) -> None:
        """Test ResourceMonitor reads hw sensor when sysfs path exists."""
        mock_power_path = MagicMock()
        with patch.object(
            QCS6490ResourceMonitor,
            "_discover_power_now_path",
            return_value=mock_power_path,
        ):
            monitor = QCS6490ResourceMonitor()
        assert monitor._hw_available is True

    def test_qcs6490_power_monitor_hw_unavailable_estimates(self) -> None:
        """Test ResourceMonitor uses estimates when sysfs unavailable."""
        with patch.object(
            QCS6490ResourceMonitor,
            "_discover_power_now_path",
            return_value=None,
        ):
            monitor = QCS6490ResourceMonitor()
        assert monitor._hw_available is False

    def test_qcs6490_power_monitor_sample_hw_available(self) -> None:
        """Test sample returns PowerSample from hardware sensor."""
        mock_power_path = MagicMock()
        mock_power_path.read_text.return_value = "5000000\n"

        with (
            patch.object(
                QCS6490ResourceMonitor,
                "_discover_power_now_path",
                return_value=mock_power_path,
            ),
            patch("psutil.cpu_percent", return_value=50.0),
        ):
            monitor = QCS6490ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, ComputeUnitUsageSample)
            assert sample.device == ComputeUnit.CPU
            assert sample.power_mw == 5000.0
            assert sample.usage_pct == 50.0

    def test_qcs6490_power_monitor_sample_hw_unavailable_fallback(self) -> None:
        """Test sample falls back to estimate when sysfs unavailable."""
        with (
            patch.object(
                QCS6490ResourceMonitor,
                "_discover_power_now_path",
                return_value=None,
            ),
            patch("psutil.cpu_percent", return_value=25.0),
        ):
            monitor = QCS6490ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, ComputeUnitUsageSample)
            assert sample.device == ComputeUnit.CPU
            assert sample.power_mw == 300.0
            assert sample.usage_pct == 25.0

    def test_qcs6490_power_monitor_sample_hw_read_error_fallback(self) -> None:
        """Test sample falls back to estimate on hardware read error."""
        mock_power_path = MagicMock()
        mock_power_path.read_text.side_effect = FileNotFoundError()

        with (
            patch.object(
                QCS6490ResourceMonitor,
                "_discover_power_now_path",
                return_value=mock_power_path,
            ),
            patch("psutil.cpu_percent", return_value=30.0),
        ):
            monitor = QCS6490ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, ComputeUnitUsageSample)
            assert sample.device == ComputeUnit.CPU
            assert sample.power_mw == 300.0
            assert sample.usage_pct == 30.0

    def test_qcs6490_power_monitor_read_utilization_cpu(self) -> None:
        """Test _read_utilization returns CPU percent for CPU unit."""
        with patch("psutil.cpu_percent", return_value=45.0):
            util = QCS6490ResourceMonitor._read_utilization(ComputeUnit.CPU)
            assert util == 45.0

    def test_qcs6490_power_monitor_read_utilization_gpu_available(self) -> None:
        """Test _read_utilization reads GPU busy percentage when available."""
        mock_gpu_path = MagicMock()
        mock_gpu_path.exists.return_value = True
        mock_gpu_path.read_text.return_value = "75\n"

        with patch(
            "moment_to_action.hardware._platforms.qcs6490._resources._KGSL_GPU_BUSY_PATH",
            mock_gpu_path,
        ):
            util = QCS6490ResourceMonitor._read_utilization(ComputeUnit.GPU)
            assert util == 75.0

    def test_qcs6490_power_monitor_read_utilization_gpu_unavailable(self) -> None:
        """Test _read_utilization returns 0.0 for GPU when path unavailable."""
        mock_gpu_path = MagicMock()
        mock_gpu_path.exists.return_value = False

        with patch(
            "moment_to_action.hardware._platforms.qcs6490._resources._KGSL_GPU_BUSY_PATH",
            mock_gpu_path,
        ):
            util = QCS6490ResourceMonitor._read_utilization(ComputeUnit.GPU)
            assert util == 0.0

    def test_qcs6490_power_monitor_read_utilization_gpu_read_error(self) -> None:
        """Test _read_utilization returns 0.0 on GPU read error."""
        mock_gpu_path = MagicMock()
        mock_gpu_path.exists.return_value = True
        mock_gpu_path.read_text.side_effect = ValueError("Invalid value")

        with patch(
            "moment_to_action.hardware._platforms.qcs6490._resources._KGSL_GPU_BUSY_PATH",
            mock_gpu_path,
        ):
            util = QCS6490ResourceMonitor._read_utilization(ComputeUnit.GPU)
            assert util == 0.0

    def test_qcs6490_power_monitor_read_utilization_npu_returns_zero(self) -> None:
        """Test _read_utilization returns 0.0 for NPU (no sysfs interface)."""
        util = QCS6490ResourceMonitor._read_utilization(ComputeUnit.NPU)
        assert util == 0.0

    def test_qcs6490_power_monitor_read_utilization_dsp_returns_zero(self) -> None:
        """Test _read_utilization returns 0.0 for DSP (no sysfs interface)."""
        util = QCS6490ResourceMonitor._read_utilization(ComputeUnit.DSP)
        assert util == 0.0

    def test_qcs6490_power_monitor_multiple_samples_npu(self) -> None:
        """Test ResourceMonitor returns consistent samples for NPU."""
        with patch.object(
            QCS6490ResourceMonitor,
            "_discover_power_now_path",
            return_value=None,
        ):
            monitor = QCS6490ResourceMonitor()
            sample1 = monitor.sample(ComputeUnit.NPU)
            sample2 = monitor.sample(ComputeUnit.NPU)

            assert sample1.power_mw == 500.0
            assert sample2.power_mw == 500.0
            assert sample1.device == ComputeUnit.NPU
            assert sample2.device == ComputeUnit.NPU

    def test_qcs6490_power_monitor_multiple_samples_gpu(self) -> None:
        """Test ResourceMonitor returns consistent samples for GPU."""
        with patch.object(
            QCS6490ResourceMonitor,
            "_discover_power_now_path",
            return_value=None,
        ):
            monitor = QCS6490ResourceMonitor()
            sample1 = monitor.sample(ComputeUnit.GPU)
            sample2 = monitor.sample(ComputeUnit.GPU)

            assert sample1.power_mw == 800.0
            assert sample2.power_mw == 800.0
            assert sample1.device == ComputeUnit.GPU
            assert sample2.device == ComputeUnit.GPU

    def test_qcs6490_discover_power_now_prefers_qcom_and_battery(self, tmp_path: Path) -> None:
        """Discovery should prioritize qcom-battmgr-bat then battery paths."""
        root = tmp_path / "power_supply"
        qcom = root / "qcom-battmgr-bat"
        qcom.mkdir(parents=True)
        qcom_power = qcom / "power_now"
        qcom_power.write_text("1", encoding="utf-8")

        with patch.object(QCS6490ResourceMonitor, "SYSFS_POWER_PATH", str(root)):
            monitor = QCS6490ResourceMonitor()
        assert monitor._power_now_path == qcom_power

    def test_qcs6490_discover_power_now_fallback_and_oserror(self, tmp_path: Path) -> None:
        """Discovery should use first iterdir power_now and handle iterdir OSError."""
        root = tmp_path / "power_supply"
        root.mkdir(parents=True)

        supply = root / "supply0"
        supply.mkdir()
        fallback_power = supply / "power_now"
        fallback_power.write_text("1", encoding="utf-8")

        with patch.object(QCS6490ResourceMonitor, "SYSFS_POWER_PATH", str(root)):
            monitor = QCS6490ResourceMonitor()
        assert monitor._power_now_path == fallback_power

        with (
            patch.object(QCS6490ResourceMonitor, "SYSFS_POWER_PATH", str(root)),
            patch("pathlib.Path.iterdir", side_effect=OSError("no access")),
        ):
            monitor2 = QCS6490ResourceMonitor()
        assert monitor2._power_now_path is None

    def test_qcs6490_discover_power_now_returns_none_without_candidates(
        self,
        tmp_path: Path,
    ) -> None:
        """Discovery should return None when power_supply exists but has no power_now files."""
        root = tmp_path / "power_supply"
        supply = root / "supply0"
        supply.mkdir(parents=True)

        with patch.object(QCS6490ResourceMonitor, "SYSFS_POWER_PATH", str(root)):
            monitor = QCS6490ResourceMonitor()

        assert monitor._power_now_path is None

    def test_qcs6490_read_hw_sensor_none_path_and_freq_none(self) -> None:
        """Hardware read should estimate when path is None and CPU freq may be None."""
        with patch.object(
            QCS6490ResourceMonitor,
            "_discover_power_now_path",
            return_value=None,
        ):
            monitor = QCS6490ResourceMonitor()

        monitor._hw_available = True
        monitor._power_now_path = None
        sample = monitor._read_hw_sensor(ComputeUnit.CPU)
        assert sample.power_mw == 300.0

        with patch("psutil.cpu_freq", return_value=None):
            assert QCS6490ResourceMonitor._read_frequency_mhz(ComputeUnit.CPU) == 0.0

    def test_qcs6490_read_frequency_mhz_cpu_error_fallback(self) -> None:
        """Test _read_frequency_mhz returns 0.0 when psutil.cpu_freq raises for CPU.

        Covers the except branch at lines 108-109 of _power.py.
        """
        with patch("psutil.cpu_freq", side_effect=OSError("cpu_freq unavailable")):
            freq = QCS6490ResourceMonitor._read_frequency_mhz(ComputeUnit.CPU)
            assert freq == 0.0

    def test_qcs6490_read_hw_sensor_typeerror_falls_back_to_estimate(self) -> None:
        """TypeError from sysfs read (e.g. NoneType concat) must not crash profiling."""
        mock_power_path = MagicMock()
        mock_power_path.read_text.side_effect = TypeError("can't concat NoneType to bytes")

        with patch.object(
            QCS6490ResourceMonitor,
            "_discover_power_now_path",
            return_value=mock_power_path,
        ):
            monitor = QCS6490ResourceMonitor()

        sample = monitor._read_hw_sensor(ComputeUnit.GPU)
        assert sample.power_mw == pytest.approx(800.0)
        assert sample.device == ComputeUnit.GPU

    def test_qcs6490_read_hw_sensor_oserror_falls_back_to_estimate(self) -> None:
        """OSError (e.g. permission denied) on power_now must not crash profiling."""
        mock_power_path = MagicMock()
        mock_power_path.read_text.side_effect = OSError("permission denied")

        with patch.object(
            QCS6490ResourceMonitor,
            "_discover_power_now_path",
            return_value=mock_power_path,
        ):
            monitor = QCS6490ResourceMonitor()

        sample = monitor._read_hw_sensor(ComputeUnit.NPU)
        assert sample.power_mw == pytest.approx(500.0)
        assert sample.device == ComputeUnit.NPU

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
