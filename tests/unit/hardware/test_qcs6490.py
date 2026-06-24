"""Unit tests for QCS6490 platform backend and resource monitoring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moment_to_action.hardware._platforms.qcs6490._resources import QCS6490ResourceMonitor
from moment_to_action.hardware._types import ComputeUnit, ComputeUnitUsageSample


@pytest.mark.unit
class TestQCS6490ResourceMonitor:
    """Test QCS6490ResourceMonitor power sampling and utilization reading."""

    def test_qcs6490_power_monitor_hw_available_reads_sensor(self) -> None:
        """Test ResourceMonitor reads hw sensor when sysfs path exists."""
        mock_sysfs = MagicMock()
        mock_sysfs.exists.return_value = True
        mock_power_path = MagicMock()
        mock_power_path.read_text.return_value = "5000000\n"

        with (
            patch("moment_to_action.hardware._platforms.qcs6490._resources.Path") as mock_path,
            patch("psutil.cpu_percent", return_value=50.0),
        ):
            mock_path.return_value = mock_sysfs
            monitor = QCS6490ResourceMonitor()
            assert monitor._hw_available is True

    def test_qcs6490_power_monitor_hw_unavailable_estimates(self) -> None:
        """Test ResourceMonitor uses estimates when sysfs unavailable."""
        mock_sysfs = MagicMock()
        mock_sysfs.exists.return_value = False

        with patch("moment_to_action.hardware._platforms.qcs6490._resources.Path") as mock_path:
            mock_path.return_value = mock_sysfs
            monitor = QCS6490ResourceMonitor()
            assert monitor._hw_available is False

    def test_qcs6490_power_monitor_sample_hw_available(self) -> None:
        """Test sample returns PowerSample from hardware sensor."""
        mock_sysfs_root = MagicMock()
        mock_sysfs_root.exists.return_value = True
        mock_power_path = MagicMock()
        mock_power_path.read_text.return_value = "5000000\n"

        with (
            patch("moment_to_action.hardware._platforms.qcs6490._resources.Path") as mock_path_cls,
            patch("psutil.cpu_percent", return_value=50.0),
        ):

            def path_side_effect(path_str: str) -> MagicMock:
                if "battery/power_now" in path_str:
                    return mock_power_path
                # Default for sysfs check in __init__
                return mock_sysfs_root

            mock_path_cls.side_effect = path_side_effect

            monitor = QCS6490ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, ComputeUnitUsageSample)
            assert sample.device == ComputeUnit.CPU
            assert sample.power_mw == 5000.0
            assert sample.usage_pct == 50.0

    def test_qcs6490_power_monitor_sample_hw_unavailable_fallback(self) -> None:
        """Test sample falls back to estimate when sysfs unavailable."""
        mock_sysfs_root = MagicMock()
        mock_sysfs_root.exists.return_value = False

        with (
            patch(
                "moment_to_action.hardware._platforms.qcs6490._resources.Path",
                return_value=mock_sysfs_root,
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
        mock_sysfs_root = MagicMock()
        mock_sysfs_root.exists.return_value = True
        mock_power_path = MagicMock()
        mock_power_path.read_text.side_effect = FileNotFoundError()

        with (
            patch("moment_to_action.hardware._platforms.qcs6490._resources.Path") as mock_path_cls,
            patch("psutil.cpu_percent", return_value=30.0),
        ):

            def path_side_effect(path_str: str) -> MagicMock:
                if "battery/power_now" in path_str:
                    return mock_power_path
                # Default for sysfs check in __init__
                return mock_sysfs_root

            mock_path_cls.side_effect = path_side_effect

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

    def test_qcs6490_power_monitor_multiple_samples_npu(self) -> None:
        """Test ResourceMonitor returns consistent samples for NPU."""
        mock_sysfs_root = MagicMock()
        mock_sysfs_root.exists.return_value = False

        with patch(
            "moment_to_action.hardware._platforms.qcs6490._resources.Path",
            return_value=mock_sysfs_root,
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
        mock_sysfs_root = MagicMock()
        mock_sysfs_root.exists.return_value = False

        with patch(
            "moment_to_action.hardware._platforms.qcs6490._resources.Path",
            return_value=mock_sysfs_root,
        ):
            monitor = QCS6490ResourceMonitor()
            sample1 = monitor.sample(ComputeUnit.GPU)
            sample2 = monitor.sample(ComputeUnit.GPU)

            assert sample1.power_mw == 800.0
            assert sample2.power_mw == 800.0
            assert sample1.device == ComputeUnit.GPU
            assert sample2.device == ComputeUnit.GPU

    def test_qcs6490_read_frequency_mhz_cpu_error_fallback(self) -> None:
        """Test _read_frequency_mhz returns 0.0 when psutil.cpu_freq raises for CPU.

        Covers the except branch at lines 108-109 of _power.py.
        """
        with patch("psutil.cpu_freq", side_effect=OSError("cpu_freq unavailable")):
            freq = QCS6490ResourceMonitor._read_frequency_mhz(ComputeUnit.CPU)
            assert freq == 0.0


# ---------------------------------------------------------------------------
# New-architecture tests for QCS6490 per-unit backends and models
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQCS6490CPUBackend:
    """Tests for QCS6490CPUBackend (new per-unit architecture)."""

    def test_construction(self) -> None:
        """QCS6490CPUBackend constructs without error."""
        from moment_to_action.hardware._platforms.qcs6490._cpu_backend import QCS6490CPUBackend

        assert QCS6490CPUBackend() is not None

    def test_unit_property(self) -> None:
        """Unit is always CPU."""
        from moment_to_action.hardware._platforms.qcs6490._cpu_backend import QCS6490CPUBackend
        from moment_to_action.hardware._types import ComputeUnit

        assert QCS6490CPUBackend().unit == ComputeUnit.CPU

    def test_supported_dtypes(self) -> None:
        """supported_dtypes returns {FP32}."""
        from moment_to_action.hardware._platforms.qcs6490._cpu_backend import QCS6490CPUBackend
        from moment_to_action.hardware._types import DataType

        assert QCS6490CPUBackend().supported_dtypes == {DataType.FP32}

    def test_supported_formats(self) -> None:
        """supported_formats includes TFLITE and ONNX."""
        from moment_to_action.hardware._platforms.qcs6490._cpu_backend import QCS6490CPUBackend
        from moment_to_action.hardware._types import ModelType

        fmts = QCS6490CPUBackend().supported_formats
        assert ModelType.TFLITE in fmts
        assert ModelType.ONNX in fmts

    def test_load_tflite_returns_tflite_model(self) -> None:
        """load_tflite returns a QCS6490TfliteModel."""
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._cpu_backend import QCS6490CPUBackend
        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490TfliteModel

        mock_interp = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._cpu_backend._load_litert_interpreter",
            return_value=mock_interp,
        ):
            model = QCS6490CPUBackend().load_tflite("/tmp/model.tflite")
        assert isinstance(model, QCS6490TfliteModel)

    def test_load_tflite_caches_interpreter(self) -> None:
        """load_tflite with the same path reuses the cached interpreter."""
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._cpu_backend import QCS6490CPUBackend

        mock_interp = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._cpu_backend._load_litert_interpreter",
            return_value=mock_interp,
        ) as mock_load:
            backend = QCS6490CPUBackend()
            backend.load_tflite("/tmp/model.tflite")
            backend.load_tflite("/tmp/model.tflite")
        assert mock_load.call_count == 1

    def test_load_onnx_returns_onnx_model(self) -> None:
        """load_onnx returns a QCS6490ONNXModel."""
        from unittest.mock import MagicMock, patch

        import onnxruntime as ort

        from moment_to_action.hardware._platforms.qcs6490._cpu_backend import QCS6490CPUBackend
        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490ONNXModel

        mock_session = MagicMock()
        with patch.object(ort, "InferenceSession", return_value=mock_session):
            model = QCS6490CPUBackend().load_onnx("/tmp/model.onnx")
        assert isinstance(model, QCS6490ONNXModel)

    def test_load_onnx_caches_session(self) -> None:
        """load_onnx with the same path reuses cached session."""
        from unittest.mock import MagicMock, patch

        import onnxruntime as ort

        from moment_to_action.hardware._platforms.qcs6490._cpu_backend import QCS6490CPUBackend

        mock_session = MagicMock()
        with patch.object(ort, "InferenceSession", return_value=mock_session) as mock_cls:
            backend = QCS6490CPUBackend()
            backend.load_onnx("/tmp/model.onnx")
            backend.load_onnx("/tmp/model.onnx")
        assert mock_cls.call_count == 1

    def test_load_litert_interpreter_allocates(self) -> None:
        """_load_litert_interpreter calls Interpreter and allocate_tensors."""
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._cpu_backend import (
            _load_litert_interpreter,
        )

        mock_interp = MagicMock()
        with patch("ai_edge_litert.interpreter.Interpreter", return_value=mock_interp) as mock_cls:
            result = _load_litert_interpreter("/tmp/model.tflite")
        mock_cls.assert_called_once_with(model_path="/tmp/model.tflite", experimental_delegates=[])
        mock_interp.allocate_tensors.assert_called_once()
        assert result is mock_interp


@pytest.mark.unit
class TestQCS6490GPUBackend:
    """Tests for QCS6490GPUBackend (new per-unit architecture)."""

    def test_construction(self) -> None:
        """QCS6490GPUBackend constructs without error."""
        from moment_to_action.hardware._platforms.qcs6490._gpu_backend import QCS6490GPUBackend

        assert QCS6490GPUBackend() is not None

    def test_unit_property(self) -> None:
        """Unit is GPU."""
        from moment_to_action.hardware._platforms.qcs6490._gpu_backend import QCS6490GPUBackend
        from moment_to_action.hardware._types import ComputeUnit

        assert QCS6490GPUBackend().unit == ComputeUnit.GPU

    def test_supported_dtypes(self) -> None:
        """supported_dtypes returns {FP16, FP32}."""
        from moment_to_action.hardware._platforms.qcs6490._gpu_backend import QCS6490GPUBackend
        from moment_to_action.hardware._types import DataType

        fmts = QCS6490GPUBackend().supported_dtypes
        assert DataType.FP16 in fmts
        assert DataType.FP32 in fmts

    def test_supported_formats(self) -> None:
        """supported_formats includes TFLITE."""
        from moment_to_action.hardware._platforms.qcs6490._gpu_backend import QCS6490GPUBackend
        from moment_to_action.hardware._types import ModelType

        assert ModelType.TFLITE in QCS6490GPUBackend().supported_formats

    def test_load_tflite_returns_tflite_model(self) -> None:
        """load_tflite returns a QCS6490TfliteModel with GPU unit."""
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._gpu_backend import QCS6490GPUBackend
        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490TfliteModel
        from moment_to_action.hardware._types import ComputeUnit

        mock_interp = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._gpu_backend._load_litert_interpreter",
            return_value=mock_interp,
        ):
            model = QCS6490GPUBackend().load_tflite("/tmp/model.tflite")
        assert isinstance(model, QCS6490TfliteModel)
        assert model.unit == ComputeUnit.GPU

    def test_load_tflite_caches_interpreter(self) -> None:
        """load_tflite with the same path reuses cached interpreter."""
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._gpu_backend import QCS6490GPUBackend

        mock_interp = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._gpu_backend._load_litert_interpreter",
            return_value=mock_interp,
        ) as mock_load:
            backend = QCS6490GPUBackend()
            backend.load_tflite("/tmp/model.tflite")
            backend.load_tflite("/tmp/model.tflite")
        assert mock_load.call_count == 1

    def test_load_litert_interpreter_allocates(self) -> None:
        """_load_litert_interpreter calls Interpreter with delegates."""
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._gpu_backend import (
            _load_litert_interpreter,
        )

        mock_interp = MagicMock()
        mock_delegate = MagicMock()
        with patch("ai_edge_litert.interpreter.Interpreter", return_value=mock_interp) as mock_cls:
            result = _load_litert_interpreter("/tmp/model.tflite", [mock_delegate])
        mock_cls.assert_called_once_with(
            model_path="/tmp/model.tflite", experimental_delegates=[mock_delegate]
        )
        mock_interp.allocate_tensors.assert_called_once()
        assert result is mock_interp


@pytest.mark.unit
class TestQCS6490HTPBackend:
    """Tests for QCS6490HTPBackend (new per-unit architecture)."""

    def _make_backend(self) -> object:
        """Return a QCS6490HTPBackend with mocked delegate loading."""
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._htp_backend import QCS6490HTPBackend

        mock_delegate = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._htp_backend._load_litert_delegate",
            return_value=[mock_delegate],
        ):
            return QCS6490HTPBackend()

    def test_construction_success(self) -> None:
        """QCS6490HTPBackend constructs when delegate loads."""
        assert self._make_backend() is not None

    def test_unit_property(self) -> None:
        """Unit is NPU."""
        from moment_to_action.hardware._types import ComputeUnit

        backend = self._make_backend()
        assert backend.unit == ComputeUnit.NPU  # type: ignore[attr-defined]

    def test_supported_dtypes(self) -> None:
        """supported_dtypes returns {W8A8, W8A16}."""
        from moment_to_action.hardware._types import DataType

        fmts = self._make_backend().supported_dtypes  # type: ignore[attr-defined]
        assert DataType.W8A8 in fmts
        assert DataType.W8A16 in fmts

    def test_supported_formats(self) -> None:
        """supported_formats includes DLC and TFLITE."""
        from moment_to_action.hardware._types import ModelType

        fmts = self._make_backend().supported_formats  # type: ignore[attr-defined]
        assert ModelType.DLC in fmts
        assert ModelType.TFLITE in fmts

    def test_construction_fails_if_delegate_unavailable(self) -> None:
        """Raises RuntimeError if _load_litert_delegate raises."""
        from unittest.mock import patch

        from moment_to_action.hardware._platforms.qcs6490._htp_backend import QCS6490HTPBackend

        with patch(
            "moment_to_action.hardware._platforms.qcs6490._htp_backend._load_litert_delegate",
            side_effect=RuntimeError("delegate not found"),
        ):
            with pytest.raises(RuntimeError):
                QCS6490HTPBackend()

    def test_load_tflite_returns_tflite_model(self) -> None:
        """load_tflite returns a QCS6490TfliteModel with NPU unit."""
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._htp_backend import QCS6490HTPBackend
        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490TfliteModel
        from moment_to_action.hardware._types import ComputeUnit

        mock_delegate = MagicMock()
        mock_interp = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._htp_backend._load_litert_delegate",
            return_value=[mock_delegate],
        ):
            backend = QCS6490HTPBackend()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._htp_backend._load_litert_interpreter",
            return_value=mock_interp,
        ):
            model = backend.load_tflite("/tmp/model.tflite")
        assert isinstance(model, QCS6490TfliteModel)
        assert model.unit == ComputeUnit.NPU

    def test_load_tflite_caches_interpreter(self) -> None:
        """load_tflite with the same path reuses cached interpreter."""
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._htp_backend import QCS6490HTPBackend

        mock_delegate = MagicMock()
        mock_interp = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._htp_backend._load_litert_delegate",
            return_value=[mock_delegate],
        ):
            backend = QCS6490HTPBackend()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._htp_backend._load_litert_interpreter",
            return_value=mock_interp,
        ) as mock_load:
            backend.load_tflite("/tmp/model.tflite")
            backend.load_tflite("/tmp/model.tflite")
        assert mock_load.call_count == 1

    def test_load_dlc_returns_dlc_model(self) -> None:
        """load_dlc returns a QCS6490DLCModel initialized on HTP."""
        import sys
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._htp_backend import QCS6490HTPBackend
        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490DLCModel
        from moment_to_action.hardware._types import ComputeUnit

        mock_delegate = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._htp_backend._load_litert_delegate",
            return_value=[mock_delegate],
        ):
            backend = QCS6490HTPBackend()

        mock_raw = MagicMock()
        mock_qairt = MagicMock()
        mock_qairt.load.return_value = mock_raw
        with patch.dict(sys.modules, {"qairt": mock_qairt}):
            model = backend.load_dlc(Path("/fake/model.dlc"))

        mock_qairt.load.assert_called_once_with("/fake/model.dlc")
        mock_raw.initialize.assert_called_once_with(backend="HTP")
        assert isinstance(model, QCS6490DLCModel)
        assert model.unit == ComputeUnit.NPU

    def test_load_dlc_raises_if_qairt_unavailable(self) -> None:
        """load_dlc raises RuntimeError when qairt cannot be imported."""
        import sys
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._htp_backend import QCS6490HTPBackend

        mock_delegate = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.qcs6490._htp_backend._load_litert_delegate",
            return_value=[mock_delegate],
        ):
            backend = QCS6490HTPBackend()

        with patch.dict(sys.modules, {"qairt": None}):  # type: ignore[dict-item]
            with pytest.raises(RuntimeError, match="QAIRT SDK is not available"):
                backend.load_dlc(Path("/fake/model.dlc"))

    def test_load_litert_delegate_returns_list_on_success(self) -> None:
        """_load_litert_delegate returns [delegate] on success."""
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._htp_backend import _load_litert_delegate

        mock_delegate = MagicMock()
        with patch("ai_edge_litert.interpreter.load_delegate", return_value=mock_delegate):
            result = _load_litert_delegate()
        assert result == [mock_delegate]

    def test_load_litert_delegate_raises_on_failure(self) -> None:
        """_load_litert_delegate raises RuntimeError when load_delegate raises."""
        from unittest.mock import patch

        from moment_to_action.hardware._platforms.qcs6490._htp_backend import _load_litert_delegate

        with patch(
            "ai_edge_litert.interpreter.load_delegate",
            side_effect=RuntimeError("library not found"),
        ):
            with pytest.raises(RuntimeError, match="QNN HTP delegate unavailable"):
                _load_litert_delegate()

    def test_load_litert_interpreter_with_delegates(self) -> None:
        """_load_litert_interpreter passes delegates to Interpreter."""
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._htp_backend import (
            _load_litert_interpreter,
        )

        mock_interp = MagicMock()
        mock_delegate = MagicMock()
        with patch("ai_edge_litert.interpreter.Interpreter", return_value=mock_interp) as mock_cls:
            result = _load_litert_interpreter("/tmp/model.tflite", [mock_delegate])
        mock_cls.assert_called_once_with(
            model_path="/tmp/model.tflite", experimental_delegates=[mock_delegate]
        )
        mock_interp.allocate_tensors.assert_called_once()
        assert result is mock_interp

    def test_load_litert_interpreter_no_delegates(self) -> None:
        """_load_litert_interpreter works with empty delegate list."""
        from unittest.mock import MagicMock, patch

        from moment_to_action.hardware._platforms.qcs6490._htp_backend import (
            _load_litert_interpreter,
        )

        mock_interp = MagicMock()
        with patch("ai_edge_litert.interpreter.Interpreter", return_value=mock_interp):
            result = _load_litert_interpreter("/tmp/model.tflite", [])
        mock_interp.allocate_tensors.assert_called_once()
        assert result is mock_interp


@pytest.mark.unit
class TestQCS6490Models:
    """Tests for QCS6490 model classes (TfliteModel, ONNXModel, DLCModel)."""

    def test_tflite_model_unit_property(self) -> None:
        """QCS6490TfliteModel.unit returns the unit passed in."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490TfliteModel
        from moment_to_action.hardware._types import ComputeUnit

        mock_interp = MagicMock()
        mock_interp.get_input_details.return_value = [
            {"index": 0, "name": "input", "dtype": np.float32}
        ]
        mock_interp.get_output_details.return_value = [{"index": 0}]
        mock_interp.get_tensor.return_value = np.zeros((1, 10))
        model = QCS6490TfliteModel(unit=ComputeUnit.NPU, interp=mock_interp)
        assert model.unit == ComputeUnit.NPU

    def test_tflite_model_dtype_default(self) -> None:
        """QCS6490TfliteModel dtype defaults to FP32."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490TfliteModel
        from moment_to_action.hardware._types import ComputeUnit, DataType

        model = QCS6490TfliteModel(unit=ComputeUnit.CPU, interp=MagicMock())
        assert model.dtype == DataType.FP32

    def test_tflite_model_type_property(self) -> None:
        """QCS6490TfliteModel model_type is TFLITE."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490TfliteModel
        from moment_to_action.hardware._types import ComputeUnit, ModelType

        model = QCS6490TfliteModel(unit=ComputeUnit.CPU, interp=MagicMock())
        assert model.model_type == ModelType.TFLITE

    def test_tflite_model_run_with_ndarray(self) -> None:
        """QCS6490TfliteModel.run() accepts np.ndarray."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490TfliteModel
        from moment_to_action.hardware._types import ComputeUnit

        mock_interp = MagicMock()
        mock_interp.get_input_details.return_value = [
            {"index": 0, "name": "input", "dtype": np.float32}
        ]
        mock_interp.get_output_details.return_value = [{"index": 0}]
        mock_interp.get_tensor.return_value = np.zeros((1, 10))
        model = QCS6490TfliteModel(unit=ComputeUnit.NPU, interp=mock_interp)
        result = model.run(np.zeros((1, 3, 224, 224), dtype=np.float32))
        assert isinstance(result, list)

    def test_tflite_model_run_with_dict(self) -> None:
        """QCS6490TfliteModel.run() accepts dict[str, np.ndarray]."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490TfliteModel
        from moment_to_action.hardware._types import ComputeUnit

        mock_interp = MagicMock()
        mock_interp.get_input_details.return_value = [
            {"index": 0, "name": "input", "dtype": np.float32}
        ]
        mock_interp.get_output_details.return_value = [{"index": 0}]
        mock_interp.get_tensor.return_value = np.zeros((1, 10))
        model = QCS6490TfliteModel(unit=ComputeUnit.NPU, interp=mock_interp)
        result = model.run({"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})
        assert isinstance(result, list)

    def test_tflite_model_unload(self) -> None:
        """QCS6490TfliteModel.unload() clears interpreter."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490TfliteModel
        from moment_to_action.hardware._types import ComputeUnit

        model = QCS6490TfliteModel(unit=ComputeUnit.CPU, interp=MagicMock())
        model.unload()
        assert model._interp is None
        assert model._unloaded is True

    def test_tflite_model_unload_idempotent(self) -> None:
        """QCS6490TfliteModel.unload() can be called twice safely."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490TfliteModel
        from moment_to_action.hardware._types import ComputeUnit

        model = QCS6490TfliteModel(unit=ComputeUnit.CPU, interp=MagicMock())
        model.unload()
        model.unload()

    def test_onnx_model_unit_property(self) -> None:
        """QCS6490ONNXModel.unit is always CPU."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490ONNXModel
        from moment_to_action.hardware._types import ComputeUnit

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.zeros((1, 10))]
        model = QCS6490ONNXModel(session=mock_session)
        assert model.unit == ComputeUnit.CPU

    def test_onnx_model_dtype_default(self) -> None:
        """QCS6490ONNXModel dtype defaults to FP32."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490ONNXModel
        from moment_to_action.hardware._types import DataType

        model = QCS6490ONNXModel(session=MagicMock())
        assert model.dtype == DataType.FP32

    def test_onnx_model_type_property(self) -> None:
        """QCS6490ONNXModel model_type is ONNX."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490ONNXModel
        from moment_to_action.hardware._types import ModelType

        model = QCS6490ONNXModel(session=MagicMock())
        assert model.model_type == ModelType.ONNX

    def test_onnx_model_run_with_ndarray(self) -> None:
        """QCS6490ONNXModel.run() accepts np.ndarray."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490ONNXModel

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.zeros((1, 10))]
        model = QCS6490ONNXModel(session=mock_session)
        result = model.run(np.zeros((1, 3, 224, 224), dtype=np.float32))
        assert isinstance(result, list)

    def test_onnx_model_run_with_dict(self) -> None:
        """QCS6490ONNXModel.run() accepts dict[str, np.ndarray]."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490ONNXModel

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.zeros((1, 10))]
        model = QCS6490ONNXModel(session=mock_session)
        result = model.run({"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})
        assert isinstance(result, list)

    def test_onnx_model_unload(self) -> None:
        """QCS6490ONNXModel.unload() clears session."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490ONNXModel

        model = QCS6490ONNXModel(session=MagicMock())
        model.unload()
        assert model._session is None
        assert model._unloaded is True

    def test_onnx_model_unload_idempotent(self) -> None:
        """QCS6490ONNXModel.unload() can be called twice safely."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490ONNXModel

        model = QCS6490ONNXModel(session=MagicMock())
        model.unload()
        model.unload()

    def test_dlc_model_unit_property(self) -> None:
        """QCS6490DLCModel.unit returns the unit passed in."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490DLCModel
        from moment_to_action.hardware._types import ComputeUnit

        mock_raw = MagicMock()
        mock_result = MagicMock()
        mock_result.data = {"output": np.zeros((1, 10))}
        mock_raw.return_value = mock_result
        model = QCS6490DLCModel(unit=ComputeUnit.NPU, raw=mock_raw)
        assert model.unit == ComputeUnit.NPU

    def test_dlc_model_dtype_default(self) -> None:
        """QCS6490DLCModel dtype defaults to W8A8."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490DLCModel
        from moment_to_action.hardware._types import ComputeUnit, DataType

        model = QCS6490DLCModel(unit=ComputeUnit.NPU, raw=MagicMock())
        assert model.dtype == DataType.W8A8

    def test_dlc_model_type_property(self) -> None:
        """QCS6490DLCModel model_type is DLC."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490DLCModel
        from moment_to_action.hardware._types import ComputeUnit, ModelType

        model = QCS6490DLCModel(unit=ComputeUnit.NPU, raw=MagicMock())
        assert model.model_type == ModelType.DLC

    def test_dlc_model_run_returns_dict(self) -> None:
        """QCS6490DLCModel.run() calls raw(inputs=...) and returns dict."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490DLCModel
        from moment_to_action.hardware._types import ComputeUnit

        mock_raw = MagicMock()
        mock_result = MagicMock()
        mock_result.data = {"output": np.zeros((1, 10))}
        mock_raw.return_value = mock_result
        model = QCS6490DLCModel(unit=ComputeUnit.NPU, raw=mock_raw)
        result = model.run(np.zeros((1, 3, 640, 640), dtype=np.float32))
        assert isinstance(result, dict)

    def test_dlc_model_unload_calls_destroy(self) -> None:
        """QCS6490DLCModel.unload() calls destroy and clears handle."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490DLCModel
        from moment_to_action.hardware._types import ComputeUnit

        model = QCS6490DLCModel(unit=ComputeUnit.NPU, raw=MagicMock())
        raw = model._raw
        model.unload()
        raw.destroy.assert_called_once()
        assert model._raw is None
        assert model._unloaded is True

    def test_dlc_model_unload_idempotent(self) -> None:
        """QCS6490DLCModel.unload() can be called twice safely."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490DLCModel
        from moment_to_action.hardware._types import ComputeUnit

        model = QCS6490DLCModel(unit=ComputeUnit.NPU, raw=MagicMock())
        model.unload()
        model.unload()

    def test_dlc_model_unload_suppress_destroy_exception(self) -> None:
        """QCS6490DLCModel.unload() suppresses destroy exceptions."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import QCS6490DLCModel
        from moment_to_action.hardware._types import ComputeUnit

        mock_raw = MagicMock()
        mock_raw.destroy.side_effect = RuntimeError("device gone")
        model = QCS6490DLCModel(unit=ComputeUnit.NPU, raw=mock_raw)
        model.unload()  # should not raise
        assert model._unloaded is True

    def test_tflite_set_inputs_missing_key(self) -> None:
        """_tflite_set_inputs raises KeyError for unknown name."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import _tflite_set_inputs

        interp = MagicMock()
        interp.get_input_details.return_value = [{"index": 0, "name": "img", "dtype": np.float32}]
        with pytest.raises(KeyError, match="wrong"):
            _tflite_set_inputs(interp, {"wrong": np.zeros((1,), dtype=np.float32)})

    def test_tflite_set_inputs_dtype_mismatch(self) -> None:
        """_tflite_set_inputs raises TypeError for dtype mismatch."""
        from unittest.mock import MagicMock

        from moment_to_action.hardware._platforms.qcs6490._models import _tflite_set_inputs

        interp = MagicMock()
        interp.get_input_details.return_value = [{"index": 0, "name": "img", "dtype": np.float32}]
        with pytest.raises(TypeError, match="dtype mismatch"):
            _tflite_set_inputs(interp, {"img": np.zeros((1,), dtype=np.int32)})
