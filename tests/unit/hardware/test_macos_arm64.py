"""Unit tests for the macOS arm64 platform backend and power monitor."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from moment_to_action.hardware._loaded_models import OnnxModel, TfliteModel, TorchModel
from moment_to_action.hardware._platforms.macos_arm64 import (
    MacOSARM64CPUBackend,
    MacOSARM64ResourceMonitor,
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
        """load_tflite returns a TfliteModel with CPU unit."""
        mock_interp = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.macos_arm64._cpu_backend._load_litert_interpreter",
            return_value=mock_interp,
        ):
            model = MacOSARM64CPUBackend().load_tflite("/tmp/model.tflite", dtype=DataType.FP32)

        assert isinstance(model, TfliteModel)
        assert model.unit == ComputeUnit.CPU

    def test_load_onnx_returns_onnx_model(self) -> None:
        """load_onnx returns an OnnxModel with CPU unit."""
        mock_session = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.macos_arm64._cpu_backend.ort.InferenceSession",
            return_value=mock_session,
        ):
            model = MacOSARM64CPUBackend().load_onnx("/tmp/model.onnx", dtype=DataType.FP32)

        assert isinstance(model, OnnxModel)
        assert model.unit == ComputeUnit.CPU

    def test_load_dlc_raises_not_implemented(self) -> None:
        """load_dlc raises NotImplementedError (not in supported_formats)."""
        with pytest.raises(NotImplementedError):
            MacOSARM64CPUBackend().load_dlc("/tmp/model.dlc", dtype=DataType.W8A8)

    def test_load_torch_returns_torch_model(self) -> None:
        """load_torch returns a TorchModel with CPU unit."""
        mock_module = MagicMock()
        with patch("torch.load", return_value=mock_module):
            model = MacOSARM64CPUBackend().load_torch("/tmp/model.pt", dtype=DataType.FP32)

        assert isinstance(model, TorchModel)
        assert model.unit == ComputeUnit.CPU

    def test_load_llama_cpp_returns_model(self) -> None:
        """load_llama_cpp calls _start_llama_model with cpu_only=True."""
        mock_model = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.macos_arm64._cpu_backend._start_llama_model",
            return_value=mock_model,
        ) as mock_start:
            result = MacOSARM64CPUBackend().load_llama_cpp(
                "/tmp/model.gguf",
                server_path="/usr/bin/llama-server",
                port=8080,
                dtype=DataType.FP32,
            )

        mock_start.assert_called_once_with(
            path="/tmp/model.gguf",
            mmproj=None,
            server_path="/usr/bin/llama-server",
            port=8080,
            unit=ComputeUnit.CPU,
            cpu_only=True,
            dtype=DataType.FP32,
        )
        assert result is mock_model


@pytest.mark.unit
class TestMacOSARM64DtypeValidation:
    """Tests that MacOSARM64CPUBackend rejects unsupported dtypes via _check_dtype."""

    def test_load_tflite_unsupported_dtype_raises(self) -> None:
        """load_tflite raises ValueError for a dtype not in supported_dtypes."""
        backend = MacOSARM64CPUBackend()
        with pytest.raises(ValueError, match="does not support dtype"):
            backend.load_tflite("/fake/model.tflite", dtype=DataType.W8A8)

    def test_load_onnx_unsupported_dtype_raises(self) -> None:
        """load_onnx raises ValueError for a dtype not in supported_dtypes."""
        backend = MacOSARM64CPUBackend()
        with pytest.raises(ValueError, match="does not support dtype"):
            backend.load_onnx("/fake/model.onnx", dtype=DataType.W8A8)


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


@pytest.mark.unit
class TestMacOSARM64TfliteModel:
    """Tests for TfliteModel on macOS arm64 (CPU unit)."""

    def _make_model(self) -> TfliteModel:
        """Return a TfliteModel with a mock interpreter."""
        import numpy as np

        mock_interp = MagicMock()
        mock_interp.get_input_details.return_value = [
            {"index": 0, "name": "input", "dtype": np.float32}
        ]
        mock_interp.get_output_details.return_value = [{"index": 0}]
        mock_interp.get_tensor.return_value = np.zeros((1, 10))
        return TfliteModel(unit=ComputeUnit.CPU, interp=mock_interp, dtype=DataType.FP32)

    def test_unit_property(self) -> None:
        """Unit is always CPU."""
        assert self._make_model().unit == ComputeUnit.CPU

    def test_dtype_property(self) -> None:
        """Dtype returns the value passed at construction."""
        assert self._make_model().dtype == DataType.FP32

    def test_model_type_property(self) -> None:
        """model_type is TFLITE."""
        assert self._make_model().model_type == ModelType.TFLITE

    def test_run_with_ndarray(self) -> None:
        """run() accepts np.ndarray."""
        import numpy as np

        model = self._make_model()
        result = model.run(np.zeros((1, 3, 224, 224), dtype=np.float32))
        assert isinstance(result, list)

    def test_run_with_dict(self) -> None:
        """run() accepts dict[str, np.ndarray]."""
        import numpy as np

        model = self._make_model()
        result = model.run({"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})
        assert isinstance(result, list)

    def test_unload_clears_interp(self) -> None:
        """unload() clears the interpreter handle."""
        model = self._make_model()
        model.unload()
        assert model._interp is None
        assert model._unloaded is True

    def test_unload_idempotent(self) -> None:
        """Calling unload() twice is safe."""
        model = self._make_model()
        model.unload()
        model.unload()


@pytest.mark.unit
class TestMacOSARM64ONNXModel:
    """Tests for OnnxModel on macOS arm64 (CPU unit)."""

    def _make_model(self) -> OnnxModel:
        """Return an OnnxModel with a mock session."""
        import numpy as np

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.zeros((1, 10))]
        return OnnxModel(unit=ComputeUnit.CPU, session=mock_session, dtype=DataType.FP32)

    def test_unit_property(self) -> None:
        """Unit is always CPU."""
        assert self._make_model().unit == ComputeUnit.CPU

    def test_dtype_property(self) -> None:
        """Dtype returns the value passed at construction."""
        assert self._make_model().dtype == DataType.FP32

    def test_model_type_property(self) -> None:
        """model_type is ONNX."""
        assert self._make_model().model_type == ModelType.ONNX

    def test_run_with_ndarray(self) -> None:
        """run() with ndarray passes it by first input name."""
        import numpy as np

        model = self._make_model()
        result = model.run(np.zeros((1, 3, 224, 224), dtype=np.float32))
        assert isinstance(result, list)

    def test_run_with_dict(self) -> None:
        """run() with dict passes it directly."""
        import numpy as np

        model = self._make_model()
        result = model.run({"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})
        assert isinstance(result, list)

    def test_unload_clears_session(self) -> None:
        """unload() clears the session handle."""
        model = self._make_model()
        model.unload()
        assert model._session is None
        assert model._unloaded is True

    def test_unload_idempotent(self) -> None:
        """Calling unload() twice is safe."""
        model = self._make_model()
        model.unload()
        model.unload()


@pytest.mark.unit
class TestMacOSARM64TfliteSetInputs:
    """Tests for _tflite_set_inputs from hardware._platforms._shared (KeyError/TypeError paths)."""

    def test_missing_key_raises_key_error(self) -> None:
        """KeyError when input name not found in model."""
        import numpy as np

        from moment_to_action.hardware._platforms._shared import _tflite_set_inputs

        interp = MagicMock()
        interp.get_input_details.return_value = [{"index": 0, "name": "img", "dtype": np.float32}]
        with pytest.raises(KeyError, match="wrong"):
            _tflite_set_inputs(interp, {"wrong": np.zeros((1,), dtype=np.float32)})

    def test_dtype_mismatch_raises_type_error(self) -> None:
        """TypeError when tensor dtype does not match model's expected dtype."""
        import numpy as np

        from moment_to_action.hardware._platforms._shared import _tflite_set_inputs

        interp = MagicMock()
        interp.get_input_details.return_value = [{"index": 0, "name": "img", "dtype": np.float32}]
        with pytest.raises(TypeError, match="dtype mismatch"):
            _tflite_set_inputs(interp, {"img": np.zeros((1,), dtype=np.int32)})


@pytest.mark.unit
class TestMacOSARM64LoadLiteRTInterpreter:
    """Tests for _load_litert_interpreter in macos_arm64._cpu_backend."""

    def test_loads_and_allocates_interpreter(self) -> None:
        """_load_litert_interpreter calls Interpreter and allocate_tensors."""
        from moment_to_action.hardware._platforms.macos_arm64._cpu_backend import (
            _load_litert_interpreter,
        )

        mock_interp = MagicMock()
        with patch(
            "ai_edge_litert.interpreter.Interpreter",
            return_value=mock_interp,
        ) as mock_cls:
            result = _load_litert_interpreter("/tmp/model.tflite")

        mock_cls.assert_called_once_with(model_path="/tmp/model.tflite", experimental_delegates=[])
        mock_interp.allocate_tensors.assert_called_once()
        assert result is mock_interp
