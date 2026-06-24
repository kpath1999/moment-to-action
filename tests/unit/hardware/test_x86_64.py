"""Unit tests for x86_64 platform backend and resource monitoring."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from moment_to_action.hardware._loaded_models import DlcModel, OnnxModel, TfliteModel, TorchModel
from moment_to_action.hardware._platforms.x86_64 import X86_64CPUBackend, X86_64ResourceMonitor
from moment_to_action.hardware._types import (
    ComputeUnit,
    ComputeUnitUsageSample,
    DataType,
    ModelType,
)


@pytest.mark.unit
class TestX86_64CPUBackend:  # noqa: N801
    """Tests for X86_64CPUBackend properties and load methods."""

    def test_construction(self) -> None:
        """X86_64CPUBackend constructs without error."""
        backend = X86_64CPUBackend()
        assert backend is not None

    def test_unit_property(self) -> None:
        """Unit property returns CPU."""
        assert X86_64CPUBackend().unit == ComputeUnit.CPU

    def test_supported_dtypes(self) -> None:
        """supported_dtypes returns {FP32}."""
        assert X86_64CPUBackend().supported_dtypes == {DataType.FP32}

    def test_supported_formats_includes_tflite_onnx_dlc(self) -> None:
        """supported_formats includes TFLITE, ONNX, and DLC."""
        fmts = X86_64CPUBackend().supported_formats
        assert ModelType.TFLITE in fmts
        assert ModelType.ONNX in fmts
        assert ModelType.DLC in fmts

    def test_supported_formats_includes_torch(self) -> None:
        """supported_formats includes TORCH."""
        assert ModelType.TORCH in X86_64CPUBackend().supported_formats

    def test_supported_formats_includes_llama_cpp(self) -> None:
        """supported_formats includes LLAMA_CPP."""
        assert ModelType.LLAMA_CPP in X86_64CPUBackend().supported_formats

    def test_load_tflite_returns_tflite_model(self) -> None:
        """load_tflite returns a TfliteModel with CPU unit."""
        mock_interp = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.x86_64._cpu_backend._load_litert_interpreter",
            return_value=mock_interp,
        ):
            backend = X86_64CPUBackend()
            model = backend.load_tflite("/tmp/model.tflite")

        assert isinstance(model, TfliteModel)
        assert model.unit == ComputeUnit.CPU

    def test_load_onnx_returns_onnx_model(self) -> None:
        """load_onnx returns an OnnxModel with CPU unit."""
        mock_session = MagicMock()
        with patch(
            "moment_to_action.hardware._platforms.x86_64._cpu_backend.ort.InferenceSession",
            return_value=mock_session,
        ):
            backend = X86_64CPUBackend()
            model = backend.load_onnx("/tmp/model.onnx")

        assert isinstance(model, OnnxModel)
        assert model.unit == ComputeUnit.CPU

    def test_load_torch_returns_torch_model(self) -> None:
        """load_torch returns a TorchModel with CPU unit."""
        mock_module = MagicMock()
        with patch("torch.load", return_value=mock_module):
            model = X86_64CPUBackend().load_torch("/tmp/model.pt")

        assert isinstance(model, TorchModel)
        assert model.unit == ComputeUnit.CPU

    def test_load_llama_cpp_returns_model(self) -> None:
        """load_llama_cpp calls _start_llama_model with cpu_only=True."""
        mock_model = MagicMock()
        with patch(
            "moment_to_action.hardware._loaded_models._llama._start_llama_model",
            return_value=mock_model,
        ) as mock_start:
            result = X86_64CPUBackend().load_llama_cpp(
                "/tmp/model.gguf", server_path="/usr/bin/llama-server", port=8080
            )

        mock_start.assert_called_once_with(
            path="/tmp/model.gguf",
            mmproj=None,
            server_path="/usr/bin/llama-server",
            port=8080,
            unit=ComputeUnit.CPU,
            cpu_only=True,
        )
        assert result is mock_model


@pytest.mark.unit
class TestX86_64DLCMethods:  # noqa: N801
    """Tests for X86_64CPUBackend DLC support via QAIRT CPU backend."""

    def test_load_dlc_returns_dlc_model(self) -> None:
        """load_dlc returns an X86_64DLCModel when qairt is available."""
        import sys
        from pathlib import Path

        mock_raw = MagicMock()
        mock_qairt = MagicMock()
        mock_qairt.load.return_value = mock_raw
        backend = X86_64CPUBackend()
        path = Path("/fake/model.dlc")
        with patch.dict(sys.modules, {"qairt": mock_qairt}):
            model = backend.load_dlc(path)

        mock_qairt.load.assert_called_once_with(str(path))
        mock_raw.initialize.assert_called_once_with(backend="CPU")
        assert isinstance(model, DlcModel)

    def test_load_dlc_raises_if_qairt_unavailable(self) -> None:
        """load_dlc raises RuntimeError when the QAIRT SDK cannot be imported."""
        import sys
        from pathlib import Path

        backend = X86_64CPUBackend()
        with (
            patch.dict(sys.modules, {"qairt": None}),  # type: ignore[dict-item]
            pytest.raises(RuntimeError, match="QAIRT SDK is not available"),
        ):
            backend.load_dlc(Path("/fake/model.dlc"))


@pytest.mark.unit
class TestX86_64ResourceMonitor:  # noqa: N801
    """Test X86_64ResourceMonitor power sampling."""

    def test_x86_64_power_monitor_sample_cpu_rapl_available(self) -> None:
        """Test X86_64ResourceMonitor.sample returns PowerSample for CPU when RAPL available."""
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = True
        mock_rapl_path.read_text.return_value = "1000000\n"
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._resources._RAPL_ENERGY_PATH",
                mock_rapl_path,
            ),
            patch("psutil.cpu_percent", return_value=50.0),
        ):
            monitor = X86_64ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, ComputeUnitUsageSample)
            assert sample.device == ComputeUnit.CPU
            assert sample.power_mw >= 0.0
            assert 0.0 <= sample.usage_pct <= 100.0
            assert isinstance(sample.timestamp, datetime)

    def test_x86_64_power_monitor_sample_cpu_rapl_fallback(self) -> None:
        """Test X86_64ResourceMonitor.sample falls back to estimate when RAPL unavailable."""
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = False
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._resources._RAPL_ENERGY_PATH",
                mock_rapl_path,
            ),
            patch("psutil.cpu_percent", return_value=25.0),
            patch("psutil.cpu_freq") as mock_freq,
        ):
            mock_freq.return_value.current = 2400  # 2.4 GHz
            monitor = X86_64ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, ComputeUnitUsageSample)
            assert sample.device == ComputeUnit.CPU
            assert sample.power_mw >= 0.0
            assert sample.usage_pct == 25.0

    def test_x86_64_power_monitor_non_cpu_unit_returns_zero(self) -> None:
        """Test X86_64ResourceMonitor returns zero power for non-CPU units."""
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = True
        with patch(
            "moment_to_action.hardware._platforms.x86_64._resources._RAPL_ENERGY_PATH",
            mock_rapl_path,
        ):
            monitor = X86_64ResourceMonitor()
            sample = monitor.sample(ComputeUnit.NPU)

            assert sample.device == ComputeUnit.NPU
            assert sample.power_mw == 0.0
            assert sample.usage_pct == 0.0

    def test_x86_64_power_monitor_rapl_read_failure_fallback(self) -> None:
        """Test X86_64ResourceMonitor falls back to estimate on RAPL read failure."""
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = True
        mock_rapl_path.read_text.side_effect = FileNotFoundError()
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._resources._RAPL_ENERGY_PATH",
                mock_rapl_path,
            ),
            patch("psutil.cpu_percent", return_value=30.0),
            patch("psutil.cpu_freq") as mock_freq,
        ):
            mock_freq.return_value.current = 2000
            monitor = X86_64ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            assert isinstance(sample, ComputeUnitUsageSample)
            assert sample.power_mw >= 0.0

    def test_x86_64_power_monitor_multiple_samples(self) -> None:
        """Test X86_64ResourceMonitor returns valid samples over multiple calls."""
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = False
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._resources._RAPL_ENERGY_PATH",
                mock_rapl_path,
            ),
            patch("psutil.cpu_percent", return_value=50.0),
            patch("psutil.cpu_freq") as mock_freq,
        ):
            mock_freq.return_value.current = 2000

            monitor = X86_64ResourceMonitor()
            sample1 = monitor.sample(ComputeUnit.CPU)
            sample2 = monitor.sample(ComputeUnit.CPU)

            assert sample1.power_mw >= 0.0
            assert sample2.power_mw >= 0.0

    def test_x86_64_power_monitor_rapl_delta_power_calculation(self) -> None:
        """Test RAPL delta power calculation (lines 103-107).

        Call sample() twice: first initializes _last_energy_uj and _last_time,
        second computes delta_energy and delta_time, then power_mw.
        """
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = True
        # First call returns 1000000 μJ, second call returns 2000000 μJ
        mock_rapl_path.read_text.side_effect = ["1000000\n", "2000000\n"]
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._resources._RAPL_ENERGY_PATH",
                mock_rapl_path,
            ),
            patch("psutil.cpu_percent", return_value=50.0),
            patch("time.time") as mock_time,
        ):
            # First call at t=0, second call at t=1 (1 second elapsed)
            mock_time.side_effect = [0.0, 1.0, 1.0]

            monitor = X86_64ResourceMonitor()
            # First call: initializes _last_energy_uj and _last_time
            sample1 = monitor.sample(ComputeUnit.CPU)
            assert sample1.power_mw == 0.0  # No previous reading yet

            # Second call: computes delta_energy (1000000 μJ) and delta_time (1.0 s)
            # power_mw = (1000000 / 1000.0) / 1.0 = 1000.0 mW
            sample2 = monitor.sample(ComputeUnit.CPU)
            assert sample2.power_mw == 1000.0

    def test_x86_64_power_monitor_estimate_fallback_on_freq_error(self) -> None:
        """Test _estimate fallback when psutil.cpu_freq() raises (lines 133-134).

        Patch psutil.cpu_freq to raise OSError, verify fallback freq_ghz=2.0
        is used in the power estimate.
        """
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = False
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._resources._RAPL_ENERGY_PATH",
                mock_rapl_path,
            ),
            patch("psutil.cpu_percent", return_value=100.0),
            patch("psutil.cpu_freq", side_effect=OSError("cpu_freq unavailable")),
        ):
            monitor = X86_64ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)

            # With fallback freq_ghz=2.0, util=100%, base=50.0 mW:
            # power_mw = 50.0 + (2.0 * 100.0 * 0.6) = 50.0 + 120.0 = 170.0 mW
            assert sample.power_mw == 170.0
            assert sample.usage_pct == 100.0

    def test_x86_64_rapl_freq_error_fallback(self) -> None:
        """Test that _read_rapl falls back to 2000 MHz when cpu_freq raises.

        Ensures the except branch at lines 119-120 of _power.py is covered.
        """
        mock_rapl_path = MagicMock()
        mock_rapl_path.exists.return_value = True
        mock_rapl_path.read_text.return_value = "5000000\n"
        with (
            patch(
                "moment_to_action.hardware._platforms.x86_64._resources._RAPL_ENERGY_PATH",
                mock_rapl_path,
            ),
            patch("psutil.cpu_percent", return_value=50.0),
            patch("psutil.virtual_memory") as mock_vmem,
            patch("psutil.cpu_freq", side_effect=OSError("unavailable")),
        ):
            mock_vmem.return_value.total = 1024 * 1024 * 4096  # 4 GB total
            mock_vmem.return_value.available = 1024 * 1024 * 3584  # 512 MB used
            monitor = X86_64ResourceMonitor()
            sample = monitor.sample(ComputeUnit.CPU)
            # RAPL path should succeed; frequency_mhz falls back to 2000.0
            assert isinstance(sample, ComputeUnitUsageSample)
            assert sample.frequency_mhz == 2000.0


@pytest.mark.unit
class TestX86_64TfliteModel:  # noqa: N801
    """Tests for TfliteModel on x86_64 (CPU unit)."""

    def _make_model(self) -> TfliteModel:
        """Return a TfliteModel with a mock interpreter."""
        mock_interp = MagicMock()
        mock_interp.get_input_details.return_value = [
            {"index": 0, "name": "input", "dtype": __import__("numpy").float32}
        ]
        mock_interp.get_output_details.return_value = [{"index": 0}]
        mock_interp.get_tensor.return_value = __import__("numpy").zeros((1, 10))
        return TfliteModel(unit=ComputeUnit.CPU, interp=mock_interp)

    def test_unit_property(self) -> None:
        """Unit is always CPU."""
        assert self._make_model().unit == ComputeUnit.CPU

    def test_dtype_property_default(self) -> None:
        """Default dtype is FP32."""
        assert self._make_model().dtype == DataType.FP32

    def test_model_type_property(self) -> None:
        """model_type is TFLITE."""
        assert self._make_model().model_type == ModelType.TFLITE

    def test_run_with_ndarray_input(self) -> None:
        """run() accepts np.ndarray input."""
        import numpy as np

        model = self._make_model()
        result = model.run(np.zeros((1, 3, 224, 224), dtype=np.float32))
        assert isinstance(result, list)

    def test_run_with_dict_input(self) -> None:
        """run() accepts dict[str, np.ndarray] input."""
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
        model.unload()  # should not raise


@pytest.mark.unit
class TestX86_64ONNXModel:  # noqa: N801
    """Tests for OnnxModel on x86_64 (CPU unit)."""

    def _make_model(self) -> OnnxModel:
        """Return an OnnxModel with a mock session."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [__import__("numpy").zeros((1, 10))]
        return OnnxModel(unit=ComputeUnit.CPU, session=mock_session)

    def test_unit_property(self) -> None:
        """Unit is always CPU."""
        assert self._make_model().unit == ComputeUnit.CPU

    def test_dtype_property_default(self) -> None:
        """Default dtype is FP32."""
        assert self._make_model().dtype == DataType.FP32

    def test_model_type_property(self) -> None:
        """model_type is ONNX."""
        assert self._make_model().model_type == ModelType.ONNX

    def test_run_with_ndarray_input(self) -> None:
        """run() with ndarray wraps it by input name."""
        import numpy as np

        model = self._make_model()
        result = model.run(np.zeros((1, 3, 224, 224), dtype=np.float32))
        assert isinstance(result, list)

    def test_run_with_dict_input(self) -> None:
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
        model.unload()  # should not raise


@pytest.mark.unit
class TestX86_64DLCModel:  # noqa: N801
    """Tests for DlcModel on x86_64 (CPU unit)."""

    def _make_model(self) -> DlcModel:
        """Return a DlcModel with a mock raw handle."""
        mock_raw = MagicMock()
        mock_result = MagicMock()
        mock_result.data = {"output": __import__("numpy").zeros((1, 10))}
        mock_raw.return_value = mock_result
        return DlcModel(unit=ComputeUnit.CPU, raw=mock_raw)

    def test_unit_property(self) -> None:
        """Unit is always CPU."""
        assert self._make_model().unit == ComputeUnit.CPU

    def test_dtype_property_default(self) -> None:
        """Default dtype is W8A8 (DLC models are quantized)."""
        assert self._make_model().dtype == DataType.W8A8

    def test_model_type_property(self) -> None:
        """model_type is DLC."""
        assert self._make_model().model_type == ModelType.DLC

    def test_run_calls_raw_and_returns_dict(self) -> None:
        """run() calls raw(inputs=...) and returns dict from result.data."""
        import numpy as np

        model = self._make_model()
        result = model.run(np.zeros((1, 3, 640, 640), dtype=np.float32))
        assert isinstance(result, dict)

    def test_unload_calls_destroy(self) -> None:
        """unload() calls raw.destroy() and clears the handle."""
        model = self._make_model()
        raw = model._raw
        assert raw is not None
        model.unload()
        raw.destroy.assert_called_once()  # type: ignore[union-attr]
        assert model._raw is None
        assert model._unloaded is True

    def test_unload_idempotent(self) -> None:
        """Calling unload() twice is safe."""
        model = self._make_model()
        model.unload()
        model.unload()  # should not raise

    def test_unload_suppress_destroy_exception(self) -> None:
        """unload() suppresses exceptions from destroy()."""
        model = self._make_model()
        raw = model._raw
        assert raw is not None
        raw.destroy.side_effect = RuntimeError("device gone")  # type: ignore[union-attr]
        model.unload()  # should not raise
        assert model._unloaded is True

    def test_run_raises_after_unload(self) -> None:
        """run() raises RuntimeError after unload() has been called."""
        model = self._make_model()
        model.unload()
        with pytest.raises(RuntimeError, match="unloaded"):
            model.run({})


@pytest.mark.unit
class TestX86_64TfliteSetInputs:  # noqa: N801
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
class TestX86_64LoadLiteRTInterpreter:  # noqa: N801
    """Tests for _load_litert_interpreter in x86_64._cpu_backend."""

    def test_loads_and_allocates_interpreter(self) -> None:
        """_load_litert_interpreter calls Interpreter and allocate_tensors."""
        from moment_to_action.hardware._platforms.x86_64._cpu_backend import (
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
