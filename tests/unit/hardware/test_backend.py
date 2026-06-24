"""Unit tests for Platform hardware entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moment_to_action.config import AppConfig
from moment_to_action.hardware import BenchmarkResult, Platform
from moment_to_action.hardware._backend import ComputeBackend
from moment_to_action.hardware._loaded_model import LoadedModel
from moment_to_action.hardware._platform import _detect_platform
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType, PlatformType


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
            return Platform(AppConfig())

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
                Platform(AppConfig())
                mock_init.assert_called_once()

    def test_init_qcs6490_called_on_qcs6490(self) -> None:
        """Platform calls _init_qcs6490 when _detect_platform returns QCS6490."""
        with patch("moment_to_action.hardware._platform._detect_platform") as mock_detect:
            mock_detect.return_value = PlatformType.QCS6490
            with patch.object(
                Platform, "_init_qcs6490", autospec=True, side_effect=self._stub_init
            ) as mock_init:
                Platform(AppConfig())
                mock_init.assert_called_once()

    def test_init_macos_arm64_called_on_macos(self) -> None:
        """Platform calls _init_macos_arm64 when _detect_platform returns MACOS_ARM64."""
        with patch("moment_to_action.hardware._platform._detect_platform") as mock_detect:
            mock_detect.return_value = PlatformType.MACOS_ARM64
            with patch.object(
                Platform, "_init_macos_arm64", autospec=True, side_effect=self._stub_init
            ) as mock_init:
                Platform(AppConfig())
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
            p = Platform(AppConfig())
        return p, mock_cpu

    def test_load_tflite_delegates(self) -> None:
        """load_tflite calls backend.load_tflite with the path and dtype."""
        p, mock_cpu = self._make_platform_with_mock_cpu()
        path = Path("/fake/model.tflite")
        mock_cpu.load_tflite.return_value = MagicMock()

        result = p.load_tflite(ComputeUnit.CPU, path, dtype=DataType.FP32)

        mock_cpu.load_tflite.assert_called_once_with(path, dtype=DataType.FP32)
        assert result is mock_cpu.load_tflite.return_value

    def test_load_onnx_delegates(self) -> None:
        """load_onnx calls backend.load_onnx with the path and dtype."""
        p, mock_cpu = self._make_platform_with_mock_cpu()
        path = Path("/fake/model.onnx")
        mock_cpu.load_onnx.return_value = MagicMock()

        result = p.load_onnx(ComputeUnit.CPU, path, dtype=DataType.FP32)

        mock_cpu.load_onnx.assert_called_once_with(path, dtype=DataType.FP32)
        assert result is mock_cpu.load_onnx.return_value

    def test_load_dlc_delegates(self) -> None:
        """load_dlc calls backend.load_dlc with the path and dtype."""
        p, mock_cpu = self._make_platform_with_mock_cpu()
        path = Path("/fake/model.dlc")
        mock_cpu.load_dlc.return_value = MagicMock()

        result = p.load_dlc(ComputeUnit.CPU, path, dtype=DataType.W8A8)

        mock_cpu.load_dlc.assert_called_once_with(path, dtype=DataType.W8A8)
        assert result is mock_cpu.load_dlc.return_value

    def test_load_torch_delegates(self) -> None:
        """load_torch calls backend.load_torch with the path and dtype."""
        p, mock_cpu = self._make_platform_with_mock_cpu()
        path = Path("/fake/model.pt")
        mock_cpu.load_torch.return_value = MagicMock()

        result = p.load_torch(ComputeUnit.CPU, path, dtype=DataType.FP32)

        mock_cpu.load_torch.assert_called_once_with(path, dtype=DataType.FP32)
        assert result is mock_cpu.load_torch.return_value

    def test_load_llama_cpp_delegates(self) -> None:
        """load_llama_cpp calls backend.load_llama_cpp with path, mmproj, and dtype."""
        p, mock_cpu = self._make_platform_with_mock_cpu()
        path = Path("/fake/model.gguf")
        mmproj = Path("/fake/mmproj.gguf")
        mock_cpu.load_llama_cpp.return_value = MagicMock()

        result = p.load_llama_cpp(ComputeUnit.CPU, path, mmproj=mmproj, dtype=DataType.FP32)

        mock_cpu.load_llama_cpp.assert_called_once_with(
            path,
            mmproj=mmproj,
            server_path=AppConfig().llama_server_path,
            port=AppConfig().llama_server_port,
            dtype=DataType.FP32,
        )
        assert result is mock_cpu.load_llama_cpp.return_value

    def test_unknown_unit_raises_value_error(self) -> None:
        """Requesting a unit with no registered backend raises ValueError."""
        p, _ = self._make_platform_with_mock_cpu()

        with pytest.raises(ValueError, match="NPU"):
            p.load_tflite(ComputeUnit.NPU, Path("/fake/model.tflite"), dtype=DataType.FP32)


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
            return Platform(AppConfig())

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


@pytest.mark.unit
class TestComputeBackendDefaults:
    """Tests for ComputeBackend default load_onnx/load_dlc/load_torch/load_tflite/load_llama_cpp."""

    def _make_backend(self) -> ComputeBackend:
        """Return a minimal concrete ComputeBackend with no supported formats."""

        class _Minimal(ComputeBackend):
            @property
            def unit(self) -> ComputeUnit:
                """Return CPU."""
                return ComputeUnit.CPU

            @property
            def supported_dtypes(self) -> set[DataType]:
                """Return empty set."""
                return set()

            @property
            def supported_formats(self) -> set[ModelType]:
                """Return empty set."""
                return set()

        return _Minimal()

    def test_load_onnx_raises_not_implemented(self) -> None:
        """load_onnx raises NotImplementedError when ONNX not in supported_formats."""
        b = self._make_backend()
        with pytest.raises(NotImplementedError, match="ONNX"):
            b.load_onnx("/tmp/model.onnx", dtype=DataType.FP32)

    def test_load_dlc_raises_not_implemented(self) -> None:
        """load_dlc raises NotImplementedError when DLC not in supported_formats."""
        b = self._make_backend()
        with pytest.raises(NotImplementedError, match="DLC"):
            b.load_dlc("/tmp/model.dlc", dtype=DataType.W8A8)

    def test_load_torch_raises_not_implemented(self) -> None:
        """load_torch raises NotImplementedError when TORCH not in supported_formats."""
        b = self._make_backend()
        with pytest.raises(NotImplementedError, match="TORCH"):
            b.load_torch("/tmp/model.pt", dtype=DataType.FP32)

    def test_load_tflite_raises_not_implemented(self) -> None:
        """load_tflite raises NotImplementedError when TFLITE not in supported_formats."""
        b = self._make_backend()
        with pytest.raises(NotImplementedError, match="TFLITE"):
            b.load_tflite("/tmp/model.tflite", dtype=DataType.FP32)

    def test_load_llama_cpp_raises_not_implemented(self) -> None:
        """load_llama_cpp raises NotImplementedError when LLAMA_CPP not in supported_formats."""
        b = self._make_backend()
        with pytest.raises(NotImplementedError, match="LLAMA_CPP"):
            b.load_llama_cpp("/tmp/model.gguf", dtype=DataType.FP32)

    def test_raise_unsupported_message_includes_supported_formats(self) -> None:
        """_raise_unsupported error message lists supported formats."""

        class _WithOnnx(ComputeBackend):
            @property
            def unit(self) -> ComputeUnit:
                """Return CPU."""
                return ComputeUnit.CPU

            @property
            def supported_dtypes(self) -> set[DataType]:
                """Return empty."""
                return set()

            @property
            def supported_formats(self) -> set[ModelType]:
                """Return ONNX only."""
                return {ModelType.ONNX}

        b = _WithOnnx()
        with pytest.raises(NotImplementedError, match="ONNX"):
            b.load_dlc("/tmp/model.dlc", dtype=DataType.W8A8)

    def test_check_dtype_raises_for_unsupported_dtype(self) -> None:
        """_check_dtype raises ValueError for dtype not in supported_dtypes."""
        b = self._make_backend()
        with pytest.raises(ValueError, match="does not support dtype"):
            b._check_dtype(DataType.FP32)

    def test_check_dtype_passes_for_supported_dtype(self) -> None:
        """_check_dtype does not raise when dtype is in supported_dtypes."""

        class _WithFP32(ComputeBackend):
            @property
            def unit(self) -> ComputeUnit:
                """Return CPU."""
                return ComputeUnit.CPU

            @property
            def supported_dtypes(self) -> set[DataType]:
                """Return FP32 only."""
                return {DataType.FP32}

            @property
            def supported_formats(self) -> set[ModelType]:
                """Return empty."""
                return set()

        b = _WithFP32()
        b._check_dtype(DataType.FP32)  # should not raise

    def test_check_dtype_error_message_includes_backend_name(self) -> None:
        """_check_dtype error message names the backend and the dtype."""
        b = self._make_backend()
        with pytest.raises(ValueError, match="_Minimal"):
            b._check_dtype(DataType.W8A8)

    def test_check_dtype_error_message_says_none_when_no_dtypes(self) -> None:
        """_check_dtype error message says 'none' when supported_dtypes is empty."""
        b = self._make_backend()
        with pytest.raises(ValueError, match="none"):
            b._check_dtype(DataType.FP32)


@pytest.mark.unit
class TestLoadedModelContextManager:
    """Tests for LoadedModel.__enter__ and __exit__."""

    def _make_model(self) -> LoadedModel:
        """Return a minimal concrete LoadedModel."""

        class _ConcreteModel(LoadedModel):
            def __init__(self) -> None:
                """Initialize."""
                self._unloaded = False

            @property
            def unit(self) -> ComputeUnit:
                """Return CPU."""
                return ComputeUnit.CPU

            @property
            def dtype(self) -> DataType:
                """Return FP32."""
                return DataType.FP32

            @property
            def model_type(self) -> ModelType:
                """Return TFLITE."""
                return ModelType.TFLITE

            def run(self, inputs: object) -> object:
                """Run inference."""
                return None

            def unload(self) -> None:
                """Unload model."""
                self._unloaded = True

        return _ConcreteModel()

    def test_enter_returns_self(self) -> None:
        """__enter__ returns the model itself."""
        model = self._make_model()
        result = model.__enter__()
        assert result is model

    def test_exit_calls_unload(self) -> None:
        """__exit__ calls unload()."""
        model = self._make_model()
        model.__exit__(None, None, None)
        assert model._unloaded is True  # type: ignore[attr-defined]

    def test_context_manager_protocol(self) -> None:
        """Using as context manager calls unload on exit."""
        model = self._make_model()
        with model as m:
            assert m is model
        assert model._unloaded is True  # type: ignore[attr-defined]

    def test_del_calls_unload(self) -> None:
        """__del__ calls unload as GC safety net."""
        model = self._make_model()
        model.__del__()
        assert model._unloaded is True  # type: ignore[attr-defined]


@pytest.mark.unit
class TestDetectPlatform:
    """Tests for _detect_platform() detection logic."""

    def test_detects_qcs6490_from_soc_file(self) -> None:
        """_detect_platform returns QCS6490 when soc file contains QCS6490."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "Qualcomm QCS6490\n"
        with (
            patch("moment_to_action.hardware._platform._QCOM_SOC_NAME_FILE", mock_path),
            patch("moment_to_action.hardware._platform._detect_platform.cache_clear"),
        ):
            _detect_platform.cache_clear()
            result = _detect_platform()
        assert result == PlatformType.QCS6490

    def test_detects_x86_64_via_machine(self) -> None:
        """_detect_platform returns X86_64 when machine() is x86_64."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with (
            patch("moment_to_action.hardware._platform._QCOM_SOC_NAME_FILE", mock_path),
            patch("moment_to_action.hardware._platform.platform.machine", return_value="x86_64"),
            patch("moment_to_action.hardware._platform.platform.system", return_value="linux"),
        ):
            _detect_platform.cache_clear()
            result = _detect_platform()
        assert result == PlatformType.X86_64

    def test_detects_x86_64_via_amd64(self) -> None:
        """_detect_platform returns X86_64 when machine() is amd64."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with (
            patch("moment_to_action.hardware._platform._QCOM_SOC_NAME_FILE", mock_path),
            patch("moment_to_action.hardware._platform.platform.machine", return_value="amd64"),
            patch("moment_to_action.hardware._platform.platform.system", return_value="linux"),
        ):
            _detect_platform.cache_clear()
            result = _detect_platform()
        assert result == PlatformType.X86_64

    def test_detects_macos_arm64(self) -> None:
        """_detect_platform returns MACOS_ARM64 when arm64 + darwin."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with (
            patch("moment_to_action.hardware._platform._QCOM_SOC_NAME_FILE", mock_path),
            patch("moment_to_action.hardware._platform.platform.machine", return_value="arm64"),
            patch("moment_to_action.hardware._platform.platform.system", return_value="darwin"),
        ):
            _detect_platform.cache_clear()
            result = _detect_platform()
        assert result == PlatformType.MACOS_ARM64


@pytest.mark.unit
class TestPlatformSupports:
    """Tests for Platform.supports()."""

    def _make_platform_with_backends(
        self,
        cpu_formats: set[ModelType],
        cpu_dtypes: set[DataType],
    ) -> Platform:
        """Build a Platform with a single CPU backend advertising specific formats and dtypes.

        Args:
            cpu_formats: Formats the mock CPU backend claims to support.
            cpu_dtypes: Data types the mock CPU backend claims to support.

        Returns:
            Constructed Platform.
        """
        with (
            patch("moment_to_action.hardware._platform._detect_platform") as mock_detect,
            patch.object(Platform, "_init_x86_64", autospec=True) as mock_init,
        ):
            mock_detect.return_value = PlatformType.X86_64
            mock_cpu = MagicMock()
            mock_cpu.supported_formats = cpu_formats
            mock_cpu.supported_dtypes = cpu_dtypes

            def _setup(self: Platform) -> None:
                self._resource_monitor = MagicMock()
                self._backends = {ComputeUnit.CPU: mock_cpu}

            mock_init.side_effect = _setup
            return Platform(AppConfig())

    def test_returns_true_when_format_supported(self) -> None:
        """Returns True when backend supports the requested model_type."""
        p = self._make_platform_with_backends(
            cpu_formats={ModelType.ONNX},
            cpu_dtypes={DataType.FP32},
        )
        assert p.supports(ComputeUnit.CPU, model_type=ModelType.ONNX) is True

    def test_returns_false_when_format_not_supported(self) -> None:
        """Returns False when backend does not support the requested model_type."""
        p = self._make_platform_with_backends(
            cpu_formats={ModelType.ONNX},
            cpu_dtypes={DataType.FP32},
        )
        assert p.supports(ComputeUnit.CPU, model_type=ModelType.DLC) is False

    def test_returns_false_for_unknown_unit(self) -> None:
        """Returns False when no backend is registered for the unit."""
        p = self._make_platform_with_backends(
            cpu_formats={ModelType.ONNX},
            cpu_dtypes={DataType.FP32},
        )
        assert p.supports(ComputeUnit.NPU, model_type=ModelType.DLC) is False

    def test_returns_true_when_format_and_dtype_both_supported(self) -> None:
        """Returns True when both model_type and data_type are supported."""
        p = self._make_platform_with_backends(
            cpu_formats={ModelType.ONNX},
            cpu_dtypes={DataType.FP32},
        )
        assert (
            p.supports(ComputeUnit.CPU, model_type=ModelType.ONNX, data_type=DataType.FP32) is True
        )

    def test_returns_false_when_dtype_not_supported(self) -> None:
        """Returns False when model_type is supported but data_type is not."""
        p = self._make_platform_with_backends(
            cpu_formats={ModelType.DLC},
            cpu_dtypes={DataType.W8A8},
        )
        assert (
            p.supports(ComputeUnit.CPU, model_type=ModelType.DLC, data_type=DataType.W8A16) is False
        )

    def test_data_type_none_skips_dtype_check(self) -> None:
        """Passing data_type=None (default) does not check dtype."""
        p = self._make_platform_with_backends(
            cpu_formats={ModelType.ONNX},
            cpu_dtypes=set(),
        )
        assert p.supports(ComputeUnit.CPU, model_type=ModelType.ONNX) is True

    def test_detects_macos_arm64_via_aarch64(self) -> None:
        """_detect_platform returns MACOS_ARM64 when aarch64 + darwin."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with (
            patch("moment_to_action.hardware._platform._QCOM_SOC_NAME_FILE", mock_path),
            patch("moment_to_action.hardware._platform.platform.machine", return_value="aarch64"),
            patch("moment_to_action.hardware._platform.platform.system", return_value="darwin"),
        ):
            _detect_platform.cache_clear()
            result = _detect_platform()
        assert result == PlatformType.MACOS_ARM64

    def test_raises_on_unknown_platform(self) -> None:
        """_detect_platform raises RuntimeError for unrecognised platform."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with (
            patch("moment_to_action.hardware._platform._QCOM_SOC_NAME_FILE", mock_path),
            patch("moment_to_action.hardware._platform.platform.machine", return_value="mips"),
            patch("moment_to_action.hardware._platform.platform.system", return_value="linux"),
        ):
            _detect_platform.cache_clear()
            with pytest.raises(RuntimeError, match="Unrecognised platform"):
                _detect_platform()

    def test_soc_file_non_qcs6490_falls_through_to_machine(self) -> None:
        """_detect_platform reads soc file but falls through when not QCS6490."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "SDM845\n"
        with (
            patch("moment_to_action.hardware._platform._QCOM_SOC_NAME_FILE", mock_path),
            patch("moment_to_action.hardware._platform.platform.machine", return_value="x86_64"),
            patch("moment_to_action.hardware._platform.platform.system", return_value="linux"),
        ):
            _detect_platform.cache_clear()
            result = _detect_platform()
        assert result == PlatformType.X86_64


@pytest.mark.unit
class TestPlatformInitMethods:
    """Tests for Platform._init_qcs6490/_init_x86_64/_init_macos_arm64 and HTP/GPU fallbacks."""

    @staticmethod
    def _make_platform_of_type(
        platform_type: PlatformType,
    ) -> Platform:
        """Build a Platform for the given type with mocked init."""
        with patch("moment_to_action.hardware._platform._detect_platform") as mock_detect:
            mock_detect.return_value = platform_type

            if platform_type == PlatformType.X86_64:
                mock_cpu = MagicMock()
                mock_monitor = MagicMock()
                with (
                    patch(
                        "moment_to_action.hardware._platform.Platform._init_x86_64",
                        autospec=True,
                    ) as mock_init,
                ):

                    def _setup(self: Platform) -> None:
                        self._resource_monitor = mock_monitor
                        self._backends = {ComputeUnit.CPU: mock_cpu}

                    mock_init.side_effect = _setup
                    return Platform(AppConfig())

            if platform_type == PlatformType.MACOS_ARM64:
                mock_cpu = MagicMock()
                mock_monitor = MagicMock()
                with (
                    patch(
                        "moment_to_action.hardware._platform.Platform._init_macos_arm64",
                        autospec=True,
                    ) as mock_init,
                ):

                    def _setup_mac(self: Platform) -> None:
                        self._resource_monitor = mock_monitor
                        self._backends = {ComputeUnit.CPU: mock_cpu}

                    mock_init.side_effect = _setup_mac
                    return Platform(AppConfig())

            msg = f"Unsupported type: {platform_type}"
            raise ValueError(msg)

    def test_init_x86_64_creates_cpu_backend(self) -> None:
        """_init_x86_64 registers a CPU backend."""
        mock_cpu = MagicMock()
        mock_monitor = MagicMock()
        with (
            patch("moment_to_action.hardware._platform._detect_platform") as mock_detect,
            patch(
                "moment_to_action.hardware._platforms.x86_64._cpu_backend.X86_64CPUBackend",
                return_value=mock_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.x86_64._resources.X86_64ResourceMonitor",
                return_value=mock_monitor,
            ),
        ):
            mock_detect.return_value = PlatformType.X86_64
            p = Platform(AppConfig())
        assert ComputeUnit.CPU in p.supported_units

    def test_init_macos_arm64_creates_cpu_backend(self) -> None:
        """_init_macos_arm64 registers a CPU backend."""
        mock_cpu = MagicMock()
        mock_monitor = MagicMock()
        with (
            patch("moment_to_action.hardware._platform._detect_platform") as mock_detect,
            patch(
                "moment_to_action.hardware._platforms.macos_arm64._cpu_backend.MacOSARM64CPUBackend",
                return_value=mock_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.macos_arm64._resources.MacOSARM64ResourceMonitor",
                return_value=mock_monitor,
            ),
        ):
            mock_detect.return_value = PlatformType.MACOS_ARM64
            p = Platform(AppConfig())
        assert ComputeUnit.CPU in p.supported_units

    def test_init_qcs6490_cpu_always_registered(self) -> None:
        """_init_qcs6490 always registers CPU; HTP and GPU are optional."""
        mock_cpu = MagicMock()
        mock_monitor = MagicMock()
        with (
            patch("moment_to_action.hardware._platform._detect_platform") as mock_detect,
            patch(
                "moment_to_action.hardware._platforms.qcs6490._cpu_backend.QCS6490CPUBackend",
                return_value=mock_cpu,
            ),
            patch(
                "moment_to_action.hardware._platforms.qcs6490._resources.QCS6490ResourceMonitor",
                return_value=mock_monitor,
            ),
            patch.object(Platform, "_try_add_htp_backend"),
            patch.object(Platform, "_try_add_qcs6490_gpu_backend"),
        ):
            mock_detect.return_value = PlatformType.QCS6490
            p = Platform(AppConfig())
        assert ComputeUnit.CPU in p.supported_units

    def test_try_add_htp_backend_registers_npu_when_available(self) -> None:
        """_try_add_htp_backend registers NPU backend when HTP imports succeed."""
        mock_htp = MagicMock()
        with (
            patch("moment_to_action.hardware._platform._detect_platform") as mock_detect,
            patch(
                "moment_to_action.hardware._platform.Platform._init_qcs6490",
                autospec=True,
            ) as mock_init,
        ):
            mock_detect.return_value = PlatformType.QCS6490

            def _setup(self: Platform) -> None:
                self._resource_monitor = MagicMock()
                self._backends = {ComputeUnit.CPU: MagicMock()}

            mock_init.side_effect = _setup
            p = Platform(AppConfig())

        # Now test _try_add_htp_backend directly
        with patch(
            "moment_to_action.hardware._platform.QCS6490HTPBackend",
            return_value=mock_htp,
            create=True,
        ):
            # Import and patch the module-level symbol used inside _try_add_htp_backend
            with patch(
                "moment_to_action.hardware._platforms.qcs6490._htp_backend.QCS6490HTPBackend",
                return_value=mock_htp,
            ):
                p._try_add_htp_backend()

    def test_try_add_htp_backend_logs_warning_when_unavailable(self) -> None:
        """_try_add_htp_backend logs warning when HTP backend raises."""
        with (
            patch("moment_to_action.hardware._platform._detect_platform") as mock_detect,
            patch(
                "moment_to_action.hardware._platform.Platform._init_qcs6490",
                autospec=True,
            ) as mock_init,
        ):
            mock_detect.return_value = PlatformType.QCS6490

            def _setup(self: Platform) -> None:
                self._resource_monitor = MagicMock()
                self._backends = {ComputeUnit.CPU: MagicMock()}

            mock_init.side_effect = _setup
            p = Platform(AppConfig())

        # Patch the import inside _try_add_htp_backend to raise
        import sys

        saved = sys.modules.get("moment_to_action.hardware._platforms.qcs6490._htp_backend")
        try:
            mock_mod = MagicMock()
            mock_mod.QCS6490HTPBackend.side_effect = RuntimeError("delegate missing")
            sys.modules["moment_to_action.hardware._platforms.qcs6490._htp_backend"] = mock_mod
            p._try_add_htp_backend()
            # NPU should NOT be in backends since import raised
            assert ComputeUnit.NPU not in p._backends  # type: ignore[attr-defined]
        finally:
            if saved is None:
                del sys.modules["moment_to_action.hardware._platforms.qcs6490._htp_backend"]
            else:
                sys.modules["moment_to_action.hardware._platforms.qcs6490._htp_backend"] = saved

    def test_try_add_qcs6490_gpu_backend_logs_warning_when_unavailable(self) -> None:
        """_try_add_qcs6490_gpu_backend logs warning when GPU backend raises."""
        with (
            patch("moment_to_action.hardware._platform._detect_platform") as mock_detect,
            patch(
                "moment_to_action.hardware._platform.Platform._init_qcs6490",
                autospec=True,
            ) as mock_init,
        ):
            mock_detect.return_value = PlatformType.QCS6490

            def _setup(self: Platform) -> None:
                self._resource_monitor = MagicMock()
                self._backends = {ComputeUnit.CPU: MagicMock()}

            mock_init.side_effect = _setup
            p = Platform(AppConfig())

        import sys

        saved = sys.modules.get("moment_to_action.hardware._platforms.qcs6490._gpu_backend")
        try:
            mock_mod = MagicMock()
            mock_mod.QCS6490GPUBackend.side_effect = RuntimeError("delegate missing")
            sys.modules["moment_to_action.hardware._platforms.qcs6490._gpu_backend"] = mock_mod
            p._try_add_qcs6490_gpu_backend()
            assert ComputeUnit.GPU not in p._backends  # type: ignore[attr-defined]
        finally:
            if saved is None:
                del sys.modules["moment_to_action.hardware._platforms.qcs6490._gpu_backend"]
            else:
                sys.modules["moment_to_action.hardware._platforms.qcs6490._gpu_backend"] = saved
