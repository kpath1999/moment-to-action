"""Unit tests for MobileNetV2Model."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from moment_to_action.hardware import ComputeUnit, Platform
from moment_to_action.hardware._types import DataType, ModelType
from moment_to_action.models.image.classification.mobilenet_v2._model import (
    MobileNetV2Model,
    _softmax,
)

_CPU_BACKENDS: dict[ComputeUnit, dict[str, str]] = {ComputeUnit.CPU: {"model": "model.onnx"}}
_DLC_BACKENDS: dict[ComputeUnit, dict[str, str]] = {
    ComputeUnit.CPU: {"model": "model.dlc"},
    ComputeUnit.NPU: {"model": "model.dlc"},
}


@pytest.fixture
def onnx_model() -> MobileNetV2Model:
    """Return an unloaded MobileNetV2Model in ONNX format."""
    return MobileNetV2Model(
        "default", Path("/fake/mnv2"), ModelType.ONNX, DataType.FP32, backends=_CPU_BACKENDS
    )


@pytest.fixture
def dlc_model() -> MobileNetV2Model:
    """Return an unloaded MobileNetV2Model in DLC format."""
    return MobileNetV2Model(
        "qcs6490", Path("/fake/mnv2_qcs"), ModelType.DLC, DataType.W8A8, backends=_DLC_BACKENDS
    )


@pytest.fixture
def mock_platform() -> MagicMock:
    """Return a mock Platform."""
    platform = MagicMock(spec=Platform)
    platform.load_onnx.return_value = MagicMock()
    platform.load_dlc.return_value = MagicMock()
    return platform


@pytest.mark.unit
class TestSoftmax:
    """Tests for _softmax helper."""

    def test_output_sums_to_one(self) -> None:
        """Softmax output sums to 1 over last axis."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _softmax(x)
        assert abs(float(result.sum()) - 1.0) < 1e-6

    def test_all_positive(self) -> None:
        """Softmax output values are in (0, 1)."""
        x = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
        result = _softmax(x)
        assert float(result.min()) > 0.0
        assert float(result.max()) < 1.0

    def test_numerically_stable_for_large_values(self) -> None:
        """Softmax does not overflow for large logits."""
        x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        result = _softmax(x)
        assert not np.any(np.isnan(result))
        assert abs(float(result.sum()) - 1.0) < 1e-6

    def test_2d_input(self) -> None:
        """Softmax works on batched input, reducing over last axis."""
        x = np.zeros((2, 10), dtype=np.float32)
        result = _softmax(x)
        assert result.shape == (2, 10)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones(2), atol=1e-6)


@pytest.mark.unit
class TestMobileNetV2ModelPrepare:
    """Tests for MobileNetV2Model.prepare()."""

    def test_output_shape(
        self, onnx_model: MobileNetV2Model, sample_image_array: np.ndarray
    ) -> None:
        """prepare() returns NCHW (1, 3, 224, 224)."""
        result = onnx_model.prepare(sample_image_array)
        assert result.shape == (1, 3, 224, 224)

    def test_output_dtype_float32(
        self, onnx_model: MobileNetV2Model, sample_image_array: np.ndarray
    ) -> None:
        """prepare() returns float32."""
        result = onnx_model.prepare(sample_image_array)
        assert result.dtype == np.float32

    def test_imagenet_normalization_applied(self, onnx_model: MobileNetV2Model) -> None:
        """prepare() applies ImageNet mean/std normalization — values outside [0,1]."""
        frame = np.zeros((224, 224, 3), dtype=np.uint8)  # all black: (0 - mean)/std < 0
        result = onnx_model.prepare(frame)
        assert float(result.min()) < 0.0

    def test_same_for_onnx_and_dlc(
        self,
        onnx_model: MobileNetV2Model,
        dlc_model: MobileNetV2Model,
        sample_image_array: np.ndarray,
    ) -> None:
        """prepare() produces identical output regardless of format."""
        r1 = onnx_model.prepare(sample_image_array)
        r2 = dlc_model.prepare(sample_image_array)
        np.testing.assert_array_equal(r1, r2)


@pytest.mark.unit
class TestMobileNetV2ModelRun:
    """Tests for MobileNetV2Model.run()."""

    def test_onnx_calls_backend_run(
        self, onnx_model: MobileNetV2Model, mock_platform: MagicMock
    ) -> None:
        """ONNX run() delegates to handle.run()."""
        onnx_model.load(mock_platform, ComputeUnit.CPU)
        prepared = np.zeros((1, 3, 224, 224), dtype=np.float32)
        onnx_model.run(prepared)
        mock_platform.load_onnx.return_value.run.assert_called_once_with(prepared)

    def test_dlc_calls_backend_infer_dlc(
        self, dlc_model: MobileNetV2Model, mock_platform: MagicMock
    ) -> None:
        """DLC run() delegates to handle.run() and returns list."""
        mock_platform.load_dlc.return_value.run.return_value = {
            "output": np.zeros((1, 1000), dtype=np.float32)
        }
        dlc_model.load(mock_platform, ComputeUnit.CPU)
        prepared = np.zeros((1, 3, 224, 224), dtype=np.float32)
        result = dlc_model.run(prepared)
        mock_platform.load_dlc.return_value.run.assert_called_once()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_run_without_load_raises(self, onnx_model: MobileNetV2Model) -> None:
        """run() without load() raises RuntimeError."""
        with pytest.raises(RuntimeError, match=r"load\(\)"):
            onnx_model.run(np.zeros((1, 3, 224, 224), dtype=np.float32))


@pytest.mark.unit
class TestMobileNetV2ModelLoadUnload:
    """Tests for MobileNetV2Model.load() and unload()."""

    def test_onnx_load_calls_load_model(
        self, onnx_model: MobileNetV2Model, mock_platform: MagicMock
    ) -> None:
        """ONNX load() calls platform.load_onnx with model.onnx path."""
        onnx_model.load(mock_platform, ComputeUnit.CPU)
        mock_platform.load_onnx.assert_called_once_with(
            ComputeUnit.CPU, onnx_model.path / "model.onnx", dtype=DataType.FP32
        )

    def test_dlc_load_calls_load_model_dlc(
        self, dlc_model: MobileNetV2Model, mock_platform: MagicMock
    ) -> None:
        """DLC load() calls platform.load_dlc with model.dlc path."""
        dlc_model.load(mock_platform, ComputeUnit.CPU)
        mock_platform.load_dlc.assert_called_once_with(
            ComputeUnit.CPU, dlc_model.path / "model.dlc", dtype=DataType.W8A8
        )

    def test_load_sets_is_loaded(
        self, onnx_model: MobileNetV2Model, mock_platform: MagicMock
    ) -> None:
        """After load(), is_loaded is True."""
        onnx_model.load(mock_platform, ComputeUnit.CPU)
        assert onnx_model.is_loaded is True

    def test_double_load_raises(
        self, onnx_model: MobileNetV2Model, mock_platform: MagicMock
    ) -> None:
        """Loading an already-loaded model raises RuntimeError."""
        onnx_model.load(mock_platform, ComputeUnit.CPU)
        with pytest.raises(RuntimeError, match="already loaded"):
            onnx_model.load(mock_platform, ComputeUnit.CPU)

    def test_unload_clears_state(
        self, onnx_model: MobileNetV2Model, mock_platform: MagicMock
    ) -> None:
        """After unload(), is_loaded is False."""
        onnx_model.load(mock_platform, ComputeUnit.CPU)
        onnx_model.unload()
        assert onnx_model.is_loaded is False

    def test_onnx_unload_calls_unload(
        self, onnx_model: MobileNetV2Model, mock_platform: MagicMock
    ) -> None:
        """ONNX unload() calls handle.unload()."""
        onnx_model.load(mock_platform, ComputeUnit.CPU)
        onnx_model.unload()
        mock_platform.load_onnx.return_value.unload.assert_called_once()

    def test_dlc_unload_calls_unload(
        self, dlc_model: MobileNetV2Model, mock_platform: MagicMock
    ) -> None:
        """DLC unload() calls handle.unload()."""
        dlc_model.load(mock_platform, ComputeUnit.CPU)
        dlc_model.unload()
        mock_platform.load_dlc.return_value.unload.assert_called_once()

    def test_unload_when_not_loaded_is_noop(self, onnx_model: MobileNetV2Model) -> None:
        """unload() when not loaded does not raise."""
        onnx_model.unload()
        assert onnx_model.is_loaded is False


@pytest.mark.unit
class TestMobileNetV2ModelPostProc:
    """Tests for MobileNetV2Model.post_proc()."""

    def _make_raw(self, logits: list[float]) -> list[np.ndarray]:
        """Build a single-element raw output list."""
        return [np.array([logits], dtype=np.float32)]

    def test_returns_top_k_classifications(self, onnx_model: MobileNetV2Model) -> None:
        """post_proc returns up to top_k results."""
        logits = [0.0] * 1000
        logits[42] = 10.0
        logits[99] = 8.0
        result = onnx_model.post_proc(self._make_raw(logits))
        assert len(result) == 5  # default top_k=5
        assert result[0].class_id == 42

    def test_top1_is_highest_confidence(self, onnx_model: MobileNetV2Model) -> None:
        """post_proc returns results ordered by descending confidence."""
        logits = [0.0] * 1000
        logits[7] = 5.0
        logits[3] = 3.0
        result = onnx_model.post_proc(self._make_raw(logits))
        assert result[0].class_id == 7
        assert result[0].confidence > result[1].confidence

    def test_confidence_sums_to_one(self, onnx_model: MobileNetV2Model) -> None:
        """Confidence values are softmax probabilities; top_k subset sums < 1."""
        logits = [1.0] * 1000
        result = onnx_model.post_proc(self._make_raw(logits))
        # Uniform distribution: each prob = 1/1000, top-5 = 0.005
        total = sum(c.confidence for c in result)
        assert total < 1.0

    def test_custom_top_k(self) -> None:
        """Custom top_k limits number of returned results."""
        model = MobileNetV2Model(
            "v", Path("/x"), ModelType.ONNX, DataType.FP32, top_k=3, backends=_CPU_BACKENDS
        )
        logits = [float(i) for i in range(1000)]
        result = model.post_proc([np.array([logits], dtype=np.float32)])
        assert len(result) == 3

    def test_empty_raw_returns_empty(self, onnx_model: MobileNetV2Model) -> None:
        """post_proc returns empty list for empty raw input."""
        assert onnx_model.post_proc([]) == []

    def test_empty_array_returns_empty(self, onnx_model: MobileNetV2Model) -> None:
        """post_proc returns empty list for zero-element array."""
        assert onnx_model.post_proc([np.array([], dtype=np.float32)]) == []

    def test_classification_has_label(self, onnx_model: MobileNetV2Model) -> None:
        """Each Classification result has a non-empty label."""
        logits = [0.0] * 1000
        logits[0] = 10.0
        result = onnx_model.post_proc(self._make_raw(logits))
        assert result[0].label != ""

    def test_confidence_in_unit_range(self, onnx_model: MobileNetV2Model) -> None:
        """All confidence values are in (0, 1)."""
        logits = list(range(1000))
        result = onnx_model.post_proc([np.array([logits], dtype=np.float32)])
        for cls in result:
            assert 0.0 < cls.confidence < 1.0


@pytest.mark.unit
class TestMobileNetV2ModelGetLabel:
    """Tests for MobileNetV2Model._get_label()."""

    def test_fallback_for_out_of_range_id(self) -> None:
        """_get_label returns 'class_<id>' for IDs outside label range."""
        original = MobileNetV2Model.IMAGENET_LABELS
        MobileNetV2Model.IMAGENET_LABELS = ("tench",)  # only 1 label
        try:
            label = MobileNetV2Model._get_label(999)
        finally:
            MobileNetV2Model.IMAGENET_LABELS = original
        assert label == "class_999"

    def test_returns_string_always(self) -> None:
        """_get_label always returns a non-empty string."""
        label = MobileNetV2Model._get_label(0)
        assert isinstance(label, str)
        assert len(label) > 0

    def test_torchvision_unavailable_uses_class_id_fallback(self) -> None:
        """_get_label falls back to 'class_<id>' when torchvision import fails."""
        import sys
        from unittest.mock import patch

        original = MobileNetV2Model.IMAGENET_LABELS
        MobileNetV2Model.IMAGENET_LABELS = ()
        try:
            with patch.dict(sys.modules, {"torchvision.models": None}):
                label = MobileNetV2Model._get_label(42)
        finally:
            MobileNetV2Model.IMAGENET_LABELS = original
        assert label == "class_42"


@pytest.mark.unit
class TestMobileNetV2ModelProperties:
    """Tests for MobileNetV2Model properties."""

    def test_top_k_default(self) -> None:
        """Default top_k is 5."""
        model = MobileNetV2Model(
            "v", Path("/x"), ModelType.ONNX, DataType.FP32, backends=_CPU_BACKENDS
        )
        assert model.top_k == 5

    def test_top_k_custom(self) -> None:
        """Custom top_k is stored correctly."""
        model = MobileNetV2Model(
            "v", Path("/x"), ModelType.ONNX, DataType.FP32, top_k=10, backends=_CPU_BACKENDS
        )
        assert model.top_k == 10

    def test_prepare_for_conversion_returns_onnx_path(self) -> None:
        """prepare_for_conversion returns path unchanged (no surgery needed)."""
        model = MobileNetV2Model(
            "v", Path("/x"), ModelType.ONNX, DataType.FP32, backends=_CPU_BACKENDS
        )
        fake_onnx = Path("/some/model.onnx")
        assert model.prepare_for_conversion(fake_onnx) == fake_onnx
