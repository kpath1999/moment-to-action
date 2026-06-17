"""Unit tests for RTMDetModel."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from moment_to_action.hardware import ComputeUnit
from moment_to_action.models._formats import ModelFormat
from moment_to_action.models.image.detection._types import BoundingBox, Detection
from moment_to_action.models.image.detection.rtmdet._model import RTMDetModel

_ONNX_BACKENDS: dict[ComputeUnit, dict[str, str]] = {ComputeUnit.CPU: {"model": "model.onnx"}}
_DLC_BACKENDS: dict[ComputeUnit, dict[str, str]] = {ComputeUnit.CPU: {"model": "model.dlc"}}


@pytest.fixture
def onnx_model() -> RTMDetModel:
    """Return an unloaded RTMDetModel in ONNX format."""
    return RTMDetModel(
        "default", Path("/fake/model.onnx"), ModelFormat.ONNX, backends=_ONNX_BACKENDS
    )


@pytest.fixture
def dlc_model() -> RTMDetModel:
    """Return an unloaded RTMDetModel in DLC format with explicit NHWC layout."""
    return RTMDetModel(
        "qcs6490",
        Path("/fake/qcs6490"),
        ModelFormat.DLC,
        input_layout="NHWC",
        backends=_DLC_BACKENDS,
    )


@pytest.fixture
def dlc_model_nchw() -> RTMDetModel:
    """Return an unloaded RTMDetModel in DLC format with NCHW layout."""
    return RTMDetModel("other", Path("/fake/other"), ModelFormat.DLC, backends=_DLC_BACKENDS)


@pytest.fixture
def mock_backend() -> MagicMock:
    """Return a mock ComputeBackend."""
    backend = MagicMock()
    backend.preferred_unit = ComputeUnit.CPU
    backend.load_model.return_value = MagicMock()
    backend.load_model_dlc.return_value = MagicMock()
    return backend


def _make_outputs(
    boxes: list[list[float]],
    scores: list[float],
    class_ids: list[int],
) -> list[np.ndarray]:
    """Build synthetic 3-tensor RTMDet output."""
    boxes_arr = np.array([boxes], dtype=np.float32)  # [1, N, 4]
    scores_arr = np.array([scores], dtype=np.float32)  # [1, N]
    ids_arr = np.array([class_ids], dtype=np.uint8)  # [1, N]
    return [boxes_arr, scores_arr, ids_arr]


@pytest.mark.unit
class TestRTMDetModelPrepare:
    """Tests for RTMDetModel.prepare()."""

    def test_nchw_output_shape(
        self, onnx_model: RTMDetModel, sample_image_array: np.ndarray
    ) -> None:
        """prepare() returns NCHW (1, 3, 640, 640) for ONNX."""
        result = onnx_model.prepare(sample_image_array)
        assert result.shape == (1, 3, 640, 640)

    def test_dlc_nchw_output_shape(
        self, dlc_model_nchw: RTMDetModel, sample_image_array: np.ndarray
    ) -> None:
        """prepare() returns NCHW (1, 3, 640, 640) for non-qcs6490 DLC."""
        result = dlc_model_nchw.prepare(sample_image_array)
        assert result.shape == (1, 3, 640, 640)

    def test_dlc_nhwc_output_shape(
        self, dlc_model: RTMDetModel, sample_image_array: np.ndarray
    ) -> None:
        """prepare() returns NHWC (1, 640, 640, 3) for qcs6490 AI Hub DLC."""
        assert dlc_model.input_layout == "NHWC"
        result = dlc_model.prepare(sample_image_array)
        assert result.shape == (1, 640, 640, 3)

    def test_output_dtype_float32(
        self, onnx_model: RTMDetModel, sample_image_array: np.ndarray
    ) -> None:
        """prepare() returns float32."""
        result = onnx_model.prepare(sample_image_array)
        assert result.dtype == np.float32

    def test_values_in_unit_range(
        self, onnx_model: RTMDetModel, sample_image_array: np.ndarray
    ) -> None:
        """prepare() normalizes pixel values to [0, 1]."""
        result = onnx_model.prepare(sample_image_array)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_stores_original_size(
        self, onnx_model: RTMDetModel, sample_image_array: np.ndarray
    ) -> None:
        """prepare() stores original frame dimensions for post_proc scaling."""
        onnx_model.prepare(sample_image_array)
        h, w = sample_image_array.shape[:2]
        assert onnx_model._last_original_size == (h, w)


@pytest.mark.unit
class TestRTMDetModelProperties:
    """Tests for RTMDetModel properties."""

    def test_confidence_threshold_default(self, onnx_model: RTMDetModel) -> None:
        """Default confidence_threshold is 0.5."""
        assert onnx_model.confidence_threshold == 0.5

    def test_confidence_threshold_custom(self) -> None:
        """Custom confidence_threshold is stored."""
        model = RTMDetModel(
            "default",
            Path("/f"),
            ModelFormat.ONNX,
            confidence_threshold=0.3,
            backends=_ONNX_BACKENDS,
        )
        assert model.confidence_threshold == 0.3

    def test_input_layout_onnx_is_nchw(self, onnx_model: RTMDetModel) -> None:
        """ONNX variant uses NCHW layout."""
        assert onnx_model.input_layout == "NCHW"

    def test_input_layout_dlc_nhwc_when_explicit(self, dlc_model: RTMDetModel) -> None:
        """DLC model with explicit NHWC has NHWC layout."""
        assert dlc_model.input_layout == "NHWC"

    def test_input_layout_other_dlc_is_nchw(self, dlc_model_nchw: RTMDetModel) -> None:
        """DLC model without explicit layout defaults to NCHW."""
        assert dlc_model_nchw.input_layout == "NCHW"


@pytest.mark.unit
class TestRTMDetModelLoadUnload:
    """Tests for RTMDetModel.load() and unload()."""

    def test_load_onnx_calls_load_model(
        self, onnx_model: RTMDetModel, mock_backend: MagicMock
    ) -> None:
        """load() with ONNX format calls backend.load_model."""
        onnx_model.load(mock_backend)
        mock_backend.load_model.assert_called_once()

    def test_load_dlc_calls_load_model_dlc(
        self, dlc_model: RTMDetModel, mock_backend: MagicMock
    ) -> None:
        """load() with DLC format calls backend.load_model_dlc."""
        dlc_model.load(mock_backend)
        mock_backend.load_model_dlc.assert_called_once_with(Path("/fake/qcs6490/model.dlc"))

    def test_load_twice_raises(self, onnx_model: RTMDetModel, mock_backend: MagicMock) -> None:
        """Loading an already-loaded model raises RuntimeError."""
        onnx_model.load(mock_backend)
        with pytest.raises(RuntimeError, match="already loaded"):
            onnx_model.load(mock_backend)

    def test_unload_onnx_calls_unload_model(
        self, onnx_model: RTMDetModel, mock_backend: MagicMock
    ) -> None:
        """unload() with ONNX format calls backend.unload_model."""
        onnx_model.load(mock_backend)
        onnx_model.unload()
        mock_backend.unload_model.assert_called_once()

    def test_unload_clears_backend(self, onnx_model: RTMDetModel, mock_backend: MagicMock) -> None:
        """unload() sets _backend to None."""
        onnx_model.load(mock_backend)
        onnx_model.unload()
        assert onnx_model._backend is None

    def test_unload_without_load_is_noop(self, onnx_model: RTMDetModel) -> None:
        """unload() without prior load() does not raise."""
        onnx_model.unload()  # Should not raise


@pytest.mark.unit
class TestRTMDetModelRun:
    """Tests for RTMDetModel.run()."""

    def test_run_without_load_raises(
        self, onnx_model: RTMDetModel, sample_image_array: np.ndarray
    ) -> None:
        """run() without prior load() raises RuntimeError."""
        prepared = onnx_model.prepare(sample_image_array)
        with pytest.raises(RuntimeError, match="load\\(\\) must be called"):
            onnx_model.run(prepared)

    def test_run_onnx_calls_backend_run(
        self, onnx_model: RTMDetModel, mock_backend: MagicMock, sample_image_array: np.ndarray
    ) -> None:
        """run() with ONNX format delegates to backend.run."""
        outputs = _make_outputs([[10, 10, 100, 100]], [0.9], [0])
        mock_backend.run.return_value = outputs
        onnx_model.load(mock_backend)
        prepared = onnx_model.prepare(sample_image_array)
        result = onnx_model.run(prepared)
        mock_backend.run.assert_called_once()
        assert result is outputs

    def test_run_dlc_returns_tensor_list(
        self, dlc_model: RTMDetModel, mock_backend: MagicMock, sample_image_array: np.ndarray
    ) -> None:
        """run() with DLC format returns [boxes, scores, class_idx] from infer_dlc."""
        boxes = np.array([[[10, 10, 100, 100]]], dtype=np.float32)
        scores = np.array([[0.9]], dtype=np.float32)
        class_idx = np.array([[0]], dtype=np.uint8)
        mock_backend.infer_dlc.return_value = {
            "boxes": boxes,
            "scores": scores,
            "class_idx": class_idx,
        }
        dlc_model.load(mock_backend)
        prepared = dlc_model.prepare(sample_image_array)
        result = dlc_model.run(prepared)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], boxes)
        np.testing.assert_array_equal(result[1], scores)
        np.testing.assert_array_equal(result[2], class_idx)


@pytest.mark.unit
class TestRTMDetModelPostProc:
    """Tests for RTMDetModel.post_proc() and _decode()."""

    def test_returns_empty_on_no_detections(self, onnx_model: RTMDetModel) -> None:
        """Empty outputs return no detections."""
        outputs = _make_outputs([], [], [])
        result = onnx_model._decode(outputs, original_size=None)
        assert result == []

    def test_filters_low_confidence(self, onnx_model: RTMDetModel) -> None:
        """Detections below confidence_threshold are discarded."""
        outputs = _make_outputs([[0, 0, 100, 100]], [0.1], [0])
        result = onnx_model._decode(outputs, original_size=None)
        assert result == []

    def test_high_confidence_detection_kept(self, onnx_model: RTMDetModel) -> None:
        """Detections at or above confidence_threshold are kept."""
        outputs = _make_outputs([[0, 0, 100, 100]], [0.9], [0])
        result = onnx_model._decode(outputs, original_size=None)
        assert len(result) == 1
        assert result[0].label == "person"
        assert result[0].confidence == pytest.approx(0.9)

    def test_returns_fewer_than_three_outputs_empty(self, onnx_model: RTMDetModel) -> None:
        """Fewer than 3 output tensors returns empty list."""
        result = onnx_model._decode([np.zeros((1, 5, 4)), np.zeros((1, 5))], original_size=None)
        assert result == []

    def test_scales_to_original_size(self, onnx_model: RTMDetModel) -> None:
        """Boxes are scaled to the original frame size."""
        outputs = _make_outputs([[0, 0, 640, 640]], [0.9], [0])
        result = onnx_model._decode(outputs, original_size=(1280, 1280))
        assert len(result) == 1
        b = result[0].bbox
        assert b.x2 == pytest.approx(1280.0)
        assert b.y2 == pytest.approx(1280.0)

    def test_coco_label_mapping(self, onnx_model: RTMDetModel) -> None:
        """Class ID maps to the correct COCO label."""
        outputs = _make_outputs([[0, 0, 100, 100]], [0.9], [2])
        result = onnx_model._decode(outputs, original_size=None)
        assert result[0].label == "car"

    def test_unknown_class_id_uses_string(self, onnx_model: RTMDetModel) -> None:
        """Class IDs beyond COCO_LABELS length are rendered as strings."""
        outputs = _make_outputs([[0, 0, 100, 100]], [0.9], [200])
        result = onnx_model._decode(outputs, original_size=None)
        assert result[0].label == "200"

    def test_nms_suppresses_overlapping(self, onnx_model: RTMDetModel) -> None:
        """Highly overlapping boxes are reduced to one by NMS."""
        outputs = _make_outputs(
            [[0, 0, 100, 100], [1, 1, 99, 99]],
            [0.9, 0.8],
            [0, 0],
        )
        result = onnx_model._decode(outputs, original_size=None)
        assert len(result) == 1

    def test_detection_has_correct_fields(self, onnx_model: RTMDetModel) -> None:
        """Detection object has all required fields populated."""
        outputs = _make_outputs([[10.0, 20.0, 110.0, 220.0]], [0.75], [1])
        result = onnx_model._decode(outputs, original_size=None)
        assert len(result) == 1
        d = result[0]
        assert isinstance(d, Detection)
        assert isinstance(d.bbox, BoundingBox)
        assert d.confidence == pytest.approx(0.75)
        assert d.label == "bicycle"

    def test_post_proc_uses_last_original_size(
        self, onnx_model: RTMDetModel, sample_image_array: np.ndarray
    ) -> None:
        """post_proc() uses the size stored by the preceding prepare() call."""
        onnx_model.prepare(sample_image_array)
        outputs = _make_outputs([[0, 0, 640, 640]], [0.9], [0])
        result = onnx_model.post_proc(outputs)
        h, w = sample_image_array.shape[:2]
        assert result[0].bbox.x2 == pytest.approx(w)
        assert result[0].bbox.y2 == pytest.approx(h)

    def test_decode_uses_provided_size(self, onnx_model: RTMDetModel) -> None:
        """decode() uses the caller-supplied original_size."""
        outputs = _make_outputs([[0, 0, 640, 640]], [0.9], [0])
        result = onnx_model.decode(outputs, original_size=(320, 320))
        assert result[0].bbox.x2 == pytest.approx(320.0)
        assert result[0].bbox.y2 == pytest.approx(320.0)


@pytest.mark.unit
class TestRTMDetModelNMS:
    """Tests for RTMDetModel._nms()."""

    def test_single_box_kept(self) -> None:
        """Single box is always kept."""
        boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        keep = RTMDetModel._nms(boxes, scores, iou_threshold=0.5)
        assert keep == [0]

    def test_non_overlapping_all_kept(self) -> None:
        """Non-overlapping boxes are all kept."""
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = RTMDetModel._nms(boxes, scores, iou_threshold=0.5)
        assert sorted(keep) == [0, 1]

    def test_identical_boxes_suppressed(self) -> None:
        """Identical boxes — lower score is suppressed."""
        boxes = np.array([[0, 0, 100, 100], [0, 0, 100, 100]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = RTMDetModel._nms(boxes, scores, iou_threshold=0.5)
        assert keep == [0]

    def test_result_is_descending_score_order(self) -> None:
        """Kept boxes are in descending score order."""
        boxes = np.array([[0, 0, 5, 5], [10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float32)
        scores = np.array([0.7, 0.9, 0.8], dtype=np.float32)
        keep = RTMDetModel._nms(boxes, scores, iou_threshold=0.5)
        kept_scores = [scores[i] for i in keep]
        assert kept_scores == sorted(kept_scores, reverse=True)
