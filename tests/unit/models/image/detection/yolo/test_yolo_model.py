"""Unit tests for YOLOModel.

Covers prepare, run, post_proc/decode, load/unload (ONNX and DLC branches),
NMS, confidence filtering, coordinate scaling, and COCO label mapping.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from moment_to_action.models._formats import ModelFormat
from moment_to_action.models.image.detection._types import BoundingBox, Detection
from moment_to_action.models.image.detection.yolo._model import YOLOModel


@pytest.fixture
def onnx_model() -> YOLOModel:
    """Return an unloaded YOLOModel in ONNX format."""
    return YOLOModel("default", Path("/fake/model.onnx"), ModelFormat.ONNX)


@pytest.fixture
def dlc_model() -> YOLOModel:
    """Return an unloaded YOLOModel in DLC format (qcs6490 → NHWC auto-detected)."""
    return YOLOModel("qcs6490", Path("/fake/qcs6490"), ModelFormat.DLC)


@pytest.fixture
def dlc_model_nchw() -> YOLOModel:
    """Return an unloaded YOLOModel in DLC format with explicit NCHW layout."""
    return YOLOModel("other", Path("/fake/other"), ModelFormat.DLC)


@pytest.fixture
def mock_backend() -> MagicMock:
    """Return a mock ComputeBackend."""
    backend = MagicMock()
    backend.load_model.return_value = MagicMock()
    backend.load_model_dlc.return_value = MagicMock()
    return backend


def _make_outputs(
    boxes: list[list[float]],
    scores: list[float],
    class_ids: list[int],
) -> list[np.ndarray]:
    """Build synthetic 3-tensor YOLO output."""
    boxes_arr = np.array([boxes], dtype=np.float32)  # [1, N, 4]
    scores_arr = np.array([scores], dtype=np.float32)  # [1, N]
    ids_arr = np.array([class_ids], dtype=np.uint8)  # [1, N]
    return [boxes_arr, scores_arr, ids_arr]


@pytest.mark.unit
class TestYOLOModelPrepare:
    """Tests for YOLOModel.prepare()."""

    def test_output_shape(self, onnx_model: YOLOModel, sample_image_array: np.ndarray) -> None:
        """prepare() returns NCHW (1, 3, 640, 640) for ONNX."""
        result = onnx_model.prepare(sample_image_array)
        assert result.shape == (1, 3, 640, 640)

    def test_dlc_nchw_output_shape(
        self, dlc_model_nchw: YOLOModel, sample_image_array: np.ndarray
    ) -> None:
        """prepare() returns NCHW (1, 3, 640, 640) for non-qcs6490 DLC."""
        result = dlc_model_nchw.prepare(sample_image_array)
        assert result.shape == (1, 3, 640, 640)

    def test_dlc_nhwc_output_shape(
        self, dlc_model: YOLOModel, sample_image_array: np.ndarray
    ) -> None:
        """prepare() returns NHWC (1, 640, 640, 3) for qcs6490 AI Hub DLC."""
        assert dlc_model.input_layout == "NHWC"
        result = dlc_model.prepare(sample_image_array)
        assert result.shape == (1, 640, 640, 3)

    def test_output_dtype_float32(
        self, onnx_model: YOLOModel, sample_image_array: np.ndarray
    ) -> None:
        """prepare() returns float32."""
        result = onnx_model.prepare(sample_image_array)
        assert result.dtype == np.float32

    def test_nhwc_output_dtype_float32(
        self, dlc_model: YOLOModel, sample_image_array: np.ndarray
    ) -> None:
        """NHWC prepare() returns float32."""
        result = dlc_model.prepare(sample_image_array)
        assert result.dtype == np.float32

    def test_values_in_unit_range(
        self, onnx_model: YOLOModel, sample_image_array: np.ndarray
    ) -> None:
        """prepare() normalizes pixel values to [0, 1]."""
        result = onnx_model.prepare(sample_image_array)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_nhwc_values_in_unit_range(
        self, dlc_model: YOLOModel, sample_image_array: np.ndarray
    ) -> None:
        """NHWC prepare() normalizes pixel values to [0, 1]."""
        result = dlc_model.prepare(sample_image_array)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0


@pytest.mark.unit
class TestYOLOModelRun:
    """Tests for YOLOModel.run()."""

    def test_onnx_calls_backend_run(self, onnx_model: YOLOModel, mock_backend: MagicMock) -> None:
        """ONNX run() delegates to backend.run()."""
        onnx_model.load(mock_backend)
        prepared = np.zeros((1, 3, 640, 640), dtype=np.float32)
        onnx_model.run(prepared)
        mock_backend.run.assert_called_once_with(mock_backend.load_model.return_value, prepared)

    def test_dlc_calls_backend_infer_dlc(
        self, dlc_model: YOLOModel, mock_backend: MagicMock
    ) -> None:
        """DLC run() delegates to backend.infer_dlc()."""
        mock_backend.infer_dlc.return_value = {
            "boxes": np.zeros((1, 8400, 4)),
            "cls": np.zeros((1, 8400, 80)),
            "class_idx": np.zeros((1, 8400)),
        }
        with mock.patch(
            "moment_to_action.models.image.detection.yolo._model.resolve_backend_artifact",
            return_value=Path("/fake/model.dlc"),
        ):
            dlc_model.load(mock_backend)
            prepared = np.zeros((1, 640, 640, 3), dtype=np.float32)
            dlc_model.run(prepared)
        mock_backend.infer_dlc.assert_called_once()

    def test_dlc_aihub_run_uses_scores_directly(
        self, dlc_model: YOLOModel, mock_backend: MagicMock
    ) -> None:
        """AI Hub DLC run() returns scores directly when 'scores' key is present."""
        mock_backend.preferred_unit = MagicMock()
        boxes = np.zeros((1, 8400, 4), dtype=np.float32)
        scores = np.full((1, 8400), 0.9, dtype=np.float32)
        class_idx = np.zeros((1, 8400), dtype=np.float32)
        mock_backend.infer_dlc.return_value = {
            "boxes": boxes,
            "scores": scores,
            "class_idx": class_idx,
        }
        with mock.patch(
            "moment_to_action.models.image.detection.yolo._model.resolve_backend_artifact",
            return_value=Path("/fake/model.dlc"),
        ):
            dlc_model.load(mock_backend)
            prepared = np.zeros((1, 640, 640, 3), dtype=np.float32)
            result = dlc_model.run(prepared)
        assert len(result) == 3
        assert result[1] is scores  # scores returned directly

    def test_dlc_local_convert_run_computes_cls_max(
        self, dlc_model: YOLOModel, mock_backend: MagicMock
    ) -> None:
        """Local-convert DLC run() computes scores from cls.max() when 'scores' absent."""
        mock_backend.preferred_unit = MagicMock()
        cls = np.zeros((1, 8400, 80), dtype=np.float32)
        cls[0, 0, 5] = 0.85  # anchor 0, class 5 is the max
        mock_backend.infer_dlc.return_value = {
            "boxes": np.zeros((1, 8400, 4), dtype=np.float32),
            "cls": cls,
            "class_idx": np.zeros((1, 8400), dtype=np.float32),
        }
        with mock.patch(
            "moment_to_action.models.image.detection.yolo._model.resolve_backend_artifact",
            return_value=Path("/fake/model.dlc"),
        ):
            dlc_model.load(mock_backend)
            prepared = np.zeros((1, 640, 640, 3), dtype=np.float32)
            result = dlc_model.run(prepared)
        assert result[1][0, 0] == pytest.approx(0.85)

    def test_raises_if_not_loaded(self, onnx_model: YOLOModel) -> None:
        """run() raises RuntimeError if load() was not called."""
        prepared = np.zeros((1, 3, 640, 640), dtype=np.float32)
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            onnx_model.run(prepared)


@pytest.mark.unit
class TestYOLOModelLoadUnload:
    """Tests for load() and unload()."""

    def test_load_onnx_calls_load_model(
        self, onnx_model: YOLOModel, mock_backend: MagicMock
    ) -> None:
        """ONNX load() calls backend.load_model with the model path."""
        onnx_model.load(mock_backend)
        mock_backend.load_model.assert_called_once_with(Path("/fake/model.onnx"))

    def test_load_dlc_calls_load_model_dlc(
        self, dlc_model: YOLOModel, mock_backend: MagicMock
    ) -> None:
        """DLC load() resolves the backend artifact and passes it to load_model_dlc."""
        artifact = Path("/fake/qcs6490/model.dlc")
        mock_backend.preferred_unit = MagicMock()
        with mock.patch(
            "moment_to_action.models.image.detection.yolo._model.resolve_backend_artifact",
            return_value=artifact,
        ) as mock_resolve:
            dlc_model.load(mock_backend)
        mock_resolve.assert_called_once_with(Path("/fake/qcs6490"), mock_backend.preferred_unit)
        mock_backend.load_model_dlc.assert_called_once_with(artifact)

    def test_load_sets_backend(self, onnx_model: YOLOModel, mock_backend: MagicMock) -> None:
        """load() stores the backend on _backend."""
        onnx_model.load(mock_backend)
        assert onnx_model._backend is mock_backend

    def test_unload_onnx_calls_unload_model(
        self, onnx_model: YOLOModel, mock_backend: MagicMock
    ) -> None:
        """ONNX unload() calls backend.unload_model(handle)."""
        onnx_model.load(mock_backend)
        handle = mock_backend.load_model.return_value
        onnx_model.unload()
        mock_backend.unload_model.assert_called_once_with(handle)

    def test_unload_dlc_calls_unload_dlc(
        self, dlc_model: YOLOModel, mock_backend: MagicMock
    ) -> None:
        """DLC unload() calls backend.unload_dlc(handle)."""
        mock_backend.preferred_unit = MagicMock()
        with mock.patch(
            "moment_to_action.models.image.detection.yolo._model.resolve_backend_artifact",
            return_value=Path("/fake/qcs6490/model.dlc"),
        ):
            dlc_model.load(mock_backend)
        handle = mock_backend.load_model_dlc.return_value
        dlc_model.unload()
        mock_backend.unload_dlc.assert_called_once_with(handle)

    def test_unload_clears_backend_and_handle(
        self, onnx_model: YOLOModel, mock_backend: MagicMock
    ) -> None:
        """unload() clears _backend and _handle."""
        onnx_model.load(mock_backend)
        onnx_model.unload()
        assert onnx_model._backend is None
        assert onnx_model._handle is None

    def test_unload_without_load_is_safe(self, onnx_model: YOLOModel) -> None:
        """unload() when _backend is None does not raise."""
        onnx_model.unload()  # Should not raise

    def test_double_load_raises(self, onnx_model: YOLOModel, mock_backend: MagicMock) -> None:
        """load() raises RuntimeError if model is already loaded."""
        onnx_model.load(mock_backend)
        with pytest.raises(RuntimeError, match="already loaded"):
            onnx_model.load(mock_backend)

    def test_backend_not_set_before_handle_loaded(
        self, onnx_model: YOLOModel, mock_backend: MagicMock
    ) -> None:
        """_backend is only set after the handle is successfully loaded."""
        load_calls: list[bool] = []

        def _slow_load(_path: object) -> MagicMock:
            load_calls.append(onnx_model._backend is None)
            return MagicMock()

        mock_backend.load_model.side_effect = _slow_load
        onnx_model.load(mock_backend)
        assert load_calls == [True]  # _backend was None during handle load

    def test_load_failure_leaves_model_unloaded(
        self, onnx_model: YOLOModel, mock_backend: MagicMock
    ) -> None:
        """If backend.load_model raises, is_loaded remains False."""
        mock_backend.load_model.side_effect = RuntimeError("backend error")
        with pytest.raises(RuntimeError, match="backend error"):
            onnx_model.load(mock_backend)
        assert onnx_model.is_loaded is False


@pytest.mark.unit
class TestYOLOModelDecode:
    """Tests for YOLOModel.decode() and post_proc()."""

    def test_decode_basic(self, onnx_model: YOLOModel) -> None:
        """decode() returns correct Detection objects from known output."""
        outputs = _make_outputs(
            boxes=[[100, 100, 200, 200], [300, 300, 400, 400]],
            scores=[0.9, 0.7],
            class_ids=[0, 1],
        )
        result = onnx_model.decode(outputs, original_size=(480, 640))
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(d, Detection) for d in result)

    def test_decode_confidence_values(self, onnx_model: YOLOModel) -> None:
        """decode() preserves confidence scores."""
        outputs = _make_outputs([[100, 100, 200, 200]], [0.95], [0])
        result = onnx_model.decode(outputs, original_size=(480, 640))
        assert result[0].confidence == pytest.approx(0.95)

    def test_decode_label_from_coco(self, onnx_model: YOLOModel) -> None:
        """decode() maps class IDs to COCO labels."""
        outputs = _make_outputs(
            [[100, 100, 200, 200], [300, 300, 400, 400], [500, 500, 600, 600]],
            [0.9, 0.85, 0.8],
            [0, 16, 17],  # person, dog, horse
        )
        result = onnx_model.decode(outputs, original_size=(480, 640))
        assert result[0].label == "person"
        assert result[1].label == "dog"
        assert result[2].label == "horse"

    def test_decode_invalid_class_id_fallback(self, onnx_model: YOLOModel) -> None:
        """Out-of-range class IDs fall back to str(class_id)."""
        outputs = _make_outputs([[100, 100, 200, 200]], [0.9], [99])
        result = onnx_model.decode(outputs, original_size=(480, 640))
        assert result[0].label == "99"

    def test_decode_coordinate_scaling(self, onnx_model: YOLOModel) -> None:
        """Coordinates are scaled from 640x640 to original_size."""
        outputs = _make_outputs([[320, 240, 480, 480]], [0.9], [0])
        result = onnx_model.decode(outputs, original_size=(480, 640))
        # x scale = 640/640 = 1.0, y scale = 480/640 = 0.75
        assert result[0].bbox.x1 == pytest.approx(320.0)  # 320 * 1.0
        assert result[0].bbox.y1 == pytest.approx(180.0)  # 240 * 0.75
        assert result[0].bbox.x2 == pytest.approx(480.0)  # 480 * 1.0
        assert result[0].bbox.y2 == pytest.approx(360.0)  # 480 * 0.75

    def test_decode_confidence_filtering(self) -> None:
        """Detections below confidence_threshold are discarded."""
        model = YOLOModel("default", Path("/x"), ModelFormat.ONNX, confidence_threshold=0.8)
        outputs = _make_outputs(
            [[100, 100, 200, 200], [300, 300, 400, 400]],
            [0.9, 0.5],
            [0, 1],
        )
        result = model.decode(outputs, original_size=(480, 640))
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)

    def test_decode_empty_outputs(self, onnx_model: YOLOModel) -> None:
        """Empty outputs return empty list."""
        outputs = _make_outputs([], [], [])
        result = onnx_model.decode(outputs, original_size=(480, 640))
        assert result == []

    def test_decode_insufficient_outputs(self, onnx_model: YOLOModel) -> None:
        """Fewer than 3 output tensors returns empty list."""
        result = onnx_model.decode([np.zeros((1, 1, 4))], original_size=(480, 640))
        assert result == []

    def test_post_proc_returns_detections(self, onnx_model: YOLOModel) -> None:
        """post_proc() returns list[Detection] without coordinate scaling."""
        outputs = _make_outputs([[100, 100, 200, 200]], [0.9], [0])
        result = onnx_model.post_proc(outputs)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Detection)
        assert isinstance(result[0].bbox, BoundingBox)

    def test_decode_bounding_box_fields(self, onnx_model: YOLOModel) -> None:
        """decode() sets bbox fields correctly."""
        outputs = _make_outputs([[100, 150, 300, 350]], [0.95], [0])
        result = onnx_model.decode(outputs, original_size=(480, 640))
        box = result[0].bbox
        # x scale = 640/640 = 1.0, y scale = 480/640 = 0.75
        assert box.x1 == pytest.approx(100.0)
        assert box.y1 == pytest.approx(112.5)  # 150 * 0.75
        assert box.x2 == pytest.approx(300.0)
        assert box.y2 == pytest.approx(262.5)  # 350 * 0.75


@pytest.mark.unit
class TestYOLOModelNMS:
    """Tests for YOLOModel._nms() (pure NumPy NMS)."""

    def test_removes_overlapping_boxes(self, onnx_model: YOLOModel) -> None:
        """NMS removes boxes with high IoU overlap."""
        boxes = np.array([[100, 100, 200, 200], [110, 110, 210, 210]], dtype=np.float32)
        scores = np.array([0.9, 0.7], dtype=np.float32)
        keep = onnx_model._nms(boxes, scores, iou_threshold=0.45)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_keeps_non_overlapping_boxes(self, onnx_model: YOLOModel) -> None:
        """NMS keeps all non-overlapping boxes."""
        boxes = np.array([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = onnx_model._nms(boxes, scores, iou_threshold=0.45)
        assert len(keep) == 2

    def test_empty_input(self, onnx_model: YOLOModel) -> None:
        """NMS with empty input returns empty list."""
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        keep = onnx_model._nms(boxes, scores, iou_threshold=0.45)
        assert keep == []

    def test_single_box(self, onnx_model: YOLOModel) -> None:
        """NMS with a single box keeps it."""
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        keep = onnx_model._nms(boxes, scores, iou_threshold=0.45)
        assert keep == [0]

    def test_score_ordering(self, onnx_model: YOLOModel) -> None:
        """NMS processes boxes in descending score order."""
        boxes = np.array(
            [[100, 100, 200, 200], [105, 105, 205, 205], [110, 110, 210, 210]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = onnx_model._nms(boxes, scores, iou_threshold=0.45)
        assert keep[0] == 0


# ---------------------------------------------------------------------------
# COCO labels
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestYOLOModelLabels:
    """Tests for COCO label mapping."""

    def test_coco_labels_count(self, onnx_model: YOLOModel) -> None:
        """80 COCO labels."""
        assert len(onnx_model.COCO_LABELS) == 80

    def test_person_is_class_0(self, onnx_model: YOLOModel) -> None:
        """Class 0 is 'person'."""
        assert onnx_model.COCO_LABELS[0] == "person"

    def test_dog_is_class_16(self, onnx_model: YOLOModel) -> None:
        """Class 16 is 'dog'."""
        assert onnx_model.COCO_LABELS[16] == "dog"


# ---------------------------------------------------------------------------
# confidence_threshold property
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestYOLOModelProperties:
    """Tests for YOLOModel properties."""

    def test_confidence_threshold_default(self, onnx_model: YOLOModel) -> None:
        """Default confidence_threshold is 0.5."""
        assert onnx_model.confidence_threshold == pytest.approx(0.5)

    def test_confidence_threshold_custom(self) -> None:
        """Custom confidence_threshold is stored."""
        model = YOLOModel("v", Path("/x"), ModelFormat.ONNX, confidence_threshold=0.75)
        assert model.confidence_threshold == pytest.approx(0.75)

    def test_input_layout_onnx_defaults_nchw(self, onnx_model: YOLOModel) -> None:
        """ONNX variant defaults to NCHW."""
        assert onnx_model.input_layout == "NCHW"

    def test_input_layout_dlc_qcs6490_auto_nhwc(self, dlc_model: YOLOModel) -> None:
        """DLC qcs6490 variant auto-detects NHWC (AI Hub export)."""
        assert dlc_model.input_layout == "NHWC"

    def test_input_layout_dlc_non_qcs6490_nchw(self, dlc_model_nchw: YOLOModel) -> None:
        """DLC non-qcs6490 variant defaults to NCHW."""
        assert dlc_model_nchw.input_layout == "NCHW"

    def test_input_layout_explicit_override(self) -> None:
        """Explicit input_layout overrides auto-detection."""
        model = YOLOModel("qcs6490", Path("/x"), ModelFormat.DLC, input_layout="NCHW")
        assert model.input_layout == "NCHW"


def _make_yolo_onnx_with_concat(path: Path, opset: int = 11) -> None:
    """Write a minimal YOLOv8-style ONNX with a mixed-range output0 Concat."""
    import onnx
    from onnx import TensorProto
    from onnx import helper as oh

    # Use distinct input names so the "cls" output produced by surgery does not
    # collide with any graph-input tensor name.
    dbox = oh.make_tensor_value_info("dbox_in", TensorProto.FLOAT, [1, 4, 8400])
    cls = oh.make_tensor_value_info("cls_in", TensorProto.FLOAT, [1, 80, 8400])
    output0 = oh.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 8400])

    concat = oh.make_node("Concat", inputs=["dbox_in", "cls_in"], outputs=["output0"], axis=1)
    graph = oh.make_graph([concat], "yolov8", [dbox, cls], [output0])
    model_proto = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    onnx.save(model_proto, str(path))


def _make_yolo_onnx_output0_no_concat(path: Path) -> None:
    """Write an ONNX with output0 produced by Identity (not Concat)."""
    import onnx
    from onnx import TensorProto
    from onnx import helper as oh

    inp = oh.make_tensor_value_info("data", TensorProto.FLOAT, [1, 84, 8400])
    output0 = oh.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 8400])

    identity = oh.make_node("Identity", inputs=["data"], outputs=["output0"])
    graph = oh.make_graph([identity], "no_concat", [inp], [output0])
    model_proto = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 11)])
    onnx.save(model_proto, str(path))


def _make_yolo_onnx_already_split(path: Path) -> None:
    """Write a minimal ONNX with three already-split outputs (no output0 Concat)."""
    import onnx
    from onnx import TensorProto
    from onnx import helper as oh

    inp = oh.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    boxes = oh.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 8400, 4])
    scores = oh.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 8400])
    class_idx = oh.make_tensor_value_info("class_idx", TensorProto.FLOAT, [1, 8400])

    identity = oh.make_node("Identity", inputs=["images"], outputs=["_dummy"])
    graph = oh.make_graph([identity], "split", [inp], [boxes, scores, class_idx])
    model_proto = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 11)])
    onnx.save(model_proto, str(path))


@pytest.mark.unit
class TestYOLOModelPrepareForConversion:
    """Tests for YOLOModel.prepare_for_conversion()."""

    def test_returns_same_path_when_already_split(self, tmp_path: Path) -> None:
        """prepare_for_conversion() returns onnx_path unchanged when outputs are already split."""
        onnx_path = tmp_path / "split.onnx"
        _make_yolo_onnx_already_split(onnx_path)
        model = YOLOModel("default", onnx_path, ModelFormat.ONNX)
        result = model.prepare_for_conversion(onnx_path)
        assert result == onnx_path

    def test_surgery_produces_different_path(self, tmp_path: Path) -> None:
        """prepare_for_conversion() returns a new temp path for a mixed-range ONNX."""
        onnx_path = tmp_path / "raw.onnx"
        _make_yolo_onnx_with_concat(onnx_path)
        model = YOLOModel("default", onnx_path, ModelFormat.ONNX)
        result = model.prepare_for_conversion(onnx_path)
        try:
            assert result != onnx_path
            assert result.exists()
        finally:
            if result != onnx_path:
                result.unlink(missing_ok=True)

    def test_surgery_output_has_three_outputs(self, tmp_path: Path) -> None:
        """Surgically modified ONNX has boxes, cls, class_idx as graph outputs."""
        import onnx

        onnx_path = tmp_path / "raw.onnx"
        _make_yolo_onnx_with_concat(onnx_path)
        model = YOLOModel("default", onnx_path, ModelFormat.ONNX)
        result = model.prepare_for_conversion(onnx_path)
        try:
            modified = onnx.load(str(result))
            out_names = [o.name for o in modified.graph.output]
            assert len(out_names) == 3
            assert "boxes" in out_names
            assert "cls" in out_names
            assert "class_idx" in out_names
        finally:
            if result != onnx_path:
                result.unlink(missing_ok=True)

    def test_surgery_removes_concat_node(self, tmp_path: Path) -> None:
        """Surgically modified ONNX does not contain the original Concat node."""
        import onnx

        onnx_path = tmp_path / "raw.onnx"
        _make_yolo_onnx_with_concat(onnx_path)
        model = YOLOModel("default", onnx_path, ModelFormat.ONNX)
        result = model.prepare_for_conversion(onnx_path)
        try:
            modified = onnx.load(str(result))
            concat_nodes = [n for n in modified.graph.node if n.op_type == "Concat"]
            assert len(concat_nodes) == 0
        finally:
            if result != onnx_path:
                result.unlink(missing_ok=True)

    def test_returns_same_path_when_output0_not_from_concat(self, tmp_path: Path) -> None:
        """prepare_for_conversion() returns onnx_path when output0 has no Concat feeding it."""
        onnx_path = tmp_path / "no_concat.onnx"
        _make_yolo_onnx_output0_no_concat(onnx_path)
        model = YOLOModel("default", onnx_path, ModelFormat.ONNX)
        result = model.prepare_for_conversion(onnx_path)
        assert result == onnx_path

    def test_surgery_opset18_produces_cls_output(self, tmp_path: Path) -> None:
        """Surgery on opset-18 models produces a cls output instead of scores+ReduceMax."""
        import onnx

        onnx_path = tmp_path / "opset18.onnx"
        _make_yolo_onnx_with_concat(onnx_path, opset=18)
        model = YOLOModel("default", onnx_path, ModelFormat.ONNX)
        result = model.prepare_for_conversion(onnx_path)
        try:
            assert result != onnx_path
            modified = onnx.load(str(result))
            out_names = [o.name for o in modified.graph.output]
            assert "cls" in out_names
            reduce_nodes = [n for n in modified.graph.node if n.op_type == "ReduceMax"]
            assert len(reduce_nodes) == 0
        finally:
            if result != onnx_path:
                result.unlink(missing_ok=True)


def _make_yolo_onnx_with_qdq(path: Path) -> None:
    """Write a minimal ONNX with Q→DQ→Identity chain.

    Structure: inp → Q → DQ → Identity → out
    After stripping: inp → Identity → out
    The Identity node's input remap exercises lines 60-63 of _strip_qdq.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper
    from onnx import helper as oh

    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1])
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, [1])
    scale = numpy_helper.from_array(np.array(0.01, dtype=np.float32), name="scale")
    zp = numpy_helper.from_array(np.array(0, dtype=np.int8), name="zp")
    q_node = oh.make_node("QuantizeLinear", ["inp", "scale", "zp"], ["inp_q"])
    dq_node = oh.make_node("DequantizeLinear", ["inp_q", "scale", "zp"], ["inp_dq"])
    identity = oh.make_node("Identity", ["inp_dq"], ["out"])
    graph = oh.make_graph([q_node, dq_node, identity], "qdq", [inp], [out], initializer=[scale, zp])
    model_proto = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    onnx.save(model_proto, str(path))


def _make_yolo_onnx_with_qdq_output(path: Path) -> None:
    """Write a minimal ONNX where the DQ output is directly a graph output.

    Structure: inp → Q → DQ → out (graph output)
    After stripping: graph output remapped from "out" to "inp".
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper
    from onnx import helper as oh

    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1])
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, [1])
    scale = numpy_helper.from_array(np.array(0.01, dtype=np.float32), name="scale")
    zp = numpy_helper.from_array(np.array(0, dtype=np.int8), name="zp")
    q_node = oh.make_node("QuantizeLinear", ["inp", "scale", "zp"], ["inp_q"])
    dq_node = oh.make_node("DequantizeLinear", ["inp_q", "scale", "zp"], ["out"])
    graph = oh.make_graph([q_node, dq_node], "qdq_out", [inp], [out], initializer=[scale, zp])
    model_proto = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    onnx.save(model_proto, str(path))


@pytest.mark.unit
class TestYOLOModelStripQDQ:
    """Tests for prepare_for_conversion() QDQ-stripping path."""

    def test_qdq_produces_different_path(self, tmp_path: Path) -> None:
        """An ONNX with Q→DQ nodes produces a new temp path."""
        onnx_path = tmp_path / "qdq.onnx"
        _make_yolo_onnx_with_qdq(onnx_path)
        model = YOLOModel("default", onnx_path, ModelFormat.ONNX)
        result = model.prepare_for_conversion(onnx_path)
        try:
            assert result != onnx_path
            assert result.exists()
        finally:
            if result != onnx_path:
                result.unlink(missing_ok=True)

    def test_qdq_nodes_removed(self, tmp_path: Path) -> None:
        """Stripped ONNX contains no QuantizeLinear or DequantizeLinear nodes."""
        import onnx

        onnx_path = tmp_path / "qdq.onnx"
        _make_yolo_onnx_with_qdq(onnx_path)
        model = YOLOModel("default", onnx_path, ModelFormat.ONNX)
        result = model.prepare_for_conversion(onnx_path)
        try:
            modified = onnx.load(str(result))
            op_types = {n.op_type for n in modified.graph.node}
            assert "QuantizeLinear" not in op_types
            assert "DequantizeLinear" not in op_types
        finally:
            if result != onnx_path:
                result.unlink(missing_ok=True)

    def test_qdq_node_input_remapped(self, tmp_path: Path) -> None:
        """Non-QDQ node that consumed a DQ output gets its input remapped to the original tensor."""
        import onnx

        onnx_path = tmp_path / "qdq.onnx"
        _make_yolo_onnx_with_qdq(onnx_path)
        model = YOLOModel("default", onnx_path, ModelFormat.ONNX)
        result = model.prepare_for_conversion(onnx_path)
        try:
            modified = onnx.load(str(result))
            identity_nodes = [n for n in modified.graph.node if n.op_type == "Identity"]
            assert len(identity_nodes) == 1
            # Identity input was "inp_dq"; after strip it should be "inp"
            assert identity_nodes[0].input[0] == "inp"
        finally:
            if result != onnx_path:
                result.unlink(missing_ok=True)

    def test_qdq_graph_output_remapped(self, tmp_path: Path) -> None:
        """Graph output whose tensor was produced by a removed DQ node is preserved via Identity."""
        import onnx

        onnx_path = tmp_path / "qdq_out.onnx"
        _make_yolo_onnx_with_qdq_output(onnx_path)
        model = YOLOModel("default", onnx_path, ModelFormat.ONNX)
        result = model.prepare_for_conversion(onnx_path)
        try:
            modified = onnx.load(str(result))
            out_names = [o.name for o in modified.graph.output]
            # Output name "out" is preserved (DLC tensor key stays correct)
            assert "out" in out_names
            assert "inp" not in out_names
            # An Identity node wires the original float tensor "inp" to "out"
            identity_nodes = [n for n in modified.graph.node if n.op_type == "Identity"]
            assert any(n.input[0] == "inp" and n.output[0] == "out" for n in identity_nodes)
        finally:
            if result != onnx_path:
                result.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# _expose_cls_for_reducemax helpers
# ---------------------------------------------------------------------------


def _make_yolo_onnx_with_reducemax(path: Path) -> None:
    """Write a minimal ONNX with a ReduceMax→scores output plus Identity→scores chain.

    Structure:
      images → Identity → cls_tensor
      cls_tensor → ReduceMax(axes=[-1]) → scores_raw → Identity → scores (output)
      images → Identity → class_idx (output)
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper
    from onnx import helper as oh

    inp = oh.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    boxes_out = oh.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 8400, 4])
    scores_out = oh.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 8400])
    class_idx_out = oh.make_tensor_value_info("class_idx", TensorProto.FLOAT, [1, 8400])

    axes_init = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="_axes")
    make_cls = oh.make_node("Identity", ["images"], ["cls_tensor"])
    reducemax = oh.make_node("ReduceMax", ["cls_tensor", "_axes"], ["scores_raw"], keepdims=0)
    id_scores = oh.make_node("Identity", ["scores_raw"], ["scores"])
    id_cidx = oh.make_node("Identity", ["images"], ["class_idx"])

    graph = oh.make_graph(
        [make_cls, reducemax, id_scores, id_cidx],
        "with_reducemax",
        [inp],
        [boxes_out, scores_out, class_idx_out],
        initializer=[axes_init],
    )
    model_proto = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    onnx.save(model_proto, str(path))


@pytest.mark.unit
class TestExposeClsForReducemax:
    """Tests for _expose_cls_for_reducemax()."""

    def test_returns_false_when_no_reducemax(self, tmp_path: Path) -> None:
        """Returns False when the graph has no ReduceMax node."""
        import onnx

        from moment_to_action.models.image.detection.yolo._model import _expose_cls_for_reducemax

        onnx_path = tmp_path / "split.onnx"
        _make_yolo_onnx_already_split(onnx_path)
        m = onnx.load(str(onnx_path))
        assert _expose_cls_for_reducemax(m) is False

    def test_returns_true_when_reducemax_found(self, tmp_path: Path) -> None:
        """Returns True when a ReduceMax node is replaced."""
        import onnx

        from moment_to_action.models.image.detection.yolo._model import _expose_cls_for_reducemax

        onnx_path = tmp_path / "rm.onnx"
        _make_yolo_onnx_with_reducemax(onnx_path)
        m = onnx.load(str(onnx_path))
        assert _expose_cls_for_reducemax(m) is True

    def test_reducemax_removed(self, tmp_path: Path) -> None:
        """ReduceMax node is removed from the graph."""
        import onnx

        from moment_to_action.models.image.detection.yolo._model import _expose_cls_for_reducemax

        onnx_path = tmp_path / "rm.onnx"
        _make_yolo_onnx_with_reducemax(onnx_path)
        m = onnx.load(str(onnx_path))
        _expose_cls_for_reducemax(m)
        op_types = [n.op_type for n in m.graph.node]
        assert "ReduceMax" not in op_types

    def test_scores_replaced_by_cls_output(self, tmp_path: Path) -> None:
        """Graph output 'scores' is replaced by 'cls'."""
        import onnx

        from moment_to_action.models.image.detection.yolo._model import _expose_cls_for_reducemax

        onnx_path = tmp_path / "rm.onnx"
        _make_yolo_onnx_with_reducemax(onnx_path)
        m = onnx.load(str(onnx_path))
        _expose_cls_for_reducemax(m)
        out_names = [o.name for o in m.graph.output]
        assert "cls" in out_names
        assert "scores" not in out_names

    def test_identity_feeding_scores_removed(self, tmp_path: Path) -> None:
        """Identity node that forwarded ReduceMax output to 'scores' is removed."""
        import onnx

        from moment_to_action.models.image.detection.yolo._model import _expose_cls_for_reducemax

        onnx_path = tmp_path / "rm.onnx"
        _make_yolo_onnx_with_reducemax(onnx_path)
        m = onnx.load(str(onnx_path))
        _expose_cls_for_reducemax(m)
        id_nodes = [n for n in m.graph.node if n.op_type == "Identity"]
        assert not any(n.output[0] == "scores" for n in id_nodes)

    def test_cls_identity_node_added(self, tmp_path: Path) -> None:
        """An Identity node mapping the ReduceMax input to 'cls' is inserted."""
        import onnx

        from moment_to_action.models.image.detection.yolo._model import _expose_cls_for_reducemax

        onnx_path = tmp_path / "rm.onnx"
        _make_yolo_onnx_with_reducemax(onnx_path)
        m = onnx.load(str(onnx_path))
        _expose_cls_for_reducemax(m)
        id_nodes = [n for n in m.graph.node if n.op_type == "Identity"]
        assert any(n.output[0] == "cls" for n in id_nodes)
