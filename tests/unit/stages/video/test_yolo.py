"""Tests for YOLO detection stage.

Tests YOLOStage including output parsing, NMS, confidence filtering,
and COCO label mapping. ComputeBackend is mocked to avoid real model loading.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from moment_to_action.messages.video import (
    BoundingBox,
    DetectionMessage,
    FrameTensorMessage,
)
from moment_to_action.stages.video._yolo import YOLOStage


@pytest.mark.unit
class TestYOLOParseOutputs:
    """Test YOLOStage._parse_outputs() with synthetic tensors."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock ComputeBackend."""
        backend = MagicMock()
        backend.load_model.return_value = MagicMock()
        return backend

    def test_parse_outputs_basic(self, mock_backend: MagicMock) -> None:
        """Test parsing basic YOLO outputs."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        # Synthetic outputs: [boxes], [scores], [class_ids]
        boxes = np.array([[[100, 100, 200, 200], [300, 300, 400, 400]]], dtype=np.float32)
        scores = np.array([[0.9, 0.7]], dtype=np.float32)
        class_ids = np.array([[0, 1]], dtype=np.uint8)  # person, bicycle

        outputs = [boxes, scores, class_ids]
        original_size = (480, 640)

        result = stage._parse_outputs(outputs, original_size)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(b, BoundingBox) for b in result)
        assert result[0].confidence == pytest.approx(0.9)
        assert result[1].confidence == pytest.approx(0.7)

    def test_parse_outputs_single_box(self, mock_backend: MagicMock) -> None:
        """Test parsing a single bounding box."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        boxes = np.array([[[100, 100, 200, 200]]], dtype=np.float32)
        scores = np.array([[0.95]], dtype=np.float32)
        class_ids = np.array([[0]], dtype=np.uint8)

        outputs = [boxes, scores, class_ids]
        original_size = (480, 640)

        result = stage._parse_outputs(outputs, original_size)

        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.95)
        assert result[0].class_id == 0
        assert result[0].label == "person"

    def test_parse_outputs_empty(self, mock_backend: MagicMock) -> None:
        """Test parsing empty outputs returns empty list."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        boxes = np.array(np.zeros((1, 0, 4)), dtype=np.float32)
        scores = np.array(np.zeros((1, 0)), dtype=np.float32)
        class_ids = np.array(np.zeros((1, 0)), dtype=np.uint8)

        outputs = [boxes, scores, class_ids]
        original_size = (480, 640)

        result = stage._parse_outputs(outputs, original_size)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_parse_outputs_insufficient_outputs(self, mock_backend: MagicMock) -> None:
        """Test with fewer than 3 outputs returns empty list."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        outputs = [np.array([[[100, 100, 200, 200]]])]  # Only 1 output instead of 3

        result = stage._parse_outputs(outputs, (480, 640))

        assert len(result) == 0

    def test_parse_outputs_coordinate_scaling(self, mock_backend: MagicMock) -> None:
        """Test that coordinates are scaled from 640x640 to original size."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        # Box at [320, 320, 320, 320] in 640x640 space
        boxes = np.array([[[320, 320, 320, 320]]], dtype=np.float32)
        scores = np.array([[0.9]], dtype=np.float32)
        class_ids = np.array([[0]], dtype=np.uint8)

        outputs = [boxes, scores, class_ids]
        original_size = (240, 320)  # Half the size

        result = stage._parse_outputs(outputs, original_size)

        assert len(result) == 1
        # Box should be scaled by (320/640, 240/640) = (0.5, 0.375)
        assert result[0].x1 == pytest.approx(160.0)
        assert result[0].y1 == pytest.approx(120.0)
        assert result[0].x2 == pytest.approx(160.0)
        assert result[0].y2 == pytest.approx(120.0)


@pytest.mark.unit
class TestYOLONMS:
    """Test _nms() (non-maximum suppression) removes overlapping boxes."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock ComputeBackend."""
        backend = MagicMock()
        backend.load_model.return_value = MagicMock()
        return backend

    def test_nms_removes_overlapping_boxes(self, mock_backend: MagicMock) -> None:
        """Test NMS removes boxes with high IoU overlap."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        # Two highly overlapping boxes
        boxes = np.array(
            [
                [100, 100, 200, 200],
                [110, 110, 210, 210],  # 90% overlap with first box
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.7], dtype=np.float32)

        keep = stage._nms(boxes, scores, iou_threshold=0.45)

        # With IoU threshold 0.45, the lower-confidence box should be removed
        assert isinstance(keep, list)
        assert len(keep) == 1
        assert keep[0] == 0  # Higher confidence box kept

    def test_nms_keeps_non_overlapping_boxes(self, mock_backend: MagicMock) -> None:
        """Test NMS keeps non-overlapping boxes."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        # Non-overlapping boxes
        boxes = np.array(
            [
                [0, 0, 100, 100],
                [200, 200, 300, 300],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)

        keep = stage._nms(boxes, scores, iou_threshold=0.45)

        assert len(keep) == 2
        assert 0 in keep
        assert 1 in keep

    def test_nms_empty_input(self, mock_backend: MagicMock) -> None:
        """Test NMS with empty input."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        boxes = np.array(np.zeros((0, 4)), dtype=np.float32)
        scores = np.array(np.zeros(0), dtype=np.float32)

        keep = stage._nms(boxes, scores, iou_threshold=0.45)

        assert len(keep) == 0

    def test_nms_single_box(self, mock_backend: MagicMock) -> None:
        """Test NMS with single box."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)

        keep = stage._nms(boxes, scores, iou_threshold=0.45)

        assert len(keep) == 1
        assert keep[0] == 0

    def test_nms_respects_score_ordering(self, mock_backend: MagicMock) -> None:
        """Test NMS processes boxes in order of descending score."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        # Three overlapping boxes, highest score first
        boxes = np.array(
            [
                [100, 100, 200, 200],  # Highest score
                [105, 105, 205, 205],  # Medium score
                [110, 110, 210, 210],  # Lowest score
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        keep = stage._nms(boxes, scores, iou_threshold=0.45)

        # Highest score box should be kept
        assert keep[0] == 0


@pytest.mark.unit
class TestYOLOConfidenceFiltering:
    """Test confidence filtering (removes boxes below threshold)."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock ComputeBackend."""
        backend = MagicMock()
        backend.load_model.return_value = MagicMock()
        return backend

    def test_confidence_threshold_filtering(self, mock_backend: MagicMock) -> None:
        """Test that boxes below threshold are filtered."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.6,
        )

        boxes = np.array(
            [
                [100, 100, 200, 200],
                [300, 300, 400, 400],
                [500, 500, 600, 600],
            ],
            dtype=np.float32,
        )
        scores = np.array([[0.9, 0.5, 0.7]], dtype=np.float32)
        class_ids = np.array([[0, 1, 2]], dtype=np.uint8)

        outputs = [boxes[np.newaxis, :], scores, class_ids]
        result = stage._parse_outputs(outputs, (480, 640))

        # Only boxes with confidence >= 0.6 should remain
        assert len(result) == 2
        assert all(b.confidence >= 0.6 for b in result)

    def test_all_boxes_below_threshold(self, mock_backend: MagicMock) -> None:
        """Test when all boxes are below confidence threshold."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.9,
        )

        boxes = np.array(
            [
                [100, 100, 200, 200],
                [300, 300, 400, 400],
            ],
            dtype=np.float32,
        )
        scores = np.array([[0.5, 0.7]], dtype=np.float32)
        class_ids = np.array([[0, 1]], dtype=np.uint8)

        outputs = [boxes[np.newaxis, :], scores, class_ids]
        result = stage._parse_outputs(outputs, (480, 640))

        assert len(result) == 0

    def test_no_filtering_at_zero_threshold(self, mock_backend: MagicMock) -> None:
        """Test with confidence threshold = 0."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.0,
        )

        boxes = np.array(
            [
                [100, 100, 200, 200],
                [300, 300, 400, 400],
            ],
            dtype=np.float32,
        )
        scores = np.array([[0.01, 0.02]], dtype=np.float32)
        class_ids = np.array([[0, 1]], dtype=np.uint8)

        outputs = [boxes[np.newaxis, :], scores, class_ids]
        result = stage._parse_outputs(outputs, (480, 640))

        assert len(result) == 2


@pytest.mark.unit
class TestYOLOLabels:
    """Test COCO label mapping (class ID → label name)."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock ComputeBackend."""
        backend = MagicMock()
        backend.load_model.return_value = MagicMock()
        return backend

    def test_coco_labels_count(self, mock_backend: MagicMock) -> None:
        """Test that COCO labels has correct count (80 classes)."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        assert len(stage.COCO_LABELS) == 80

    def test_coco_label_person(self, mock_backend: MagicMock) -> None:
        """Test that class 0 is 'person'."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        assert stage.COCO_LABELS[0] == "person"

    def test_coco_label_dog(self, mock_backend: MagicMock) -> None:
        """Test that class 16 is 'dog'."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        assert stage.COCO_LABELS[16] == "dog"

    def test_label_mapping_in_parse_outputs(self, mock_backend: MagicMock) -> None:
        """Test that labels are correctly assigned in _parse_outputs."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        # Class IDs: 0 (person), 16 (dog), 17 (horse)
        boxes = np.array(
            [
                [100, 100, 200, 200],
                [300, 300, 400, 400],
                [500, 500, 600, 600],
            ],
            dtype=np.float32,
        )
        scores = np.array([[0.9, 0.85, 0.8]], dtype=np.float32)
        class_ids = np.array([[0, 16, 17]], dtype=np.uint8)

        outputs = [boxes[np.newaxis, :], scores, class_ids]
        result = stage._parse_outputs(outputs, (480, 640))

        assert result[0].label == "person"
        assert result[0].class_id == 0
        assert result[1].label == "dog"
        assert result[1].class_id == 16
        assert result[2].label == "horse"
        assert result[2].class_id == 17

    def test_invalid_class_id_fallback(self, mock_backend: MagicMock) -> None:
        """Test handling of out-of-range class IDs."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        # Class ID 99 is out of range
        boxes = np.array([[[100, 100, 200, 200]]], dtype=np.float32)
        scores = np.array([[0.9]], dtype=np.float32)
        class_ids = np.array([[99]], dtype=np.uint8)

        outputs = [boxes, scores, class_ids]
        result = stage._parse_outputs(outputs, (480, 640))

        assert len(result) == 1
        assert result[0].class_id == 99
        # Should use string representation of class ID as fallback
        assert result[0].label == "99"


@pytest.mark.unit
class TestYOLOStageE2E:
    """Test YOLOStage end-to-end with mocked ComputeBackend."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock ComputeBackend."""
        backend = MagicMock()
        backend.load_model.return_value = MagicMock()
        return backend

    def test_yolo_stage_process_valid_input(self, mock_backend: MagicMock) -> None:
        """Test YOLOStage.process with valid input."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        # Mock backend to return YOLO outputs
        boxes = np.array([[[100, 100, 200, 200], [300, 300, 400, 400]]], dtype=np.float32)
        scores = np.array([[0.9, 0.7]], dtype=np.float32)
        class_ids = np.array([[0, 1]], dtype=np.uint8)
        mock_backend.run.return_value = [boxes, scores, class_ids]

        rng = np.random.default_rng()

        tensor = rng.standard_normal((1, 3, 640, 640)).astype(np.float32)
        msg = FrameTensorMessage(
            tensor=tensor,
            timestamp=time.time(),
            original_size=(480, 640),
        )

        result = stage.process(msg)

        assert isinstance(result, DetectionMessage)
        assert len(result.boxes) == 2
        assert all(isinstance(b, BoundingBox) for b in result.boxes)

    def test_yolo_stage_process_no_detections(self, mock_backend: MagicMock) -> None:
        """Test YOLOStage returns None when no detections above threshold."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.9,
        )

        # All boxes below threshold
        boxes = np.array([[[100, 100, 200, 200]]], dtype=np.float32)
        scores = np.array([[0.5]], dtype=np.float32)
        class_ids = np.array([[0]], dtype=np.uint8)
        mock_backend.run.return_value = [boxes, scores, class_ids]

        rng = np.random.default_rng()

        tensor = rng.standard_normal((1, 3, 640, 640)).astype(np.float32)
        msg = FrameTensorMessage(
            tensor=tensor,
            timestamp=time.time(),
            original_size=(480, 640),
        )

        result = stage.process(msg)

        assert result is None

    def test_yolo_stage_invalid_input_type(self, mock_backend: MagicMock) -> None:
        """Test YOLOStage rejects non-FrameTensorMessage input."""
        from moment_to_action.messages.sensor import RawFrameMessage

        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        rng = np.random.default_rng()

        img_array = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        with pytest.raises(TypeError, match="expects FrameTensorMessage"):
            stage.process(msg)

    def test_yolo_stage_confidence_property(self, mock_backend: MagicMock) -> None:
        """Test confidence_threshold property."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.75,
        )

        assert stage.confidence_threshold == 0.75

    def test_yolo_stage_model_loading(self, mock_backend: MagicMock) -> None:
        """Test that YOLOStage calls backend.load_model during init."""
        _stage = YOLOStage(
            model_path="test_model.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        mock_backend.load_model.assert_called_once_with("test_model.onnx")

    def test_yolo_stage_box_attributes(self, mock_backend: MagicMock) -> None:
        """Test that BoundingBox attributes are set correctly."""
        stage = YOLOStage(
            model_path="dummy.onnx",
            backend=mock_backend,
            confidence_threshold=0.5,
        )

        boxes = np.array([[[100, 150, 300, 350]]], dtype=np.float32)
        scores = np.array([[0.95]], dtype=np.float32)
        class_ids = np.array([[0]], dtype=np.uint8)

        outputs = [boxes, scores, class_ids]
        # Original size (480, 640) means scale factors are (640/640=1.0, 480/640=0.75)
        result = stage._parse_outputs(outputs, (480, 640))

        assert len(result) == 1
        box = result[0]
        assert box.x1 == pytest.approx(100.0)
        # y1 scaled from 150 in 640x640 space to original 480 height: 150 * 480/640 = 112.5
        assert box.y1 == pytest.approx(112.5)
        assert box.x2 == pytest.approx(300.0)
        # y2 scaled: 350 * 480/640 = 262.5
        assert box.y2 == pytest.approx(262.5)
        assert box.confidence == pytest.approx(0.95)
        assert box.class_id == 0
        assert box.label == "person"
