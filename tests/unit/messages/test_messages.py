"""Unit tests for pipeline messages."""

from __future__ import annotations

import time

import numpy as np
import pytest

from moment_to_action.messages.llm import ReasoningMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.messages.video import BoundingBox, DetectionMessage, FrameTensorMessage
from moment_to_action.messages.vlm import ClassificationMessage


@pytest.mark.unit
class TestRawFrameMessage:
    """Tests for RawFrameMessage."""

    def test_rawframe_construction_with_frame(self, sample_image_array: np.ndarray) -> None:
        """Test RawFrameMessage construction with a valid frame array."""
        timestamp = time.time()
        msg = RawFrameMessage(
            timestamp=timestamp,
            frame=sample_image_array,
            source="cam0",
            width=640,
            height=480,
        )
        assert msg.timestamp == timestamp
        assert msg.frame is not None
        assert msg.frame.shape == (480, 640, 3)
        assert msg.source == "cam0"
        assert msg.width == 640
        assert msg.height == 480

    def test_rawframe_construction_with_none_frame(self) -> None:
        """Test RawFrameMessage construction with None frame (dropped frame)."""
        timestamp = time.time()
        msg = RawFrameMessage(
            timestamp=timestamp,
            frame=None,
            source="cam1",
            width=1920,
            height=1080,
        )
        assert msg.frame is None
        assert msg.source == "cam1"
        assert msg.width == 1920
        assert msg.height == 1080

    def test_rawframe_default_values(self, sample_image_array: np.ndarray) -> None:
        """Test RawFrameMessage default values for optional fields."""
        timestamp = time.time()
        msg = RawFrameMessage(
            timestamp=timestamp,
            frame=sample_image_array,
        )
        assert msg.source == ""
        assert msg.width == 0
        assert msg.height == 0

    def test_rawframe_field_access(self, sample_image_array: np.ndarray) -> None:
        """Test field access on RawFrameMessage."""
        timestamp = time.time()
        msg = RawFrameMessage(
            timestamp=timestamp,
            frame=sample_image_array,
            source="cam_test",
            width=1280,
            height=720,
        )
        assert hasattr(msg, "timestamp")
        assert hasattr(msg, "frame")
        assert hasattr(msg, "source")
        assert hasattr(msg, "width")
        assert hasattr(msg, "height")
        assert hasattr(msg, "latency_ms")


@pytest.mark.unit
class TestFrameTensorMessage:
    """Tests for FrameTensorMessage."""

    def test_frametensor_construction(self, sample_frame_tensor: np.ndarray) -> None:
        """Test FrameTensorMessage construction with valid tensor."""
        timestamp = time.time()
        msg = FrameTensorMessage(
            timestamp=timestamp,
            tensor=sample_frame_tensor,
            original_size=(640, 480),
        )
        assert msg.timestamp == timestamp
        assert msg.tensor.shape == (1, 3, 256, 256)
        assert msg.original_size == (640, 480)

    def test_frametensor_with_different_tensor_size(self) -> None:
        """Test FrameTensorMessage with different tensor dimensions."""
        timestamp = time.time()
        tensor = np.random.randn(4, 3, 512, 512).astype(np.float32)  # noqa: NPY002
        msg = FrameTensorMessage(
            timestamp=timestamp,
            tensor=tensor,
            original_size=(1920, 1080),
        )
        assert msg.tensor.shape == (4, 3, 512, 512)
        assert msg.original_size == (1920, 1080)

    def test_frametensor_field_access(self, sample_frame_tensor: np.ndarray) -> None:
        """Test field access on FrameTensorMessage."""
        timestamp = time.time()
        msg = FrameTensorMessage(
            timestamp=timestamp,
            tensor=sample_frame_tensor,
            original_size=(800, 600),
        )
        assert hasattr(msg, "timestamp")
        assert hasattr(msg, "tensor")
        assert hasattr(msg, "original_size")
        assert hasattr(msg, "latency_ms")


@pytest.mark.unit
class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_boundingbox_construction(self) -> None:
        """Test BoundingBox construction with valid values."""
        box = BoundingBox(
            x1=100.0,
            y1=150.0,
            x2=500.0,
            y2=600.0,
            confidence=0.95,
            class_id=0,
            label="person",
        )
        assert box.x1 == 100.0
        assert box.y1 == 150.0
        assert box.x2 == 500.0
        assert box.y2 == 600.0
        assert box.confidence == 0.95
        assert box.class_id == 0
        assert box.label == "person"

    def test_boundingbox_field_types(self) -> None:
        """Test BoundingBox field types."""
        box = BoundingBox(
            x1=50.5,
            y1=75.25,
            x2=450.75,
            y2=550.5,
            confidence=0.87,
            class_id=5,
            label="dog",
        )
        assert isinstance(box.x1, float)
        assert isinstance(box.y1, float)
        assert isinstance(box.x2, float)
        assert isinstance(box.y2, float)
        assert isinstance(box.confidence, float)
        assert isinstance(box.class_id, int)
        assert isinstance(box.label, str)

    def test_boundingbox_with_zero_confidence(self) -> None:
        """Test BoundingBox with zero confidence."""
        box = BoundingBox(
            x1=0.0,
            y1=0.0,
            x2=100.0,
            y2=100.0,
            confidence=0.0,
            class_id=0,
            label="unknown",
        )
        assert box.confidence == 0.0

    def test_boundingbox_with_full_confidence(self) -> None:
        """Test BoundingBox with full confidence."""
        box = BoundingBox(
            x1=0.0,
            y1=0.0,
            x2=100.0,
            y2=100.0,
            confidence=1.0,
            class_id=1,
            label="car",
        )
        assert box.confidence == 1.0


@pytest.mark.unit
class TestDetectionMessage:
    """Tests for DetectionMessage."""

    def test_detection_construction_with_boxes(self) -> None:
        """Test DetectionMessage construction with bounding boxes."""
        timestamp = time.time()
        boxes = [
            BoundingBox(
                x1=100.0,
                y1=150.0,
                x2=500.0,
                y2=600.0,
                confidence=0.95,
                class_id=0,
                label="person",
            ),
            BoundingBox(
                x1=550.0,
                y1=200.0,
                x2=750.0,
                y2=650.0,
                confidence=0.88,
                class_id=2,
                label="car",
            ),
        ]
        msg = DetectionMessage(timestamp=timestamp, boxes=boxes)
        assert msg.timestamp == timestamp
        assert len(msg.boxes) == 2
        assert msg.boxes[0].label == "person"
        assert msg.boxes[1].label == "car"

    def test_detection_construction_empty_boxes(self) -> None:
        """Test DetectionMessage construction with empty box list."""
        timestamp = time.time()
        msg = DetectionMessage(timestamp=timestamp, boxes=[])
        assert msg.timestamp == timestamp
        assert len(msg.boxes) == 0
        assert msg.boxes == []

    def test_detection_has_detections_true(self) -> None:
        """Test has_detections property returns True when boxes exist."""
        timestamp = time.time()
        boxes = [
            BoundingBox(
                x1=0.0,
                y1=0.0,
                x2=100.0,
                y2=100.0,
                confidence=0.9,
                class_id=0,
                label="object",
            )
        ]
        msg = DetectionMessage(timestamp=timestamp, boxes=boxes)
        assert msg.has_detections is True

    def test_detection_has_detections_false(self) -> None:
        """Test has_detections property returns False when no boxes."""
        timestamp = time.time()
        msg = DetectionMessage(timestamp=timestamp, boxes=[])
        assert msg.has_detections is False

    def test_detection_top_method(self) -> None:
        """Test top() method returns highest confidence boxes."""
        timestamp = time.time()
        boxes = [
            BoundingBox(
                x1=0.0,
                y1=0.0,
                x2=100.0,
                y2=100.0,
                confidence=0.7,
                class_id=0,
                label="low",
            ),
            BoundingBox(
                x1=100.0,
                y1=100.0,
                x2=200.0,
                y2=200.0,
                confidence=0.95,
                class_id=1,
                label="high",
            ),
            BoundingBox(
                x1=200.0,
                y1=200.0,
                x2=300.0,
                y2=300.0,
                confidence=0.85,
                class_id=2,
                label="medium",
            ),
        ]
        msg = DetectionMessage(timestamp=timestamp, boxes=boxes)
        top_1 = msg.top(1)
        assert len(top_1) == 1
        assert top_1[0].label == "high"
        assert top_1[0].confidence == 0.95

    def test_detection_top_method_multiple(self) -> None:
        """Test top() method with multiple boxes."""
        timestamp = time.time()
        boxes = [
            BoundingBox(
                x1=0.0,
                y1=0.0,
                x2=100.0,
                y2=100.0,
                confidence=0.5,
                class_id=0,
                label="low",
            ),
            BoundingBox(
                x1=100.0,
                y1=100.0,
                x2=200.0,
                y2=200.0,
                confidence=0.99,
                class_id=1,
                label="highest",
            ),
            BoundingBox(
                x1=200.0,
                y1=200.0,
                x2=300.0,
                y2=300.0,
                confidence=0.9,
                class_id=2,
                label="high",
            ),
        ]
        msg = DetectionMessage(timestamp=timestamp, boxes=boxes)
        top_2 = msg.top(2)
        assert len(top_2) == 2
        assert top_2[0].confidence == 0.99
        assert top_2[1].confidence == 0.9


@pytest.mark.unit
class TestReasoningMessage:
    """Tests for ReasoningMessage."""

    def test_reasoning_construction(self) -> None:
        """Test ReasoningMessage construction."""
        timestamp = time.time()
        response = "The image shows a dog sitting on a bench."
        prompt = "Describe what you see in this image."
        msg = ReasoningMessage(
            timestamp=timestamp,
            response=response,
            prompt=prompt,
        )
        assert msg.timestamp == timestamp
        assert msg.response == response
        assert msg.prompt == prompt

    def test_reasoning_field_access(self) -> None:
        """Test field access on ReasoningMessage."""
        timestamp = time.time()
        msg = ReasoningMessage(
            timestamp=timestamp,
            response="Response text",
            prompt="Prompt text",
        )
        assert hasattr(msg, "timestamp")
        assert hasattr(msg, "response")
        assert hasattr(msg, "prompt")
        assert hasattr(msg, "latency_ms")

    def test_reasoning_with_long_response(self) -> None:
        """Test ReasoningMessage with long response text."""
        timestamp = time.time()
        long_response = "A" * 10000
        msg = ReasoningMessage(
            timestamp=timestamp,
            response=long_response,
            prompt="Short prompt",
        )
        assert len(msg.response) == 10000


@pytest.mark.unit
class TestClassificationMessage:
    """Tests for ClassificationMessage."""

    def test_classification_construction(self) -> None:
        """Test ClassificationMessage construction."""
        timestamp = time.time()
        all_scores = {
            "cat": 0.5,
            "dog": 0.35,
            "bird": 0.1,
            "fish": 0.05,
        }
        msg = ClassificationMessage(
            timestamp=timestamp,
            label="cat",
            confidence=0.5,
            all_scores=all_scores,
        )
        assert msg.timestamp == timestamp
        assert msg.label == "cat"
        assert msg.confidence == 0.5
        assert msg.all_scores == all_scores

    def test_classification_field_access(self) -> None:
        """Test field access on ClassificationMessage."""
        timestamp = time.time()
        msg = ClassificationMessage(
            timestamp=timestamp,
            label="dog",
            confidence=0.9,
            all_scores={"dog": 0.9, "cat": 0.1},
        )
        assert hasattr(msg, "timestamp")
        assert hasattr(msg, "label")
        assert hasattr(msg, "confidence")
        assert hasattr(msg, "all_scores")
        assert hasattr(msg, "latency_ms")

    def test_classification_full_distribution(self) -> None:
        """Test ClassificationMessage with full probability distribution."""
        timestamp = time.time()
        classes = ["apple", "banana", "orange", "grape", "kiwi"]
        scores = {cls: round(1.0 / len(classes), 3) for cls in classes}
        msg = ClassificationMessage(
            timestamp=timestamp,
            label="apple",
            confidence=scores["apple"],
            all_scores=scores,
        )
        assert len(msg.all_scores) == 5
        assert all(cls in msg.all_scores for cls in classes)

    def test_classification_with_zero_confidence(self) -> None:
        """Test ClassificationMessage with zero confidence."""
        timestamp = time.time()
        msg = ClassificationMessage(
            timestamp=timestamp,
            label="unknown",
            confidence=0.0,
            all_scores={"unknown": 0.0},
        )
        assert msg.confidence == 0.0


@pytest.mark.unit
class TestMessageModelCopy:
    """Tests for message model_copy and latency stamping."""

    def test_message_model_copy_for_latency(self, sample_image_array: np.ndarray) -> None:
        """Test model_copy for latency stamping on messages."""
        timestamp = time.time()
        msg = RawFrameMessage(
            timestamp=timestamp,
            frame=sample_image_array,
            source="cam0",
        )
        assert msg.latency_ms == 0.0

        # Simulate latency stamping via model_copy
        updated_msg = msg.model_copy(update={"latency_ms": 42.5})
        assert updated_msg.latency_ms == 42.5
        assert msg.latency_ms == 0.0  # original unchanged

    def test_detection_message_model_copy(self) -> None:
        """Test model_copy on DetectionMessage."""
        timestamp = time.time()
        boxes = [
            BoundingBox(
                x1=0.0,
                y1=0.0,
                x2=100.0,
                y2=100.0,
                confidence=0.9,
                class_id=0,
                label="test",
            )
        ]
        msg = DetectionMessage(timestamp=timestamp, boxes=boxes)
        updated_msg = msg.model_copy(update={"latency_ms": 25.0})
        assert updated_msg.latency_ms == 25.0
        assert len(updated_msg.boxes) == 1

    def test_reasoning_message_model_copy(self) -> None:
        """Test model_copy on ReasoningMessage."""
        timestamp = time.time()
        msg = ReasoningMessage(
            timestamp=timestamp,
            response="Test response",
            prompt="Test prompt",
        )
        updated_msg = msg.model_copy(update={"latency_ms": 150.5})
        assert updated_msg.latency_ms == 150.5
        assert updated_msg.response == "Test response"

    def test_classification_message_model_copy(self) -> None:
        """Test model_copy on ClassificationMessage."""
        timestamp = time.time()
        msg = ClassificationMessage(
            timestamp=timestamp,
            label="test",
            confidence=0.85,
            all_scores={"test": 0.85, "other": 0.15},
        )
        updated_msg = msg.model_copy(update={"latency_ms": 75.0})
        assert updated_msg.latency_ms == 75.0
        assert updated_msg.label == "test"
