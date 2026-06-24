"""Unit tests for pipeline messages."""

from __future__ import annotations

import time

import numpy as np
import pytest

from moment_to_action.messages import DetectionMessage
from moment_to_action.messages.llm import ReasoningMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.messages.video import FrameTensorMessage, VideoClipMessage
from moment_to_action.messages.vlm import ClassificationMessage
from moment_to_action.models.image.detection._types import BoundingBox, Detection


def _make_detection(
    label: str = "person",
    confidence: float = 0.9,
    x1: float = 0.0,
    y1: float = 0.0,
    x2: float = 100.0,
    y2: float = 100.0,
) -> Detection:
    """Build a Detection with given fields."""
    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return Detection(label=label, confidence=confidence, bbox=bbox)


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
class TestDetectionMessage:
    """Tests for DetectionMessage."""

    def test_construction_with_detections(self) -> None:
        """DetectionMessage stores a list of Detection objects."""
        detections = [_make_detection("person", 0.9), _make_detection("car", 0.8)]
        msg = DetectionMessage(timestamp=time.time(), detections=detections)
        assert len(msg.detections) == 2
        assert msg.detections[0].label == "person"
        assert msg.detections[1].label == "car"

    def test_construction_empty_detections(self) -> None:
        """DetectionMessage with empty list is valid."""
        msg = DetectionMessage(timestamp=time.time(), detections=[])
        assert msg.detections == []

    def test_detection_confidence_preserved(self) -> None:
        """DetectionMessage preserves detection confidence values."""
        d = _make_detection("dog", 0.77)
        msg = DetectionMessage(timestamp=time.time(), detections=[d])
        assert msg.detections[0].confidence == pytest.approx(0.77)

    def test_detection_bbox_preserved(self) -> None:
        """DetectionMessage preserves bounding box coordinates."""
        d = _make_detection(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        msg = DetectionMessage(timestamp=time.time(), detections=[d])
        box = msg.detections[0].bbox
        assert box.x1 == pytest.approx(10.0)
        assert box.y1 == pytest.approx(20.0)
        assert box.x2 == pytest.approx(100.0)
        assert box.y2 == pytest.approx(200.0)

    def test_isinstance_message(self) -> None:
        """DetectionMessage is a member of the Message union."""
        from moment_to_action.messages import Message

        msg = DetectionMessage(timestamp=time.time(), detections=[])
        assert isinstance(msg, Message.__args__)

    def test_has_latency_ms(self) -> None:
        """DetectionMessage inherits latency_ms from BaseMessage."""
        msg = DetectionMessage(timestamp=time.time(), detections=[])
        assert msg.latency_ms == 0.0


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
        detections = [_make_detection("cat", 0.9)]
        msg = DetectionMessage(timestamp=timestamp, detections=detections)
        updated_msg = msg.model_copy(update={"latency_ms": 25.0})
        assert updated_msg.latency_ms == 25.0
        assert len(updated_msg.detections) == 1

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


@pytest.mark.unit
class TestVideoClipMessage:
    """Tests for VideoClipMessage."""

    def test_num_frames_property(self) -> None:
        """num_frames returns the number of frames in the clip."""
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        msg = VideoClipMessage(timestamp=time.time(), frames=frames)
        assert msg.num_frames == 5

    def test_num_frames_empty_clip(self) -> None:
        """num_frames returns 0 for an empty clip."""
        msg = VideoClipMessage(timestamp=time.time(), frames=[])
        assert msg.num_frames == 0
