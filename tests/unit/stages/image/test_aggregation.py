"""Unit tests for DetectionAggregationStage."""

from __future__ import annotations

import pytest

from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.models.image.detection._types import BoundingBox, Detection
from moment_to_action.stages._base import Stage
from moment_to_action.stages.image._aggregation import DetectionAggregationStage

_BOX = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)


def _det(label: str, confidence: float) -> Detection:
    """Build a Detection with a fixed dummy bounding box."""
    return Detection(label=label, confidence=confidence, bbox=_BOX)


def _detection_msg(
    detections: list[Detection], timestamp: float = 0.0, question: str = ""
) -> DetectionMessage:
    """Build a DetectionMessage for testing."""
    return DetectionMessage(timestamp=timestamp, detections=detections, question=question)


@pytest.mark.unit
class TestDetectionAggregationStage:
    """Tests for DetectionAggregationStage."""

    def test_is_stage_subclass(self) -> None:
        """DetectionAggregationStage extends the base Stage."""
        assert issubclass(DetectionAggregationStage, Stage)

    def test_keeps_highest_confidence_per_label(self) -> None:
        """Across the window, only the highest-confidence detection per label survives."""
        stage = DetectionAggregationStage(window=3)
        messages = [
            _detection_msg([_det("person", 0.5)]),
            _detection_msg([_det("person", 0.9), _det("dog", 0.4)]),
            _detection_msg([_det("dog", 0.2)]),
        ]

        (result,) = list(stage.process(iter(messages)))

        assert isinstance(result, DetectionMessage)
        by_label = {d.label: d.confidence for d in result.detections}
        assert by_label == {"person": 0.9, "dog": 0.4}

    def test_emits_once_for_whole_window(self) -> None:
        """A window of N messages emits exactly one aggregated message."""
        stage = DetectionAggregationStage(window=5)
        messages = [_detection_msg([_det("a", 0.1 * i)]) for i in range(5)]

        results = list(stage.process(iter(messages)))

        assert len(results) == 1

    def test_timestamp_and_question_from_last_message(self) -> None:
        """Output timestamp/question come from the last buffered message."""
        stage = DetectionAggregationStage(window=2)
        messages = [
            _detection_msg([], timestamp=1.0, question="first?"),
            _detection_msg([], timestamp=2.0, question="second?"),
        ]

        (result,) = list(stage.process(iter(messages)))

        assert isinstance(result, DetectionMessage)
        assert result.timestamp == 2.0
        assert result.question == "second?"

    def test_empty_detections_in_every_frame_yields_empty(self) -> None:
        """No detections across the whole window aggregates to an empty list."""
        stage = DetectionAggregationStage(window=2)
        messages = [_detection_msg([]), _detection_msg([])]

        (result,) = list(stage.process(iter(messages)))

        assert isinstance(result, DetectionMessage)
        assert result.detections == []

    def test_single_frame_window(self) -> None:
        """window=1 passes through a single frame's detections unchanged (per label)."""
        stage = DetectionAggregationStage(window=1)
        messages = [_detection_msg([_det("cat", 0.7)])]

        (result,) = list(stage.process(iter(messages)))

        assert isinstance(result, DetectionMessage)
        assert len(result.detections) == 1
        assert result.detections[0].label == "cat"
        assert result.detections[0].confidence == 0.7
