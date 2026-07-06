"""Unit tests for DetectionAggregationStage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from moment_to_action.messages.control import EndOfClipMessage
from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.models.image.detection._types import BoundingBox, Detection
from moment_to_action.stages._base import Stage
from moment_to_action.stages.image._aggregation import DetectionAggregationStage

if TYPE_CHECKING:
    from moment_to_action.messages import Message

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

    def test_emits_nothing_until_end_of_clip(self) -> None:
        """Per-frame DetectionMessages alone produce no output."""
        stage = DetectionAggregationStage()
        messages: list[Message] = [
            _detection_msg([_det("a", 0.1)]),
            _detection_msg([_det("b", 0.2)]),
        ]

        results = list(stage.process(iter(messages)))

        assert results == []

    def test_keeps_highest_confidence_per_label(self) -> None:
        """Across a clip, only the highest-confidence detection per label survives."""
        stage = DetectionAggregationStage()
        messages: list[Message] = [
            _detection_msg([_det("person", 0.5)]),
            _detection_msg([_det("person", 0.9), _det("dog", 0.4)]),
            _detection_msg([_det("dog", 0.2)]),
            EndOfClipMessage(timestamp=0.0),
        ]

        (result,) = list(stage.process(iter(messages)))

        assert isinstance(result, DetectionMessage)
        by_label = {d.label: d.confidence for d in result.detections}
        assert by_label == {"person": 0.9, "dog": 0.4}

    def test_emits_exactly_one_message_per_clip(self) -> None:
        """A clip of N detection messages + one EndOfClipMessage emits exactly one message."""
        stage = DetectionAggregationStage()
        messages: list[Message] = [_detection_msg([_det("a", 0.1 * i)]) for i in range(5)]
        messages.append(EndOfClipMessage(timestamp=0.0))

        results = list(stage.process(iter(messages)))

        assert len(results) == 1

    def test_timestamp_and_question_from_clip(self) -> None:
        """Output timestamp comes from the EndOfClipMessage; question from the last frame seen."""
        stage = DetectionAggregationStage()
        messages: list[Message] = [
            _detection_msg([], timestamp=1.0, question="first?"),
            _detection_msg([], timestamp=2.0, question="second?"),
            EndOfClipMessage(timestamp=3.0),
        ]

        (result,) = list(stage.process(iter(messages)))

        assert isinstance(result, DetectionMessage)
        assert result.timestamp == 3.0
        assert result.question == "second?"

    def test_empty_detections_in_every_frame_yields_empty(self) -> None:
        """No detections across the whole clip aggregates to an empty list."""
        stage = DetectionAggregationStage()
        messages: list[Message] = [
            _detection_msg([]),
            _detection_msg([]),
            EndOfClipMessage(timestamp=0.0),
        ]

        (result,) = list(stage.process(iter(messages)))

        assert isinstance(result, DetectionMessage)
        assert result.detections == []

    def test_single_frame_clip(self) -> None:
        """A one-frame clip passes through that frame's detections unchanged (per label)."""
        stage = DetectionAggregationStage()
        messages: list[Message] = [
            _detection_msg([_det("cat", 0.7)]),
            EndOfClipMessage(timestamp=0.0),
        ]

        (result,) = list(stage.process(iter(messages)))

        assert isinstance(result, DetectionMessage)
        assert len(result.detections) == 1
        assert result.detections[0].label == "cat"
        assert result.detections[0].confidence == 0.7

    def test_resets_accumulation_after_each_clip(self) -> None:
        """One stage instance correctly aggregates multiple sequential clips."""
        stage = DetectionAggregationStage()
        messages: list[Message] = [
            _detection_msg([_det("person", 0.9)]),
            EndOfClipMessage(timestamp=1.0),
            _detection_msg([_det("dog", 0.3)]),
            EndOfClipMessage(timestamp=2.0),
        ]

        results = list(stage.process(iter(messages)))

        assert len(results) == 2
        assert isinstance(results[0], DetectionMessage)
        assert isinstance(results[1], DetectionMessage)
        assert [d.label for d in results[0].detections] == ["person"]
        assert [d.label for d in results[1].detections] == ["dog"]
