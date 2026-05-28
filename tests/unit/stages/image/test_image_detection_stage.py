"""Unit tests for ImageDetectionStage."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.messages.video import FrameTensorMessage
from moment_to_action.models.image.detection._base import ImageDetectionModel
from moment_to_action.models.image.detection._types import BoundingBox, Detection
from moment_to_action.stages.image._base import ImageStage
from moment_to_action.stages.image._detection import ImageDetectionStage


@pytest.mark.unit
class TestImageDetectionStage:
    """Tests for ImageDetectionStage."""

    @pytest.fixture
    def sample_detection(self) -> Detection:
        """Return a single Detection for use in mock returns."""
        return Detection(
            label="person",
            confidence=0.9,
            bbox=BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0),
        )

    @pytest.fixture
    def mock_model(self, sample_detection: Detection) -> MagicMock:
        """Return a mock ImageDetectionModel with reasonable defaults."""
        model = MagicMock(spec=ImageDetectionModel)
        model.prepare.return_value = np.zeros((1, 3, 640, 640), dtype=np.float32)
        model.run.return_value = [
            np.zeros((1, 1, 4), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.uint8),
        ]
        model.post_proc.return_value = [sample_detection]
        return model

    @pytest.fixture
    def raw_frame_msg(self) -> RawFrameMessage:
        """Return a RawFrameMessage with a valid frame."""
        return RawFrameMessage(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=1234.5,
            width=640,
            height=480,
        )

    @pytest.fixture
    def dropped_frame_msg(self) -> RawFrameMessage:
        """Return a RawFrameMessage with frame=None (dropped frame)."""
        return RawFrameMessage(
            frame=None,
            timestamp=1234.5,
            width=640,
            height=480,
        )

    def test_is_subclass_of_image_stage(self) -> None:
        """ImageDetectionStage must extend ImageStage."""
        assert issubclass(ImageDetectionStage, ImageStage)

    def test_stage_name(self, mock_model: MagicMock) -> None:
        """Stage name must be the class name."""
        stage = ImageDetectionStage(model=mock_model)
        assert stage.name == "ImageDetectionStage"

    def test_happy_path_returns_detection_message(
        self,
        mock_model: MagicMock,
        raw_frame_msg: RawFrameMessage,
        sample_detection: Detection,
    ) -> None:
        """Valid RawFrameMessage with frame → DetectionMessage with expected detections."""
        stage = ImageDetectionStage(model=mock_model)
        result = stage.process(raw_frame_msg)

        assert isinstance(result, DetectionMessage)
        assert len(result.detections) == 1
        assert result.detections[0].label == sample_detection.label
        assert result.detections[0].confidence == pytest.approx(sample_detection.confidence)

    def test_happy_path_calls_model_methods_in_order(
        self,
        mock_model: MagicMock,
        raw_frame_msg: RawFrameMessage,
    ) -> None:
        """Verify prepare → run → post_proc call chain is correct."""
        stage = ImageDetectionStage(model=mock_model)
        stage.process(raw_frame_msg)

        mock_model.prepare.assert_called_once()
        assert raw_frame_msg.frame is not None
        np.testing.assert_array_equal(mock_model.prepare.call_args[0][0], raw_frame_msg.frame)
        mock_model.run.assert_called_once_with(mock_model.prepare.return_value)
        mock_model.post_proc.assert_called_once_with(mock_model.run.return_value)

    def test_wrong_message_type_returns_none_and_warns(
        self, mock_model: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-RawFrameMessage input must return None and log a warning."""
        import logging

        stage = ImageDetectionStage(model=mock_model)
        wrong_msg = FrameTensorMessage(
            tensor=np.zeros((1, 3, 640, 640), dtype=np.float32),
            original_size=(640, 480),
            timestamp=time.time(),
        )
        with caplog.at_level(logging.WARNING):
            result = stage.process(wrong_msg)

        assert result is None
        mock_model.prepare.assert_not_called()
        assert any("RawFrameMessage" in r.message for r in caplog.records)

    def test_dropped_frame_returns_none(
        self,
        mock_model: MagicMock,
        dropped_frame_msg: RawFrameMessage,
    ) -> None:
        """RawFrameMessage with frame=None must return None without calling model."""
        stage = ImageDetectionStage(model=mock_model)
        result = stage.process(dropped_frame_msg)

        assert result is None
        mock_model.prepare.assert_not_called()

    def test_empty_detections_returns_detection_message(
        self,
        mock_model: MagicMock,
        raw_frame_msg: RawFrameMessage,
    ) -> None:
        """Model returning empty list → DetectionMessage with empty detections list."""
        mock_model.post_proc.return_value = []
        stage = ImageDetectionStage(model=mock_model)
        result = stage.process(raw_frame_msg)

        assert isinstance(result, DetectionMessage)
        assert result.detections == []

    def test_metrics_spans_recorded(
        self,
        mock_model: MagicMock,
        raw_frame_msg: RawFrameMessage,
    ) -> None:
        """Happy path records prepare, run, and post_proc spans in metrics."""
        from moment_to_action.metrics import MetricsCollector, SpanType

        stage = ImageDetectionStage(model=mock_model)
        metrics = MetricsCollector(session_id="test_detection_spans")
        with metrics.start_trace():
            stage.process(raw_frame_msg, metrics=metrics)

        span_names = {s.name for s in metrics.spans}
        span_types = {s.type_ for s in metrics.spans}
        assert "prepare" in span_names
        assert "run" in span_names
        assert "post_process" in span_names
        assert SpanType.MODEL_PREPROCESS in span_types
        assert SpanType.MODEL_INFERENCE in span_types
        assert SpanType.MODEL_POST_PROCESS in span_types

    def test_timestamp_preserved(
        self,
        mock_model: MagicMock,
        raw_frame_msg: RawFrameMessage,
    ) -> None:
        """Output DetectionMessage.timestamp must match input message timestamp."""
        stage = ImageDetectionStage(model=mock_model)
        result = stage.process(raw_frame_msg)

        assert isinstance(result, DetectionMessage)
        assert result.timestamp == raw_frame_msg.timestamp
