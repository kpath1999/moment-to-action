"""Unit tests for ImageDetectionStage."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from moment_to_action.hardware import ComputeUnit, DataType, ModelType
from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.messages.video import FrameTensorMessage
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models.image.detection._base import ImageDetectionModel
from moment_to_action.models.image.detection._types import BoundingBox, Detection
from moment_to_action.stages.image._base import ImageStage
from moment_to_action.stages.image._detection import ImageDetectionStage


class _StubDetectionModel(ImageDetectionModel):
    """Minimal concrete detection model for span-recording tests."""

    def __init__(self, *, metrics: MetricsCollector | None = None) -> None:
        """Initialize with fixed output."""
        super().__init__(
            "default",
            Path("/x"),
            ModelType.ONNX,
            DataType.FP32,
            backends={},
            metrics=metrics,
        )
        self._platform = MagicMock()

    def _load(self, platform: object, unit: object) -> None:
        """No-op load."""

    def _unload(self) -> None:
        """No-op unload."""

    def _prepare(self, inputs: np.ndarray) -> np.ndarray:
        """Pass through."""
        return inputs

    def _run(self, prepared: np.ndarray) -> list[np.ndarray]:
        """Return dummy output."""
        return [np.zeros((1, 1, 4)), np.zeros((1, 1)), np.zeros((1, 1), dtype=np.uint8)]

    def _post_proc(self, raw: list[np.ndarray]) -> list[Detection]:
        """Return fixed detection."""
        return [Detection("person", 0.9, BoundingBox(10, 20, 100, 200))]

    def verify_outputs(
        self,
        inputs: np.ndarray,
        ref_outputs: list[np.ndarray],
        *,
        tol: float,
        is_npu: bool,
    ) -> tuple[bool, str]:
        """Always pass."""
        return True, ""


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
        (result,) = list(stage.process(iter([raw_frame_msg])))

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
        list(stage.process(iter([raw_frame_msg])))

        mock_model.prepare.assert_called_once()
        assert raw_frame_msg.frame is not None
        np.testing.assert_array_equal(mock_model.prepare.call_args[0][0], raw_frame_msg.frame)
        assert mock_model.run.call_args[0][0] is mock_model.prepare.return_value
        assert mock_model.post_proc.call_args[0][0] is mock_model.run.return_value

    def test_wrong_message_type_yields_nothing(self, mock_model: MagicMock) -> None:
        """Non-RawFrameMessage input must be dropped before it reaches _process."""
        stage = ImageDetectionStage(model=mock_model)
        wrong_msg = FrameTensorMessage(
            tensor=np.zeros((1, 3, 640, 640), dtype=np.float32),
            original_size=(640, 480),
            timestamp=time.time(),
        )
        results = list(stage.process(iter([wrong_msg])))

        assert results == []
        mock_model.prepare.assert_not_called()

    def test_dropped_frame_yields_nothing(
        self,
        mock_model: MagicMock,
        dropped_frame_msg: RawFrameMessage,
    ) -> None:
        """RawFrameMessage with frame=None must be dropped without calling model."""
        stage = ImageDetectionStage(model=mock_model)
        results = list(stage.process(iter([dropped_frame_msg])))

        assert results == []
        mock_model.prepare.assert_not_called()

    def test_empty_detections_returns_detection_message(
        self,
        mock_model: MagicMock,
        raw_frame_msg: RawFrameMessage,
    ) -> None:
        """Model returning empty list → DetectionMessage with empty detections list."""
        mock_model.post_proc.return_value = []
        stage = ImageDetectionStage(model=mock_model)
        (result,) = list(stage.process(iter([raw_frame_msg])))

        assert isinstance(result, DetectionMessage)
        assert result.detections == []

    def test_metrics_spans_recorded(
        self,
        raw_frame_msg: RawFrameMessage,
    ) -> None:
        """Happy path records prepare, run, and post_proc spans in metrics."""
        from moment_to_action.metrics import MetricsCollector, SpanType

        metrics = MetricsCollector(session_id="test_detection_spans")
        stub = _StubDetectionModel(metrics=metrics)
        stage = ImageDetectionStage(model=stub, metrics=metrics)
        with metrics.start_trace():
            list(stage.process(iter([raw_frame_msg])))

        span_names = {s.name for s in metrics.spans}
        span_types = {s.type_ for s in metrics.spans}
        assert "_StubDetectionModel.prepare" in span_names
        assert "_StubDetectionModel.run" in span_names
        assert "_StubDetectionModel.post_proc" in span_names
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
        (result,) = list(stage.process(iter([raw_frame_msg])))

        assert isinstance(result, DetectionMessage)
        assert result.timestamp == raw_frame_msg.timestamp

    def test_question_passed_through(
        self,
        mock_model: MagicMock,
    ) -> None:
        """Output DetectionMessage.question must match the input message's question."""
        msg = RawFrameMessage(
            frame=np.zeros((480, 640, 3), dtype=np.uint8), timestamp=1.0, question="Is this safe?"
        )
        stage = ImageDetectionStage(model=mock_model)
        (result,) = list(stage.process(iter([msg])))

        assert isinstance(result, DetectionMessage)
        assert result.question == "Is this safe?"

    def test_load_calls_model_load(self, mock_model: MagicMock) -> None:
        """load() forwards platform and unit to the wrapped model."""
        stage = ImageDetectionStage(model=mock_model)
        platform = MagicMock()
        stage.load(platform, ComputeUnit.CPU)
        mock_model.load.assert_called_once_with(platform, ComputeUnit.CPU)

    def test_load_without_unit_raises(self, mock_model: MagicMock) -> None:
        """load() without a compute unit raises ValueError."""
        stage = ImageDetectionStage(model=mock_model)
        with pytest.raises(ValueError, match="compute unit"):
            stage.load(MagicMock())

    def test_unload_calls_model_unload(self, mock_model: MagicMock) -> None:
        """unload() delegates to the wrapped model."""
        stage = ImageDetectionStage(model=mock_model)
        stage.unload()
        mock_model.unload.assert_called_once()
