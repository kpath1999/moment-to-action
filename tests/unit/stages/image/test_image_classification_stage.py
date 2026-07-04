"""Unit tests for ImageClassificationStage."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages._image_classification import ImageClassificationMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.messages.video import FrameTensorMessage
from moment_to_action.models.image.classification._base import ImageClassificationModel
from moment_to_action.models.image.classification._types import Classification
from moment_to_action.stages.image._base import ImageStage
from moment_to_action.stages.image._classification import ImageClassificationStage


@pytest.mark.unit
class TestImageClassificationStage:
    """Tests for ImageClassificationStage."""

    @pytest.fixture
    def sample_classification(self) -> Classification:
        """Return a single Classification for use in mock returns."""
        return Classification(label="tench", confidence=0.95, class_id=0)

    @pytest.fixture
    def mock_model(self, sample_classification: Classification) -> MagicMock:
        """Return a mock ImageClassificationModel with reasonable defaults."""
        model = MagicMock(spec=ImageClassificationModel)
        model.prepare.return_value = np.zeros((1, 3, 224, 224), dtype=np.float32)
        model.run.return_value = [np.zeros((1, 1000), dtype=np.float32)]
        model.post_proc.return_value = [sample_classification]
        return model

    @pytest.fixture
    def raw_frame_msg(self) -> RawFrameMessage:
        """Return a RawFrameMessage with a valid frame."""
        return RawFrameMessage(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=time.time(),
        )

    def test_is_image_stage_subclass(self, mock_model: MagicMock) -> None:
        """ImageClassificationStage is a subclass of ImageStage."""
        stage = ImageClassificationStage(model=mock_model)
        assert isinstance(stage, ImageStage)

    def test_returns_classification_message(
        self, mock_model: MagicMock, raw_frame_msg: RawFrameMessage
    ) -> None:
        """_process returns an ImageClassificationMessage for a valid frame."""
        stage = ImageClassificationStage(model=mock_model)
        (result,) = list(stage.process(iter([raw_frame_msg])))
        assert isinstance(result, ImageClassificationMessage)

    def test_classifications_forwarded(
        self,
        mock_model: MagicMock,
        raw_frame_msg: RawFrameMessage,
        sample_classification: Classification,
    ) -> None:
        """Returned message contains the classifications from post_proc."""
        stage = ImageClassificationStage(model=mock_model)
        (result,) = list(stage.process(iter([raw_frame_msg])))
        assert isinstance(result, ImageClassificationMessage)
        assert result.classifications == [sample_classification]

    def test_calls_prepare_run_post_proc(
        self, mock_model: MagicMock, raw_frame_msg: RawFrameMessage
    ) -> None:
        """_process calls prepare, run, and post_proc in order."""
        stage = ImageClassificationStage(model=mock_model)
        list(stage.process(iter([raw_frame_msg])))
        mock_model.prepare.assert_called_once()
        mock_model.run.assert_called_once()
        mock_model.post_proc.assert_called_once()

    def test_non_frame_message_yields_nothing(self, mock_model: MagicMock) -> None:
        """_process yields nothing for non-RawFrameMessage input."""
        stage = ImageClassificationStage(model=mock_model)
        other_msg = FrameTensorMessage(
            tensor=np.zeros((1, 3, 224, 224), dtype=np.float32),
            original_size=(640, 480),
            timestamp=time.time(),
        )
        results = list(stage.process(iter([other_msg])))
        assert results == []

    def test_none_frame_yields_nothing(self, mock_model: MagicMock) -> None:
        """_process yields nothing when RawFrameMessage.frame is None."""
        stage = ImageClassificationStage(model=mock_model)
        msg = RawFrameMessage(frame=None, timestamp=time.time())
        results = list(stage.process(iter([msg])))
        assert results == []

    def test_timestamp_preserved(
        self, mock_model: MagicMock, raw_frame_msg: RawFrameMessage
    ) -> None:
        """Output message timestamp matches input message timestamp."""
        stage = ImageClassificationStage(model=mock_model)
        (result,) = list(stage.process(iter([raw_frame_msg])))
        assert isinstance(result, ImageClassificationMessage)
        assert result.timestamp == raw_frame_msg.timestamp

    def test_load_calls_model_load(self, mock_model: MagicMock) -> None:
        """load() forwards platform and unit to the wrapped model."""
        stage = ImageClassificationStage(model=mock_model)
        platform = MagicMock()
        stage.load(platform, ComputeUnit.CPU)
        mock_model.load.assert_called_once_with(platform, ComputeUnit.CPU)

    def test_load_without_unit_raises(self, mock_model: MagicMock) -> None:
        """load() without a compute unit raises ValueError."""
        stage = ImageClassificationStage(model=mock_model)
        with pytest.raises(ValueError, match="compute unit"):
            stage.load(MagicMock())

    def test_unload_calls_model_unload(self, mock_model: MagicMock) -> None:
        """unload() delegates to the wrapped model."""
        stage = ImageClassificationStage(model=mock_model)
        stage.unload()
        mock_model.unload.assert_called_once()
