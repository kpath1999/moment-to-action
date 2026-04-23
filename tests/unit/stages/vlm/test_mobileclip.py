"""Unit tests for the low-latency MobileCLIP stage."""

from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from moment_to_action.messages.video import FrameTensorMessage
from moment_to_action.messages.vlm import ClassificationMessage
from moment_to_action.models import ModelID, ModelManager
from moment_to_action.stages.vlm._mobileclip import MobileCLIPStage


def _unit_vector(index: int, size: int = 512) -> np.ndarray:
    """Create a unit vector with a single active dimension."""
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


@pytest.mark.unit
class TestMobileCLIPStage:
    """Tests for the restored cached-embedding MobileCLIP stage."""

    @pytest.fixture
    def mock_backend(self) -> mock.MagicMock:
        """Create a mocked ComputeBackend with a safe default embedding output."""
        backend = mock.MagicMock()
        backend.load_model.return_value = "mock_model_handle"
        backend.run.return_value = [
            np.ones((1, 512), dtype=np.float32),
            np.ones((1, 512), dtype=np.float32),
        ]
        return backend

    @pytest.fixture
    def mock_manager(self) -> mock.MagicMock:
        """Create a mock ModelManager that returns a fake model path."""
        manager = mock.MagicMock(spec=ModelManager)
        manager.get_path.return_value = Path("/fake/mobileclip.tflite")
        return manager

    @pytest.fixture
    def mock_tokenizer(self) -> mock.MagicMock:
        """Create a mocked tokenizer that returns token arrays."""
        tokenizer = mock.MagicMock()

        def tokenize_fn(prompts: list[str] | str) -> np.ndarray:
            if isinstance(prompts, list):
                return np.random.default_rng().integers(0, 1000, (len(prompts), 77), dtype=np.int64)
            return np.random.default_rng().integers(0, 1000, (1, 77), dtype=np.int64)

        tokenizer.side_effect = tokenize_fn
        return tokenizer

    @pytest.fixture
    def sample_frame_tensor(self) -> FrameTensorMessage:
        """Create a sample frame tensor message."""
        tensor = np.random.default_rng().standard_normal((1, 3, 256, 256)).astype(np.float32)
        return FrameTensorMessage(
            tensor=tensor,
            original_size=(640, 480),
            timestamp=time.time(),
        )

    @pytest.fixture
    def text_prompts(self) -> list[str]:
        """Sample text prompts for classification."""
        return ["person", "hand", "face"]

    def test_mobileclip_stage_initialization(
        self,
        mock_backend: mock.MagicMock,
        mock_tokenizer: mock.MagicMock,
        text_prompts: list[str],
        mock_manager: mock.MagicMock,
    ) -> None:
        """Initialization precomputes one cached text embedding per prompt."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                text_prompts=text_prompts,
                backend=mock_backend,
                manager=mock_manager,
            )

            assert stage._text_prompts == text_prompts
            assert len(stage._text_tokens) == len(text_prompts)
            assert stage._text_tokens.dtype == np.int64
            assert stage._text_embeddings.shape == (len(text_prompts), 512)
            mock_manager.get_path.assert_called_once_with(ModelID.MOBILECLIP_S2)
            mock_backend.load_model.assert_called_once_with(mock_manager.get_path.return_value)
            assert mock_backend.run.call_count == len(text_prompts)

    def test_mobileclip_zero_shot_classification_uses_cached_text_embeddings(
        self,
        mock_backend: mock.MagicMock,
        mock_tokenizer: mock.MagicMock,
        sample_frame_tensor: FrameTensorMessage,
        text_prompts: list[str],
        mock_manager: mock.MagicMock,
    ) -> None:
        """Process runs one image-side inference and scores against cached prompts."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                text_prompts=text_prompts,
                backend=mock_backend,
                manager=mock_manager,
            )

            stage._text_embeddings = np.stack(
                [_unit_vector(0), _unit_vector(1), _unit_vector(2)]
            ).astype(np.float32)
            mock_backend.run.reset_mock()
            mock_backend.run.return_value = [
                np.zeros((1, 512), dtype=np.float32),
                _unit_vector(1)[np.newaxis, :],
            ]

            result = stage.process(sample_frame_tensor)

            assert isinstance(result, ClassificationMessage)
            assert result.label == text_prompts[1]
            assert 0.0 <= result.confidence <= 1.0
            assert all(prompt in result.all_scores for prompt in text_prompts)
            assert sum(result.all_scores.values()) == pytest.approx(1.0, abs=0.01)

    def test_mobileclip_backend_called_correctly(
        self,
        mock_backend: mock.MagicMock,
        mock_tokenizer: mock.MagicMock,
        sample_frame_tensor: FrameTensorMessage,
        text_prompts: list[str],
        mock_manager: mock.MagicMock,
    ) -> None:
        """Processing performs one backend call using the cached first token set."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                text_prompts=text_prompts,
                backend=mock_backend,
                manager=mock_manager,
            )

            mock_backend.run.reset_mock()
            mock_backend.run.return_value = [
                np.zeros((1, 512), dtype=np.float32),
                _unit_vector(0)[np.newaxis, :],
            ]

            stage.process(sample_frame_tensor)

            assert mock_backend.run.call_count == 1
            inputs_dict = mock_backend.run.call_args.args[1]
            assert "serving_default_args_0:0" in inputs_dict
            assert "serving_default_args_1:0" in inputs_dict
            np.testing.assert_array_equal(
                inputs_dict["serving_default_args_0:0"], sample_frame_tensor.tensor
            )
            np.testing.assert_array_equal(
                inputs_dict["serving_default_args_1:0"],
                stage._text_tokens[0][np.newaxis, ...].astype(np.int64),
            )

    def test_update_prompts_swaps_without_reloading(
        self,
        mock_backend: mock.MagicMock,
        mock_tokenizer: mock.MagicMock,
        mock_manager: mock.MagicMock,
    ) -> None:
        """Updating prompts refreshes the cached text embeddings without reloading the model."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            initial_prompts = ["person", "hand"]
            stage = MobileCLIPStage(
                text_prompts=initial_prompts,
                backend=mock_backend,
                manager=mock_manager,
            )

            initial_load_count = mock_backend.load_model.call_count
            mock_backend.run.reset_mock()

            new_prompts = ["car", "bike", "dog"]
            stage.update_prompts(new_prompts)

            assert mock_backend.load_model.call_count == initial_load_count
            assert stage._text_prompts == new_prompts
            assert len(stage._text_tokens) == len(new_prompts)
            assert stage._text_embeddings.shape == (len(new_prompts), 512)
            assert mock_backend.run.call_count == len(new_prompts)
            assert mock_get_tokenizer.call_count == 2

    def test_classification_message_output_format(
        self,
        mock_backend: mock.MagicMock,
        mock_tokenizer: mock.MagicMock,
        sample_frame_tensor: FrameTensorMessage,
        text_prompts: list[str],
        mock_manager: mock.MagicMock,
    ) -> None:
        """Output remains a normalized ClassificationMessage."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                text_prompts=text_prompts,
                backend=mock_backend,
                manager=mock_manager,
            )

            stage._text_embeddings = np.stack(
                [_unit_vector(0), _unit_vector(1), _unit_vector(2)]
            ).astype(np.float32)
            mock_backend.run.reset_mock()
            mock_backend.run.return_value = [
                np.zeros((1, 512), dtype=np.float32),
                _unit_vector(0)[np.newaxis, :],
            ]

            result = stage.process(sample_frame_tensor)

            assert isinstance(result, ClassificationMessage)
            assert hasattr(result, "label")
            assert hasattr(result, "confidence")
            assert hasattr(result, "all_scores")
            assert hasattr(result, "timestamp")
            assert hasattr(result, "latency_ms")
            assert result.label in text_prompts
            assert 0.0 <= result.confidence <= 1.0
            assert len(result.all_scores) == len(text_prompts)
            assert set(result.all_scores.keys()) == set(text_prompts)
            assert sum(result.all_scores.values()) == pytest.approx(1.0, abs=0.01)

    def test_mobileclip_rejects_non_frame_tensor_message(
        self,
        mock_backend: mock.MagicMock,
        mock_tokenizer: mock.MagicMock,
        text_prompts: list[str],
        mock_manager: mock.MagicMock,
    ) -> None:
        """The stage still rejects the wrong message type."""
        from moment_to_action.messages.sensor import RawFrameMessage

        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                text_prompts=text_prompts,
                backend=mock_backend,
                manager=mock_manager,
            )

            wrong_msg = RawFrameMessage(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=time.time(),
                width=640,
                height=480,
            )

            with pytest.raises(TypeError, match="expects FrameTensorMessage"):
                stage.process(wrong_msg)

    def test_mobileclip_preserves_timestamp(
        self,
        mock_backend: mock.MagicMock,
        mock_tokenizer: mock.MagicMock,
        sample_frame_tensor: FrameTensorMessage,
        text_prompts: list[str],
        mock_manager: mock.MagicMock,
    ) -> None:
        """Timestamp is preserved from input to output."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                text_prompts=text_prompts,
                backend=mock_backend,
                manager=mock_manager,
            )

            stage._text_embeddings = np.stack(
                [_unit_vector(0), _unit_vector(1), _unit_vector(2)]
            ).astype(np.float32)
            mock_backend.run.reset_mock()
            mock_backend.run.return_value = [
                np.zeros((1, 512), dtype=np.float32),
                _unit_vector(2)[np.newaxis, :],
            ]

            result = stage.process(sample_frame_tensor)

            assert result is not None
            assert isinstance(result, ClassificationMessage)
            assert result.timestamp == sample_frame_tensor.timestamp

    def test_mobileclip_high_confidence_prediction(
        self,
        mock_backend: mock.MagicMock,
        mock_tokenizer: mock.MagicMock,
        sample_frame_tensor: FrameTensorMessage,
        text_prompts: list[str],
        mock_manager: mock.MagicMock,
    ) -> None:
        """A close image/prompt match still yields a high-confidence winner."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                text_prompts=text_prompts,
                backend=mock_backend,
                manager=mock_manager,
            )

            stage._text_embeddings = np.stack(
                [_unit_vector(0), _unit_vector(1), _unit_vector(2)]
            ).astype(np.float32)
            mock_backend.run.reset_mock()
            mock_backend.run.return_value = [
                np.zeros((1, 512), dtype=np.float32),
                _unit_vector(0)[np.newaxis, :],
            ]

            result = stage.process(sample_frame_tensor)

            assert result is not None
            assert isinstance(result, ClassificationMessage)
            assert result.label == text_prompts[0]
            assert result.confidence > 0.5

    def test_mobileclip_stage_name(
        self,
        mock_backend: mock.MagicMock,
        mock_tokenizer: mock.MagicMock,
        text_prompts: list[str],
        mock_manager: mock.MagicMock,
    ) -> None:
        """The stage name is unchanged."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                text_prompts=text_prompts,
                backend=mock_backend,
                manager=mock_manager,
            )

            assert stage.name == "MobileCLIPStage"
