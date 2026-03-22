"""Unit tests for MobileCLIPStage.

Tests zero-shot classification pipeline with mocked backend.
"""

from __future__ import annotations

import time
from unittest import mock

import numpy as np
import pytest

from moment_to_action.messages.video import FrameTensorMessage
from moment_to_action.messages.vlm import ClassificationMessage
from moment_to_action.stages.vlm._mobileclip import MobileCLIPStage


@pytest.mark.unit
class TestMobileCLIPStage:
    """Tests for MobileCLIPStage."""

    @pytest.fixture
    def mock_backend(self) -> mock.MagicMock:
        """Create a mocked ComputeBackend."""
        backend = mock.MagicMock()
        backend.load_model.return_value = "mock_model_handle"
        return backend

    @pytest.fixture
    def mock_tokenizer(self) -> mock.MagicMock:
        """Create a mocked tokenizer that returns token arrays."""
        tokenizer = mock.MagicMock()

        def tokenize_fn(prompts: list[str] | str) -> np.ndarray:
            """Return a mock token array for each prompt."""
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
        self, mock_backend: mock.MagicMock, mock_tokenizer: mock.MagicMock, text_prompts: list[str]
    ) -> None:
        """Test MobileCLIPStage initialization with mocked backend."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                model_path="mock_model.tflite",
                text_prompts=text_prompts,
                backend=mock_backend,
            )

            assert stage._text_prompts == text_prompts
            assert len(stage._text_tokens) == len(text_prompts)
            assert stage._text_tokens.dtype == np.int64
            mock_backend.load_model.assert_called_once_with("mock_model.tflite")

    def test_mobileclip_zero_shot_classification(
        self,
        mock_backend: mock.MagicMock,
        mock_tokenizer: mock.MagicMock,
        sample_frame_tensor: FrameTensorMessage,
        text_prompts: list[str],
    ) -> None:
        """Test zero-shot classification: image → text embeddings → similarity."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                model_path="mock_model.tflite",
                text_prompts=text_prompts,
                backend=mock_backend,
            )

            # Mock the backend.run() to return embeddings [image_emb, text_emb]
            image_emb = np.random.default_rng().standard_normal(512).astype(np.float32)
            image_emb = image_emb / np.linalg.norm(image_emb)
            text_emb = np.random.default_rng().standard_normal(512).astype(np.float32)
            text_emb = text_emb / np.linalg.norm(text_emb)

            # Return embeddings for each prompt
            mock_backend.run.side_effect = [
                [text_emb, image_emb],  # First prompt
                [text_emb, image_emb],  # Second prompt
                [text_emb, image_emb],  # Third prompt
            ]

            result = stage.process(sample_frame_tensor)

            assert isinstance(result, ClassificationMessage)
            assert result.label in text_prompts
            assert 0.0 <= result.confidence <= 1.0
            assert all(prompt in result.all_scores for prompt in text_prompts)
            assert sum(result.all_scores.values()) == pytest.approx(1.0, abs=0.01)

    def test_mobileclip_backend_called_correctly(
        self,
        mock_backend: mock.MagicMock,
        mock_tokenizer: mock.MagicMock,
        sample_frame_tensor: FrameTensorMessage,
        text_prompts: list[str],
    ) -> None:
        """Test that backend.run() is called with correct inputs for each prompt."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                model_path="mock_model.tflite",
                text_prompts=text_prompts,
                backend=mock_backend,
            )

            # Mock backend.run()
            embeddings = [
                [
                    np.random.default_rng().standard_normal(512).astype(np.float32),
                    np.random.default_rng().standard_normal(512).astype(np.float32),
                ]
                for _ in text_prompts
            ]
            mock_backend.run.side_effect = embeddings

            stage.process(sample_frame_tensor)

            # Verify backend.run() was called once per prompt
            assert mock_backend.run.call_count == len(text_prompts)

            # Verify input tensors in calls
            for call in mock_backend.run.call_args_list:
                inputs_dict = call[0][1]
                assert "serving_default_args_0:0" in inputs_dict  # image tensor
                assert "serving_default_args_1:0" in inputs_dict  # token tensor

                # Image tensor should be the input tensor
                np.testing.assert_array_equal(
                    inputs_dict["serving_default_args_0:0"], sample_frame_tensor.tensor
                )

    def test_update_prompts_swaps_without_reloading(
        self, mock_backend: mock.MagicMock, mock_tokenizer: mock.MagicMock
    ) -> None:
        """Test update_prompts() swaps text prompts without reloading model."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            initial_prompts = ["person", "hand"]
            stage = MobileCLIPStage(
                model_path="mock_model.tflite",
                text_prompts=initial_prompts,
                backend=mock_backend,
            )

            # Verify model was loaded once
            initial_load_count = mock_backend.load_model.call_count
            assert initial_load_count == 1

            # Update prompts
            new_prompts = ["car", "bike", "dog"]
            stage.update_prompts(new_prompts)

            # Verify model was not reloaded
            assert mock_backend.load_model.call_count == initial_load_count

            # Verify prompts were swapped
            assert stage._text_prompts == new_prompts
            assert len(stage._text_tokens) == len(new_prompts)

            # Verify tokenizer was called again
            assert mock_get_tokenizer.call_count == 2  # Once in __init__, once in update_prompts

    def test_classification_message_output_format(
        self,
        mock_backend: mock.MagicMock,
        mock_tokenizer: mock.MagicMock,
        sample_frame_tensor: FrameTensorMessage,
        text_prompts: list[str],
    ) -> None:
        """Test that output is ClassificationMessage with label and confidence."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                model_path="mock_model.tflite",
                text_prompts=text_prompts,
                backend=mock_backend,
            )

            # Mock backend.run() to return embeddings with high similarity to first prompt
            embeddings_list = []
            for i in range(len(text_prompts)):
                image_emb = np.random.default_rng().standard_normal(512).astype(np.float32)
                image_emb = image_emb / np.linalg.norm(image_emb)

                # Create text embeddings where first one is most similar to image
                if i == 0:
                    # High similarity: similar vector
                    text_emb = (
                        image_emb
                        + np.random.default_rng().standard_normal(512).astype(np.float32) * 0.1
                    )
                else:
                    # Lower similarity: random vector
                    text_emb = np.random.default_rng().standard_normal(512).astype(np.float32)

                text_emb = text_emb / np.linalg.norm(text_emb)
                embeddings_list.append([text_emb, image_emb])

            mock_backend.run.side_effect = embeddings_list

            result = stage.process(sample_frame_tensor)

            # Verify ClassificationMessage structure
            assert isinstance(result, ClassificationMessage)
            assert hasattr(result, "label")
            assert hasattr(result, "confidence")
            assert hasattr(result, "all_scores")
            assert hasattr(result, "timestamp")
            assert hasattr(result, "latency_ms")

            # Verify label is one of the prompts
            assert result.label in text_prompts

            # Verify confidence is in valid range
            assert 0.0 <= result.confidence <= 1.0

            # Verify all_scores contains all prompts
            assert len(result.all_scores) == len(text_prompts)
            assert set(result.all_scores.keys()) == set(text_prompts)

            # Verify scores sum to 1.0 (softmax normalized)
            total_score = sum(result.all_scores.values())
            assert total_score == pytest.approx(1.0, abs=0.01)

    def test_mobileclip_rejects_non_frame_tensor_message(
        self, mock_backend: mock.MagicMock, mock_tokenizer: mock.MagicMock, text_prompts: list[str]
    ) -> None:
        """Test that MobileCLIPStage rejects non-FrameTensorMessage input."""
        from moment_to_action.messages.sensor import RawFrameMessage

        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                model_path="mock_model.tflite",
                text_prompts=text_prompts,
                backend=mock_backend,
            )

            # Create a non-FrameTensorMessage
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
    ) -> None:
        """Test that timestamp is preserved from input message."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                model_path="mock_model.tflite",
                text_prompts=text_prompts,
                backend=mock_backend,
            )

            # Mock backend.run()
            embeddings = [
                [
                    np.random.default_rng().standard_normal(512).astype(np.float32),
                    np.random.default_rng().standard_normal(512).astype(np.float32),
                ]
                for _ in text_prompts
            ]
            mock_backend.run.side_effect = embeddings

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
    ) -> None:
        """Test classification with high confidence for top prediction."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                model_path="mock_model.tflite",
                text_prompts=text_prompts,
                backend=mock_backend,
            )

            # Mock embeddings where first prompt is clearly the best match
            image_emb = np.array([1.0] + [0.0] * 511, dtype=np.float32)
            image_emb = image_emb / np.linalg.norm(image_emb)

            embeddings_list = [
                [image_emb, image_emb],  # Perfect match for first prompt
                [np.zeros(512, dtype=np.float32), image_emb],  # No match
                [np.zeros(512, dtype=np.float32), image_emb],  # No match
            ]
            mock_backend.run.side_effect = embeddings_list

            result = stage.process(sample_frame_tensor)

            assert result is not None
            assert isinstance(result, ClassificationMessage)
            assert result.label == text_prompts[0]
            assert result.confidence > 0.5  # Should be high confidence

    def test_mobileclip_stage_name(
        self, mock_backend: mock.MagicMock, mock_tokenizer: mock.MagicMock, text_prompts: list[str]
    ) -> None:
        """Test that stage name is correct."""
        with mock.patch(
            "moment_to_action.stages.vlm._mobileclip.open_clip.get_tokenizer"
        ) as mock_get_tokenizer:
            mock_get_tokenizer.return_value = mock_tokenizer

            stage = MobileCLIPStage(
                model_path="mock_model.tflite",
                text_prompts=text_prompts,
                backend=mock_backend,
            )

            assert stage.name == "MobileCLIPStage"
