"""Tests for video preprocessing stages.

Tests ImagePreprocessor (color space, resize, crop, normalize) and
PreprocessorStage (end-to-end pipeline).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.messages.video import FrameTensorMessage
from moment_to_action.stages.video._preprocess import (
    ImagePreprocessConfig,
    ImagePreprocessor,
    PreprocessorStage,
    ProcessedFrame,
)


@pytest.mark.unit
class TestImagePreprocessorDefault:
    """Test ImagePreprocessor with default config (256x256, RGB, normalized)."""

    def test_default_config(self) -> None:
        """Verify default config parameters."""
        config = ImagePreprocessConfig()
        assert config.target_size == (256, 256)
        assert config.crop_size is None
        assert config.letterbox is False
        assert config.to_rgb is True
        assert config.mean == (0.485, 0.456, 0.406)
        assert config.std == (0.229, 0.224, 0.225)

    def test_process_basic_frame(self) -> None:
        """Test preprocessing a basic RGB frame."""
        # Create a 480x640 RGB image
        img_array = np.full((480, 640, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        preprocessor = ImagePreprocessor(config=ImagePreprocessConfig())
        result = preprocessor.process(msg)

        assert isinstance(result, ProcessedFrame)
        assert result.data.shape == (256, 256, 3)
        assert result.data.dtype == np.float32
        assert result.original_size == (480, 640)
        assert result.timestamp == msg.timestamp

    def test_process_grayscale_to_rgb(self) -> None:
        """Test that grayscale frames are handled."""
        # Create a 480x640 grayscale image
        img_array = np.full((480, 640), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        preprocessor = ImagePreprocessor(config=ImagePreprocessConfig())
        result = preprocessor.process(msg)

        assert result.data.shape == (256, 256, 3)
        assert result.data.dtype == np.float32

    def test_normalization_applied(self) -> None:
        """Test that normalization is applied correctly."""
        # Create a frame with all white pixels (255)
        img_array = np.full((480, 640, 3), 255, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        config = ImagePreprocessConfig(
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )
        preprocessor = ImagePreprocessor(config=config)
        result = preprocessor.process(msg)

        # After normalization: (255/255 - 0) / 1 = 1.0
        assert np.allclose(result.data, 1.0, atol=0.01)

    def test_output_shape_and_dtype(self) -> None:
        """Test output shape and dtype."""
        rng = np.random.default_rng()

        img_array = rng.integers(0, 256, (100, 200, 3), dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=200,
            height=100,
        )

        preprocessor = ImagePreprocessor(config=ImagePreprocessConfig())
        result = preprocessor.process(msg)

        assert result.data.dtype == np.float32
        assert len(result.data.shape) == 3
        assert result.data.shape[2] == 3


@pytest.mark.unit
class TestImagePreprocessorLetterbox:
    """Test ImagePreprocessor with letterbox=True (aspect ratio preserved)."""

    def test_letterbox_aspect_ratio(self) -> None:
        """Test that letterbox preserves aspect ratio."""
        # Create a wide 480x960 frame (2:1 aspect)
        img_array = np.full((480, 960, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=960,
            height=480,
        )

        config = ImagePreprocessConfig(
            target_size=(256, 256),
            letterbox=True,
        )
        preprocessor = ImagePreprocessor(config=config)
        result = preprocessor.process(msg)

        assert result.data.shape == (256, 256, 3)

    def test_letterbox_square_image(self) -> None:
        """Test letterbox on a square image."""
        img_array = np.full((256, 256, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=256,
            height=256,
        )

        config = ImagePreprocessConfig(
            target_size=(256, 256),
            letterbox=True,
        )
        preprocessor = ImagePreprocessor(config=config)
        result = preprocessor.process(msg)

        assert result.data.shape == (256, 256, 3)

    def test_letterbox_tall_image(self) -> None:
        """Test letterbox on a tall image."""
        # Create a tall 960x480 frame (1:2 aspect)
        img_array = np.full((960, 480, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=480,
            height=960,
        )

        config = ImagePreprocessConfig(
            target_size=(256, 256),
            letterbox=True,
        )
        preprocessor = ImagePreprocessor(config=config)
        result = preprocessor.process(msg)

        assert result.data.shape == (256, 256, 3)
        assert result.data.dtype == np.float32


@pytest.mark.unit
class TestImagePreprocessorCrop:
    """Test ImagePreprocessor with crop_size (center crop after resize)."""

    def test_center_crop(self) -> None:
        """Test center crop after resize."""
        img_array = np.full((480, 640, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        config = ImagePreprocessConfig(
            target_size=(256, 256),
            crop_size=(224, 224),
        )
        preprocessor = ImagePreprocessor(config=config)
        result = preprocessor.process(msg)

        assert result.data.shape == (224, 224, 3)
        assert result.data.dtype == np.float32

    def test_crop_with_different_target_size(self) -> None:
        """Test crop with various target/crop combinations."""
        img_array = np.full((480, 640, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        config = ImagePreprocessConfig(
            target_size=(512, 512),
            crop_size=(384, 384),
        )
        preprocessor = ImagePreprocessor(config=config)
        result = preprocessor.process(msg)

        assert result.data.shape == (384, 384, 3)

    def test_crop_size_larger_than_target_fails(self) -> None:
        """Test that crop_size > target_size raises an error."""
        img_array = np.full((480, 640, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        config = ImagePreprocessConfig(
            target_size=(256, 256),
            crop_size=(512, 512),  # Larger than target
        )
        preprocessor = ImagePreprocessor(config=config)

        with pytest.raises(ValueError, match="smaller than crop size"):
            preprocessor.process(msg)

    def test_center_crop_extracts_center(self) -> None:
        """Test that center crop actually extracts from the center."""
        # Create a frame with distinct horizontal bands
        img_array = np.zeros((256, 256, 3), dtype=np.uint8)
        img_array[:85, :] = 50  # top band
        img_array[85:171, :] = 150  # middle band
        img_array[171:, :] = 50  # bottom band

        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=256,
            height=256,
        )

        config = ImagePreprocessConfig(
            target_size=(256, 256),
            crop_size=(150, 256),
        )
        preprocessor = ImagePreprocessor(config=config)
        result = preprocessor.process(msg)

        # The center crop should primarily contain the middle band
        assert result.data.shape == (150, 256, 3)


@pytest.mark.unit
class TestPreprocessorStageE2E:
    """Test PreprocessorStage end-to-end: RawFrameMessage → FrameTensorMessage."""

    def test_preprocessor_stage_basic(self) -> None:
        """Test basic PreprocessorStage operation."""
        rng = np.random.default_rng()

        img_array = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        stage = PreprocessorStage(
            target_size=(256, 256),
            letterbox=False,
            channels_first=True,
        )
        result = stage.process(msg)

        assert isinstance(result, FrameTensorMessage)
        assert result.tensor.shape == (1, 3, 256, 256)
        assert result.tensor.dtype == np.float32
        assert result.original_size == (480, 640)
        assert result.timestamp == msg.timestamp

    def test_preprocessor_stage_channels_first(self) -> None:
        """Test PreprocessorStage with channels_first=True."""
        rng = np.random.default_rng()

        img_array = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        stage = PreprocessorStage(
            target_size=(256, 256),
            channels_first=True,
        )
        result = stage.process(msg)

        assert isinstance(result, FrameTensorMessage)
        assert result.tensor.shape == (1, 3, 256, 256)
        assert result.tensor.ndim == 4
        assert result.tensor.dtype == np.float32

    def test_preprocessor_stage_channels_last(self) -> None:
        """Test PreprocessorStage with channels_first=False."""
        rng = np.random.default_rng()

        img_array = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        stage = PreprocessorStage(
            target_size=(256, 256),
            channels_first=False,
        )
        result = stage.process(msg)

        assert isinstance(result, FrameTensorMessage)
        assert result.tensor.shape == (1, 256, 256, 3)
        assert result.tensor.ndim == 4
        assert result.tensor.dtype == np.float32

    def test_preprocessor_stage_with_letterbox(self) -> None:
        """Test PreprocessorStage with letterbox option."""
        rng = np.random.default_rng()

        img_array = rng.integers(0, 256, (480, 960, 3), dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=960,
            height=480,
        )

        stage = PreprocessorStage(
            target_size=(640, 640),
            letterbox=True,
            channels_first=True,
        )
        result = stage.process(msg)

        assert isinstance(result, FrameTensorMessage)
        assert result.tensor.shape == (1, 3, 640, 640)

    def test_preprocessor_stage_none_frame(self) -> None:
        """Test PreprocessorStage with None frame returns None."""
        msg = RawFrameMessage(
            frame=None,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        stage = PreprocessorStage(target_size=(256, 256))
        result = stage.process(msg)

        assert result is None

    def test_preprocessor_stage_invalid_input_type(self) -> None:
        """Test PreprocessorStage rejects non-RawFrameMessage."""
        from moment_to_action.messages.video import FrameTensorMessage

        rng = np.random.default_rng()

        tensor = rng.standard_normal((1, 3, 256, 256)).astype(np.float32)
        msg = FrameTensorMessage(
            tensor=tensor,
            timestamp=time.time(),
            original_size=(480, 640),
        )

        stage = PreprocessorStage(target_size=(256, 256))

        with pytest.raises(TypeError, match="expects RawFrameMessage"):
            stage.process(msg)

    def test_preprocessor_stage_output_type_name(self) -> None:
        """Test that FrameTensorMessage type is correct."""
        rng = np.random.default_rng()

        img_array = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        stage = PreprocessorStage(target_size=(256, 256))
        result = stage.process(msg)

        assert isinstance(result, FrameTensorMessage)
        assert type(result).__name__ == "FrameTensorMessage"

    def test_output_shape_is_correct_channels_first(self) -> None:
        """Test output shape is (1, C, H, W) when channels_first=True."""
        rng = np.random.default_rng()

        img_array = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        stage = PreprocessorStage(
            target_size=(640, 640),
            channels_first=True,
        )
        result = stage.process(msg)

        assert isinstance(result, FrameTensorMessage)
        assert result.tensor.ndim == 4
        assert result.tensor.shape[0] == 1
        assert result.tensor.shape[1] == 3
        assert result.tensor.shape[2] == 640
        assert result.tensor.shape[3] == 640

    def test_output_shape_is_correct_channels_last(self) -> None:
        """Test output shape is (1, H, W, C) when channels_first=False."""
        rng = np.random.default_rng()

        img_array = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )

        stage = PreprocessorStage(
            target_size=(640, 640),
            channels_first=False,
        )
        result = stage.process(msg)

        assert isinstance(result, FrameTensorMessage)
        assert result.tensor.ndim == 4
        assert result.tensor.shape[0] == 1
        assert result.tensor.shape[1] == 640
        assert result.tensor.shape[2] == 640
        assert result.tensor.shape[3] == 3
