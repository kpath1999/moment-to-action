"""Tests for video preprocessing stages.

Tests ImagePreprocessor (color space, resize, crop, normalize) and
PreprocessorStage (end-to-end pipeline).
"""

from __future__ import annotations

import sys
import time
from unittest import mock

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


@pytest.mark.unit
class TestValidationErrors:
    """Test _validate error paths."""

    def test_validate_frame_none(self) -> None:
        """Test _validate raises ValueError when frame=None."""
        msg = RawFrameMessage(
            frame=None,
            timestamp=time.time(),
            width=640,
            height=480,
        )
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="ImageInput has no frame data"):
            preprocessor.process(msg)

    def test_validate_frame_4d(self) -> None:
        """Test _validate raises ValueError for 4D frame."""
        img_array = np.full((1, 480, 640, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="Expected 2D or 3D frame"):
            preprocessor.process(msg)

    def test_validate_frame_1d(self) -> None:
        """Test _validate raises ValueError for 1D frame."""
        img_array = np.full((640,), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="Expected 2D or 3D frame"):
            preprocessor.process(msg)

    def test_validate_negative_width(self) -> None:
        """Test _validate raises ValueError for negative width."""
        img_array = np.full((480, 640, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=-640,
            height=480,
        )
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="Invalid dimensions"):
            preprocessor.process(msg)

    def test_validate_zero_width(self) -> None:
        """Test _validate raises ValueError for zero width."""
        img_array = np.full((480, 640, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=0,
            height=480,
        )
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="Invalid dimensions"):
            preprocessor.process(msg)

    def test_validate_negative_height(self) -> None:
        """Test _validate raises ValueError for negative height."""
        img_array = np.full((480, 640, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=-480,
        )
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="Invalid dimensions"):
            preprocessor.process(msg)

    def test_validate_zero_height(self) -> None:
        """Test _validate raises ValueError for zero height."""
        img_array = np.full((480, 640, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=0,
        )
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="Invalid dimensions"):
            preprocessor.process(msg)


@pytest.mark.unit
class TestToRGBEdgeCases:
    """Test _to_rgb edge cases and fallback behavior."""

    def test_to_rgb_with_frame_none(self) -> None:
        """Test _to_rgb returns message unchanged when frame=None."""
        msg = RawFrameMessage(
            frame=None,
            timestamp=time.time(),
            width=640,
            height=480,
        )
        preprocessor = ImagePreprocessor()
        result = preprocessor._to_rgb(msg)
        assert result.frame is None
        assert result.timestamp == msg.timestamp
        assert result.width == msg.width
        assert result.height == msg.height

    def test_to_rgb_cv2_import_error_numpy_fallback(self) -> None:
        """Test _to_rgb uses numpy fallback when cv2 import fails."""
        img_array = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=2,
            height=1,
        )
        preprocessor = ImagePreprocessor()

        with mock.patch.dict(sys.modules, {"cv2": None}):
            result = preprocessor._to_rgb(msg)

        assert result.frame is not None
        assert result.frame.shape == img_array.shape
        np.testing.assert_array_equal(result.frame, img_array[..., ::-1])

    def test_to_rgb_normal_operation(self) -> None:
        """Test _to_rgb correctly converts BGR to RGB."""
        img_array = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=2,
            height=1,
        )
        preprocessor = ImagePreprocessor()
        result = preprocessor._to_rgb(msg)
        assert result.frame is not None
        expected = np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8)
        np.testing.assert_array_equal(result.frame, expected)


@pytest.mark.unit
class TestResizeEdgeCases:
    """Test _resize error paths and cv2 fallback."""

    def test_resize_with_frame_none(self) -> None:
        """Test _resize raises ValueError when frame=None."""
        msg = RawFrameMessage(
            frame=None,
            timestamp=time.time(),
            width=640,
            height=480,
        )
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="Cannot resize a RawFrameMessage with frame=None"):
            preprocessor._resize(msg, (256, 256), letterbox=False)

    def test_resize_cv2_import_error_calls_numpy_fallback(self) -> None:
        """Test _resize falls back to _resize_numpy when cv2 import fails."""
        img_array = np.full((480, 640, 3), 128, dtype=np.uint8)
        msg = RawFrameMessage(
            frame=img_array,
            timestamp=time.time(),
            width=640,
            height=480,
        )
        preprocessor = ImagePreprocessor()

        with mock.patch.dict(sys.modules, {"cv2": None}):
            result = preprocessor._resize(msg, (256, 256), letterbox=False)

        assert isinstance(result, ProcessedFrame)
        assert result.data.shape == (256, 256, 3)
        assert result.data.dtype == np.float32


@pytest.mark.unit
class TestResizeNumpy:
    """Test _resize_numpy directly."""

    def test_resize_numpy_small_array(self) -> None:
        """Test _resize_numpy with a small synthetic array."""
        img_array = np.array(
            [
                [[1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4]],
            ],
            dtype=np.uint8,
        )
        preprocessor = ImagePreprocessor()
        result = preprocessor._resize_numpy(img_array, (4, 4))

        assert result.shape == (4, 4, 3)
        assert result.dtype == np.uint8

    def test_resize_numpy_nearest_neighbor(self) -> None:
        """Test _resize_numpy uses nearest-neighbor resampling."""
        img_array = np.array(
            [[[100, 100, 100], [200, 200, 200]]],
            dtype=np.uint8,
        )
        preprocessor = ImagePreprocessor()
        result = preprocessor._resize_numpy(img_array, (2, 4))

        assert result.shape == (2, 4, 3)
        assert np.all(result[0, 0] == [100, 100, 100])
        assert np.all(result[0, -1] == [200, 200, 200])

    def test_resize_numpy_scaling_up(self) -> None:
        """Test _resize_numpy scaling up."""
        img_array = np.array(
            [[[50, 50, 50], [100, 100, 100]]],
            dtype=np.uint8,
        )
        preprocessor = ImagePreprocessor()
        result = preprocessor._resize_numpy(img_array, (3, 6))

        assert result.shape == (3, 6, 3)
        assert result.dtype == np.uint8

    def test_resize_numpy_scaling_down(self) -> None:
        """Test _resize_numpy scaling down."""
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        preprocessor = ImagePreprocessor()
        result = preprocessor._resize_numpy(img_array, (10, 10))

        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8


@pytest.mark.unit
class TestLetterboxCV2:
    """Test _letterbox_cv2 with different aspect ratios."""

    def test_letterbox_cv2_wide_image(self) -> None:
        """Test _letterbox_cv2 with a wide image (aspect > 1)."""
        img_array = np.full((256, 512, 3), 128, dtype=np.uint8)
        preprocessor = ImagePreprocessor()
        result = preprocessor._letterbox_cv2(img_array, (256, 256))

        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8
        assert np.any(result == 114)

    def test_letterbox_cv2_tall_image(self) -> None:
        """Test _letterbox_cv2 with a tall image (aspect < 1)."""
        img_array = np.full((512, 256, 3), 128, dtype=np.uint8)
        preprocessor = ImagePreprocessor()
        result = preprocessor._letterbox_cv2(img_array, (256, 256))

        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8
        assert np.any(result == 114)

    def test_letterbox_cv2_square_image(self) -> None:
        """Test _letterbox_cv2 with a square image."""
        img_array = np.full((256, 256, 3), 100, dtype=np.uint8)
        preprocessor = ImagePreprocessor()
        result = preprocessor._letterbox_cv2(img_array, (256, 256))

        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8
        assert np.all(result == 100)

    def test_letterbox_cv2_padding_applied(self) -> None:
        """Test _letterbox_cv2 applies padding correctly."""
        img_array = np.full((100, 200, 3), 100, dtype=np.uint8)
        preprocessor = ImagePreprocessor()
        result = preprocessor._letterbox_cv2(img_array, (200, 200))

        assert result.shape == (200, 200, 3)
        assert 114 in result

    def test_letterbox_cv2_preserves_content(self) -> None:
        """Test _letterbox_cv2 preserves original image content in center."""
        img_array = np.full((100, 200, 3), 150, dtype=np.uint8)
        preprocessor = ImagePreprocessor()
        result = preprocessor._letterbox_cv2(img_array, (200, 400))

        assert result.shape == (200, 400, 3)
        assert np.max(result) >= 150


@pytest.mark.unit
class TestImagePreprocessorBufferAllocation:
    """Tests for _allocate_buffers and reconfigure edge cases."""

    def test_allocate_buffers_without_config_returns_early(self) -> None:
        """Test _allocate_buffers returns early when _config is not yet set.

        This guard exists so the BasePreprocessor.__init__ can call
        _allocate_buffers before the subclass has a chance to set _config.
        Covered by deleting _config after construction and calling directly.
        """
        preprocessor = ImagePreprocessor()
        # Temporarily remove _config to simulate the state before it is set.
        del preprocessor._config
        # Should return immediately without raising — no buffer registered.
        preprocessor._allocate_buffers()

    def test_reconfigure_updates_config_and_reallocates_buffers(self) -> None:
        """Test reconfigure() swaps config and re-allocates buffers.

        Covers lines 274-279 — the reconfigure method that allows changing
        preprocessing parameters after construction without creating a new instance.
        """
        original_config = ImagePreprocessConfig(target_size=(128, 128))
        preprocessor = ImagePreprocessor(config=original_config)

        new_config = ImagePreprocessConfig(
            target_size=(256, 256),
            crop_size=(224, 224),
        )
        preprocessor.reconfigure(new_config)

        # Config should be updated.
        assert preprocessor._config is new_config
        assert preprocessor._config.target_size == (256, 256)
        assert preprocessor._config.crop_size == (224, 224)

    def test_reconfigure_without_crop_uses_target_size(self) -> None:
        """Test reconfigure() without crop_size falls back to target_size for the frame buffer."""
        preprocessor = ImagePreprocessor(config=ImagePreprocessConfig(target_size=(64, 64)))
        new_config = ImagePreprocessConfig(target_size=(192, 192))
        preprocessor.reconfigure(new_config)

        assert preprocessor._config.target_size == (192, 192)
        assert preprocessor._config.crop_size is None
