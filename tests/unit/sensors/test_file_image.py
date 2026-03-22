"""Unit tests for FileImageSensor."""

from __future__ import annotations

import pathlib
import tempfile
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.sensors._file_image import FileImageSensor

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.mark.unit
class TestFileImageSensor:
    """Tests for FileImageSensor class."""

    @pytest.fixture
    def temp_image_file(self) -> Generator[Path, None, None]:
        """Create a temporary image file for testing.

        Yields:
            Path to a temporary JPEG image file.
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            # Create a simple 480x640 BGR image
            rng = np.random.default_rng()
            image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(tmp.name, image)
            yield pathlib.Path(tmp.name)
        # Clean up after test
        pathlib.Path(tmp.name).unlink(missing_ok=True)

    def test_open_with_valid_path(self, temp_image_file: Path) -> None:
        """Test FileImageSensor.open() with valid path — no error."""
        sensor = FileImageSensor(temp_image_file)
        # Should not raise
        sensor.open()
        sensor.close()

    def test_open_with_missing_file(self) -> None:
        """Test FileImageSensor.open() with missing file — raises FileNotFoundError."""
        sensor = FileImageSensor("/nonexistent/path/to/file.jpg")
        with pytest.raises(FileNotFoundError):
            sensor.open()

    def test_read_returns_raw_frame_message(self, temp_image_file: Path) -> None:
        """Test FileImageSensor.read() returns RawFrameMessage with correct shape."""
        sensor = FileImageSensor(temp_image_file)
        sensor.open()
        msg = sensor.read()
        sensor.close()

        # Verify it's a RawFrameMessage
        assert isinstance(msg, RawFrameMessage)

        # Verify frame is not None
        assert msg.frame is not None

        # Verify shape matches what we created (480x640x3)
        assert msg.frame.shape == (480, 640, 3)

        # Verify frame dtype is uint8 (as read by cv2)
        assert msg.frame.dtype == np.uint8

        # Verify width and height metadata
        assert msg.width == 640
        assert msg.height == 480

        # Verify source is set correctly
        assert str(temp_image_file) in msg.source

        # Verify timestamp is present
        assert msg.timestamp > 0

    def test_close_is_noop(self, temp_image_file: Path) -> None:
        """Test FileImageSensor.close() is no-op (doesn't raise)."""
        sensor = FileImageSensor(temp_image_file)
        sensor.open()
        # Should not raise
        sensor.close()
        # Calling close again should also not raise
        sensor.close()

    def test_context_manager_protocol(self, temp_image_file: Path) -> None:
        """Test context manager protocol: with FileImageSensor(...) as sensor."""
        with FileImageSensor(temp_image_file) as sensor:
            msg = sensor.read()

        # Verify we got a valid message
        assert isinstance(msg, RawFrameMessage)
        assert msg.frame is not None
        assert msg.frame.shape == (480, 640, 3)

    def test_read_with_real_test_image(self) -> None:
        """Test FileImageSensor.read() with real test image from tests/int/images/."""
        # Use one of the real test images if available
        test_image = pathlib.Path(
            "/home/nikola/code/moment-to-action/tests/int/images/pedestrian.jpg"
        )
        if not test_image.exists():
            pytest.skip("Real test image not found")

        with FileImageSensor(test_image) as sensor:
            msg = sensor.read()

        # Verify message structure
        assert isinstance(msg, RawFrameMessage)
        assert msg.frame is not None
        assert len(msg.frame.shape) == 3
        assert msg.frame.shape[2] == 3  # BGR channels
        assert msg.width > 0
        assert msg.height > 0
        assert msg.source == str(test_image)

    def test_string_path_conversion(self, temp_image_file: Path) -> None:
        """Test that FileImageSensor accepts both str and Path objects."""
        # Test with string path
        sensor_str = FileImageSensor(str(temp_image_file))
        sensor_str.open()
        msg_str = sensor_str.read()
        sensor_str.close()

        # Test with pathlib.Path
        sensor_path = FileImageSensor(temp_image_file)
        sensor_path.open()
        msg_path = sensor_path.read()
        sensor_path.close()

        # Both should produce same dimensions
        assert msg_str.width == msg_path.width
        assert msg_str.height == msg_path.height
