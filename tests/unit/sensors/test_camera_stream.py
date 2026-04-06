"""Unit tests for CameraStreamSensor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moment_to_action.sensors._camera_stream import CameraStreamSensor


@pytest.mark.unit
class TestCameraStreamSensorInit:
    """Tests for CameraStreamSensor initialization."""

    def test_default_construction(self) -> None:
        """Sensor can be constructed with default arguments."""
        sensor = CameraStreamSensor()
        assert sensor._device == 0
        assert sensor._requested_width == 0
        assert sensor._requested_height == 0
        assert sensor._requested_fps == 0.0
        assert sensor._backend == "auto"
        assert sensor._cap is None

    def test_construction_with_custom_params(self) -> None:
        """Sensor stores custom device, resolution, fps, and backend."""
        sensor = CameraStreamSensor(
            device="/dev/video1",
            width=1280,
            height=720,
            fps=30.0,
            backend=200,
        )
        assert sensor._device == "/dev/video1"
        assert sensor._requested_width == 1280
        assert sensor._requested_height == 720
        assert sensor._requested_fps == 30.0
        assert sensor._backend == 200


@pytest.mark.unit
class TestCameraStreamSensorOpen:
    """Tests for CameraStreamSensor.open()."""

    @patch("moment_to_action.sensors._camera_stream.cv2.VideoCapture")
    def test_open_auto_backend(self, mock_video_capture: MagicMock) -> None:
        """Open with auto backend calls VideoCapture with device only."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_video_capture.return_value = mock_cap

        sensor = CameraStreamSensor(device=0)
        sensor.open()

        mock_video_capture.assert_called_once_with(0)
        assert sensor._cap is mock_cap

    @patch("moment_to_action.sensors._camera_stream.cv2.VideoCapture")
    def test_open_explicit_backend(self, mock_video_capture: MagicMock) -> None:
        """Open with explicit backend passes it to VideoCapture."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_video_capture.return_value = mock_cap

        sensor = CameraStreamSensor(device=0, backend=200)
        sensor.open()

        mock_video_capture.assert_called_once_with(0, 200)

    @patch("moment_to_action.sensors._camera_stream.cv2.VideoCapture")
    def test_open_raises_os_error_when_device_fails(self, mock_video_capture: MagicMock) -> None:
        """Open raises OSError when the capture device cannot be opened."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        sensor = CameraStreamSensor(device=99)
        with pytest.raises(OSError, match="could not open device"):
            sensor.open()

    @patch("moment_to_action.sensors._camera_stream.cv2.VideoCapture")
    def test_open_applies_resolution_and_fps(self, mock_video_capture: MagicMock) -> None:
        """Open sets width, height, and fps on the capture device."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_video_capture.return_value = mock_cap

        sensor = CameraStreamSensor(device=0, width=1280, height=720, fps=30.0)
        sensor.open()

        from moment_to_action.sensors._camera_stream import cv2

        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 30.0)

    @patch("moment_to_action.sensors._camera_stream.cv2.VideoCapture")
    def test_open_skips_zero_resolution_and_fps(self, mock_video_capture: MagicMock) -> None:
        """Open does not set properties when they are zero (default)."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_video_capture.return_value = mock_cap

        sensor = CameraStreamSensor(device=0, width=0, height=0, fps=0.0)
        sensor.open()

        mock_cap.set.assert_not_called()


@pytest.mark.unit
class TestCameraStreamSensorRead:
    """Tests for CameraStreamSensor.read()."""

    def test_read_raises_os_error_before_open(self) -> None:
        """Read raises OSError if the sensor has not been opened."""
        sensor = CameraStreamSensor()
        with pytest.raises(OSError, match="call open"):
            sensor.read()

    @patch("moment_to_action.sensors._camera_stream.cv2.VideoCapture")
    def test_read_returns_frame_message(self, mock_video_capture: MagicMock) -> None:
        """Successful read returns a RawFrameMessage with frame data."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_cap.read.return_value = (True, frame)
        mock_video_capture.return_value = mock_cap

        sensor = CameraStreamSensor(device=0)
        sensor.open()
        msg = sensor.read()

        assert msg.frame is not None
        assert msg.width == 640
        assert msg.height == 480
        assert msg.source == "0"

    @patch("moment_to_action.sensors._camera_stream.cv2.VideoCapture")
    def test_read_returns_none_frame_on_dropped_capture(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Dropped frame returns RawFrameMessage with frame=None."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        sensor = CameraStreamSensor(device=0)
        sensor.open()
        msg = sensor.read()

        assert msg.frame is None
        assert msg.source == "0"


@pytest.mark.unit
class TestCameraStreamSensorClose:
    """Tests for CameraStreamSensor.close()."""

    def test_close_when_not_opened(self) -> None:
        """Close on an unopened sensor does nothing."""
        sensor = CameraStreamSensor()
        sensor.close()  # should not raise

    @patch("moment_to_action.sensors._camera_stream.cv2.VideoCapture")
    def test_close_releases_capture(self, mock_video_capture: MagicMock) -> None:
        """Close releases the capture device and sets _cap to None."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_video_capture.return_value = mock_cap

        sensor = CameraStreamSensor(device=0)
        sensor.open()
        sensor.close()

        mock_cap.release.assert_called_once()
        assert sensor._cap is None


@pytest.mark.unit
class TestCameraStreamSensorProperties:
    """Tests for CameraStreamSensor convenience properties."""

    def test_fps_before_open_returns_zero(self) -> None:
        """Fps property returns 0.0 before the sensor is opened."""
        sensor = CameraStreamSensor()
        assert sensor.fps == 0.0

    @patch("moment_to_action.sensors._camera_stream.cv2.VideoCapture")
    def test_fps_after_open(self, mock_video_capture: MagicMock) -> None:
        """Fps property returns the device-reported value after open."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        from moment_to_action.sensors._camera_stream import cv2

        def fake_get(prop: int) -> float:
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

        mock_cap.get.side_effect = fake_get
        mock_video_capture.return_value = mock_cap

        sensor = CameraStreamSensor(device=0)
        sensor.open()
        assert sensor.fps == 30.0

    def test_resolution_before_open_returns_zero(self) -> None:
        """Resolution property returns (0, 0) before the sensor is opened."""
        sensor = CameraStreamSensor()
        assert sensor.resolution == (0, 0)

    @patch("moment_to_action.sensors._camera_stream.cv2.VideoCapture")
    def test_resolution_after_open(self, mock_video_capture: MagicMock) -> None:
        """Resolution property returns (width, height) from the device."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        from moment_to_action.sensors._camera_stream import cv2

        def fake_get(prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 1280.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 720.0
            return 0.0

        mock_cap.get.side_effect = fake_get
        mock_video_capture.return_value = mock_cap

        sensor = CameraStreamSensor(device=0)
        sensor.open()
        assert sensor.resolution == (1280, 720)


@pytest.mark.unit
class TestCameraStreamSensorContextManager:
    """Tests for CameraStreamSensor used as a context manager."""

    @patch("moment_to_action.sensors._camera_stream.cv2.VideoCapture")
    def test_context_manager_opens_and_closes(self, mock_video_capture: MagicMock) -> None:
        """Using the sensor as a context manager calls open and close."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_video_capture.return_value = mock_cap

        with CameraStreamSensor(device=0) as sensor:
            assert sensor._cap is not None

        mock_cap.release.assert_called_once()
