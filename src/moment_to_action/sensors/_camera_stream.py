"""Live camera-stream sensor backed by OpenCV VideoCapture.

Supports:
- USB/built-in webcams (integer device index)
- RTSP / HTTP / file-based video streams (string URL or path)

The sensor is designed to be used as a context manager so that the
capture device is always released cleanly::

    with CameraStreamSensor(device=0) as cam:
        while True:
            msg = cam.read()          # RawFrameMessage, frame is None on drop
            if msg.frame is not None:
                pipeline.run(msg)

Thread safety: :meth:`read` is *not* thread-safe; callers that share a
sensor across threads must synchronise externally.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from moment_to_action.messages.sensor import RawFrameMessage

from ._base import BaseSensor

logger = logging.getLogger(__name__)

# VideoCapture is imported lazily so that cv2 is not a hard import-time
# requirement (mirrors the pattern in FileImageSensor and SensorStage).


class CameraStreamSensor(BaseSensor):
    """Live camera sensor that yields one ``RawFrameMessage`` per :meth:`read` call.

    Wraps :class:`~moment_to_action.messages.sensor.RawFrameMessage`.

    Args:
        device: Camera index (``0`` for the default webcam) **or** a video
            stream URL / file path accepted by ``cv2.VideoCapture``.
        width:  Requested capture width in pixels.  The actual width may
            differ if the device does not support the requested resolution.
        height: Requested capture height in pixels.
        fps:    Requested frame-rate.  ``0`` keeps the device default.
        backend: OpenCV ``CAP_*`` backend constant.  ``"auto"`` lets OpenCV
            pick the best backend for *device*.

    Example:
        >>> with CameraStreamSensor(device=0, width=1280, height=720) as cam:
        ...     msg = cam.read()
        ...     print(msg.width, msg.height)
    """

    def __init__(
        self,
        device: int | str = 0,
        *,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        backend: int | Literal["auto"] = "auto",
    ) -> None:
        self._device = device
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps
        self._backend = backend
        self._cap: object = None  # cv2.VideoCapture — type hidden to avoid hard import

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the capture device and apply the requested resolution / fps.

        Raises:
            RuntimeError: If ``opencv-python`` is not installed.
            OSError: If the device cannot be opened.
        """
        try:
            import cv2
        except ImportError as exc:
            msg = "CameraStreamSensor requires opencv-python: pip install opencv-python"
            raise RuntimeError(msg) from exc

        if self._backend == "auto":
            cap = cv2.VideoCapture(self._device)
        else:
            cap = cv2.VideoCapture(self._device, self._backend)

        if not cap.isOpened():
            msg = f"CameraStreamSensor: could not open device {self._device!r}"
            raise OSError(msg)

        # Apply resolution / fps hints — the device ignores requests it
        # cannot satisfy, so we read back the actual values afterwards.
        if self._requested_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        if self._requested_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        if self._requested_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self._requested_fps)

        self._cap = cap
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "CameraStreamSensor opened device=%r  resolution=%dx%d  fps=%.1f",
            self._device,
            actual_w,
            actual_h,
            actual_fps,
        )

    def read(self) -> RawFrameMessage:
        """Capture one frame from the device.

        Returns:
            A :class:`~moment_to_action.messages.sensor.RawFrameMessage`.
            ``frame`` is ``None`` when the device drops a frame; the
            pipeline should handle this gracefully (e.g. :class:`ClipBufferStage`
            simply discards ``None`` frames).

        Raises:
            OSError: If the sensor has not been opened via :meth:`open`.
        """
        if self._cap is None:
            msg = "CameraStreamSensor: call open() before read()"
            raise OSError(msg)

        cap = self._cap  # type: ignore[assignment]
        ok, frame = cap.read()  # type: ignore[attr-defined]
        ts = time.time()

        if not ok or frame is None:
            logger.debug("CameraStreamSensor: dropped frame from device %r", self._device)
            return RawFrameMessage(
                frame=None,
                timestamp=ts,
                source=str(self._device),
            )

        h, w = frame.shape[:2]
        return RawFrameMessage(
            frame=frame,
            timestamp=ts,
            source=str(self._device),
            width=w,
            height=h,
        )

    def close(self) -> None:
        """Release the capture device."""
        if self._cap is not None:
            self._cap.release()  # type: ignore[attr-defined]
            logger.info("CameraStreamSensor: released device %r", self._device)
            self._cap = None

    # ------------------------------------------------------------------
    # Convenience property — available after open()
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Actual capture frame-rate reported by the device; ``0.0`` before open."""
        if self._cap is None:
            return 0.0
        import cv2

        return float(self._cap.get(cv2.CAP_PROP_FPS))  # type: ignore[attr-defined]

    @property
    def resolution(self) -> tuple[int, int]:
        """``(width, height)`` reported by the device; ``(0, 0)`` before open."""
        if self._cap is None:
            return (0, 0)
        import cv2

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # type: ignore[attr-defined]
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # type: ignore[attr-defined]
        return (w, h)
