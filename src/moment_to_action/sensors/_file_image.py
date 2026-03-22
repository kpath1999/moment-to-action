"""File-based sensor: reads a single image from disk."""

from __future__ import annotations

import logging
import pathlib
import time

import cv2

from moment_to_action.messages.sensor import RawFrameMessage

from ._base import BaseSensor

logger = logging.getLogger(__name__)


class FileImageSensor(BaseSensor):
    """Sensor that reads a single image frame from a file on disk.

    This sensor is extracted from the original ``SensorStage`` pipeline
    step. Unlike that stage it is not tied to the pipeline machinery; it
    is a plain Python object that can be used anywhere.

    Args:
        path: Path to the image file. Both ``str`` and ``pathlib.Path``
            are accepted; internally the path is stored as a
            ``pathlib.Path`` for consistency.

    Example:
        >>> with FileImageSensor("frame.jpg") as sensor:
        ...     msg = sensor.read()
        ...     print(msg.width, msg.height)
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        # Normalise to Path immediately so all subsequent code is uniform.
        self._path: pathlib.Path = pathlib.Path(path)

    def open(self) -> None:
        """Validate that the image file exists.

        Raises:
            FileNotFoundError: If ``path`` does not point to an existing file.
        """
        if not self._path.is_file():
            msg = f"FileImageSensor: image file not found: {self._path}"
            raise FileNotFoundError(msg)
        logger.debug("FileImageSensor opened: %s", self._path)

    def read(self) -> RawFrameMessage:
        """Load the image from disk and return it as a ``RawFrameMessage``.

        Uses ``cv2.imread`` which returns a BGR NumPy array. The timestamp
        is captured immediately after the read so it reflects when the data
        became available to the pipeline.

        Returns:
            A ``RawFrameMessage`` with the loaded frame and provenance info.

        Raises:
            IOError: If ``cv2.imread`` returns ``None`` (unsupported format,
                corrupt file, or permission error).
        """
        frame = cv2.imread(str(self._path))
        if frame is None:
            logger.error("FileImageSensor: could not read %s", self._path)
            msg = f"FileImageSensor: could not load image: {self._path}"
            raise OSError(msg)

        h, w = frame.shape[:2]
        return RawFrameMessage(
            frame=frame,
            timestamp=time.time(),
            source=str(self._path),
            width=w,
            height=h,
        )

    def close(self) -> None:
        """No-op: file-based reads hold no persistent resources.

        Implemented to satisfy the ``BaseSensor`` contract and to allow
        ``FileImageSensor`` to be used safely as a context manager.
        """
        logger.debug("FileImageSensor closed: %s", self._path)
