"""Abstract base class for all sensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from moment_to_action.messages.sensor import RawFrameMessage


class BaseSensor(ABC):
    """Abstract base for all sensors.

    Sensors are the entry point for raw data into the pipeline. Each
    concrete sensor wraps a single data source (file, camera, network
    stream, etc.) and normalises its output into a ``RawFrameMessage``.

    Use as a context manager to ensure ``open``/``close`` are paired:

    Example:
        >>> with FileSensor("frame.jpg") as sensor:
        ...     msg = sensor.read()
    """

    @abstractmethod
    def read(self) -> RawFrameMessage:
        """Read one frame/sample from the sensor.

        Returns:
            A ``RawFrameMessage`` containing the captured frame and metadata.

        Raises:
            IOError: If the frame cannot be read.
        """
        ...

    @abstractmethod
    def open(self) -> None:
        """Initialise the sensor (open device, file handle, etc.).

        Must be called before :meth:`read`. When using the sensor as a
        context manager this is called automatically by ``__enter__``.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release sensor resources.

        Safe to call even if :meth:`open` was never called. When using
        the sensor as a context manager this is called automatically by
        ``__exit__``.
        """
        ...

    def __enter__(self) -> Self:
        """Open the sensor and return self for use in a ``with`` block."""
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        """Close the sensor on exit from a ``with`` block."""
        self.close()
