"""Sensor-layer messages for raw camera / frame data."""

from __future__ import annotations

from numpy.typing import NDArray  # noqa: TC002

from ._base import BaseMessage


class RawFrameMessage(BaseMessage):
    """Raw frame captured directly from a sensor or video source."""

    frame: NDArray | None
    """Raw image as a NumPy array (HxWxC, BGR or RGB). ``None`` signals a dropped frame."""

    source: str = ""
    """Identifier for the capture device or stream (e.g. ``"cam0"``)."""

    width: int = 0
    """Frame width in pixels; ``0`` when unknown."""

    height: int = 0
    """Frame height in pixels; ``0`` when unknown."""

class AudioInput(BaseMessage):
    """Raw audio captured directly from a sensor or stream."""

    waveform: NDArray | None
    """Raw mono audio as a NumPy array. ``None`` signals a dropped chunk."""

    source: str = ""
    """Identifier for the capture device or stream (e.g. ``"mic0"``)."""

    sample_rate: int = 16000
    """Sampling rate in Hz."""

    num_samples: int = 0
    """Number of samples in ``waveform``; ``0`` when unknown."""
