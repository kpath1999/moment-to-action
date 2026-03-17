"""Sensor-layer messages for raw camera / frame data."""

from __future__ import annotations

import numpy as np  # noqa: TC002

from ._base import BaseMessage


class RawFrameMessage(BaseMessage):
    """Raw frame captured directly from a sensor or video source.

    Wraps an unprocessed image array together with provenance metadata
    so downstream stages can validate or log the origin of each frame.

    Attributes:
        frame: Raw image as a NumPy array (HxWxC, BGR or RGB).
               ``None`` signals a dropped or unavailable frame.
        source: Identifier for the capture device or stream (e.g. ``"cam0"``).
        width: Frame width in pixels; ``0`` when unknown.
        height: Frame height in pixels; ``0`` when unknown.
    """

    frame: np.ndarray | None
    source: str = ""
    width: int = 0
    height: int = 0
