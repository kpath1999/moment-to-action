"""Frame encoding helpers for VLM stages."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import TYPE_CHECKING

import cv2
from PIL import Image

if TYPE_CHECKING:
    import numpy as np

_DEFAULT_JPEG_QUALITY = 85


def bgr_to_b64(frame: np.ndarray, *, quality: int = _DEFAULT_JPEG_QUALITY) -> str:
    """Convert a BGR uint8 frame to a base64-encoded JPEG string.

    Args:
        frame: BGR uint8 image array (as stored on
            :class:`~moment_to_action.messages.sensor.RawFrameMessage` /
            :class:`~moment_to_action.messages.video.VideoClipMessage`).
        quality: JPEG encoding quality, 1-95.

    Returns:
        Base64-encoded JPEG bytes as a UTF-8 string (no ``data:`` prefix).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()
