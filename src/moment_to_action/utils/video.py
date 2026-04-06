"""Video utility functions for frame processing and sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
from PIL import Image

if TYPE_CHECKING:
    import numpy as np


def to_pil_rgb(bgr_frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR frame into a PIL RGB image.

    Args:
        bgr_frame: A ``(H, W, 3)`` uint8 array in BGR channel order.

    Returns:
        A PIL ``Image`` in RGB mode.
    """
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def sample_frames(frames: list[np.ndarray], max_images: int) -> list[np.ndarray]:
    """Uniformly sample up to *max_images* frames, preserving temporal order.

    Args:
        frames: Ordered list of video frames.
        max_images: Maximum number of frames to return.

    Returns:
        A list of at most *max_images* frames uniformly spaced from *frames*.
    """
    if len(frames) <= max_images:
        return frames
    if max_images == 1:
        return [frames[0]]
    step = (len(frames) - 1) / (max_images - 1)
    indices = [round(i * step) for i in range(max_images)]
    return [frames[idx] for idx in indices]
