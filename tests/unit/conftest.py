"""Conftest for unit tests.

Provides fixtures for common test utilities.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_image_array() -> np.ndarray:
    """Return a sample 480x640x3 BGR image array (uint8)."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_tensor() -> np.ndarray:
    """Return a sample 1x3x256x256 normalized float32 tensor (channels-first)."""
    return np.random.randn(1, 3, 256, 256).astype(np.float32)


@pytest.fixture
def sample_detection_box() -> dict:
    """Return a sample detection bounding box dict."""
    return {
        "label": "person",
        "confidence": 0.95,
        "box": [100, 150, 500, 600],  # [x1, y1, x2, y2]
    }
