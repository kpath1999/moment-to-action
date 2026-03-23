"""Conftest for integration tests.

Provides fixtures for test images and compute infrastructure used in integration tests.
Model paths are resolved by the stages themselves via ModelManager.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from moment_to_action.hardware import ComputeBackend

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_image_path() -> Path:
    """Return the path to the test image (pedestrain.jpg in tests/int/images/)."""
    image_path = Path(__file__).parent / "images" / "pedestrian.jpg"
    if not image_path.exists():
        msg = f"Test image not found: {image_path}"
        raise FileNotFoundError(msg)
    return image_path


@pytest.fixture(scope="session")
def compute_backend() -> ComputeBackend:
    """Return a ComputeBackend instance for inference.

    Session-scoped so the same backend is reused across all tests.
    """
    return ComputeBackend()
