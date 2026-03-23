"""Conftest for integration tests.

Provides fixtures for ML models and test images used in integration tests.
Models are loaded via ModelManager, which handles caching and downloads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from moment_to_action.hardware import ComputeBackend
from moment_to_action.models import ModelID, ModelManager

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def yolo_model_path() -> Path:
    """Return the path to the YOLO model via ModelManager.

    Loads the vendored YOLO V8 model which is already in the
    correct output format for YOLOStage.
    """
    manager = ModelManager()
    model_path = manager.get_path(ModelID.YOLO_V8)
    logger.info("Using YOLO model: %s", model_path)
    return model_path


@pytest.fixture(scope="session")
def mobileclip_model_path() -> Generator[Path, None, None]:
    """Download and cache the MobileCLIP model via ModelManager.

    Downloads mobileclip_s2_datacompdr_last.tflite from HuggingFace Hub
    if not already cached.

    Skips the test if download fails (network issues in CI).
    """
    manager = ModelManager()
    try:
        model_path = manager.get_path(ModelID.MOBILECLIP_S2)
        logger.info("Using MobileCLIP model: %s", model_path)
        yield model_path
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to download MobileCLIP model: %s. Skipping test.", e)
        pytest.skip(f"Model download failed: {e}")


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
