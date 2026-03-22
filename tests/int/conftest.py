"""Conftest for integration tests.

Provides fixtures for ML models and test images used in integration tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import platformdirs
import pytest

from moment_to_action.hardware import ComputeBackend

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


def _get_model_cache_dir() -> Path:
    """Return the model cache directory, creating it if needed."""
    cache_dir = Path(platformdirs.user_cache_path("moment-to-action")) / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture(scope="session")
def yolo_model_path() -> Generator[Path, None, None]:
    """Return the path to the YOLO ONNX model.

    Uses the model already present in models/yolo/model.onnx (with its
    external .data file) since that matches the output format expected by
    YOLOStage: outputs[0]=[1,N,4] boxes, outputs[1]=[1,N] scores.
    """
    # Use the vendored model in tests/int/models/yolo/ — small enough to commit
    # and already in the correct output format for YOLOStage.
    vendored = Path(__file__).parent / "models" / "yolo" / "model.onnx"
    logger.info("Using vendored YOLO model: %s", vendored)
    return vendored


@pytest.fixture(scope="session")
def mobileclip_model_path() -> Generator[Path, None, None]:
    """Download and cache the MobileCLIP model from HuggingFace.

    Downloads mobileclip_s2_datacompdr_last.tflite from HuggingFace Hub
    if not already cached.

    Skips the test if download fails (network issues in CI).
    """
    cache_dir = _get_model_cache_dir()
    model_path = cache_dir / "mobileclip_s2" / "mobileclip_s2_datacompdr_last.tflite"

    if model_path.exists():
        logger.info("Using cached MobileCLIP model: %s", model_path)
        yield model_path
        return

    try:
        from huggingface_hub import hf_hub_download

        logger.info("Downloading MobileCLIP model from HuggingFace...")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        hf_hub_download(
            repo_id="anton96vice/mobileclip2_tflite",
            filename="mobileclip_s2_datacompdr_last.tflite",
            local_dir=str(model_path.parent),
        )

        logger.info("Cached MobileCLIP model to %s", model_path)
        yield model_path
    except Exception as e:  # noqa: BLE001
        msg = f"Model download failed: {e}"
        logger.warning("Failed to download MobileCLIP model: %s. Skipping test.", e)
        pytest.skip(msg)


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
