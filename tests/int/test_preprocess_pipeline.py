"""Integration tests for the preprocessing pipeline stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from moment_to_action.messages import FrameTensorMessage
from moment_to_action.pipeline import Pipeline
from moment_to_action.sensors import FileImageSensor
from moment_to_action.stages.video import PreprocessorStage

if TYPE_CHECKING:
    from pathlib import Path


def _preprocess_stage() -> PreprocessorStage:
    """Return a PreprocessorStage configured for YOLO (640x640, channels-first)."""
    return PreprocessorStage(
        target_size=(640, 640),
        letterbox=True,
        channels_first=True,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )


@pytest.mark.integration
def test_preprocess_stage_output(test_image_path: Path) -> None:
    """Preprocessing pipeline produces a correct YOLO-ready tensor.

    Asserts:
    - Output tensor has shape [1, 3, 640, 640] (channels-first)
    - Tensor dtype is float32
    - Latency >= 0 ms
    - original_size is preserved
    """
    sensor = FileImageSensor(test_image_path)
    sensor.open()
    raw_msg = sensor.read()
    sensor.close()

    pipeline = Pipeline([_preprocess_stage()])
    result = pipeline.run(raw_msg)

    assert isinstance(result, FrameTensorMessage)
    assert result.latency_ms > -0.01, "Preprocessing latency should be ~0 or positive"
    assert result.tensor.dtype == np.float32
    assert result.tensor.shape == (1, 3, 640, 640), f"Unexpected shape: {result.tensor.shape}"
    assert len(result.original_size) == 2, "original_size should be (W, H)"
