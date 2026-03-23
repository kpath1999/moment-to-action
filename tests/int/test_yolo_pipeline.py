"""Integration tests for the full YOLO detection pipeline.

Tests the complete pipeline: FileImageSensor → PreprocessorStage → YOLOStage → ReasoningStage
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from moment_to_action.hardware import ComputeBackend
from moment_to_action.messages import DetectionMessage, FrameTensorMessage, ReasoningMessage
from moment_to_action.models import ModelManager
from moment_to_action.pipeline import Pipeline
from moment_to_action.sensors import FileImageSensor
from moment_to_action.stages.llm import ReasoningStage
from moment_to_action.stages.video import PreprocessorStage, YOLOStage

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
def test_yolo_pipeline_full(test_image_path: Path) -> None:
    """Test the complete YOLO pipeline from image load to reasoning output.

    Uses pedestrian.jpg which produces a person detection at 0.91 confidence,
    guaranteeing a non-None result from the full pipeline.

    Asserts:
    - result is ReasoningMessage with latency > 0
    - Reasoning response is non-empty
    """
    sensor = FileImageSensor(test_image_path)
    sensor.open()
    raw_msg = sensor.read()
    sensor.close()

    backend = ComputeBackend()
    manager = ModelManager()
    # Stages resolve their own model paths via ModelManager.
    pipeline = Pipeline(
        [
            _preprocess_stage(),
            YOLOStage(backend=backend, manager=manager),
            ReasoningStage(),
        ]
    )

    result = pipeline.run(raw_msg)

    assert isinstance(result, ReasoningMessage), (
        f"Final message should be ReasoningMessage, got {type(result).__name__}"
    )
    assert result.latency_ms > 0, "Pipeline latency should be > 0"
    assert result.response, "Reasoning response should be non-empty"


@pytest.mark.integration
def test_yolo_detections(test_image_path: Path) -> None:
    """Test that the preprocess → YOLO pipeline detects objects in pedestrian.jpg.

    pedestrian.jpg produces a person detection at 0.91 confidence,
    well above the default 0.5 threshold.

    Asserts:
    - At least 1 detection is returned
    - All bounding box properties are valid
    - Latency > 0 ms
    """
    sensor = FileImageSensor(test_image_path)
    sensor.open()
    raw_msg = sensor.read()
    sensor.close()

    backend = ComputeBackend()
    manager = ModelManager()
    pipeline = Pipeline(
        [
            _preprocess_stage(),
            YOLOStage(backend=backend, manager=manager),
        ]
    )

    result = pipeline.run(raw_msg)

    assert isinstance(result, DetectionMessage)
    assert len(result.boxes) >= 1, "Should have at least 1 detection"
    assert result.latency_ms > 0, "YOLO latency should be > 0"

    for box in result.boxes:
        assert 0 <= box.confidence <= 1, f"Confidence should be in [0, 1], got {box.confidence}"
        assert box.label, "Label should be non-empty"
        assert box.class_id >= 0, "Class ID should be non-negative"
        assert box.x1 < box.x2, "x1 should be less than x2"
        assert box.y1 < box.y2, "y1 should be less than y2"


@pytest.mark.integration
def test_preprocess_stage_output(test_image_path: Path) -> None:
    """Test that a single-stage preprocessing pipeline produces the correct tensor.

    Asserts:
    - Output tensor has shape [1, 3, 640, 640] (channels-first)
    - Tensor dtype is float32
    - Latency > 0 ms
    - original_size is preserved
    """
    sensor = FileImageSensor(test_image_path)
    sensor.open()
    raw_msg = sensor.read()
    sensor.close()

    pipeline = Pipeline([_preprocess_stage()])
    result = pipeline.run(raw_msg)

    assert isinstance(result, FrameTensorMessage)
    assert result.latency_ms > 0, "Preprocessing latency should be > 0"
    assert result.tensor.dtype == np.float32
    assert result.tensor.shape == (1, 3, 640, 640), f"Unexpected shape: {result.tensor.shape}"
    assert len(result.original_size) == 2, "original_size should be (W, H)"
