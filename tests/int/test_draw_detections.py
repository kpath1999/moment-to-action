"""Integration tests for drawing bounding boxes on images.

Tests saving detection results with visual annotations to files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import pytest

from moment_to_action.sensors import FileImageSensor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
def test_detection_visualization_no_detections(
    test_image_path: Path,
    tmp_path: Path,
) -> None:
    """Test that saving a frame with no detections (no drawing) works correctly.

    Asserts:
    - File is saved successfully
    - File exists and is a valid JPEG
    """
    sensor = FileImageSensor(test_image_path)
    sensor.open()
    raw_msg = sensor.read()
    sensor.close()

    assert raw_msg.frame is not None

    output_path = tmp_path / "no_detections.jpg"
    assert cv2.imwrite(str(output_path), raw_msg.frame.copy()), "Failed to save JPEG"

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    loaded = cv2.imread(str(output_path))
    assert loaded is not None, "Output is not a valid image"
