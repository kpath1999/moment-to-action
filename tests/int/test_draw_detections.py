"""Integration tests for drawing bounding boxes on images.

Tests saving detection results with visual annotations to files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import pytest

from moment_to_action._pipeline import Pipeline
from moment_to_action.hardware import ComputeBackend
from moment_to_action.messages import DetectionMessage
from moment_to_action.sensors import FileImageSensor
from moment_to_action.stages.video import PreprocessorStage, YOLOStage

if TYPE_CHECKING:
    from pathlib import Path


def _detection_pipeline(yolo_model_path: Path) -> Pipeline:
    """Return a preprocess → YOLO pipeline ready to produce DetectionMessages."""
    backend = ComputeBackend()
    return Pipeline(
        [
            PreprocessorStage(target_size=(640, 640), letterbox=True, channels_first=True),
            YOLOStage(model_path=str(yolo_model_path), backend=backend),
        ]
    )


@pytest.mark.integration
def test_draw_detections(
    yolo_model_path: Path,
    test_image_path: Path,
    tmp_path: Path,
) -> None:
    """Test drawing bounding boxes on detected objects and saving to file.

    Uses pedestrian.jpg which reliably yields a person detection at 0.91 confidence.
    Runs the preprocess → YOLO pipeline, draws boxes, saves to tmp_path, and
    verifies the output is a valid JPEG.

    Asserts:
    - At least one detection is found
    - Output file exists and is non-empty
    - Saved image can be loaded by OpenCV
    """
    sensor = FileImageSensor(test_image_path)
    sensor.open()
    raw_msg = sensor.read()
    sensor.close()

    assert raw_msg.frame is not None
    input_frame = raw_msg.frame.copy()

    result = _detection_pipeline(yolo_model_path).run(raw_msg)

    assert isinstance(result, DetectionMessage), "pedestrian.jpg should yield detections"
    assert len(result.boxes) > 0

    # Draw bounding boxes onto the original frame
    frame_with_boxes = input_frame.copy()
    for box in result.boxes:
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        label = f"{box.label} {box.confidence:.2f}"

        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 0.5, 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x, text_y = x1, max(y1 - 5, text_size[1])

        cv2.rectangle(
            frame_with_boxes,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0], text_y + 5),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame_with_boxes, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness
        )

    output_path = tmp_path / "detections.jpg"
    assert cv2.imwrite(str(output_path), frame_with_boxes), f"Failed to write JPEG to {output_path}"

    assert output_path.exists()
    assert output_path.stat().st_size > 0, "Output file is empty"

    loaded = cv2.imread(str(output_path))
    assert loaded is not None, "Output file is not a valid image"
    assert loaded.shape == frame_with_boxes.shape, "Output shape mismatch"


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
