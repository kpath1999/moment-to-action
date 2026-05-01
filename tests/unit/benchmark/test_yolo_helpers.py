from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moment_to_action.benchmark._benchmarks._yolo import (
    _build_oracle_boxes,
    _letterbox_resize,
    _load_yolo_tensor,
    _nms_numpy,
    _parse_yolo_boxes,
)


@pytest.mark.unit
def test_nms_numpy_keeps_best_overlapping_box() -> None:
    boxes = np.array(
        [
            [10.0, 10.0, 20.0, 20.0],
            [10.5, 10.5, 20.5, 20.5],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8], dtype=np.float32)
    kept = _nms_numpy(boxes, scores, iou_threshold=0.4)
    assert kept == [0]


@pytest.mark.unit
def test_build_oracle_boxes_clamps_and_labels() -> None:
    boxes = np.array([[-1.0, -2.0, 1000.0, 2000.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    class_ids = np.array([1], dtype=np.int32)
    result = _build_oracle_boxes(
        boxes,
        scores,
        class_ids,
        img_w=640,
        img_h=480,
        class_labels=("person", "car"),
    )
    assert len(result) == 1
    assert result[0].x1 == pytest.approx(0.0)
    assert result[0].y1 == pytest.approx(0.0)
    assert result[0].x2 == pytest.approx(640.0)
    assert result[0].y2 == pytest.approx(480.0)
    assert result[0].label == "car"


@pytest.mark.unit
def test_parse_yolo_boxes_three_tensor_format() -> None:
    boxes = np.array([[[10.0, 20.0, 30.0, 40.0]]], dtype=np.float32)
    scores = np.array([[0.8]], dtype=np.float32)
    class_ids = np.array([[0]], dtype=np.int32)
    parsed = _parse_yolo_boxes([boxes, scores, class_ids], (1, 3, 640, 640), conf_threshold=0.5)
    assert len(parsed) == 1
    assert parsed[0].x1 == pytest.approx(10.0)


@pytest.mark.unit
def test_parse_yolo_boxes_combined_format_transposed() -> None:
    arr = np.zeros((1, 84, 1), dtype=np.float32)
    arr[0, 0, 0] = 320.0
    arr[0, 1, 0] = 320.0
    arr[0, 2, 0] = 100.0
    arr[0, 3, 0] = 100.0
    arr[0, 4, 0] = 0.9
    parsed = _parse_yolo_boxes([arr], (1, 3, 640, 640), conf_threshold=0.5)
    assert len(parsed) == 1
    assert parsed[0].confidence == pytest.approx(0.9)


@pytest.mark.unit
def test_parse_yolo_boxes_invalid_inputs_return_empty() -> None:
    assert _parse_yolo_boxes("bad", (1, 3, 640, 640)) == []
    assert _parse_yolo_boxes([np.zeros((3,), dtype=np.float32)], (1, 3, 640, 640)) == []


@pytest.mark.unit
def test_letterbox_resize_shape() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = _letterbox_resize(img, 320, 320)
    assert out.shape == (320, 320, 3)


@pytest.mark.unit
def test_nms_numpy_single_box_kept() -> None:
    boxes = np.array([[1.0, 1.0, 2.0, 2.0]], dtype=np.float32)
    scores = np.array([0.5], dtype=np.float32)
    assert _nms_numpy(boxes, scores) == [0]


@pytest.mark.unit
def test_parse_yolo_boxes_dict_output_and_empty_after_threshold() -> None:
    arr = np.zeros((1, 84, 1), dtype=np.float32)
    arr[0, 0, 0] = 10.0
    arr[0, 1, 0] = 10.0
    arr[0, 2, 0] = 2.0
    arr[0, 3, 0] = 2.0
    arr[0, 4, 0] = 0.1
    parsed = _parse_yolo_boxes({"out": arr}, (1, 3, 640, 640), conf_threshold=0.9)
    assert parsed == []


@pytest.mark.unit
def test_load_yolo_tensor_supports_nchw_and_nhwc(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "img.jpg"
    Image.fromarray(np.zeros((10, 20, 3), dtype=np.uint8)).save(image_path)

    nchw = _load_yolo_tensor(image_path, (1, 3, 64, 64))
    nhwc = _load_yolo_tensor(image_path, (1, 64, 64, 3))

    assert nchw.shape == (1, 3, 64, 64)
    assert nhwc.shape == (1, 64, 64, 3)


@pytest.mark.unit
def test_parse_yolo_boxes_nhwc_input_shape_path() -> None:
    arr = np.zeros((1, 84, 1), dtype=np.float32)
    arr[0, 0, 0] = 10.0
    arr[0, 1, 0] = 10.0
    arr[0, 2, 0] = 4.0
    arr[0, 3, 0] = 4.0
    arr[0, 4, 0] = 0.95
    parsed = _parse_yolo_boxes([arr], (1, 64, 64, 3), conf_threshold=0.5)
    assert len(parsed) == 1


@pytest.mark.unit
def test_parse_yolo_boxes_three_tensor_empty_after_mask() -> None:
    boxes = np.array([[[1.0, 1.0, 2.0, 2.0]]], dtype=np.float32)
    scores = np.array([[0.1]], dtype=np.float32)
    class_ids = np.array([[0]], dtype=np.int32)
    parsed = _parse_yolo_boxes([boxes, scores, class_ids], (1, 3, 64, 64), conf_threshold=0.9)
    assert parsed == []


@pytest.mark.unit
def test_parse_yolo_boxes_rejects_small_feature_matrix() -> None:
    arr = np.zeros((1, 4), dtype=np.float32)
    parsed = _parse_yolo_boxes(arr, (1, 3, 64, 64), conf_threshold=0.1)
    assert parsed == []


@pytest.mark.unit
def test_parse_yolo_boxes_gpu_low_confidence_passes_with_low_threshold() -> None:
    """Simulates GPU TFLite output: max class prob ~0.08, sparse distribution.

    With conf_threshold=0.25 (CPU default) no boxes are returned.
    With conf_threshold=0.05 (GPU default) the anchor is recovered.
    """
    # Shape (1, 84, 1): one anchor with cx=0.5, cy=0.5, w=0.2, h=0.2,
    # class[0] score = 0.08 (typical for a sparse GPU FP16 model).
    arr = np.zeros((1, 84, 1), dtype=np.float32)
    arr[0, 0, 0] = 0.5  # cx (normalized)
    arr[0, 1, 0] = 0.5  # cy
    arr[0, 2, 0] = 0.2  # w
    arr[0, 3, 0] = 0.2  # h
    arr[0, 4, 0] = 0.08  # class[0] score — low but valid for GPU path

    # CPU default threshold: no boxes.
    assert _parse_yolo_boxes([arr], (1, 3, 640, 640), conf_threshold=0.25) == []

    # GPU threshold: box is recovered.
    result = _parse_yolo_boxes([arr], (1, 3, 640, 640), conf_threshold=0.05)
    assert len(result) == 1
    assert result[0].confidence == pytest.approx(0.08)


@pytest.mark.unit
def test_parse_yolo_boxes_cpu_and_gpu_outputs_share_same_decoder() -> None:
    """CPU and GPU backends produce OracleBoxes via identical decode logic.

    Verifies that when the same anchor data is present in both a "high-confidence
    CPU-like" tensor and a "low-confidence GPU-like" tensor, the decoder returns
    consistent geometry and label regardless of the confidence magnitude.
    """

    def _make_tensor(class_score: float) -> list[np.ndarray]:
        arr = np.zeros((1, 84, 1), dtype=np.float32)
        arr[0, 0, 0] = 320.0  # cx in input pixels
        arr[0, 1, 0] = 240.0  # cy
        arr[0, 2, 0] = 100.0  # w
        arr[0, 3, 0] = 80.0  # h
        arr[0, 4, 0] = class_score  # class[0]
        return [arr]

    cpu_result = _parse_yolo_boxes(_make_tensor(0.6), (1, 3, 640, 640), conf_threshold=0.25)
    gpu_result = _parse_yolo_boxes(_make_tensor(0.08), (1, 3, 640, 640), conf_threshold=0.05)

    assert len(cpu_result) == 1
    assert len(gpu_result) == 1

    # Geometry must be identical — only confidence differs.
    assert cpu_result[0].x1 == pytest.approx(gpu_result[0].x1)
    assert cpu_result[0].y1 == pytest.approx(gpu_result[0].y1)
    assert cpu_result[0].x2 == pytest.approx(gpu_result[0].x2)
    assert cpu_result[0].y2 == pytest.approx(gpu_result[0].y2)
    assert cpu_result[0].label == gpu_result[0].label
