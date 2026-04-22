"""Unit tests for benchmark accuracy utilities (_accuracy.py)."""

from __future__ import annotations

import numpy as np
import pytest

from moment_to_action.benchmark._accuracy import (
    compute_iou,
    compute_map50,
    cosine_similarity,
    match_detections,
    mean_embedding_similarity,
    parse_yolo_outputs,
)

# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_iou_identical_boxes() -> None:
    box = np.array([10.0, 20.0, 50.0, 60.0], dtype=np.float32)
    assert compute_iou(box, box) == pytest.approx(1.0)


@pytest.mark.unit
def test_iou_non_overlapping_boxes() -> None:
    a = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    b = np.array([20.0, 20.0, 30.0, 30.0], dtype=np.float32)
    assert compute_iou(a, b) == pytest.approx(0.0)


@pytest.mark.unit
def test_iou_partial_overlap() -> None:
    # a = [0,0,4,4], b = [2,2,6,6]  =>  inter = 4, area_a=16, area_b=16, union=28
    a = np.array([0.0, 0.0, 4.0, 4.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)
    expected = 4.0 / 28.0
    assert compute_iou(a, b) == pytest.approx(expected, rel=1e-5)


@pytest.mark.unit
def test_iou_zero_area_box() -> None:
    a = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    b = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    assert compute_iou(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# match_detections
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_match_all_correct() -> None:
    box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    tp, fp, fn = match_detections([box], [box])
    assert tp == 1
    assert fp == 0
    assert fn == 0


@pytest.mark.unit
def test_match_no_gt() -> None:
    box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    tp, fp, fn = match_detections([box], [])
    assert tp == 0
    assert fp == 1
    assert fn == 0


@pytest.mark.unit
def test_match_no_pred() -> None:
    box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    tp, fp, fn = match_detections([], [box])
    assert tp == 0
    assert fp == 0
    assert fn == 1


@pytest.mark.unit
def test_match_low_iou_counts_as_fp() -> None:
    # Boxes barely overlap — IoU << 0.5
    pred = np.array([0.0, 0.0, 3.0, 3.0], dtype=np.float32)
    gt = np.array([5.0, 5.0, 10.0, 10.0], dtype=np.float32)
    tp, fp, fn = match_detections([pred], [gt], iou_threshold=0.5)
    assert tp == 0
    assert fp == 1
    assert fn == 1


@pytest.mark.unit
def test_match_detections_skips_already_matched_gt() -> None:
    gt = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    pred1 = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    pred2 = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    tp, fp, fn = match_detections([pred1, pred2], [gt], iou_threshold=0.5)
    assert tp == 1
    assert fp == 1
    assert fn == 0


# ---------------------------------------------------------------------------
# compute_map50
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_map50_perfect_predictions() -> None:
    box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    score = compute_map50([[box]], [[box]])
    assert score == pytest.approx(1.0)


@pytest.mark.unit
def test_map50_no_predictions_no_gt() -> None:
    score = compute_map50([[]], [[]])
    assert score == pytest.approx(0.0)


@pytest.mark.unit
def test_map50_mismatched_lists_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_map50([[]], [[], []])


@pytest.mark.unit
def test_map50_all_wrong() -> None:
    pred = np.array([0.0, 0.0, 3.0, 3.0], dtype=np.float32)
    gt = np.array([50.0, 50.0, 100.0, 100.0], dtype=np.float32)
    score = compute_map50([[pred]], [[gt]])
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cosine_similarity_identical() -> None:
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


@pytest.mark.unit
def test_cosine_similarity_orthogonal() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-7)


@pytest.mark.unit
def test_cosine_similarity_zero_vector() -> None:
    z = np.zeros(3, dtype=np.float32)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(z, v) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mean_embedding_similarity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mean_embedding_similarity_identical() -> None:
    v = np.ones((1, 128), dtype=np.float32)
    score = mean_embedding_similarity([v], [v])
    assert score == pytest.approx(1.0)


@pytest.mark.unit
def test_mean_embedding_similarity_empty_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        mean_embedding_similarity([np.ones(3, dtype=np.float32)], [])


@pytest.mark.unit
def test_mean_embedding_similarity_no_inputs() -> None:
    assert mean_embedding_similarity([], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# parse_yolo_outputs — 1-tensor format
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_yolo_1tensor_no_detections() -> None:
    # Raw [1, 84, N] tensor with all-zero class scores → no predictions
    raw = np.zeros((1, 84, 5), dtype=np.float32)
    result = parse_yolo_outputs([raw], confidence_threshold=0.25)
    assert result == []


@pytest.mark.unit
def test_parse_yolo_1tensor_one_detection() -> None:
    # cx=320, cy=320, w=100, h=100, class-0 score=0.9
    raw = np.zeros((1, 84, 1), dtype=np.float32)
    raw[0, 0, 0] = 320.0  # cx
    raw[0, 1, 0] = 320.0  # cy
    raw[0, 2, 0] = 100.0  # w
    raw[0, 3, 0] = 100.0  # h
    raw[0, 4, 0] = 0.9  # class-0 confidence
    result = parse_yolo_outputs([raw], confidence_threshold=0.5)
    assert len(result) == 1
    box = result[0]
    assert float(box[0]) == pytest.approx(270.0)  # x1 = cx - w/2
    assert float(box[1]) == pytest.approx(270.0)  # y1
    assert float(box[2]) == pytest.approx(370.0)  # x2 = cx + w/2
    assert float(box[3]) == pytest.approx(370.0)  # y2


@pytest.mark.unit
def test_parse_yolo_3tensor_one_detection() -> None:
    boxes = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype=np.float32)  # [1, 1, 4]
    scores = np.array([[0.8]], dtype=np.float32)  # [1, 1]
    class_ids = np.array([[0]], dtype=np.uint8)  # [1, 1]
    result = parse_yolo_outputs([boxes, scores, class_ids], confidence_threshold=0.5)  # type: ignore[list-item]
    assert len(result) == 1
    assert result[0].tolist() == pytest.approx([10.0, 20.0, 50.0, 60.0])


@pytest.mark.unit
def test_parse_yolo_malformed_tensor_returns_empty() -> None:
    # Only 2 outputs — not a valid 3-tensor format
    dummy = np.zeros((1, 10, 4), dtype=np.float32)
    result = parse_yolo_outputs([dummy, dummy])
    assert result == []


@pytest.mark.unit
def test_parse_yolo_1tensor_malformed_matrix_returns_empty() -> None:
    raw = np.zeros((1, 10, 2), dtype=np.float32)
    assert parse_yolo_outputs([raw]) == []


@pytest.mark.unit
def test_parse_yolo_3tensor_all_below_threshold_returns_empty() -> None:
    boxes = np.array([[[10.0, 20.0, 30.0, 40.0]]], dtype=np.float32)
    scores = np.array([[0.1]], dtype=np.float32)
    class_ids = np.array([[0]], dtype=np.uint8)
    assert parse_yolo_outputs([boxes, scores, class_ids], confidence_threshold=0.5) == []  # type: ignore[list-item]


@pytest.mark.unit
def test_parse_yolo_3tensor_runs_nms_branch() -> None:
    boxes = np.array([[[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5]]], dtype=np.float32)
    scores = np.array([[0.9, 0.8]], dtype=np.float32)
    class_ids = np.array([[0, 0]], dtype=np.uint8)
    parsed = parse_yolo_outputs([boxes, scores, class_ids], confidence_threshold=0.1)  # type: ignore[list-item]
    assert len(parsed) == 1
