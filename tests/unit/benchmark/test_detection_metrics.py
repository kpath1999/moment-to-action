"""Unit tests for detection metric computation helpers."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from moment_to_action.benchmark._detection_metrics import _recall_at_iou_50, compute_detection_map
from moment_to_action.benchmark._oracle_ground_truth import OracleBox, OracleDetection


class _FakeCOCO:
    def __init__(self) -> None:
        self.dataset: dict[str, object] = {}

    def createIndex(self) -> None:  # noqa: N802
        return

    def loadRes(self, data: object) -> object:  # noqa: N802
        return data


class _FakeCOCOeval:
    def __init__(self, coco_gt: object, coco_dt: object, iouType: str) -> None:  # noqa: N803
        del coco_gt, coco_dt, iouType
        self.stats = np.array([0.42, 0.61, 0.53] + [0.0] * 9, dtype=np.float32)
        self.eval = {
            "precision": np.array([[[[[0.8]], [[0.6]]], [[[0.8]], [[0.6]]]]], dtype=np.float32)
        }

    def evaluate(self) -> None:
        return

    def accumulate(self) -> None:
        return

    def summarize(self) -> None:
        return


@pytest.mark.unit
def test_compute_detection_map_with_fake_pycocotools(monkeypatch: pytest.MonkeyPatch) -> None:
    """MAP helper should consume pycocotools outputs and compute expected metrics."""
    pycocotools = types.ModuleType("pycocotools")
    coco_mod = types.ModuleType("pycocotools.coco")
    cocoeval_mod = types.ModuleType("pycocotools.cocoeval")
    coco_mod.COCO = _FakeCOCO  # type: ignore[attr-defined]
    cocoeval_mod.COCOeval = _FakeCOCOeval  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pycocotools", pycocotools)
    monkeypatch.setitem(sys.modules, "pycocotools.coco", coco_mod)
    monkeypatch.setitem(sys.modules, "pycocotools.cocoeval", cocoeval_mod)

    gt = [
        OracleDetection(
            image_name="img.jpg",
            boxes=[OracleBox(0.0, 0.0, 10.0, 10.0, "person", 1.0)],
        )
    ]
    preds = [
        OracleDetection(
            image_name="img.jpg",
            boxes=[OracleBox(0.0, 0.0, 10.0, 10.0, "person", 0.9)],
        )
    ]

    metrics = compute_detection_map(predictions=preds, ground_truth=gt)

    assert metrics.map_50 == pytest.approx(0.61)
    assert metrics.map_75 == pytest.approx(0.53)
    assert metrics.recall_50 == pytest.approx(1.0)
    assert metrics.per_class_ap["person"] == pytest.approx(0.8)


@pytest.mark.unit
def test_compute_detection_map_returns_zero_for_empty_inputs() -> None:
    metrics = compute_detection_map(predictions=[], ground_truth=[])
    assert metrics.map_50 == 0.0
    assert metrics.map_75 == 0.0
    assert metrics.recall_50 == 0.0


@pytest.mark.unit
def test_compute_detection_map_returns_zero_when_category_names_empty() -> None:
    gt = [OracleDetection(image_name="img.jpg", boxes=[])]
    preds = [OracleDetection(image_name="img.jpg", boxes=[])]
    metrics = compute_detection_map(predictions=preds, ground_truth=gt)
    assert metrics.map_50 == 0.0


@pytest.mark.unit
def test_compute_detection_map_returns_zero_when_no_valid_annotations() -> None:
    gt = [
        OracleDetection(
            image_name="img.jpg",
            boxes=[OracleBox(1.0, 1.0, 1.0, 2.0, "person", 1.0)],
        )
    ]
    preds = [
        OracleDetection(
            image_name="img.jpg",
            boxes=[OracleBox(0.0, 0.0, 0.0, 0.0, "person", 0.5)],
        )
    ]
    metrics = compute_detection_map(predictions=preds, ground_truth=gt)
    assert metrics.map_50 == 0.0


@pytest.mark.unit
def test_compute_detection_map_returns_zero_when_no_matching_predictions() -> None:
    gt = [
        OracleDetection(
            image_name="img.jpg",
            boxes=[OracleBox(0.0, 0.0, 10.0, 10.0, "person", 1.0)],
        )
    ]
    preds = [
        OracleDetection(
            image_name="other.jpg",
            boxes=[OracleBox(0.0, 0.0, 10.0, 10.0, "person", 0.9)],
        )
    ]
    metrics = compute_detection_map(predictions=preds, ground_truth=gt)
    assert metrics.map_50 == 0.0


@pytest.mark.unit
def test_compute_detection_map_returns_zero_when_labels_missing() -> None:
    gt = [OracleDetection(image_name="img.jpg", boxes=[OracleBox(0, 0, 10, 10, "", 1.0)])]
    metrics = compute_detection_map(predictions=[], ground_truth=gt)
    assert metrics.map_50 == 0.0


@pytest.mark.unit
def test_compute_detection_map_skips_prediction_with_missing_category_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pycocotools = types.ModuleType("pycocotools")
    coco_mod = types.ModuleType("pycocotools.coco")
    cocoeval_mod = types.ModuleType("pycocotools.cocoeval")
    coco_mod.COCO = _FakeCOCO  # type: ignore[attr-defined]
    cocoeval_mod.COCOeval = _FakeCOCOeval  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycocotools", pycocotools)
    monkeypatch.setitem(sys.modules, "pycocotools.coco", coco_mod)
    monkeypatch.setitem(sys.modules, "pycocotools.cocoeval", cocoeval_mod)

    original_sorted = sorted

    def _sorted_override(
        iterable: Iterable[object],
        *args: Any,
        **kwargs: Any,
    ) -> list[object]:
        values = list(iterable)
        if set(values) == {"person", "unknown"}:
            return ["person"]
        return original_sorted(values, *args, **kwargs)

    monkeypatch.setattr("builtins.sorted", _sorted_override)

    gt = [
        OracleDetection(
            image_name="img.jpg",
            boxes=[OracleBox(0.0, 0.0, 10.0, 10.0, "person", 1.0)],
        )
    ]
    preds = [
        OracleDetection(
            image_name="img.jpg",
            boxes=[OracleBox(0.0, 0.0, 10.0, 10.0, "unknown", 0.9)],
        )
    ]
    metrics = compute_detection_map(predictions=preds, ground_truth=gt)
    assert metrics.map_50 == 0.0


@pytest.mark.unit
def test_compute_detection_map_returns_zero_when_evaluator_stats_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EmptyStatsCOCOeval(_FakeCOCOeval):
        def __init__(self, coco_gt: object, coco_dt: object, iouType: str) -> None:  # noqa: N803
            super().__init__(coco_gt, coco_dt, iouType)
            self.stats = np.array([], dtype=np.float32)

    pycocotools = types.ModuleType("pycocotools")
    coco_mod = types.ModuleType("pycocotools.coco")
    cocoeval_mod = types.ModuleType("pycocotools.cocoeval")
    coco_mod.COCO = _FakeCOCO  # type: ignore[attr-defined]
    cocoeval_mod.COCOeval = _EmptyStatsCOCOeval  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycocotools", pycocotools)
    monkeypatch.setitem(sys.modules, "pycocotools.coco", coco_mod)
    monkeypatch.setitem(sys.modules, "pycocotools.cocoeval", cocoeval_mod)

    gt = [OracleDetection(image_name="img.jpg", boxes=[OracleBox(0, 0, 10, 10, "person", 1.0)])]
    preds = [OracleDetection(image_name="img.jpg", boxes=[OracleBox(0, 0, 10, 10, "person", 0.9)])]
    metrics = compute_detection_map(predictions=preds, ground_truth=gt)
    assert metrics.map_50 == 0.0


@pytest.mark.unit
def test_compute_detection_map_recall_zero_when_gt_has_no_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pycocotools = types.ModuleType("pycocotools")
    coco_mod = types.ModuleType("pycocotools.coco")
    cocoeval_mod = types.ModuleType("pycocotools.cocoeval")
    coco_mod.COCO = _FakeCOCO  # type: ignore[attr-defined]
    cocoeval_mod.COCOeval = _FakeCOCOeval  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycocotools", pycocotools)
    monkeypatch.setitem(sys.modules, "pycocotools.coco", coco_mod)
    monkeypatch.setitem(sys.modules, "pycocotools.cocoeval", cocoeval_mod)

    gt = [OracleDetection(image_name="img.jpg", boxes=[])]
    preds = [OracleDetection(image_name="img.jpg", boxes=[OracleBox(0, 0, 10, 10, "person", 0.9)])]
    metrics = compute_detection_map(predictions=preds, ground_truth=gt)
    assert metrics.recall_50 == 0.0


@pytest.mark.unit
def test_recall_at_iou_50_returns_zero_when_no_gt_boxes() -> None:
    gt = [OracleDetection(image_name="img.jpg", boxes=[])]
    preds = [OracleDetection(image_name="img.jpg", boxes=[OracleBox(0, 0, 1, 1, "a", 0.9)])]
    assert _recall_at_iou_50(predictions=preds, ground_truth=gt) == 0.0
