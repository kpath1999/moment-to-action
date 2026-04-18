"""Unit tests for detection metric computation helpers."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from moment_to_action.benchmark._detection_metrics import compute_detection_map
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
        self.stats = np.array([0.42, 0.61] + [0.0] * 10, dtype=np.float32)
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

    assert metrics.map_50_95 == pytest.approx(0.42)
    assert metrics.map_50 == pytest.approx(0.61)
    assert metrics.recall_50 == pytest.approx(1.0)
    assert metrics.per_class_ap["person"] == pytest.approx(0.8)
