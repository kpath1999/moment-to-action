from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from moment_to_action.benchmark._benchmarks._rf_detr_n import RFDETRBenchmark, _parse_rfdetr_boxes
from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_rfdetr_model_id() -> None:
    assert RFDETRBenchmark(coco_dataset=mock.MagicMock()).model_id == ModelID.RF_DETR_N


@pytest.mark.unit
def test_rfdetr_load_always_uses_rf_detr_n() -> None:
    benchmark = RFDETRBenchmark(coco_dataset=mock.MagicMock())
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.NPU
    backend.get_input_details.return_value = [{"shape": [1, 3, 640, 640]}]

    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/rf_detr_n.onnx")

    benchmark._load_model(backend=backend, manager=manager)
    manager.get_path.assert_called_once_with(ModelID.RF_DETR_N)


@pytest.mark.unit
def test_rfdetr_coco_accuracy_emits_map_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = RFDETRBenchmark(coco_dataset=mock.MagicMock())
    benchmark._coco_dataset.images.return_value = [Path("/tmp/img.jpg")]  # type: ignore[union-attr]

    from moment_to_action.benchmark._oracle_ground_truth import OracleBox, OracleDetection

    benchmark._coco_dataset.instance_detections.return_value = [  # type: ignore[union-attr]
        OracleDetection(
            image_name="img.jpg",
            boxes=[OracleBox(0.0, 0.0, 10.0, 10.0, "person", 1.0)],
        )
    ]

    backend = mock.MagicMock()
    backend.run.return_value = [
        np.array([[[2.0, -2.0]]], dtype=np.float32),
        np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32),
    ]

    image = mock.MagicMock()
    image.__enter__.return_value = image
    image.__exit__.return_value = None
    image.size = (640, 480)
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._rf_detr_n.Image.open",
        lambda _path: image,
    )
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._rf_detr_n._load_rfdetr_tensor",
        lambda *_args, **_kwargs: np.zeros((1, 3, 640, 640), dtype=np.float32),
    )

    fake_metrics = mock.MagicMock(map_50=0.57, map_75=0.44, recall_50=0.68)
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._rf_detr_n.compute_detection_map",
        lambda **_kwargs: fake_metrics,
    )

    result = benchmark._evaluate_coco_accuracy(handle=object(), backend=backend)
    assert result == pytest.approx(0.57)
    details = benchmark._accuracy_details()
    assert details is not None
    assert details["map_50"] == pytest.approx(0.57)
    assert details["map_75"] == pytest.approx(0.44)


@pytest.mark.unit
def test_parse_rfdetr_boxes_basic_decode() -> None:
    parsed = _parse_rfdetr_boxes(
        [
            np.array([[[2.0, -2.0]]], dtype=np.float32),
            np.array([[[0.5, 0.5, 0.2, 0.4]]], dtype=np.float32),
        ],
        image_width=200,
        image_height=100,
        conf_threshold=0.5,
        class_labels=("person", "car"),
    )
    assert len(parsed) == 1
    assert parsed[0].label == "person"
    assert parsed[0].x1 == pytest.approx(80.0)
    assert parsed[0].x2 == pytest.approx(120.0)


@pytest.mark.unit
def test_parse_rfdetr_boxes_invalid_payload_returns_empty() -> None:
    assert _parse_rfdetr_boxes("bad", image_width=100, image_height=100) == []
