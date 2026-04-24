from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from moment_to_action.benchmark._benchmarks._ssd_mobilenetv2 import (
    SSDMobileNetV2Benchmark,
    _parse_ssd_boxes,
)
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_ssd_model_id() -> None:
    benchmark = SSDMobileNetV2Benchmark(coco_dataset=mock.MagicMock())
    assert benchmark.model_id == ModelID.SSD_MOBILENETV2


@pytest.mark.unit
def test_ssd_load_uses_model_manager() -> None:
    benchmark = SSDMobileNetV2Benchmark(coco_dataset=mock.MagicMock())
    backend = mock.MagicMock()
    backend.get_input_details.return_value = [{"shape": [1, 3, 300, 300]}]
    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/ssd.onnx")

    benchmark._load_model(backend=backend, manager=manager)
    manager.get_path.assert_called_once_with(ModelID.SSD_MOBILENETV2)


@pytest.mark.unit
def test_ssd_coco_accuracy_emits_map_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = SSDMobileNetV2Benchmark(coco_dataset=mock.MagicMock())
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
        np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.9]], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    ]

    image = mock.MagicMock()
    image.__enter__.return_value = image
    image.__exit__.return_value = None
    image.size = (640, 480)
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._ssd_mobilenetv2.Image.open",
        lambda _path: image,
    )
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._ssd_mobilenetv2._load_ssd_tensor",
        lambda *_args, **_kwargs: np.zeros((1, 3, 300, 300), dtype=np.float32),
    )

    fake_metrics = mock.MagicMock(map_50=0.62, map_75=0.51, recall_50=0.73)
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._ssd_mobilenetv2.compute_detection_map",
        lambda **_kwargs: fake_metrics,
    )

    result = benchmark._evaluate_coco_accuracy(handle=object(), backend=backend)
    assert result == pytest.approx(0.62)
    details = benchmark._accuracy_details()
    assert details is not None
    assert details["map_50"] == pytest.approx(0.62)
    assert details["map_75"] == pytest.approx(0.51)


@pytest.mark.unit
def test_parse_ssd_boxes_parses_normalized_boxes() -> None:
    parsed = _parse_ssd_boxes(
        [
            np.array([[[0.2, 0.1, 0.8, 0.7]]], dtype=np.float32),
            np.array([[1.0]], dtype=np.float32),
            np.array([[0.95]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        ],
        image_width=100,
        image_height=200,
        conf_threshold=0.5,
    )
    assert len(parsed) == 1
    assert parsed[0].label == "person"
    assert parsed[0].x1 == pytest.approx(10.0)
    assert parsed[0].y1 == pytest.approx(40.0)
    assert parsed[0].x2 == pytest.approx(70.0)
    assert parsed[0].y2 == pytest.approx(160.0)


@pytest.mark.unit
def test_parse_ssd_boxes_invalid_payload_returns_empty() -> None:
    assert _parse_ssd_boxes("bad", image_width=100, image_height=100) == []
