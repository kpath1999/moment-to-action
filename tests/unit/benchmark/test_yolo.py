from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from moment_to_action.benchmark import YOLOBenchmark
from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_yolo_model_id_is_v12() -> None:
    assert YOLOBenchmark().model_id == ModelID.YOLO_V12_N


@pytest.mark.unit
def test_yolo_load_prefers_v12_int8_320_on_npu() -> None:
    benchmark = YOLOBenchmark(coco_dataset=mock.MagicMock())
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.NPU
    backend.get_input_details.return_value = [{"shape": [1, 320, 320, 3]}]

    manager = mock.MagicMock(spec=ModelManager)
    manager.is_available.side_effect = lambda model: model == ModelID.YOLO_V12_N_TFLITE_INT8_320
    manager.get_path.return_value = Path("/tmp/model_int8_320.tflite")

    benchmark._load_model(backend=backend, manager=manager)
    manager.get_path.assert_called_once_with(ModelID.YOLO_V12_N_TFLITE_INT8_320)


@pytest.mark.unit
def test_yolo_coco_accuracy_emits_map50_and_map75(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = YOLOBenchmark(coco_dataset=mock.MagicMock())
    benchmark._coco_dataset.images.return_value = [Path("/tmp/img.jpg")]  # type: ignore[union-attr]

    from moment_to_action.benchmark._oracle_ground_truth import OracleBox, OracleDetection

    benchmark._coco_dataset.instance_detections.return_value = [  # type: ignore[union-attr]
        OracleDetection(
            image_name="img.jpg",
            boxes=[OracleBox(0.0, 0.0, 10.0, 10.0, "person", 1.0)],
        )
    ]

    backend = mock.MagicMock()
    backend.run.return_value = [np.zeros((1, 84, 1), dtype=np.float32)]

    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._yolo._load_yolo_tensor",
        lambda *_args, **_kwargs: np.zeros((1, 3, 640, 640), dtype=np.float32),
    )
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._yolo._parse_yolo_boxes",
        lambda *_args, **_kwargs: [],
    )

    fake_metrics = mock.MagicMock(map_50=0.6, map_75=0.5, recall_50=0.7)
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._yolo.compute_detection_map",
        lambda **_kwargs: fake_metrics,
    )

    result = benchmark._evaluate_coco_accuracy(handle=object(), backend=backend)
    assert result == pytest.approx(0.6)
    details = benchmark._accuracy_details()
    assert details is not None
    assert details["map_50"] == pytest.approx(0.6)
    assert details["map_75"] == pytest.approx(0.5)
