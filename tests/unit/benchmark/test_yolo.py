from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image as PILImage

from moment_to_action.benchmark import YOLOBenchmark
from moment_to_action.benchmark._benchmarks._yolo import (
    _effective_conf_threshold,
    _project_tflite_box_to_original,
)
from moment_to_action.benchmark._oracle_ground_truth import OracleBox, OracleDetection
from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_yolo_model_id_is_v12() -> None:
    assert YOLOBenchmark().model_id == ModelID.YOLO_V12_N


@pytest.mark.unit
def test_yolo_load_always_uses_v12_n() -> None:
    benchmark = YOLOBenchmark(coco_dataset=mock.MagicMock())
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.NPU

    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/yolo12n.onnx")

    benchmark._load_model(backend=backend, manager=manager)
    manager.get_path.assert_called_once_with(ModelID.YOLO_V12_N)


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


@pytest.mark.unit
def test_yolo_default_has_no_per_unit_threshold_override() -> None:
    """Default YOLO benchmark config should compare units at the same threshold."""
    benchmark = YOLOBenchmark(conf_threshold=0.25)
    assert benchmark._per_unit_conf_thresholds is None


@pytest.mark.unit
def test_yolo_per_unit_conf_thresholds_override_respected() -> None:
    """Explicit per_unit_conf_thresholds must override the built-in defaults."""
    benchmark = YOLOBenchmark(conf_threshold=0.25, per_unit_conf_thresholds={"gpu": 0.01})
    assert benchmark._per_unit_conf_thresholds == {"gpu": 0.01}


@pytest.mark.unit
def test_effective_conf_threshold_returns_override_for_matching_unit() -> None:
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.GPU
    result = _effective_conf_threshold(0.25, {"gpu": 0.05}, backend)
    assert result == pytest.approx(0.05)


@pytest.mark.unit
def test_effective_conf_threshold_returns_base_when_no_override() -> None:
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.CPU
    result = _effective_conf_threshold(0.25, {"gpu": 0.05}, backend)
    assert result == pytest.approx(0.25)


@pytest.mark.unit
def test_effective_conf_threshold_returns_base_when_dict_is_none() -> None:
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.GPU
    result = _effective_conf_threshold(0.25, None, backend)
    assert result == pytest.approx(0.25)


@pytest.mark.unit
def test_effective_conf_threshold_returns_base_when_backend_has_no_active_unit() -> None:
    # Backend without active_unit attribute (e.g., plain mock without spec)
    backend = object()
    result = _effective_conf_threshold(0.3, {"gpu": 0.05}, backend)
    assert result == pytest.approx(0.3)


@pytest.mark.unit
def test_yolo_gpu_threshold_applied_in_predict_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """With GPU backend, _predict_image must pass the per-unit GPU threshold."""
    benchmark = YOLOBenchmark(
        coco_dataset=mock.MagicMock(),
        conf_threshold=0.25,
        per_unit_conf_thresholds={"gpu": 0.05},
    )
    benchmark._input_shape = (1, 640, 640, 3)
    benchmark._is_tflite = True
    benchmark._debug_image_counter = 0

    used_threshold: list[float] = []

    def fake_parse(
        _raw_outputs: object,
        _input_shape: object,
        *,
        conf_threshold: float = 0.25,
        **_kw: object,
    ) -> list[object]:
        used_threshold.append(conf_threshold)
        return []

    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._yolo._parse_yolo_boxes",
        fake_parse,
    )
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._yolo._load_yolo_tensor",
        lambda *_a, **_k: np.zeros((1, 640, 640, 3), dtype=np.float32),
    )

    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.GPU
    backend.run.return_value = [np.zeros((1, 84, 1), dtype=np.float32)]

    gt_det = OracleDetection(image_name="img.jpg", boxes=[])
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = Path(f.name)
    PILImage.new("RGB", (640, 640)).save(tmp_path)

    benchmark._predict_image(tmp_path, gt_det, object(), backend)

    assert used_threshold == [pytest.approx(0.05)]


@pytest.mark.unit
def test_project_tflite_box_to_original_undoes_wide_image_letterbox() -> None:
    box = OracleBox(
        x1=0.25,
        y1=0.3125,
        x2=0.75,
        y2=0.6875,
        label="person",
        confidence=0.9,
    )

    projected = _project_tflite_box_to_original(
        box,
        orig_w=400,
        orig_h=200,
        model_w=640,
        model_h=640,
    )

    assert projected.x1 == pytest.approx(100.0)
    assert projected.y1 == pytest.approx(25.0)
    assert projected.x2 == pytest.approx(300.0)
    assert projected.y2 == pytest.approx(175.0)


@pytest.mark.unit
def test_project_tflite_box_to_original_square_image_is_direct_scale() -> None:
    box = OracleBox(
        x1=0.1,
        y1=0.2,
        x2=0.6,
        y2=0.8,
        label="person",
        confidence=0.9,
    )

    projected = _project_tflite_box_to_original(
        box,
        orig_w=640,
        orig_h=640,
        model_w=640,
        model_h=640,
    )

    assert projected.x1 == pytest.approx(64.0)
    assert projected.y1 == pytest.approx(128.0)
    assert projected.x2 == pytest.approx(384.0)
    assert projected.y2 == pytest.approx(512.0)
