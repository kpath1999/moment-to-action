from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from moment_to_action.benchmark import YOLOBenchmark
from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_yolo_load_and_input_shape_cpu_uses_onnx() -> None:
    """On CPU, _load_model picks the ONNX variant and input is NCHW."""
    benchmark = YOLOBenchmark()
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.CPU
    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/yolo.onnx")

    handle = benchmark._load_model(backend=backend, manager=manager)
    manager.get_path.assert_called_once_with(ModelID.YOLO_V8)
    backend.load_model.assert_called_once_with(Path("/tmp/yolo.onnx"))

    inputs = benchmark._make_dummy_input(handle, batch_size=3)
    assert isinstance(inputs, np.ndarray)
    assert inputs.shape == (3, 3, 640, 640)


@pytest.mark.unit
def test_yolo_load_npu_prefers_int8_320_when_available() -> None:
    """On NPU, _load_model picks YOLO_V8_TFLITE_INT8_320 first (fits TCM)."""
    benchmark = YOLOBenchmark()
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.NPU
    manager = mock.MagicMock(spec=ModelManager)
    manager.is_available.return_value = True
    tflite_path = Path("/tmp/yolo_npu_int8_320.tflite")
    manager.get_path.return_value = tflite_path
    # 320x320 TFLite model reports NHWC shape
    backend.get_input_details.return_value = [{"shape": [1, 320, 320, 3], "name": "input"}]

    handle = benchmark._load_model(backend=backend, manager=manager)
    manager.is_available.assert_called_once_with(ModelID.YOLO_V8_TFLITE_INT8_320)
    manager.get_path.assert_called_once_with(ModelID.YOLO_V8_TFLITE_INT8_320)
    backend.load_model.assert_called_once_with(tflite_path)

    inputs = benchmark._make_dummy_input(handle, batch_size=2)
    assert isinstance(inputs, np.ndarray)
    assert inputs.shape == (2, 320, 320, 3)


@pytest.mark.unit
def test_yolo_load_npu_falls_back_to_int8_640_when_320_unavailable() -> None:
    """On NPU with only 640-INT8 available, _load_model picks YOLO_V8_TFLITE_INT8."""
    benchmark = YOLOBenchmark()
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.NPU
    manager = mock.MagicMock(spec=ModelManager)
    # is_available: False for 320, True for 640-int8
    manager.is_available.side_effect = lambda m: m == ModelID.YOLO_V8_TFLITE_INT8
    tflite_path = Path("/tmp/yolo_npu_int8.tflite")
    manager.get_path.return_value = tflite_path
    backend.get_input_details.return_value = [{"shape": [1, 640, 640, 3], "name": "input"}]

    handle = benchmark._load_model(backend=backend, manager=manager)
    manager.is_available.assert_any_call(ModelID.YOLO_V8_TFLITE_INT8_320)
    manager.is_available.assert_any_call(ModelID.YOLO_V8_TFLITE_INT8)
    manager.get_path.assert_called_once_with(ModelID.YOLO_V8_TFLITE_INT8)
    backend.load_model.assert_called_once_with(tflite_path)

    inputs = benchmark._make_dummy_input(handle, batch_size=2)
    assert inputs.shape == (2, 640, 640, 3)  # type: ignore[attr-defined]


@pytest.mark.unit
def test_yolo_load_falls_back_to_onnx_when_tflite_unavailable() -> None:
    """On NPU without any TFLite model, _load_model falls back to ONNX."""
    benchmark = YOLOBenchmark()
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.NPU
    manager = mock.MagicMock(spec=ModelManager)
    manager.is_available.return_value = False
    onnx_path = Path("/tmp/yolo.onnx")
    manager.get_path.return_value = onnx_path

    benchmark._load_model(backend=backend, manager=manager)
    # NPU path checks 320 and 640-int8; both unavailable
    # then non-CPU path checks YOLO_V8_TFLITE; also unavailable
    manager.is_available.assert_any_call(ModelID.YOLO_V8_TFLITE_INT8_320)
    manager.is_available.assert_any_call(ModelID.YOLO_V8_TFLITE_INT8)
    manager.is_available.assert_any_call(ModelID.YOLO_V8_TFLITE)
    manager.get_path.assert_called_once_with(ModelID.YOLO_V8)
    backend.load_model.assert_called_once_with(onnx_path)

    inputs = benchmark._make_dummy_input(None, batch_size=1)
    assert inputs.shape == (1, 3, 640, 640)  # type: ignore[attr-defined]


@pytest.mark.unit
def test_yolo_model_id_returns_yolo_v8() -> None:
    """model_id property always returns YOLO_V8."""
    assert YOLOBenchmark().model_id == ModelID.YOLO_V8


@pytest.mark.unit
def test_yolo_run_inference_calls_backend_run() -> None:
    """_run_inference delegates to backend.run."""
    benchmark = YOLOBenchmark()
    backend = mock.MagicMock()
    inputs = np.zeros((1, 3, 640, 640), dtype=np.float32)
    handle = object()
    benchmark._run_inference(handle, inputs, backend)
    backend.run.assert_called_once_with(handle, inputs)


@pytest.mark.unit
def test_yolo_run_inference_raises_for_non_ndarray() -> None:
    """_run_inference raises TypeError when inputs is not an ndarray."""
    benchmark = YOLOBenchmark()
    backend = mock.MagicMock()
    with pytest.raises(TypeError, match="expects ndarray"):
        benchmark._run_inference(object(), {"data": 1}, backend)


@pytest.mark.unit
def test_yolo_evaluate_accuracy_returns_none_without_eval_images() -> None:
    """_evaluate_accuracy returns None when no eval images are configured."""
    benchmark = YOLOBenchmark()
    backend = mock.MagicMock()
    manager = mock.MagicMock(spec=ModelManager)
    result = benchmark._evaluate_accuracy(object(), backend, manager)
    assert result is None


@pytest.mark.unit
def test_yolo_evaluate_accuracy_returns_none_when_cv2_missing() -> None:
    """_evaluate_accuracy returns None gracefully when opencv is not installed."""
    img_path = Path("/tmp/dummy.jpg")
    benchmark = YOLOBenchmark(eval_image_paths=[img_path])
    backend = mock.MagicMock()
    manager = mock.MagicMock(spec=ModelManager)

    with mock.patch("builtins.__import__", side_effect=ImportError("cv2")):
        result = benchmark._evaluate_accuracy(object(), backend, manager)

    assert result is None


@pytest.mark.unit
def test_yolo_evaluate_accuracy_with_mocked_pipeline(tmp_path: Path) -> None:
    """_evaluate_accuracy computes mAP50 when oracle and variant agree perfectly."""
    import numpy as np

    # Create a placeholder image file (content doesn't matter; cv2 is mocked)
    img_file = tmp_path / "img.jpg"
    img_file.write_bytes(b"\xff\xd8\xff")

    benchmark = YOLOBenchmark(eval_image_paths=[img_file])

    # Both oracle and eval backends return the same raw 1-tensor output:
    # one box at [270,270,370,370] with score 0.9 for class 0
    raw_output = np.zeros((1, 84, 1), dtype=np.float32)
    raw_output[0, 0, 0] = 320.0
    raw_output[0, 1, 0] = 320.0
    raw_output[0, 2, 0] = 100.0
    raw_output[0, 3, 0] = 100.0
    raw_output[0, 4, 0] = 0.9

    fake_image = np.zeros((480, 640, 3), dtype=np.uint8)

    cpu_backend = mock.MagicMock()
    cpu_backend.run.return_value = [raw_output]

    eval_backend = mock.MagicMock()
    eval_backend.get_input_details.return_value = [{"shape": [1, 3, 640, 640]}]
    eval_backend.run.return_value = [raw_output]
    eval_backend.load_model.return_value = object()
    eval_backend.active_unit = ComputeUnit.CPU

    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/yolo.onnx")

    import cv2

    with (
        mock.patch(
            "moment_to_action.hardware.ComputeBackend",
            return_value=cpu_backend,
        ),
        mock.patch.object(cv2, "imread", return_value=fake_image),
        mock.patch.object(cv2, "resize", return_value=fake_image[:640, :640]),
    ):
        result = benchmark._evaluate_accuracy(object(), eval_backend, manager)

    # Perfect agreement → mAP50 = 1.0
    assert result == pytest.approx(1.0)


@pytest.mark.unit
def test_yolo_evaluate_coco_accuracy_uses_native_dataset_gt() -> None:
    """COCO path should use native dataset detections and compute detection metrics."""
    from moment_to_action.benchmark._oracle_ground_truth import OracleBox, OracleDetection

    dataset = mock.MagicMock()
    dataset.images.return_value = [Path("/tmp/000000000001.jpg")]
    dataset.instance_detections.return_value = [
        OracleDetection(
            image_name="000000000001.jpg",
            boxes=[
                OracleBox(
                    x1=10.0,
                    y1=20.0,
                    x2=30.0,
                    y2=40.0,
                    label="person",
                    confidence=1.0,
                )
            ],
        )
    ]

    benchmark = YOLOBenchmark(coco_dataset=dataset)
    benchmark._input_shape = (1, 3, 640, 640)

    backend = mock.MagicMock()
    backend.run.return_value = [np.zeros((1, 84, 1), dtype=np.float32)]

    metrics = mock.MagicMock(map_50=0.7, map_50_95=0.6, recall_50=0.8)
    with (
        mock.patch(
            "moment_to_action.benchmark._yolo._load_yolo_tensor",
            return_value=np.zeros((1, 3, 640, 640), dtype=np.float32),
        ),
        mock.patch("moment_to_action.benchmark._yolo._parse_yolo_boxes", return_value=[]),
        mock.patch("moment_to_action.benchmark._yolo.compute_detection_map", return_value=metrics),
    ):
        result = benchmark._evaluate_coco_accuracy(handle=object(), backend=backend)

    dataset.instance_detections.assert_called_once()
    assert result == pytest.approx(0.6)
