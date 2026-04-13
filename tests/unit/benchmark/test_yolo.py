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
def test_yolo_load_and_input_shape_npu_uses_tflite_when_available() -> None:
    """On NPU with INT8 available, _load_model picks YOLO_V8_TFLITE_INT8."""
    benchmark = YOLOBenchmark()
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.NPU
    manager = mock.MagicMock(spec=ModelManager)
    manager.is_available.return_value = True
    tflite_path = Path("/tmp/yolo_npu_int8.tflite")
    manager.get_path.return_value = tflite_path
    # TFLite model reports NHWC input shape
    backend.get_input_details.return_value = [{"shape": [1, 640, 640, 3], "name": "input"}]

    handle = benchmark._load_model(backend=backend, manager=manager)
    manager.is_available.assert_called_once_with(ModelID.YOLO_V8_TFLITE_INT8)
    manager.get_path.assert_called_once_with(ModelID.YOLO_V8_TFLITE_INT8)
    backend.load_model.assert_called_once_with(tflite_path)

    inputs = benchmark._make_dummy_input(handle, batch_size=2)
    assert isinstance(inputs, np.ndarray)
    assert inputs.shape == (2, 640, 640, 3)


@pytest.mark.unit
def test_yolo_load_falls_back_to_onnx_when_tflite_unavailable() -> None:
    """On NPU without TFLite model, _load_model falls back to ONNX."""
    benchmark = YOLOBenchmark()
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.NPU
    manager = mock.MagicMock(spec=ModelManager)
    manager.is_available.return_value = False
    onnx_path = Path("/tmp/yolo.onnx")
    manager.get_path.return_value = onnx_path

    benchmark._load_model(backend=backend, manager=manager)
    assert manager.is_available.call_count == 2
    manager.is_available.assert_has_calls(
        [
            mock.call(ModelID.YOLO_V8_TFLITE_INT8),
            mock.call(ModelID.YOLO_V8_TFLITE),
        ]
    )
    manager.get_path.assert_called_once_with(ModelID.YOLO_V8)
    backend.load_model.assert_called_once_with(onnx_path)

    inputs = benchmark._make_dummy_input(None, batch_size=1)
    assert inputs.shape == (1, 3, 640, 640)


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
