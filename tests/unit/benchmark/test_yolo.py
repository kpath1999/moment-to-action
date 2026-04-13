from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from moment_to_action.benchmark import YOLOBenchmark
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_yolo_load_and_input_shape() -> None:
    benchmark = YOLOBenchmark()
    backend = mock.MagicMock()
    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/yolo.onnx")

    handle = benchmark._load_model(backend=backend, manager=manager)
    manager.get_path.assert_called_once_with(ModelID.YOLO_V8)
    backend.load_model.assert_called_once_with(Path("/tmp/yolo.onnx"))

    inputs = benchmark._make_dummy_input(handle, batch_size=3)
    assert isinstance(inputs, np.ndarray)
    assert inputs.shape == (3, 3, 640, 640)
