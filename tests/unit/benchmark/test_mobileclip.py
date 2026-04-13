from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from moment_to_action.benchmark import MobileCLIPBenchmark
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_mobileclip_load_and_multi_input_shape() -> None:
    benchmark = MobileCLIPBenchmark()
    backend = mock.MagicMock()
    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/mobileclip.tflite")

    handle = benchmark._load_model(backend=backend, manager=manager)
    manager.get_path.assert_called_once_with(ModelID.MOBILECLIP_S2)
    backend.load_model.assert_called_once_with(Path("/tmp/mobileclip.tflite"))

    inputs = benchmark._make_dummy_input(handle, batch_size=2)
    assert isinstance(inputs, dict)
    assert set(inputs) == {"serving_default_args_0:0", "serving_default_args_1:0"}
    assert isinstance(inputs["serving_default_args_0:0"], np.ndarray)
    assert isinstance(inputs["serving_default_args_1:0"], np.ndarray)
    assert inputs["serving_default_args_0:0"].shape == (2, 3, 256, 256)
    assert inputs["serving_default_args_1:0"].shape == (2, 77)
