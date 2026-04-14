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


@pytest.mark.unit
def test_mobileclip_run_inference_raises_for_non_dict() -> None:
    benchmark = MobileCLIPBenchmark()
    backend = mock.MagicMock()
    with pytest.raises(TypeError, match="expects dict"):
        benchmark._run_inference(object(), np.zeros((1, 3, 256, 256)), backend)


@pytest.mark.unit
def test_mobileclip_evaluate_accuracy_returns_none_without_eval_images() -> None:
    """_evaluate_accuracy returns None when no eval images are configured."""
    benchmark = MobileCLIPBenchmark()
    backend = mock.MagicMock()
    manager = mock.MagicMock(spec=ModelManager)
    result = benchmark._evaluate_accuracy(object(), backend, manager)
    assert result is None


@pytest.mark.unit
def test_mobileclip_evaluate_accuracy_returns_none_when_cv2_missing() -> None:
    """_evaluate_accuracy returns None gracefully when opencv is not installed."""
    benchmark = MobileCLIPBenchmark(eval_image_paths=[Path("/tmp/dummy.jpg")])
    backend = mock.MagicMock()
    manager = mock.MagicMock(spec=ModelManager)

    with mock.patch("builtins.__import__", side_effect=ImportError("cv2")):
        result = benchmark._evaluate_accuracy(object(), backend, manager)

    assert result is None


@pytest.mark.unit
def test_mobileclip_evaluate_accuracy_with_mocked_pipeline(tmp_path: Path) -> None:
    """_evaluate_accuracy returns mean cosine similarity = 1.0 when outputs match."""
    img_file = tmp_path / "test.jpg"
    img_file.write_bytes(b"\xff\xd8\xff")

    benchmark = MobileCLIPBenchmark(eval_image_paths=[img_file])

    # Both oracle and eval produce the same embedding
    embedding = np.ones((1, 512), dtype=np.float32)
    oracle_backend = mock.MagicMock()
    oracle_backend.run.return_value = [embedding]
    oracle_backend.load_model.return_value = object()

    eval_backend = mock.MagicMock()
    eval_backend.run.return_value = [embedding]

    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/mobileclip.tflite")

    fake_image = np.zeros((256, 256, 3), dtype=np.uint8)

    import cv2  # noqa: PLC0415

    with (
        mock.patch(
            "moment_to_action.hardware.ComputeBackend",
            return_value=oracle_backend,
        ),
        mock.patch.object(cv2, "imread", return_value=fake_image),
        mock.patch.object(cv2, "resize", return_value=fake_image),
    ):
        result = benchmark._evaluate_accuracy(object(), eval_backend, manager)

    assert result == pytest.approx(1.0)


@pytest.mark.unit
def test_mobileclip_evaluate_accuracy_returns_none_for_nan_gpu_output(tmp_path: Path) -> None:
    """_evaluate_accuracy returns None when the eval backend produces NaN embeddings (GPU FP16)."""
    img_file = tmp_path / "test.jpg"
    img_file.write_bytes(b"\xff\xd8\xff")

    benchmark = MobileCLIPBenchmark(eval_image_paths=[img_file])

    oracle_backend = mock.MagicMock()
    oracle_backend.run.return_value = [np.ones((1, 512), dtype=np.float32)]
    oracle_backend.load_model.return_value = object()

    # GPU backend returns NaN embeddings (FP16 overflow)
    nan_embedding = np.full((1, 512), float("nan"), dtype=np.float32)
    eval_backend = mock.MagicMock()
    eval_backend.run.return_value = [nan_embedding]
    eval_backend.active_unit.name = "GPU"

    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/mobileclip.tflite")

    fake_image = np.zeros((256, 256, 3), dtype=np.uint8)

    import cv2  # noqa: PLC0415

    with (
        mock.patch(
            "moment_to_action.hardware.ComputeBackend",
            return_value=oracle_backend,
        ),
        mock.patch.object(cv2, "imread", return_value=fake_image),
        mock.patch.object(cv2, "resize", return_value=fake_image),
    ):
        result = benchmark._evaluate_accuracy(object(), eval_backend, manager)

    assert result is None
