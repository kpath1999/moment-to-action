from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from moment_to_action.benchmark import MobileCLIPBenchmark
from moment_to_action.benchmark._benchmarks._mobileclip import (
    _default_sample_images,
    _load_mobileclip_tensor,
)
from moment_to_action.benchmark._oracle_ground_truth import (
    OracleClassification,
    OracleGroundTruth,
)
from moment_to_action.benchmark._retrieval_metrics import RetrievalMetrics
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

    import cv2

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

    import cv2

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


@pytest.mark.unit
def test_mobileclip_project_oracle_accuracy_path(tmp_path: Path) -> None:
    benchmark = MobileCLIPBenchmark()
    backend = mock.MagicMock()
    handle = object()

    gt = OracleGroundTruth(
        detections=[],
        classifications=[
            OracleClassification(
                image_name="img.jpg",
                top_label="person",
                scores={"person": 0.9, "car": 0.1},
            )
        ],
        text_queries=[],
        text_prompts=["person", "car"],
        hardware_target="x86_64",
        recorded_at="now",
        dataset_name="project",
    )

    img_file = tmp_path / "img.jpg"
    img_file.write_bytes(b"jpg")

    # outputs[1] is image_emb, outputs[0] is text_emb in benchmark logic
    backend.run.side_effect = [
        [np.array([[1.0, 0.0]], dtype=np.float32), np.array([[1.0, 0.0]], dtype=np.float32)],
        [np.array([[0.0, 1.0]], dtype=np.float32), np.array([[1.0, 0.0]], dtype=np.float32)],
    ]

    with (
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip.OracleStore"
        ) as mock_store_cls,
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip.open_clip.get_tokenizer",
            return_value=lambda prompts: np.zeros((len(prompts), 77), dtype=np.int64),
        ),
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip._default_sample_images",
            return_value=[img_file],
        ),
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip._load_mobileclip_tensor",
            return_value=np.zeros((1, 3, 256, 256), dtype=np.float32),
        ),
    ):
        mock_store_cls.return_value.load.return_value = gt
        acc = benchmark._evaluate_project_oracle_accuracy(handle=handle, backend=backend)

    assert acc == pytest.approx(1.0)


@pytest.mark.unit
def test_mobileclip_coco_accuracy_path(tmp_path: Path) -> None:
    img_file = tmp_path / "img.jpg"
    img_file.write_bytes(b"jpg")

    dataset = mock.MagicMock()
    dataset.dataset_name = "coco_val2017"
    dataset.images.return_value = [img_file]

    gt = OracleGroundTruth(
        detections=[],
        classifications=[
            OracleClassification(
                image_name=img_file.name,
                top_label="person",
                scores={"person": 0.9, "car": 0.1},
            )
        ],
        text_queries=[],
        text_prompts=[],
        hardware_target="x86_64",
        recorded_at="now",
        dataset_name="coco_val2017",
    )

    store = mock.MagicMock()
    store.load.return_value = gt
    backend = mock.MagicMock()
    backend.run.return_value = [
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    ]

    benchmark = MobileCLIPBenchmark(coco_dataset=dataset, oracle_store=store)

    metrics = RetrievalMetrics(
        recall_at_1=0.8,
        recall_at_5=1.0,
        recall_at_10=1.0,
        kendall_tau=0.7,
        mean_rank_delta=0.2,
    )

    with (
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip.open_clip.get_tokenizer",
            return_value=lambda prompts: np.zeros((len(prompts), 77), dtype=np.int64),
        ),
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip._load_mobileclip_tensor",
            return_value=np.zeros((1, 3, 256, 256), dtype=np.float32),
        ),
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip.compute_retrieval_metrics",
            return_value=metrics,
        ),
    ):
        acc = benchmark._evaluate_coco_accuracy(handle=object(), backend=backend)

    assert acc == pytest.approx(0.8)


@pytest.mark.unit
def test_mobileclip_coco_accuracy_returns_none_without_scores(tmp_path: Path) -> None:
    img_file = tmp_path / "img.jpg"
    img_file.write_bytes(b"jpg")

    dataset = mock.MagicMock()
    dataset.dataset_name = "coco_val2017"
    dataset.images.return_value = [img_file]

    gt = OracleGroundTruth(
        detections=[],
        classifications=[
            OracleClassification(
                image_name=img_file.name,
                top_label="person",
                scores={},
            )
        ],
        text_queries=[],
        text_prompts=[],
        hardware_target="x86_64",
        recorded_at="now",
        dataset_name="coco_val2017",
    )

    store = mock.MagicMock()
    store.load.return_value = gt
    benchmark = MobileCLIPBenchmark(coco_dataset=dataset, oracle_store=store)
    result = benchmark._evaluate_coco_accuracy(handle=object(), backend=mock.MagicMock())
    assert result is None


@pytest.mark.unit
def test_mobileclip_load_tensor_and_default_images(tmp_path: Path) -> None:
    image_path = tmp_path / "img.jpg"

    from PIL import Image

    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)

    tensor = _load_mobileclip_tensor(image_path)
    assert tensor.shape == (1, 3, 256, 256)

    with mock.patch(
        "moment_to_action.benchmark._benchmarks._mobileclip.Path.is_dir",
        return_value=False,
    ):
        assert _default_sample_images() == []


@pytest.mark.unit
def test_mobileclip_run_inference_happy_path_calls_backend() -> None:
    benchmark = MobileCLIPBenchmark()
    backend = mock.MagicMock()
    inputs = {
        "serving_default_args_0:0": np.zeros((1, 3, 256, 256), dtype=np.float32),
        "serving_default_args_1:0": np.zeros((1, 77), dtype=np.int64),
    }
    benchmark._run_inference(object(), inputs, backend)
    backend.run.assert_called_once()


@pytest.mark.unit
def test_mobileclip_evaluate_accuracy_delegates_to_coco() -> None:
    benchmark = MobileCLIPBenchmark(coco_dataset=mock.MagicMock())
    with mock.patch.object(benchmark, "_evaluate_coco_accuracy", return_value=0.4) as coco_eval:
        result = benchmark._evaluate_accuracy(object(), mock.MagicMock(), mock.MagicMock())
    assert result == pytest.approx(0.4)
    coco_eval.assert_called_once()


@pytest.mark.unit
def test_mobileclip_coco_accuracy_returns_none_without_dataset() -> None:
    benchmark = MobileCLIPBenchmark(coco_dataset=None)
    assert benchmark._evaluate_coco_accuracy(object(), mock.MagicMock()) is None


@pytest.mark.unit
def test_mobileclip_project_oracle_returns_none_without_prompts_or_images() -> None:
    benchmark = MobileCLIPBenchmark()
    gt_no_prompts = OracleGroundTruth(
        detections=[],
        classifications=[
            OracleClassification(image_name="a.jpg", top_label="x", scores={"x": 1.0})
        ],
        text_queries=[],
        text_prompts=[],
        hardware_target="x86_64",
        recorded_at="now",
        dataset_name="project",
    )

    with mock.patch("moment_to_action.benchmark._benchmarks._mobileclip.OracleStore") as store:
        store.return_value.load.return_value = gt_no_prompts
        assert benchmark._evaluate_project_oracle_accuracy(object(), mock.MagicMock()) is None

    gt_with_prompts = OracleGroundTruth(
        detections=[],
        classifications=[
            OracleClassification(image_name="missing.jpg", top_label="x", scores={"x": 1.0})
        ],
        text_queries=[],
        text_prompts=["x"],
        hardware_target="x86_64",
        recorded_at="now",
        dataset_name="project",
    )
    with (
        mock.patch("moment_to_action.benchmark._benchmarks._mobileclip.OracleStore") as store,
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip.open_clip.get_tokenizer",
            return_value=lambda prompts: np.zeros((len(prompts), 77), dtype=np.int64),
        ),
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip._default_sample_images",
            return_value=[],
        ),
    ):
        store.return_value.load.return_value = gt_with_prompts
        assert benchmark._evaluate_project_oracle_accuracy(object(), mock.MagicMock()) is None


@pytest.mark.unit
def test_mobileclip_project_oracle_returns_none_when_no_matched_images(tmp_path: Path) -> None:
    benchmark = MobileCLIPBenchmark()
    gt = OracleGroundTruth(
        detections=[],
        classifications=[
            OracleClassification(image_name="not_found.jpg", top_label="x", scores={"x": 1.0})
        ],
        text_queries=[],
        text_prompts=["x"],
        hardware_target="x86_64",
        recorded_at="now",
        dataset_name="project",
    )
    local_image = tmp_path / "local.jpg"
    local_image.write_bytes(b"jpg")
    with (
        mock.patch("moment_to_action.benchmark._benchmarks._mobileclip.OracleStore") as store,
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip.open_clip.get_tokenizer",
            return_value=lambda prompts: np.zeros((len(prompts), 77), dtype=np.int64),
        ),
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip._default_sample_images",
            return_value=[local_image],
        ),
    ):
        store.return_value.load.return_value = gt
        assert benchmark._evaluate_project_oracle_accuracy(object(), mock.MagicMock()) is None


@pytest.mark.unit
def test_mobileclip_coco_accuracy_returns_none_for_missing_store_data() -> None:
    dataset = mock.MagicMock()
    dataset.dataset_name = "coco_val2017"
    benchmark = MobileCLIPBenchmark(
        coco_dataset=dataset,
        oracle_store=mock.MagicMock(load=lambda: None),
    )
    assert benchmark._evaluate_coco_accuracy(object(), mock.MagicMock()) is None


@pytest.mark.unit
def test_mobileclip_embedding_consistency_skips_unreadable_images() -> None:
    benchmark = MobileCLIPBenchmark(eval_image_paths=[Path("/tmp/missing.jpg")])
    backend = mock.MagicMock()
    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/mobileclip.tflite")
    cpu_backend = mock.MagicMock()
    cpu_backend.load_model.return_value = object()

    with (
        mock.patch("moment_to_action.hardware.ComputeBackend", return_value=cpu_backend),
        mock.patch("cv2.imread", return_value=None),
    ):
        assert benchmark._evaluate_embedding_consistency(object(), backend, manager) is None


@pytest.mark.unit
def test_mobileclip_default_sample_images_true_branch(tmp_path: Path) -> None:
    with (
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip.Path.is_dir",
            return_value=True,
        ),
        mock.patch(
            "moment_to_action.benchmark._benchmarks._mobileclip.Path.glob",
            return_value=[tmp_path / "a.jpg"],
        ),
    ):
        result = _default_sample_images()
    assert result == [tmp_path / "a.jpg"]
