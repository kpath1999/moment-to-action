from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from moment_to_action.benchmark._benchmarks._grounding_dino import (
    GroundingDINOBenchmark,
    _DinoHandle,
)
from moment_to_action.benchmark._benchmarks._grounding_dino import (
    _default_sample_images as _dino_default_sample_images,
)
from moment_to_action.benchmark._benchmarks._siglip import SigLIPBenchmark, _SigLIPHandle
from moment_to_action.benchmark._oracle_ground_truth import OracleGroundTruth, OracleStore
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_grounding_dino_cast_handle_type_error() -> None:
    with pytest.raises(TypeError, match="Expected _DinoHandle"):
        GroundingDINOBenchmark._cast_handle(object())


@pytest.mark.unit
def test_siglip_cast_handle_type_error() -> None:
    with pytest.raises(TypeError, match="Expected _SigLIPHandle"):
        SigLIPBenchmark._cast_handle(object())


@pytest.mark.unit
def test_grounding_dino_load_dummy_and_run() -> None:
    bench = GroundingDINOBenchmark(text_queries=["person"])
    backend = mock.MagicMock()
    backend.resolve_torch_policy.return_value = mock.MagicMock(device=torch.device("cpu"))
    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/dino")

    processor = mock.MagicMock()

    class _Inputs(dict):
        def to(self, device: object) -> _Inputs:
            del device
            return self

    processor.return_value = _Inputs({"input_ids": torch.tensor([[1]])})
    model = mock.MagicMock()
    model.to.return_value = model

    with (
        mock.patch(
            "moment_to_action.benchmark._benchmarks._grounding_dino.AutoProcessor.from_pretrained",
            return_value=processor,
        ),
        mock.patch(
            "moment_to_action.benchmark._benchmarks._grounding_dino.AutoModelForZeroShotObjectDetection.from_pretrained",
            return_value=model,
        ),
    ):
        handle = bench._load_model(backend=backend, manager=manager)
        assert isinstance(handle, _DinoHandle)
        dummy = bench._make_dummy_input(handle)
        assert isinstance(dummy, dict)
        assert "input_ids" in dummy
        bench._run_inference(handle, dummy, backend)

    with pytest.raises(TypeError, match="expects dict"):
        bench._run_inference(handle, object(), backend)


@pytest.mark.unit
def test_grounding_dino_evaluate_accuracy_saves_oracle(tmp_path: Path) -> None:
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    img_path = tmp_path / "img.jpg"
    img.save(img_path)

    store = mock.MagicMock(spec=OracleStore)
    store.path = tmp_path / "oracle.json"
    store.load.return_value = OracleGroundTruth(
        detections=[],
        classifications=[],
        text_queries=[],
        text_prompts=[],
        hardware_target="cpu",
        recorded_at="now",
        dataset_name="project",
    )

    bench = GroundingDINOBenchmark(
        text_queries=["person"],
        sample_images=[img_path],
        oracle_store=store,
    )
    processor = mock.MagicMock()

    class _Inputs(dict):
        def to(self, device: object) -> _Inputs:
            del device
            return self

    processor.return_value = _Inputs({"input_ids": torch.tensor([[1]], dtype=torch.int64)})
    processor.post_process_grounded_object_detection.return_value = [
        {
            "boxes": [torch.tensor([1.0, 1.0, 2.0, 2.0])],
            "scores": [torch.tensor(0.9)],
            "labels": ["person"],
        }
    ]
    handle = _DinoHandle(
        model=mock.MagicMock(return_value=mock.MagicMock()),
        processor=processor,
        device="cpu",
    )

    with mock.patch(
        "moment_to_action.benchmark._benchmarks._grounding_dino.detect_platform",
        return_value=mock.MagicMock(name="x86_64"),
    ):
        result = bench._evaluate_accuracy(
            handle=handle,
            backend=mock.MagicMock(),
            manager=mock.MagicMock(),
        )

    assert result is None
    store.save.assert_called_once()


@pytest.mark.unit
def test_siglip_load_dummy_and_run() -> None:
    bench = SigLIPBenchmark(text_prompts=["a person"])
    backend = mock.MagicMock()
    backend.resolve_torch_policy.return_value = mock.MagicMock(device=torch.device("cpu"))
    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/siglip")

    processor = mock.MagicMock()

    class _Inputs(dict):
        def to(self, device: object) -> _Inputs:
            del device
            return self

    processor.return_value = _Inputs({"input_ids": torch.tensor([[1]])})
    model = mock.MagicMock()
    model.to.return_value = model

    with (
        mock.patch(
            "moment_to_action.benchmark._benchmarks._siglip.AutoProcessor.from_pretrained",
            return_value=processor,
        ),
        mock.patch(
            "moment_to_action.benchmark._benchmarks._siglip.AutoModel.from_pretrained",
            return_value=model,
        ),
    ):
        handle = bench._load_model(backend=backend, manager=manager)
        assert isinstance(handle, _SigLIPHandle)
        dummy = bench._make_dummy_input(handle)
        assert isinstance(dummy, dict)
        assert "input_ids" in dummy
        bench._run_inference(handle, dummy, backend)

    with pytest.raises(TypeError, match="expects dict"):
        bench._run_inference(handle, object(), backend)


@pytest.mark.unit
def test_siglip_evaluate_accuracy_skips_empty_prompts_and_saves(tmp_path: Path) -> None:
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    img_path = tmp_path / "img.jpg"
    img.save(img_path)

    dataset = mock.MagicMock()
    dataset.captions.return_value = []
    dataset.dataset_name = "coco_val2017"

    store = mock.MagicMock(spec=OracleStore)
    store.path = tmp_path / "oracle.json"
    store.load.return_value = OracleGroundTruth(
        detections=[],
        classifications=[],
        text_queries=[],
        text_prompts=[],
        hardware_target="cpu",
        recorded_at="now",
        dataset_name="coco_val2017",
    )

    bench = SigLIPBenchmark(sample_images=[img_path], oracle_store=store, coco_dataset=dataset)
    handle = _SigLIPHandle(model=mock.MagicMock(), processor=mock.MagicMock(), device="cpu")

    with mock.patch(
        "moment_to_action.benchmark._benchmarks._siglip.detect_platform",
        return_value=mock.MagicMock(name="x86_64"),
    ):
        result = bench._evaluate_accuracy(
            handle=handle,
            backend=mock.MagicMock(),
            manager=mock.MagicMock(),
        )

    assert result is None
    store.save.assert_called_once()


@pytest.mark.unit
def test_benchmark_model_ids() -> None:
    assert GroundingDINOBenchmark().model_id == ModelID.GROUNDING_DINO_BASE
    assert SigLIPBenchmark().model_id == ModelID.SIGLIP_SO400M


@pytest.mark.unit
def test_grounding_dino_uses_yolo_labels_for_coco_dataset() -> None:
    dataset = mock.MagicMock()
    dataset.images.return_value = []
    dataset.dataset_name = "coco_val2017"

    bench = GroundingDINOBenchmark(coco_dataset=dataset)
    assert "person" in bench._text_queries


@pytest.mark.unit
def test_grounding_dino_coco_evaluate_uses_track(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)

    dataset = mock.MagicMock()
    dataset.images.return_value = [img_path]
    dataset.dataset_name = "coco_val2017"

    store = mock.MagicMock(spec=OracleStore)
    store.path = tmp_path / "oracle.json"
    store.load.return_value = None

    bench = GroundingDINOBenchmark(coco_dataset=dataset, oracle_store=store)

    processor = mock.MagicMock()

    class _Inputs(dict):
        def to(self, device: object) -> _Inputs:
            del device
            return self

    processor.return_value = _Inputs({"input_ids": torch.tensor([[1]], dtype=torch.int64)})
    processor.post_process_grounded_object_detection.return_value = [
        {"boxes": [], "scores": [], "labels": []}
    ]

    handle = _DinoHandle(
        model=mock.MagicMock(return_value=mock.MagicMock()),
        processor=processor,
        device="cpu",
    )

    with (
        mock.patch(
            "moment_to_action.benchmark._benchmarks._grounding_dino.track",
            side_effect=lambda it, **_kwargs: it,
        ) as track_mock,
        mock.patch(
            "moment_to_action.benchmark._benchmarks._grounding_dino.detect_platform",
            return_value=mock.MagicMock(name="x86_64"),
        ),
    ):
        bench._evaluate_accuracy(handle, mock.MagicMock(), mock.MagicMock())

    track_mock.assert_called_once()


@pytest.mark.unit
def test_grounding_dino_default_sample_images_missing_dir() -> None:
    with mock.patch(
        "moment_to_action.benchmark._benchmarks._grounding_dino.Path.is_dir",
        return_value=False,
    ):
        assert _dino_default_sample_images() == []
