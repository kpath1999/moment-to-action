from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest import mock

import numpy as np
import pytest
import torch

from moment_to_action.benchmark._benchmarks._siglip import (
    SigLIPBenchmark,
    _default_sample_images,
    _SigLIPHandle,
)
from moment_to_action.benchmark._oracle_ground_truth import OracleGroundTruth
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_siglip_load_and_dummy_input() -> None:
    benchmark = SigLIPBenchmark(text_prompts=["a person"])
    backend = mock.MagicMock()
    policy = mock.MagicMock(device=torch.device("cpu"))
    backend.resolve_torch_policy.return_value = policy

    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/siglip")

    processor = mock.MagicMock()
    processor.return_value = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
    }

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
        handle = benchmark._load_model(backend=backend, manager=manager)

    manager.get_path.assert_called_once_with(ModelID.SIGLIP_SO400M)
    inputs = cast("dict[str, torch.Tensor]", benchmark._make_dummy_input(handle, batch_size=1))
    assert "input_ids" in inputs


@pytest.mark.unit
def test_siglip_run_inference_requires_dict() -> None:
    benchmark = SigLIPBenchmark()
    handle = _SigLIPHandle(
        model=mock.MagicMock(),
        processor=mock.MagicMock(),
        device="cpu",
    )
    with pytest.raises(TypeError, match="expects dict"):
        benchmark._run_inference(handle, np.zeros((1, 3)), mock.MagicMock())


@pytest.mark.unit
def test_siglip_evaluate_accuracy_records_and_merges(tmp_path: Path) -> None:
    image_path = tmp_path / "img.jpg"
    from PIL import Image

    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)

    store = mock.MagicMock()
    store.path = tmp_path / "oracle.json"
    store.load.return_value = OracleGroundTruth(
        detections=[],
        classifications=[],
        text_queries=["q"],
        text_prompts=["old"],
        hardware_target="x86_64",
        recorded_at="now",
        dataset_name="project",
    )

    benchmark = SigLIPBenchmark(
        text_prompts=["a person", "a car"],
        sample_images=[image_path],
        oracle_store=store,
    )

    model = mock.MagicMock()
    model.return_value = mock.MagicMock(logits_per_image=torch.tensor([[0.1, 0.9]]))
    processor = mock.MagicMock()
    processor.return_value = {"input_ids": torch.zeros((1, 2), dtype=torch.long)}
    handle = mock.MagicMock(model=model, processor=processor, device="cpu")

    with mock.patch.object(SigLIPBenchmark, "_cast_handle", return_value=handle):
        result = benchmark._evaluate_accuracy(object(), mock.MagicMock(), mock.MagicMock())

    assert result is None
    store.save.assert_called_once()


@pytest.mark.unit
def test_siglip_default_sample_images_returns_list() -> None:
    images = _default_sample_images()
    assert isinstance(images, list)
    assert all(isinstance(path, Path) for path in images)


@pytest.mark.unit
def test_siglip_default_sample_images_missing_dir_returns_empty() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(Path, "is_dir", lambda _self: False)
        assert _default_sample_images() == []
