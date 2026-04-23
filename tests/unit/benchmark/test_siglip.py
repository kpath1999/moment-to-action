from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch

from moment_to_action.benchmark._benchmarks._siglip import SigLIPBenchmark
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_siglip_model_id() -> None:
    assert SigLIPBenchmark(coco_dataset=mock.MagicMock()).model_id == ModelID.SIGLIP_SO400M


@pytest.mark.unit
def test_siglip_load_uses_model_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = SigLIPBenchmark(coco_dataset=mock.MagicMock())
    backend = mock.MagicMock()
    backend.resolve_torch_policy.return_value = mock.MagicMock(device=torch.device("cpu"))
    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/siglip")

    processor = mock.MagicMock()
    model = mock.MagicMock()
    model.to.return_value = model

    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._siglip.AutoProcessor.from_pretrained",
        lambda _path: processor,
    )
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._siglip.AutoModel.from_pretrained",
        lambda _path: model,
    )

    benchmark._load_model(backend=backend, manager=manager)
    manager.get_path.assert_called_once_with(ModelID.SIGLIP_SO400M)


@pytest.mark.unit
def test_siglip_coco_retrieval_sets_recall_at_1(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = mock.MagicMock()
    dataset.images.return_value = [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]
    dataset.captions.side_effect = [["cap a"], ["cap b"]]

    benchmark = SigLIPBenchmark(coco_dataset=dataset)

    image = mock.MagicMock()
    image.convert.return_value = image
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._siglip.Image.open", lambda _p: image
    )

    processor = mock.MagicMock()
    processor.return_value = {"input_ids": torch.zeros((1, 1), dtype=torch.long)}
    model = mock.MagicMock()
    model.return_value = mock.MagicMock(
        logits_per_image=torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    )
    handle = mock.MagicMock(model=model, processor=processor, device="cpu")
    monkeypatch.setattr(SigLIPBenchmark, "_cast_handle", lambda _self, _h: handle)

    fake_metrics = mock.MagicMock(recall_at_1=0.5)
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._siglip.compute_retrieval_metrics",
        lambda **_kwargs: fake_metrics,
    )

    score = benchmark._evaluate_accuracy(
        handle=object(), backend=mock.MagicMock(), manager=mock.MagicMock()
    )
    assert score == pytest.approx(0.5)
    details = benchmark._accuracy_details()
    assert details is not None
    assert details["recall_at_1"] == pytest.approx(0.5)
