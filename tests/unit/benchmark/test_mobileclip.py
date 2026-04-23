from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from moment_to_action.benchmark import MobileCLIPBenchmark
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_mobileclip_model_id() -> None:
    assert MobileCLIPBenchmark(coco_dataset=mock.MagicMock()).model_id == ModelID.MOBILECLIP_S2


@pytest.mark.unit
def test_mobileclip_load_uses_model_manager() -> None:
    benchmark = MobileCLIPBenchmark(coco_dataset=mock.MagicMock())
    backend = mock.MagicMock()
    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/mobileclip.tflite")

    benchmark._load_model(backend=backend, manager=manager)
    manager.get_path.assert_called_once_with(ModelID.MOBILECLIP_S2)


@pytest.mark.unit
def test_mobileclip_coco_retrieval_sets_recall_at_1(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = mock.MagicMock()
    dataset.images.return_value = [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]
    dataset.captions.side_effect = [["cap a"], ["cap b"]]

    benchmark = MobileCLIPBenchmark(coco_dataset=dataset)
    backend = mock.MagicMock()
    backend.run.return_value = [
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    ]

    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._mobileclip.open_clip.get_tokenizer",
        lambda _name: lambda prompts: np.zeros((len(prompts), 77), dtype=np.int64),
    )
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._mobileclip._load_mobileclip_tensor",
        lambda _path: np.zeros((1, 3, 256, 256), dtype=np.float32),
    )

    fake_metrics = mock.MagicMock(recall_at_1=0.5)
    monkeypatch.setattr(
        "moment_to_action.benchmark._benchmarks._mobileclip.compute_retrieval_metrics",
        lambda **_kwargs: fake_metrics,
    )

    score = benchmark._evaluate_coco_accuracy(handle=object(), backend=backend)
    assert score == pytest.approx(0.5)
    details = benchmark._accuracy_details()
    assert details is not None
    assert details["recall_at_1"] == pytest.approx(0.5)
