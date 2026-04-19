from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch

from moment_to_action.benchmark import SmolVLM2Benchmark
from moment_to_action.benchmark._datasets._msrvtt_dataset import MsrvttItem
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_smolvlm2_load_uses_torch_policy() -> None:
    benchmark = SmolVLM2Benchmark()
    backend = mock.MagicMock()
    policy = mock.MagicMock(device=torch.device("cpu"), dtype=torch.float32)
    backend.resolve_torch_policy.return_value = policy

    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/smolvlm2")

    processor = mock.MagicMock()
    processor.apply_chat_template.return_value = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
    }

    model = mock.MagicMock()
    model.to.return_value = model
    model.device = torch.device("cpu")

    with (
        mock.patch(
            "moment_to_action.benchmark._benchmarks._smolvlm2.AutoProcessor.from_pretrained",
            return_value=processor,
        ) as mock_processor,
        mock.patch(
            "moment_to_action.benchmark._benchmarks._smolvlm2.AutoModelForImageTextToText.from_pretrained",
            return_value=model,
        ) as mock_model,
    ):
        handle = benchmark._load_model(backend=backend, manager=manager)

    manager.get_path.assert_called_once_with(ModelID.SMOLVLM2_2_2B)
    mock_processor.assert_called_once()
    mock_model.assert_called_once()

    inputs = benchmark._make_dummy_input(handle, batch_size=1)
    assert "input_ids" in inputs  # type: ignore[operator]


@pytest.mark.unit
def test_smolvlm2_evaluate_accuracy_exact_match() -> None:
    dataset = mock.MagicMock()
    dataset.items.return_value = [
        MsrvttItem(video_path=Path("/tmp/a.mp4"), question="q1", answer="person running"),
        MsrvttItem(video_path=Path("/tmp/b.mp4"), question="q2", answer="red car"),
    ]
    benchmark = SmolVLM2Benchmark(msrvtt_dataset=dataset)

    with (
        mock.patch.object(SmolVLM2Benchmark, "_cast_handle", return_value=mock.MagicMock()),
        mock.patch(
            "moment_to_action.benchmark._benchmarks._smolvlm2._sample_video_frames",
            return_value=[mock.MagicMock()],
        ),
        mock.patch.object(
            SmolVLM2Benchmark,
            "_generate_answer",
            side_effect=["person running", "different answer"],
        ),
    ):
        accuracy = benchmark._evaluate_accuracy(
            handle=object(),
            backend=mock.MagicMock(),
            manager=mock.MagicMock(),
        )

    assert accuracy == pytest.approx(0.5)
