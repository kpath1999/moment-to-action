from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch

from moment_to_action.benchmark import Qwen3Benchmark
from moment_to_action.benchmark._gsm8k_dataset import GSM8KItem
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_qwen3_load_uses_torch_policy() -> None:
    benchmark = Qwen3Benchmark()
    backend = mock.MagicMock()
    policy = mock.MagicMock(device=torch.device("cpu"), dtype=torch.float32)
    backend.resolve_torch_policy.return_value = policy

    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/qwen3")

    tokenizer = mock.MagicMock()
    tokenizer.return_value = {"input_ids": torch.zeros((1, 8), dtype=torch.long)}

    model = mock.MagicMock()
    model.to.return_value = model
    model.device = torch.device("cpu")

    with (
        mock.patch(
            "moment_to_action.benchmark._qwen3.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as mock_tokenizer,
        mock.patch(
            "moment_to_action.benchmark._qwen3.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as mock_model,
    ):
        handle = benchmark._load_model(backend=backend, manager=manager)

    manager.get_path.assert_called_once_with(ModelID.QWEN2_5_4B)
    mock_tokenizer.assert_called_once()
    mock_model.assert_called_once()

    inputs = benchmark._make_dummy_input(handle, batch_size=2)
    assert "input_ids" in inputs  # type: ignore[operator]


@pytest.mark.unit
def test_qwen3_evaluate_accuracy_exact_match() -> None:
    dataset = mock.MagicMock()
    dataset.items.return_value = [
        GSM8KItem(question="q1", answer="42"),
        GSM8KItem(question="q2", answer="7"),
    ]
    benchmark = Qwen3Benchmark(gsm8k_dataset=dataset)

    with (
        mock.patch.object(Qwen3Benchmark, "_cast_handle", return_value=mock.MagicMock()),
        mock.patch.object(Qwen3Benchmark, "_generate_answer", side_effect=["#### 42", "answer: 8"]),
    ):
        accuracy = benchmark._evaluate_accuracy(
            handle=object(),
            backend=mock.MagicMock(),
            manager=mock.MagicMock(),
        )

    assert accuracy == pytest.approx(0.5)
