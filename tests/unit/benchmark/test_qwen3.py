from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch

from moment_to_action.benchmark import Qwen3Benchmark
from moment_to_action.benchmark._benchmarks._qwen3 import _extract_numeric_answer, _Qwen3Handle
from moment_to_action.benchmark._datasets._gsm8k_dataset import GSM8KItem
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
            "moment_to_action.benchmark._benchmarks._qwen3.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as mock_tokenizer,
        mock.patch(
            "moment_to_action.benchmark._benchmarks._qwen3.AutoModelForCausalLM.from_pretrained",
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


@pytest.mark.unit
def test_qwen3_evaluate_accuracy_none_without_dataset() -> None:
    benchmark = Qwen3Benchmark()
    accuracy = benchmark._evaluate_accuracy(
        handle=object(),
        backend=mock.MagicMock(),
        manager=mock.MagicMock(),
    )
    assert accuracy is None


@pytest.mark.unit
def test_qwen3_run_inference_requires_mapping() -> None:
    tokenizer = mock.MagicMock()
    model = mock.MagicMock()
    model.device = torch.device("cpu")
    handle = _Qwen3Handle(model=model, tokenizer=tokenizer)

    with pytest.raises(TypeError, match="expects mapping"):
        Qwen3Benchmark()._run_inference(handle, [1, 2, 3], mock.MagicMock())


@pytest.mark.unit
def test_qwen3_run_inference_happy_path() -> None:
    tokenizer = mock.MagicMock()
    model = mock.MagicMock()
    model.device = torch.device("cpu")
    handle = _Qwen3Handle(model=model, tokenizer=tokenizer)

    Qwen3Benchmark()._run_inference(
        handle,
        {"input_ids": torch.zeros((1, 2), dtype=torch.long)},
        mock.MagicMock(),
    )
    model.generate.assert_called_once()


@pytest.mark.unit
def test_qwen3_evaluate_accuracy_none_when_no_numeric_predictions() -> None:
    dataset = mock.MagicMock()
    dataset.items.return_value = [GSM8KItem(question="q", answer="1")]
    benchmark = Qwen3Benchmark(gsm8k_dataset=dataset)

    with (
        mock.patch.object(Qwen3Benchmark, "_cast_handle", return_value=mock.MagicMock()),
        mock.patch.object(Qwen3Benchmark, "_generate_answer", return_value="no number"),
    ):
        accuracy = benchmark._evaluate_accuracy(object(), mock.MagicMock(), mock.MagicMock())

    assert accuracy is None


@pytest.mark.unit
def test_qwen3_generate_answer_empty_decode_returns_empty() -> None:
    tokenizer = mock.MagicMock()
    tokenizer.return_value = {"input_ids": torch.zeros((1, 2), dtype=torch.long)}
    tokenizer.batch_decode.return_value = []
    model = mock.MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.zeros((1, 2), dtype=torch.long)
    handle = _Qwen3Handle(model=model, tokenizer=tokenizer)

    result = Qwen3Benchmark._generate_answer(handle, "hello")
    assert result == ""


@pytest.mark.unit
def test_qwen3_generate_answer_returns_stripped_text() -> None:
    tokenizer = mock.MagicMock()
    tokenizer.return_value = {"input_ids": torch.zeros((1, 2), dtype=torch.long)}
    tokenizer.batch_decode.return_value = [" 42 "]
    model = mock.MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.tensor([[0, 1, 2]], dtype=torch.long)
    handle = _Qwen3Handle(model=model, tokenizer=tokenizer)

    assert Qwen3Benchmark._generate_answer(handle, "q") == "42"


@pytest.mark.unit
def test_qwen3_cast_handle_type_error() -> None:
    with pytest.raises(TypeError, match="Invalid Qwen3 benchmark handle"):
        Qwen3Benchmark._cast_handle(object())


@pytest.mark.unit
def test_extract_numeric_answer_edges() -> None:
    assert _extract_numeric_answer("no number") is None
    assert _extract_numeric_answer("#### 1,234") == "1234"
    assert _extract_numeric_answer("value 3.50") == "3.5"


@pytest.mark.unit
def test_extract_numeric_answer_value_error_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRegex:
        @staticmethod
        def findall(_text: str) -> list[str]:
            return ["not_a_float"]

    monkeypatch.setattr("moment_to_action.benchmark._benchmarks._qwen3._NUMBER_RE", _FakeRegex())
    assert _extract_numeric_answer("anything") == "not_a_float"


@pytest.mark.unit
def test_extract_numeric_answer_fallbacks_to_full_text_search() -> None:
    assert _extract_numeric_answer("prefix 17 #### no digits") == "17"
