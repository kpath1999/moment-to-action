from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from moment_to_action.benchmark import WhisperTinyBenchmark
from moment_to_action.benchmark._datasets._librispeech_dataset import LibriSpeechItem
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_whisper_load_uses_torch_policy() -> None:
    benchmark = WhisperTinyBenchmark()
    backend = mock.MagicMock()
    policy = mock.MagicMock(device=torch.device("cpu"), dtype=torch.float32)
    backend.resolve_torch_policy.return_value = policy

    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/whisper")

    processor = mock.MagicMock()
    processor.return_value = {
        "input_features": torch.zeros((1, 80, 300), dtype=torch.float32),
    }

    model = mock.MagicMock()
    model.to.return_value = model
    model.device = torch.device("cpu")

    with (
        mock.patch(
            "moment_to_action.benchmark._benchmarks._whisper.AutoProcessor.from_pretrained",
            return_value=processor,
        ) as mock_processor,
        mock.patch(
            "moment_to_action.benchmark._benchmarks._whisper.AutoModelForSpeechSeq2Seq.from_pretrained",
            return_value=model,
        ) as mock_model,
    ):
        handle = benchmark._load_model(backend=backend, manager=manager)

    manager.get_path.assert_called_once_with(ModelID.WHISPER_TINY)
    mock_processor.assert_called_once()
    mock_model.assert_called_once()
    inputs = benchmark._make_dummy_input(handle, batch_size=1)
    assert "input_features" in inputs  # type: ignore[operator]


@pytest.mark.unit
def test_whisper_evaluate_accuracy_sets_wer() -> None:
    dataset = mock.MagicMock()
    dataset.items.return_value = [
        LibriSpeechItem(
            audio_path=Path("/tmp/a.wav"),
            sample_rate=16000,
            transcript="hello world",
        ),
        LibriSpeechItem(
            audio_path=Path("/tmp/b.wav"),
            sample_rate=16000,
            transcript="good morning",
        ),
    ]

    benchmark = WhisperTinyBenchmark(librispeech_dataset=dataset)

    fake_handle = mock.MagicMock()
    fake_handle.model.device = torch.device("cpu")
    fake_handle.processor.return_value = {
        "input_features": torch.zeros((1, 80, 300), dtype=torch.float32),
    }
    fake_handle.model.generate.return_value = torch.tensor([[1, 2, 3]])
    fake_handle.processor.batch_decode.side_effect = [["hello world"], ["good evening"]]

    with (
        mock.patch.object(WhisperTinyBenchmark, "_cast_handle", return_value=fake_handle),
        mock.patch(
            "soundfile.read",
            return_value=(np.zeros((16000,), dtype=np.float32), 16000),
        ),
    ):
        accuracy = benchmark._evaluate_accuracy(
            handle=object(),
            backend=mock.MagicMock(),
            manager=mock.MagicMock(),
        )

    assert accuracy is not None
    details = benchmark._accuracy_details()
    assert details is not None
    assert "wer" in details
