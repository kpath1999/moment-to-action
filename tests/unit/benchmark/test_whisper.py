from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from moment_to_action.benchmark import WhisperTinyBenchmark
from moment_to_action.benchmark._benchmarks._whisper import _normalize_text, _WhisperHandle
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


@pytest.mark.unit
def test_whisper_run_inference_type_error() -> None:
    benchmark = WhisperTinyBenchmark()
    with pytest.raises(TypeError, match="Invalid Whisper benchmark handle"):
        benchmark._run_inference(object(), mock.MagicMock(), mock.MagicMock())


@pytest.mark.unit
def test_whisper_run_inference_requires_mapping_inputs() -> None:
    model = mock.MagicMock()
    model.device = torch.device("cpu")
    processor = mock.MagicMock()
    handle = _WhisperHandle(model=model, processor=processor)

    with pytest.raises(TypeError, match="expects mapping"):
        WhisperTinyBenchmark()._run_inference(handle, [1, 2, 3], mock.MagicMock())


@pytest.mark.unit
def test_whisper_run_inference_happy_path() -> None:
    model = mock.MagicMock()
    model.device = torch.device("cpu")
    processor = mock.MagicMock()
    handle = _WhisperHandle(model=model, processor=processor)

    WhisperTinyBenchmark()._run_inference(
        handle,
        {"input_features": torch.zeros((1, 80, 300), dtype=torch.float32)},
        mock.MagicMock(),
    )
    model.generate.assert_called_once()


@pytest.mark.unit
def test_whisper_evaluate_accuracy_none_without_dataset() -> None:
    benchmark = WhisperTinyBenchmark(librispeech_dataset=None)
    result = benchmark._evaluate_accuracy(object(), mock.MagicMock(), mock.MagicMock())
    assert result is None


@pytest.mark.unit
def test_whisper_cast_handle_type_error() -> None:
    with pytest.raises(TypeError, match="Invalid Whisper benchmark handle"):
        WhisperTinyBenchmark._cast_handle(object())


@pytest.mark.unit
def test_whisper_normalize_text_removes_punctuation() -> None:
    assert _normalize_text(" Hello,   WORLD! ") == "hello world"


@pytest.mark.unit
def test_whisper_evaluate_accuracy_none_when_all_decodes_empty() -> None:
    dataset = mock.MagicMock()
    dataset.items.return_value = [
        LibriSpeechItem(audio_path=Path("/tmp/a.wav"), sample_rate=16000, transcript="hello"),
    ]
    benchmark = WhisperTinyBenchmark(librispeech_dataset=dataset)

    fake_handle = mock.MagicMock()
    fake_handle.model.device = torch.device("cpu")
    fake_handle.processor.return_value = {
        "input_features": torch.zeros((1, 80, 300), dtype=torch.float32),
    }
    fake_handle.model.generate.return_value = torch.tensor([[1, 2, 3]])
    fake_handle.processor.batch_decode.return_value = []

    with (
        mock.patch.object(WhisperTinyBenchmark, "_cast_handle", return_value=fake_handle),
        mock.patch(
            "soundfile.read",
            return_value=(np.zeros((16000,), dtype=np.float32), 16000),
        ),
    ):
        accuracy = benchmark._evaluate_accuracy(object(), mock.MagicMock(), mock.MagicMock())

    assert accuracy is None


@pytest.mark.unit
def test_whisper_evaluate_accuracy_handles_stereo_audio() -> None:
    dataset = mock.MagicMock()
    dataset.items.return_value = [
        LibriSpeechItem(audio_path=Path("/tmp/a.wav"), sample_rate=16000, transcript="hello")
    ]
    benchmark = WhisperTinyBenchmark(librispeech_dataset=dataset)

    fake_handle = mock.MagicMock()
    fake_handle.model.device = torch.device("cpu")
    fake_handle.processor.return_value = {
        "input_features": torch.zeros((1, 80, 300), dtype=torch.float32),
    }
    fake_handle.model.generate.return_value = torch.tensor([[1, 2]], dtype=torch.long)
    fake_handle.processor.batch_decode.return_value = ["hello"]

    stereo = np.zeros((100, 2), dtype=np.float32)
    with (
        mock.patch.object(WhisperTinyBenchmark, "_cast_handle", return_value=fake_handle),
        mock.patch("soundfile.read", return_value=(stereo, 16000)),
    ):
        accuracy = benchmark._evaluate_accuracy(object(), mock.MagicMock(), mock.MagicMock())

    assert accuracy == pytest.approx(1.0)
