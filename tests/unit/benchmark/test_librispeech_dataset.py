from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from moment_to_action.benchmark._datasets._librispeech_dataset import (
    LibriSpeechDataset,
    LibriSpeechItem,
)


@pytest.mark.unit
def test_librispeech_dataset_writes_wav(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    samples = np.zeros((160,), dtype=np.float32)
    LibriSpeechDataset._write_wav(path, samples, sample_rate=16000)
    assert path.exists()
    assert path.stat().st_size > 0


@pytest.mark.unit
def test_librispeech_dataset_rejects_non_positive_n_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LibriSpeechDataset, "_load_items", lambda _self: [])
    with pytest.raises(ValueError, match="n_items must be greater than 0"):
        LibriSpeechDataset(n_items=0)


@pytest.mark.unit
def test_librispeech_pick_transcript_and_sample_rate_helpers() -> None:
    assert LibriSpeechDataset._pick_transcript({"text": " hello "}) == "hello"
    assert LibriSpeechDataset._pick_transcript({"sentence": "a"}) == "a"
    assert LibriSpeechDataset._pick_transcript({"text": ""}) == ""

    assert LibriSpeechDataset._sample_rate_from_record({"sampling_rate": 22050}) == 22050
    assert LibriSpeechDataset._sample_rate_from_record({"sampling_rate": 0}) == 16000


@pytest.mark.unit
def test_librispeech_audio_path_from_record_existing_path(tmp_path: Path) -> None:
    audio_file = tmp_path / "a.wav"
    audio_file.write_bytes(b"wav")

    path = LibriSpeechDataset._audio_path_from_record(
        audio={"path": str(audio_file), "sampling_rate": 16000},
        index=0,
        audio_cache_dir=tmp_path,
    )
    assert path == audio_file


@pytest.mark.unit
def test_librispeech_parse_row_and_load_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_rows = [
        {"audio": {"array": [0.1, 0.2], "sampling_rate": 16000}, "text": "one"},
        {"audio": {"array": np.array([0.3, 0.4], dtype=np.float32)}, "transcript": "two"},
        {"audio": {"bad": True}, "text": "skip"},
        "not-dict",
    ]

    fake_datasets = mock.MagicMock()
    fake_datasets.load_dataset.return_value = fake_rows
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    dataset = LibriSpeechDataset(n_items=2, cache_dir=tmp_path)
    items = dataset.items()
    assert len(items) == 2
    assert all(isinstance(item, LibriSpeechItem) for item in items)


@pytest.mark.unit
def test_librispeech_parse_row_invalid_paths(tmp_path: Path) -> None:
    audio_cache = tmp_path / "audio"
    audio_cache.mkdir()

    assert LibriSpeechDataset._parse_row("x", 0, audio_cache) is None
    assert LibriSpeechDataset._parse_row({"audio": {"array": [0.1]}}, 0, audio_cache) is None
    assert LibriSpeechDataset._parse_row({"text": "ok", "audio": {}}, 0, audio_cache) is None
    assert LibriSpeechDataset._parse_row({"text": "ok", "audio": "bad"}, 0, audio_cache) is None


@pytest.mark.unit
def test_librispeech_load_items_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace())

    with pytest.raises(RuntimeError, match="datasets package is required"):
        LibriSpeechDataset(n_items=1)


@pytest.mark.unit
def test_librispeech_load_items_raises_when_no_valid_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_datasets = mock.MagicMock()
    fake_datasets.load_dataset.return_value = [{"audio": {"bad": True}, "text": ""}]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="No valid LibriSpeech items"):
        LibriSpeechDataset(n_items=1, cache_dir=tmp_path)


@pytest.mark.unit
def test_librispeech_dataset_name_property(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LibriSpeechDataset, "_load_items", lambda _self: [mock.MagicMock()])
    dataset = LibriSpeechDataset(n_items=1)
    assert dataset.dataset_name == "librispeech_test_clean"
