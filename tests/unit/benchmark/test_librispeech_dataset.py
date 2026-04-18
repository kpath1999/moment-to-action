from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moment_to_action.benchmark._librispeech_dataset import LibriSpeechDataset


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
