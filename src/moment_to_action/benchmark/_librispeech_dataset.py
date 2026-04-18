from __future__ import annotations

import random
import wave
from pathlib import Path

import attrs
import numpy as np
import platformdirs

_DEFAULT_DATASET_ID = "librispeech_asr"
_DEFAULT_CONFIG = "clean"
_DEFAULT_SPLIT = "test"
_PCM16_MAX = 32767.0


def _default_cache_dir() -> Path:
    return platformdirs.user_cache_path("moment_to_action", "GATech") / "librispeech_test_clean"


@attrs.frozen
class LibriSpeechItem:
    """One LibriSpeech utterance with transcript."""

    audio_path: Path
    sample_rate: int
    transcript: str


@attrs.define
class LibriSpeechDataset:
    """LibriSpeech test-clean loader backed by the HuggingFace datasets library."""

    n_items: int = 500
    cache_dir: Path = attrs.Factory(_default_cache_dir)
    seed: int = 42
    dataset_id: str = _DEFAULT_DATASET_ID
    config_name: str = _DEFAULT_CONFIG
    split: str = _DEFAULT_SPLIT
    _items: list[LibriSpeechItem] = attrs.field(factory=list, init=False)

    def __attrs_post_init__(self) -> None:
        if self.n_items <= 0:
            msg = "n_items must be greater than 0"
            raise ValueError(msg)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._items = self._load_items()

    @property
    def dataset_name(self) -> str:
        """Dataset identifier used in benchmark output payloads."""
        return "librispeech_test_clean"

    def items(self) -> list[LibriSpeechItem]:
        """Return sampled utterances for evaluation."""
        return list(self._items)

    def _load_items(self) -> list[LibriSpeechItem]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - guarded by dependency
            msg = "datasets package is required for LibriSpeechDataset"
            raise RuntimeError(msg) from exc

        dataset = load_dataset(
            self.dataset_id,
            self.config_name,
            split=self.split,
            cache_dir=str(self.cache_dir),
        )

        audio_cache_dir = self.cache_dir / "audio"
        audio_cache_dir.mkdir(parents=True, exist_ok=True)

        parsed: list[LibriSpeechItem] = []
        for idx, row in enumerate(dataset):
            item = self._parse_row(row=row, index=idx, audio_cache_dir=audio_cache_dir)
            if item is not None:
                parsed.append(item)

        if not parsed:
            msg = "No valid LibriSpeech items were parsed from dataset rows"
            raise RuntimeError(msg)

        sample_size = min(self.n_items, len(parsed))
        rng = random.Random(self.seed)  # noqa: S311
        return rng.sample(parsed, sample_size)

    @staticmethod
    def _parse_row(
        row: object,
        index: int,
        audio_cache_dir: Path,
    ) -> LibriSpeechItem | None:
        if not isinstance(row, dict):
            return None

        transcript = LibriSpeechDataset._pick_transcript(row)
        if not transcript:
            return None

        audio = row.get("audio")
        if isinstance(audio, dict):
            audio_path = LibriSpeechDataset._audio_path_from_record(
                audio=audio,
                index=index,
                audio_cache_dir=audio_cache_dir,
            )
            if audio_path is None:
                return None
            sample_rate = LibriSpeechDataset._sample_rate_from_record(audio)
            return LibriSpeechItem(
                audio_path=audio_path,
                sample_rate=sample_rate,
                transcript=transcript,
            )

        return None

    @staticmethod
    def _pick_transcript(row: dict[str, object]) -> str:
        for key in ("text", "transcript", "sentence"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _sample_rate_from_record(audio: dict[str, object]) -> int:
        sample_rate = audio.get("sampling_rate")
        if isinstance(sample_rate, int) and sample_rate > 0:
            return sample_rate
        return 16000

    @staticmethod
    def _audio_path_from_record(
        audio: dict[str, object],
        index: int,
        audio_cache_dir: Path,
    ) -> Path | None:
        path = audio.get("path")
        if isinstance(path, str) and path:
            maybe_path = Path(path)
            if maybe_path.exists():
                return maybe_path

        array = audio.get("array")
        if isinstance(array, np.ndarray):
            values = array.astype(np.float32)
        elif isinstance(array, list):
            values = np.asarray(array, dtype=np.float32)
        else:
            return None

        sample_rate = LibriSpeechDataset._sample_rate_from_record(audio)
        out_path = audio_cache_dir / f"sample_{index:07d}.wav"
        LibriSpeechDataset._write_wav(out_path, values, sample_rate)
        return out_path

    @staticmethod
    def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * _PCM16_MAX).astype(np.int16)
        with wave.open(str(path), mode="wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
