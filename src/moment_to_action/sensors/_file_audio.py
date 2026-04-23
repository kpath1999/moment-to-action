"""File-based sensor: reads audio from disk."""

from __future__ import annotations

import logging
import pathlib
import subprocess  # noqa: S404
import time

import numpy as np

from moment_to_action.messages.sensor import AudioInput

from ._base import BaseSensor

logger = logging.getLogger(__name__)
_MP4_SAMPLE_RATE = 16_000
_UINT8_SAMPLE_WIDTH = 1
_INT16_SAMPLE_WIDTH = 2
_INT24_SAMPLE_WIDTH = 3
_INT32_SAMPLE_WIDTH = 4
_UINT8_PCM_ZERO = 128.0
_UINT8_PCM_SCALE = 128.0
_INT16_PCM_SCALE = 32768.0
_INT24_SIGN_BIT = 0x800000
_INT24_SIGN_EXTEND_MASK = ~0xFFFFFF
_INT24_PCM_SCALE = 8388608.0
_INT32_PCM_SCALE = 2147483648.0


class FileAudioSensor(BaseSensor):
    """Sensor that reads audio from a file on disk."""

    def __init__(self, path: str | pathlib.Path) -> None:
        self._path: pathlib.Path = pathlib.Path(path)

    def open(self) -> None:
        """Validate that the audio file exists."""
        if not self._path.is_file():
            msg = f"FileAudioSensor: audio file not found: {self._path}"
            raise FileNotFoundError(msg)
        logger.debug("FileAudioSensor opened: %s", self._path)

    def read(self) -> AudioInput:
        """Load the audio file from disk and return it as an ``AudioInput``."""
        suffix = self._path.suffix.lower()
        if suffix == ".wav":
            waveform, sample_rate = self._load_wav()
        elif suffix == ".mp4":
            waveform, sample_rate = self._load_mp4()
        else:
            msg = f"FileAudioSensor: unsupported audio format: {self._path.suffix}"
            raise ValueError(msg)

        return AudioInput(
            waveform=waveform,
            source=str(self._path),
            sample_rate=sample_rate,
            num_samples=len(waveform),
            timestamp=time.time(),
        )

    def close(self) -> None:
        """No-op: file-based reads hold no persistent resources."""
        logger.debug("FileAudioSensor closed: %s", self._path)

    def _load_wav(self) -> tuple[np.ndarray, int]:
        try:
            import soundfile as sf
        except ImportError:
            return self._load_wav_stdlib()

        waveform, sample_rate = sf.read(str(self._path), dtype="float32", always_2d=True)
        return waveform.mean(axis=1).astype(np.float32), int(sample_rate)

    def _load_wav_stdlib(self) -> tuple[np.ndarray, int]:
        import wave

        with wave.open(str(self._path), "rb") as wav:
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            num_channels = wav.getnchannels()
            raw = wav.readframes(wav.getnframes())

        waveform = self._pcm_bytes_to_float32(raw, sample_width)
        if num_channels > 1:
            waveform = waveform.reshape(-1, num_channels).mean(axis=1)
        return waveform.astype(np.float32), sample_rate

    def _load_mp4(self) -> tuple[np.ndarray, int]:
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(self._path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(_MP4_SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ]
        try:
            result = subprocess.run(  # noqa: S603, S607
                command,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            msg = "FileAudioSensor: loading .mp4 audio requires ffmpeg on PATH."
            raise RuntimeError(msg) from exc

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            msg = f"FileAudioSensor: could not load audio from {self._path}: {stderr}"
            raise OSError(msg)

        return np.frombuffer(result.stdout, dtype="<f4").astype(np.float32), _MP4_SAMPLE_RATE

    @staticmethod
    def _pcm_bytes_to_float32(raw: bytes, sample_width: int) -> np.ndarray:
        if sample_width == _UINT8_SAMPLE_WIDTH:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            return (data - _UINT8_PCM_ZERO) / _UINT8_PCM_SCALE
        if sample_width == _INT16_SAMPLE_WIDTH:
            return np.frombuffer(raw, dtype="<i2").astype(np.float32) / _INT16_PCM_SCALE
        if sample_width == _INT24_SAMPLE_WIDTH:
            bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, _INT24_SAMPLE_WIDTH)
            data = (
                bytes_[:, 0].astype(np.int32)
                | (bytes_[:, 1].astype(np.int32) << 8)
                | (bytes_[:, 2].astype(np.int32) << 16)
            )
            data = np.where(data & _INT24_SIGN_BIT, data | _INT24_SIGN_EXTEND_MASK, data)
            return data.astype(np.float32) / _INT24_PCM_SCALE
        if sample_width == _INT32_SAMPLE_WIDTH:
            return np.frombuffer(raw, dtype="<i4").astype(np.float32) / _INT32_PCM_SCALE

        msg = f"FileAudioSensor: unsupported WAV sample width: {sample_width} bytes"
        raise ValueError(msg)
