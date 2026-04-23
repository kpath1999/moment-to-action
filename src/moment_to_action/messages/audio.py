"""Audio-layer messages.

Note: This module is a **placeholder** pending full audio-pipeline work.
      The shape of ``AudioTensorMessage`` may change significantly once the
      audio preprocessing and feature-extraction stages are designed.
"""

from __future__ import annotations

from numpy.typing import NDArray  # noqa: TC002
from pydantic import Field

from ._base import BaseMessage
from .sensor import AudioInput


class RawAudioMessage(AudioInput):
    """Raw audio loaded directly from a file source."""

    source_path: str = ""
    """Path to the source audio file on disk."""


class AudioTensorMessage(BaseMessage):
    """Preprocessed audio tensor ready for model inference."""

    data: NDArray
    """Audio samples or feature tensor as a NumPy array."""

    sample_rate: int
    """Sampling rate in Hz used when ``data`` holds raw PCM samples."""

    source: str
    """Identifier for the audio capture device or stream."""


class AudioClassificationMessage(BaseMessage):
    """Audio event classification output."""

    top_predictions: dict[str, float]
    """Top predicted classes and their scores, highest first."""

    sample_rate: int
    """Sampling rate in Hz for the analyzed clip."""

    source: str
    """Identifier for the audio capture device or stream."""


class AudioTranscriptionMessage(BaseMessage):
    """Speech-to-text output for an audio clip."""

    text: str
    """Transcribed text."""

    language: str | None = None
    """Detected or forced language code, if available."""

    confidence: float | None = None
    """Optional confidence score reported by the transcriber."""

    segments: list[dict] = Field(default_factory=list)
    """Optional per-segment metadata from the transcription backend."""

    sample_rate: int
    """Sampling rate in Hz for the analyzed clip."""

    source: str
    """Identifier for the audio capture device or stream."""
