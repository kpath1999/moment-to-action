"""Audio-layer messages.

Note: This module is a **placeholder** pending full audio-pipeline work.
      The shape of ``AudioTensorMessage`` may change significantly once the
      audio preprocessing and feature-extraction stages are designed.
"""

from __future__ import annotations

from numpy.typing import NDArray  # noqa: TC002

from ._base import BaseMessage


class AudioTensorMessage(BaseMessage):
    """Preprocessed audio tensor ready for model inference.

    Placeholder — not yet used in the active pipeline.
    """

    data: NDArray
    """Audio samples or feature tensor as a NumPy array.
    Shape is intentionally unconstrained until the pipeline is finalised."""

    sample_rate: int
    """Sampling rate in Hz used when ``data`` holds raw PCM samples."""

    source: str
    """Identifier for the audio capture device or stream."""
