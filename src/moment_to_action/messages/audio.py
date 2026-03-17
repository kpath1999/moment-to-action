"""Audio-layer messages.

Note: This module is a **placeholder** pending full audio-pipeline work.
      The shape of ``AudioTensorMessage`` may change significantly once the
      audio preprocessing and feature-extraction stages are designed.
"""

from __future__ import annotations

import numpy as np  # noqa: TC002

from ._base import BaseMessage


class AudioTensorMessage(BaseMessage):
    """Preprocessed audio tensor ready for model inference.

    Placeholder — not yet used in the active pipeline.  Fields reflect a
    minimal PCM / feature-tensor contract and are subject to revision.

    Attributes:
        data: Audio samples or feature tensor as a NumPy array.
              Shape is intentionally unconstrained until the pipeline is finalised.
        sample_rate: Sampling rate in Hz used when ``data`` holds raw PCM samples.
        source: Identifier for the audio capture device or stream.
    """

    data: np.ndarray
    sample_rate: int = 16000
    source: str = ""
