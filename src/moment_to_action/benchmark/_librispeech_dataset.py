"""Compatibility shim for the LibriSpeech benchmark dataset."""

from __future__ import annotations

import sys

from ._datasets import _librispeech_dataset as _impl
from ._datasets._librispeech_dataset import LibriSpeechDataset, LibriSpeechItem

__all__ = ["LibriSpeechDataset", "LibriSpeechItem"]

sys.modules[__name__] = _impl
