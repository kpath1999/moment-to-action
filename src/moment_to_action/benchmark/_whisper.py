"""Compatibility shim for Whisper benchmarking."""

from __future__ import annotations

import sys

from ._benchmarks import _whisper as _impl
from ._benchmarks._whisper import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTinyBenchmark

__all__ = ["AutoModelForSpeechSeq2Seq", "AutoProcessor", "WhisperTinyBenchmark"]

sys.modules[__name__] = _impl
