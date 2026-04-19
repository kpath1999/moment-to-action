"""Compatibility shim for SmolVLM2 benchmarking."""

from __future__ import annotations

import sys

from ._benchmarks import _smolvlm2 as _impl
from ._benchmarks._smolvlm2 import (
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLM2Benchmark,
    _sample_video_frames,
)

__all__ = [
    "AutoModelForImageTextToText",
    "AutoProcessor",
    "SmolVLM2Benchmark",
    "_sample_video_frames",
]

sys.modules[__name__] = _impl
