"""Compatibility shim for benchmark base types."""

from __future__ import annotations

import sys

from ._benchmarks import _base as _impl
from ._benchmarks._base import ModelBenchmark, detect_platform, psutil

__all__ = ["ModelBenchmark", "detect_platform", "psutil"]

sys.modules[__name__] = _impl
