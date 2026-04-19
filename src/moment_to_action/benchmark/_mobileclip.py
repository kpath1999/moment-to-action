"""Compatibility shim for MobileCLIP benchmarking."""

from __future__ import annotations

import sys

from ._benchmarks import _mobileclip as _impl
from ._benchmarks._mobileclip import MobileCLIPBenchmark

__all__ = ["MobileCLIPBenchmark"]

sys.modules[__name__] = _impl
