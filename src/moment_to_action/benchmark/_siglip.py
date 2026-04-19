"""Compatibility shim for SigLIP benchmarking."""

from __future__ import annotations

import sys

from ._benchmarks import _siglip as _impl
from ._benchmarks._siglip import SigLIPBenchmark

__all__ = ["SigLIPBenchmark"]

sys.modules[__name__] = _impl
