"""Compatibility shim for Grounding DINO benchmarking."""

from __future__ import annotations

import sys

from ._benchmarks import _grounding_dino as _impl
from ._benchmarks._grounding_dino import GroundingDINOBenchmark

__all__ = ["GroundingDINOBenchmark"]

sys.modules[__name__] = _impl
