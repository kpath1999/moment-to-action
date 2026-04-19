"""Compatibility shim for Qwen3 benchmarking."""

from __future__ import annotations

import sys

from ._benchmarks import _qwen3 as _impl
from ._benchmarks._qwen3 import AutoModelForCausalLM, AutoTokenizer, Qwen3Benchmark

__all__ = ["AutoModelForCausalLM", "AutoTokenizer", "Qwen3Benchmark"]

sys.modules[__name__] = _impl
