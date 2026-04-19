"""Compatibility shim for the GSM8K benchmark dataset."""

from __future__ import annotations

import sys

from ._datasets import _gsm8k_dataset as _impl
from ._datasets._gsm8k_dataset import GSM8KDataset, GSM8KItem

__all__ = ["GSM8KDataset", "GSM8KItem"]

sys.modules[__name__] = _impl
