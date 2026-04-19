"""Compatibility shim for the MSRVTT benchmark dataset."""

from __future__ import annotations

import sys

from ._datasets import _msrvtt_dataset as _impl
from ._datasets._msrvtt_dataset import MsrvttDataset, MsrvttItem

__all__ = ["MsrvttDataset", "MsrvttItem"]

sys.modules[__name__] = _impl
