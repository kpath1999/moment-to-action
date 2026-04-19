"""Compatibility shim for the COCO benchmark dataset."""

from __future__ import annotations

import sys

from ._datasets import _coco_dataset as _impl
from ._datasets._coco_dataset import CocoDataset

__all__ = ["CocoDataset"]

sys.modules[__name__] = _impl
