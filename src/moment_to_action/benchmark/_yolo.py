"""Compatibility shim for YOLO benchmarking."""

from __future__ import annotations

import sys

from ._benchmarks import _yolo as _impl
from ._benchmarks._yolo import (
    YOLOBenchmark,
    _load_yolo_tensor,
    _parse_yolo_boxes,
    compute_detection_map,
)

__all__ = ["YOLOBenchmark", "_load_yolo_tensor", "_parse_yolo_boxes", "compute_detection_map"]

sys.modules[__name__] = _impl
