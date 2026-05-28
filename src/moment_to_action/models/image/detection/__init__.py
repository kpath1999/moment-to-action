"""Image detection model subpackage."""

from __future__ import annotations

from ._base import ImageDetectionModel
from ._types import BoundingBox, Detection

__all__ = ["BoundingBox", "Detection", "ImageDetectionModel"]
