"""Image stages package."""

from __future__ import annotations

from ._base import ImageStage
from ._detection import ImageDetectionStage

__all__ = ["ImageDetectionStage", "ImageStage"]
