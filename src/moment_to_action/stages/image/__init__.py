"""Image stages package."""

from __future__ import annotations

from ._base import ImageStage
from ._classification import ImageClassificationStage
from ._detection import ImageDetectionStage

__all__ = ["ImageClassificationStage", "ImageDetectionStage", "ImageStage"]
