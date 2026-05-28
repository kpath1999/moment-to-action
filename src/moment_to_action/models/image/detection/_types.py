"""Detection output types for image-based object detectors."""

from __future__ import annotations

import attrs


@attrs.frozen
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates.

    Attributes:
        x1: Left edge in pixels.
        y1: Top edge in pixels.
        x2: Right edge in pixels.
        y2: Bottom edge in pixels.
    """

    x1: float
    y1: float
    x2: float
    y2: float


@attrs.frozen
class Detection:
    """Single object detection result.

    Attributes:
        label: Human-readable class name.
        confidence: Detection confidence in [0, 1].
        bbox: Bounding box in original image pixel coordinates.
    """

    label: str
    confidence: float
    bbox: BoundingBox
