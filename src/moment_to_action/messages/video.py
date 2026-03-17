"""Video-pipeline messages: tensors, bounding boxes, and detections."""

from __future__ import annotations

from numpy.typing import NDArray  # noqa: TC002
from pydantic import BaseModel

from ._base import BaseMessage


class FrameTensorMessage(BaseMessage):
    """Preprocessed video frame ready for model inference."""

    tensor: NDArray
    """Preprocessed image tensor as a NumPy array (CxHxW or HxWxC)."""

    original_size: tuple[int, int]
    """``(width, height)`` of the source frame before preprocessing."""


class BoundingBox(BaseModel):
    """A single object-detection bounding box produced by a YOLO model.

    Coordinates are expressed in **absolute pixels** relative to the
    original (pre-resize) frame dimensions.
    """

    x1: float
    """Left edge of the box (pixels)."""

    y1: float
    """Top edge of the box (pixels)."""

    x2: float
    """Right edge of the box (pixels)."""

    y2: float
    """Bottom edge of the box (pixels)."""

    confidence: float
    """Detection confidence score in ``[0, 1]``."""

    class_id: int
    """Integer class index assigned by the model."""

    label: str
    """Human-readable class name corresponding to ``class_id``."""


class DetectionMessage(BaseMessage):
    """YOLO detection output for a single frame."""

    boxes: list[BoundingBox]
    """All bounding boxes returned by the detector (may be empty)."""

    latency_ms: float  # type: ignore[assignment]  # override optional base field as required
    """Wall-clock time taken by the detection stage in milliseconds."""

    @property
    def has_detections(self) -> bool:
        """Return ``True`` when at least one bounding box was detected."""
        return len(self.boxes) > 0

    def top(self, n: int) -> list[BoundingBox]:
        """Return the *n* highest-confidence detections.

        Args:
            n: Maximum number of boxes to return.

        Returns:
            Up to *n* :class:`BoundingBox` instances sorted by descending confidence.
        """
        return sorted(self.boxes, key=lambda b: b.confidence, reverse=True)[:n]
