"""Video-pipeline messages: tensors, bounding boxes, and detections."""

from __future__ import annotations

import numpy as np  # noqa: TC002
from pydantic import BaseModel

from ._base import BaseMessage


class FrameTensorMessage(BaseMessage):
    """Preprocessed video frame ready for model inference.

    Produced by the frame-preprocessing stage after resizing, normalising,
    and converting a :class:`~moment_to_action.messages.sensor.RawFrameMessage`
    into a format suitable for a neural-network forward pass.

    Attributes:
        tensor: Preprocessed image tensor as a NumPy array (CxHxW or HxWxC).
        original_size: ``(width, height)`` of the source frame before preprocessing,
                       used to map model outputs back to pixel coordinates.
    """

    tensor: np.ndarray
    original_size: tuple[int, int]


class BoundingBox(BaseModel):
    """A single object-detection bounding box produced by a YOLO model.

    Coordinates are expressed in **absolute pixels** relative to the
    *original* (pre-resize) frame dimensions.

    Attributes:
        x1: Left edge of the box (pixels).
        y1: Top edge of the box (pixels).
        x2: Right edge of the box (pixels).
        y2: Bottom edge of the box (pixels).
        confidence: Detection confidence score in ``[0, 1]``.
        class_id: Integer class index assigned by the model.
        label: Human-readable class name corresponding to ``class_id``.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str

    model_config = {"arbitrary_types_allowed": True}


class DetectionMessage(BaseMessage):
    """YOLO detection output for a single frame.

    Bundles all bounding boxes produced by one inference pass along with
    timing metadata so latency can be tracked end-to-end.

    Attributes:
        boxes: All bounding boxes returned by the detector (may be empty).
        latency_ms: Wall-clock time taken by the detection stage in milliseconds.
    """

    boxes: list[BoundingBox]
    latency_ms: float

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
