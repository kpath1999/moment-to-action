"""Video-pipeline messages: tensors, bounding boxes, detections, clips, and actions."""

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


class VideoClipMessage(BaseMessage):
    """A temporal window of raw frames captured from a live stream or file.

    Populated by :class:`~moment_to_action.stages.video.ClipBufferStage`.
    Consumed by temporal action-recognition stages (e.g. MoViNet, VideoMAE).

    All frames share the same spatial dimensions (width x height) and are
    stored in capture order (oldest first).
    """

    frames: list[NDArray]
    """Ordered list of raw BGR frames (HxWxC, uint8).  Oldest frame first."""

    source: str = ""
    """Identifier of the originating sensor (camera index, device path, etc.)."""

    width: int = 0
    """Frame width in pixels; ``0`` when unknown."""

    height: int = 0
    """Frame height in pixels; ``0`` when unknown."""

    fps: float = 0.0
    """Capture frame-rate reported by the sensor; ``0.0`` when unknown."""

    @property
    def num_frames(self) -> int:
        """Number of frames in the clip."""
        return len(self.frames)


class ActionPrediction(BaseModel):
    """A single action class prediction from a temporal model."""

    label: str
    """Human-readable action label (e.g. ``"walking"``, ``"running"``)."""

    confidence: float
    """Probability / softmax score in ``[0, 1]``."""

    class_id: int = -1
    """Integer class index from the model vocabulary; ``-1`` when unavailable."""


class ActionMessage(BaseMessage):
    """Output of a temporal action-recognition stage.

    Produced by :class:`~moment_to_action.stages.video.TemporalActionStage`.
    """

    predictions: list[ActionPrediction]
    """All action predictions, sorted by descending confidence."""

    clip_num_frames: int
    """Number of frames in the input clip that produced these predictions."""

    model_name: str = ""
    """Name/path of the model that produced this output (for logging)."""

    @property
    def top_action(self) -> ActionPrediction | None:
        """Return the highest-confidence prediction, or ``None`` if empty."""
        return self.predictions[0] if self.predictions else None
