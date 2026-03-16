"""Messages are the data contracts between stages.

Each stage consumes one message type and produces another.
Plain dataclasses — no logic, no imports from the rest of the framework.

Creating messages focused around YoLo and MobileCLIP for now.
The idea is to make it easy to extend the message space in two ways:
    1. Add new classes of messages (mostly for different sensors)
    2. Add/Remove new parameters to existing message classes if different information is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class RawFrameMessage:
    """Out of SensorStage. Raw image, nothing done to it yet."""

    frame: np.ndarray  # HxWxC uint8, BGR (OpenCV default)
    timestamp: float
    source: str = ""  # path or device name, useful for debugging
    width: int = 0
    height: int = 0


@dataclass
class TensorMessage:
    """Out of PreprocessorStage. Frame is resized, normalized, ready for a model."""

    tensor: np.ndarray  # model-ready float32 tensor, shape depends on model
    original_size: tuple  # (H, W) before preprocessing
    timestamp: float


@dataclass
class BoundingBox:
    """A single YOLO detection."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str


@dataclass
class DetectionMessage:
    """Out of ModelStage (YOLO). Zero or more bounding boxes."""

    boxes: list[BoundingBox]
    latency_ms: float
    timestamp: float

    @property
    def has_detections(self) -> bool:
        """Return True if any bounding boxes are present."""
        return len(self.boxes) > 0

    def top(self, n: int = 1) -> list[BoundingBox]:
        """Return top-n detections by confidence."""
        return sorted(self.boxes, key=lambda b: -b.confidence)[:n]


@dataclass
class ReasoningMessage:
    """Out of ReasoningStage (LLM). Final text output."""

    response: str
    prompt: str  # what was sent to the LLM, useful for debugging
    latency_ms: float
    timestamp: float


@dataclass
class ClassificationMessage:
    """Out of MobileCLIPStage. Best matching prompt and all scores."""

    label: str  # best matching prompt
    confidence: float  # softmax score for best match
    all_scores: dict[str, float]  # prompt → score for all prompts
    latency_ms: float
    timestamp: float


# Union of all message types — what a stage can receive or emit
# (this will be useful for multimodal stages, where we may have to combine data
# from different sensors)
Message = (
    RawFrameMessage | TensorMessage | DetectionMessage | ReasoningMessage | ClassificationMessage
)
