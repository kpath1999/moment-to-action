"""Public API for the ``moment_to_action.messages`` package.

All pipeline message types are re-exported from this module so consumers
only need a single import path::

    from moment_to_action.messages import DetectionMessage, RawFrameMessage

A :data:`Message` union type alias is provided for type-checker exhaustiveness
checks and ``isinstance`` guards across the full message hierarchy.
"""

from __future__ import annotations

from typing import TypeAlias

from moment_to_action.models.image.detection._types import BoundingBox, Detection

from ._image_classification import ImageClassificationMessage
from .audio import AudioTensorMessage
from .control import EndOfClipMessage
from .detection import DetectionMessage
from .llm import (
    DecisionMessage,
    DecisionReasoningMessage,
    EndOfGenerationMessage,
    GenerationMessage,
)
from .sensor import RawFrameMessage
from .video import FrameTensorMessage, VideoClipMessage
from .vlm import ClassificationMessage

# Union of every concrete message type in the pipeline.
# Use this alias for ``isinstance`` checks or exhaustive ``match`` statements.
Message: TypeAlias = (
    RawFrameMessage
    | AudioTensorMessage
    | FrameTensorMessage
    | VideoClipMessage
    | DetectionMessage
    | GenerationMessage
    | DecisionMessage
    | DecisionReasoningMessage
    | ClassificationMessage
    | ImageClassificationMessage
    | EndOfClipMessage
    | EndOfGenerationMessage
)

__all__ = [
    "AudioTensorMessage",
    "BoundingBox",
    "ClassificationMessage",
    "DecisionMessage",
    "DecisionReasoningMessage",
    "Detection",
    "DetectionMessage",
    "EndOfClipMessage",
    "EndOfGenerationMessage",
    "FrameTensorMessage",
    "GenerationMessage",
    "ImageClassificationMessage",
    "Message",
    "RawFrameMessage",
    "VideoClipMessage",
]
