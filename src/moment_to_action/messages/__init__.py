"""Public API for the ``moment_to_action.messages`` package.

All pipeline message types are re-exported from this module so consumers
only need a single import path::

    from moment_to_action.messages import DetectionMessage, RawFrameMessage

A :data:`Message` union type alias is provided for type-checker exhaustiveness
checks and ``isinstance`` guards across the full message hierarchy.
"""

from __future__ import annotations

from .audio import AudioTensorMessage
from .llm import ReasoningMessage
from .prompt import PromptMessage
from .sensor import AudioInput, RawFrameMessage
from .video import BoundingBox, DetectionMessage, FrameTensorMessage, VideoClipMessage
from .vlm import ClassificationMessage

# Union of every concrete message type in the pipeline.
# Use this alias for ``isinstance`` checks or exhaustive ``match`` statements.
type Message = (
    RawFrameMessage
    | AudioInput
    | AudioTensorMessage
    | FrameTensorMessage
    | VideoClipMessage
    | DetectionMessage
    | ReasoningMessage
    | ClassificationMessage
    | PromptMessage
)

__all__ = [
    "AudioInput",
    "AudioTensorMessage",
    "BoundingBox",
    "ClassificationMessage",
    "DetectionMessage",
    "FrameTensorMessage",
    "Message",
    "PromptMessage",
    "RawFrameMessage",
    "ReasoningMessage",
    "VideoClipMessage",
]
