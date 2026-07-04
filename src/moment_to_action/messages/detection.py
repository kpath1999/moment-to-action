"""Detection pipeline message."""

from __future__ import annotations

from moment_to_action.models.image.detection._types import Detection  # noqa: TC001

from ._base import BaseMessage


class DetectionMessage(BaseMessage):
    """Output of an object detection model for a single frame.

    Attributes:
        detections: All detections returned by the model (may be empty).
        question: The task question to pose to a downstream LLM stage about
            these detections. Carried through from the originating frame
            message so one loaded ``LLMStage`` can serve any question.
    """

    detections: list[Detection]
    question: str = ""
