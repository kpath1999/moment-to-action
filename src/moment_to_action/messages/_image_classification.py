"""Image classification pipeline message."""

from __future__ import annotations

from moment_to_action.models.image.classification._types import Classification  # noqa: TC001

from ._base import BaseMessage


class ImageClassificationMessage(BaseMessage):
    """Output of an image classification model for a single frame.

    Attributes:
        classifications: Top-k predictions ordered by descending confidence.
    """

    classifications: list[Classification]
