"""Abstract base class for image detection models."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from moment_to_action.models.image._base import ImageModel

if TYPE_CHECKING:
    from moment_to_action.models.image.detection._types import Detection


class ImageDetectionModel(ImageModel):
    """Abstract base for models that detect objects in images.

    Narrows :meth:`post_proc` to return ``list[Detection]`` instead of
    the generic ``list[object]`` defined by :class:`~moment_to_action.models.image.ImageModel`.
    """

    @abstractmethod
    def post_proc(self, raw: object) -> list[Detection]:  # type: ignore[override]
        """Decode raw model output into a list of detections.

        Args:
            raw: Output returned by :meth:`~moment_to_action.models.image.ImageModel.run`.

        Returns:
            List of :class:`~moment_to_action.models.image.detection.Detection` objects.
        """
        ...
