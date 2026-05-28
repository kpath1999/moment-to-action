"""Marker base class for image-input pipeline stages."""

from __future__ import annotations

from moment_to_action.stages._base import Stage


class ImageStage(Stage):
    """Marker base class for stages that process image-based messages.

    Extends :class:`~moment_to_action.stages._base.Stage` without adding new
    abstract methods.  Subclasses must implement
    :meth:`~moment_to_action.stages._base.Stage._process`.
    """
