"""Marker base class for image-input pipeline stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from moment_to_action.stages._base import ModelStage

if TYPE_CHECKING:
    from moment_to_action.models._base import BaseModel

_ModelT = TypeVar("_ModelT", bound="BaseModel")


class ImageStage(ModelStage[_ModelT], Generic[_ModelT]):
    """Marker base class for model-backed stages that process image-based messages.

    Extends :class:`~moment_to_action.stages._base.ModelStage` without adding new
    abstract methods.  Subclasses must implement
    :meth:`~moment_to_action.stages._base.Stage._process`, parametrizing with
    their concrete model type (e.g. ``ImageStage[ImageDetectionModel]``).
    """
