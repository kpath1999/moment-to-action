"""Stages package — expose core abstractions and submodules.

Consumers import from the submodules directly::

    from moment_to_action.stages.image import ImageStage, ImageDetectionStage
    from moment_to_action.stages.llm import LLMStage, DecisionStage
    from moment_to_action.stages.vlm import VLMDescriptionStage
"""

from __future__ import annotations

from moment_to_action.pipeline import Pipeline

from . import image, llm, vlm
from ._base import Stage

__all__ = ["Pipeline", "Stage", "image", "llm", "vlm"]
