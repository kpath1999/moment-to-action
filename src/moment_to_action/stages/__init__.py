"""Stages package — expose core abstractions and submodules.

Consumers import from the submodules directly::

    from moment_to_action.stages.video import YOLOStage, PreprocessorStage
    from moment_to_action.stages.vlm import MobileCLIPStage
    from moment_to_action.stages.llm import ReasoningStage
"""

from __future__ import annotations

from moment_to_action._pipeline import Pipeline

from . import llm, video, vlm
from ._base import Stage

__all__ = ["Pipeline", "Stage", "llm", "video", "vlm"]
