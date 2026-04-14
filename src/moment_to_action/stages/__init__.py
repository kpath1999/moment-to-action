"""Stages package — expose core abstractions and submodules.

Consumers import from the submodules directly::

    from moment_to_action.stages.video import YOLOStage, PreprocessorStage, ClipBufferStage
    from moment_to_action.stages.vlm import MobileCLIPStage, SmolVLM2Stage
    from moment_to_action.stages.llm import ReasoningStage, LLMStage
"""

from __future__ import annotations

from moment_to_action.pipeline import Pipeline

from . import llm, video, vlm, audio
from ._base import Stage
from ._formatter import PromptFormatterStage

__all__ = ["Pipeline", "Stage", "llm", "video", "vlm"]
