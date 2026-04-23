"""Stages package — expose core abstractions and submodules.

Consumers import from the submodules directly::

    from moment_to_action.stages.video import YOLOStage, PreprocessorStage, ClipBufferStage
    from moment_to_action.stages.vlm import MobileCLIPStage, SmolVLM2Stage
    from moment_to_action.stages.llm import ReasoningStage, LLMStage
"""

from __future__ import annotations

from moment_to_action.pipeline import Pipeline

from . import audio, llm, video, vlm
from ._base import Stage
from ._formatter import PromptFormatterStage
from ._trigger import TriggerStage
from .sources import AudioSourceStage, ImageSourceStage

__all__ = [
    "AudioSourceStage",
    "ImageSourceStage",
    "Pipeline",
    "PromptFormatterStage",
    "Stage",
    "TriggerStage",
    "audio",
    "llm",
    "video",
    "vlm",
]
