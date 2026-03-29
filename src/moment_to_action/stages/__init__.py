"""Stages package — expose core abstractions and submodules.

Consumers import from the submodules directly::

    from moment_to_action.stages.video import YOLOStage, PreprocessorStage, ClipBufferStage
    from moment_to_action.stages.vlm import MobileCLIPStage, SmolVLM2Stage
    from moment_to_action.stages.llm import ReasoningStage
"""

from __future__ import annotations

import importlib

from moment_to_action.pipeline import Pipeline

from ._base import Stage

__all__ = ["Pipeline", "Stage", "llm", "video", "vlm"]


def __getattr__(name: str) -> object:
    """Load stage subpackages lazily."""
    if name not in {"llm", "video", "vlm"}:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    return importlib.import_module(f"{__name__}.{name}")
