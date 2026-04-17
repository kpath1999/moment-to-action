"""PromptMessage — carries a formatted LLM prompt through the pipeline.

Emitted by PromptFormatterStage, consumed by LLMStage.
"""

from __future__ import annotations

from pydantic import Field

from ._base import BaseMessage


class PromptMessage(BaseMessage):
    """A fully-formatted prompt ready for LLM inference.

    Produced by PromptFormatterStage from any upstream vision message
    (DetectionMessage, ClassificationMessage, etc.).
    """

    prompt: str
    """The formatted prompt string to be submitted to the LLM."""

    source_stage: str = ""
    """Name of the upstream stage that produced the raw data (e.g. 'YOLOStage')."""

    raw_context: dict = Field(default_factory=dict)
    """Optional structured representation of the data used to build the prompt.
    Useful for debugging and logging without re-parsing the prompt string."""
