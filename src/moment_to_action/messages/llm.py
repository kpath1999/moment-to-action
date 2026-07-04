"""LLM-layer messages for streaming language-model generation and yes/no decisions."""

from __future__ import annotations

from typing import Literal

from ._base import BaseMessage


class GenerationMessage(BaseMessage):
    """One partial or final chunk of a streaming LLM generation.

    Emitted by :class:`~moment_to_action.stages.llm.LLMStage` once per token: ``text``
    is the accumulated text of the current ``type`` phase up to and including this
    token (not just the new token), so each message is a complete snapshot of that
    phase so far. ``done`` is ``True`` only on the terminal message for a prompt.
    """

    prompt: str
    """The exact prompt that was submitted to the model."""

    text: str
    """Accumulated text of the current phase (``type``) up to this token."""

    type: Literal["think", "response"]
    """Which phase the current tokens belong to.

    Models without a ``<think>`` block never emit ``"think"`` — every message
    has ``type="response"``.
    """

    done: bool = False
    """True only on the terminal message for this prompt."""


class DecisionMessage(BaseMessage):
    """A yes/no decision extracted from a grammar-constrained LLM generation."""

    decision: Literal["yes", "no"]
    """The extracted decision."""

    prompt: str
    """The exact prompt that produced this decision."""


class DecisionReasoningMessage(BaseMessage):
    """The free-text rationale following a yes/no decision, streamed incrementally."""

    text: str
    """Accumulated rationale text up to this token."""

    prompt: str
    """The exact prompt that produced this rationale."""

    done: bool = False
    """True only on the terminal message for this prompt."""
