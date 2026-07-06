"""LLM-layer messages for streaming language-model generation and yes/no decisions."""

from __future__ import annotations

from typing import Literal

from ._base import BaseMessage


class GenerationMessage(BaseMessage):
    """One chunk of a streaming LLM generation.

    Emitted by :class:`~moment_to_action.stages.llm.LLMStage` once per token: ``text``
    is the accumulated text of the current ``type`` phase up to and including this
    token (not just the new token), so each message is a complete snapshot of that
    phase so far. The stream for one prompt always ends with an
    :class:`EndOfGenerationMessage` carrying the same ``prompt``, rather than a
    ``done`` flag on the last content message.
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


class EndOfGenerationMessage(BaseMessage):
    """Sentinel marking the end of one prompt's streamed generation.

    Emitted once, after the last :class:`GenerationMessage` for a given
    ``prompt``, by :class:`~moment_to_action.stages.llm.LLMStage` /
    :class:`~moment_to_action.stages.vlm.VLMDescriptionStage`.
    :class:`~moment_to_action.stages.llm.DecisionStage` forwards it unchanged,
    since its filtered/stripped reasoning stream shares the same lifecycle as the
    raw generation it's derived from — so this single sentinel also marks the end
    of the corresponding :class:`DecisionReasoningMessage` stream.
    """

    prompt: str
    """The exact prompt whose generation just ended."""


class DecisionMessage(BaseMessage):
    """A yes/no decision extracted from a grammar-constrained LLM generation."""

    decision: Literal["yes", "no"]
    """The extracted decision."""

    prompt: str
    """The exact prompt that produced this decision."""


class DecisionReasoningMessage(BaseMessage):
    """The free-text rationale following a yes/no decision, streamed incrementally.

    The stream for one prompt ends when an :class:`EndOfGenerationMessage` carrying
    the same ``prompt`` arrives (forwarded by
    :class:`~moment_to_action.stages.llm.DecisionStage` from the underlying
    generation it's reasoning over), rather than a ``done`` flag on the last
    rationale message.
    """

    text: str
    """Accumulated rationale text up to this token."""

    prompt: str
    """The exact prompt that produced this rationale."""
