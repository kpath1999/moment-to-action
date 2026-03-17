"""VLM-layer messages for vision-language-model classification outputs."""

from __future__ import annotations

from ._base import BaseMessage


class ClassificationMessage(BaseMessage):
    """Classification result produced by a vision-language model (VLM)."""

    label: str
    """Winning class label selected by the model."""

    confidence: float
    """Confidence score for ``label`` in ``[0, 1]``."""

    all_scores: dict[str, float]
    """Mapping of every candidate label to its score (full probability distribution)."""

    latency_ms: float  # type: ignore[assignment]  # override optional base field as required
    """Wall-clock inference time in milliseconds."""
