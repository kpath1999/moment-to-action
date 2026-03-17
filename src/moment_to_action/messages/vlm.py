"""VLM-layer messages for vision-language-model classification outputs."""

from __future__ import annotations

from ._base import BaseMessage


class ClassificationMessage(BaseMessage):
    """Classification result produced by a vision-language model (VLM).

    Represents the outcome of a single VLM inference pass where the model
    assigns a label (and associated confidence) to a visual input.

    Attributes:
        label: Winning class label selected by the model.
        confidence: Confidence score for ``label`` in ``[0, 1]``.
        all_scores: Mapping of every candidate label to its score, allowing
                    callers to inspect the full probability distribution.
        latency_ms: Wall-clock inference time in milliseconds.
    """

    label: str
    confidence: float
    all_scores: dict[str, float]
    latency_ms: float
