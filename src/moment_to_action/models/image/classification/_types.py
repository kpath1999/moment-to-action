"""Types for image classification models."""

from __future__ import annotations

import attrs


@attrs.frozen
class Classification:
    """A single image classification result.

    Args:
        label: Human-readable class label.
        confidence: Softmax probability in ``[0, 1]``.
        class_id: Integer class index in the model's label list.
    """

    label: str
    """Human-readable class label."""

    confidence: float
    """Softmax probability in ``[0, 1]``."""

    class_id: int
    """Integer class index in the model's label list."""
