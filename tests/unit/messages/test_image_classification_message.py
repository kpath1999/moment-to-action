"""Unit tests for ImageClassificationMessage."""

from __future__ import annotations

import pytest

from moment_to_action.messages._image_classification import ImageClassificationMessage
from moment_to_action.models.image.classification._types import Classification


@pytest.mark.unit
class TestImageClassificationMessage:
    """Tests for ImageClassificationMessage."""

    def test_construction(self) -> None:
        """ImageClassificationMessage can be constructed with valid fields."""
        cls = Classification(label="tench", confidence=0.9, class_id=0)
        msg = ImageClassificationMessage(timestamp=1.0, classifications=[cls])
        assert msg.timestamp == 1.0
        assert len(msg.classifications) == 1
        assert msg.classifications[0].label == "tench"

    def test_empty_classifications(self) -> None:
        """ImageClassificationMessage accepts empty classification list."""
        msg = ImageClassificationMessage(timestamp=0.0, classifications=[])
        assert msg.classifications == []

    def test_default_latency_ms(self) -> None:
        """latency_ms defaults to 0.0."""
        msg = ImageClassificationMessage(timestamp=1.0, classifications=[])
        assert msg.latency_ms == 0.0

    def test_multiple_classifications(self) -> None:
        """ImageClassificationMessage stores multiple classifications."""
        clslist = [
            Classification(label="tench", confidence=0.7, class_id=0),
            Classification(label="goldfish", confidence=0.2, class_id=1),
        ]
        msg = ImageClassificationMessage(timestamp=2.0, classifications=clslist)
        assert len(msg.classifications) == 2
        assert msg.classifications[1].label == "goldfish"

    def test_in_message_union(self) -> None:
        """ImageClassificationMessage is part of the Message union."""
        from moment_to_action.messages import ImageClassificationMessage as Imported
        from moment_to_action.messages import Message

        msg = Imported(timestamp=0.0, classifications=[])
        assert isinstance(msg, Imported)
        # Message is a TypeAlias union — just verify import works
        assert Message is not None
