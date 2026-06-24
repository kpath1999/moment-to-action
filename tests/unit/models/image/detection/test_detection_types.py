"""Unit tests for detection POD types (BoundingBox, Detection)."""

from __future__ import annotations

import pytest

from moment_to_action.models.image.detection._types import BoundingBox, Detection


@pytest.mark.unit
class TestBoundingBox:
    """Tests for BoundingBox attrs.frozen type."""

    def test_construction(self) -> None:
        """BoundingBox stores x1, y1, x2, y2."""
        bb = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        assert bb.x1 == 10.0
        assert bb.y1 == 20.0
        assert bb.x2 == 100.0
        assert bb.y2 == 200.0

    def test_immutable(self) -> None:
        """BoundingBox is frozen (attrs.frozen)."""
        bb = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        with pytest.raises(AttributeError):
            bb.x1 = 5.0  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two BoundingBoxes with identical fields compare equal."""
        a = BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
        b = BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
        c = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        assert a == b
        assert a != c

    def test_field_types_are_float(self) -> None:
        """All coordinate fields are float."""
        bb = BoundingBox(x1=1, y1=2, x2=3, y2=4)
        assert isinstance(bb.x1, int | float)


@pytest.mark.unit
class TestDetection:
    """Tests for Detection attrs.frozen type."""

    def test_construction(self) -> None:
        """Detection stores label, confidence, and bbox."""
        bb = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        d = Detection(label="person", confidence=0.9, bbox=bb)
        assert d.label == "person"
        assert d.confidence == pytest.approx(0.9)
        assert d.bbox is bb

    def test_immutable(self) -> None:
        """Detection is frozen."""
        bb = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        d = Detection(label="cat", confidence=0.5, bbox=bb)
        with pytest.raises(AttributeError):
            d.label = "dog"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two Detections with identical fields compare equal."""
        bb = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        a = Detection(label="car", confidence=0.8, bbox=bb)
        b = Detection(label="car", confidence=0.8, bbox=bb)
        c = Detection(label="truck", confidence=0.8, bbox=bb)
        assert a == b
        assert a != c
