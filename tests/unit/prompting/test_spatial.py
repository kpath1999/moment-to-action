"""Unit tests for prompting._spatial."""

from __future__ import annotations

import pytest

from moment_to_action.models.image.detection._types import BoundingBox
from moment_to_action.prompting import _spatial


def _bb(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    """Shorthand BoundingBox constructor for tests."""
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


@pytest.mark.unit
class TestArea:
    """Tests for area()."""

    def test_area_computes_width_times_height(self) -> None:
        """area() multiplies width by height."""
        assert _spatial.area(_bb(0, 0, 10, 5)) == 50.0

    def test_area_zero_for_degenerate_box(self) -> None:
        """area() is zero when width or height collapses to zero."""
        assert _spatial.area(_bb(5, 5, 5, 10)) == 0.0


@pytest.mark.unit
class TestIou:
    """Tests for iou()."""

    def test_iou_identical_boxes_is_one(self) -> None:
        """Identical boxes have IoU 1.0."""
        b = _bb(0, 0, 10, 10)
        assert _spatial.iou(b, b) == pytest.approx(1.0)

    def test_iou_non_overlapping_is_zero(self) -> None:
        """Disjoint boxes have IoU 0.0."""
        a = _bb(0, 0, 10, 10)
        b = _bb(20, 20, 30, 30)
        assert _spatial.iou(a, b) == 0.0

    def test_iou_partial_overlap(self) -> None:
        """Partially overlapping boxes give a value strictly between 0 and 1."""
        a = _bb(0, 0, 10, 10)
        b = _bb(5, 5, 15, 15)
        result = _spatial.iou(a, b)
        assert 0.0 < result < 1.0


@pytest.mark.unit
class TestFrameZone:
    """Tests for frame_zone()."""

    def test_top_left(self) -> None:
        """A box centered near the top-left maps to 'top-left'."""
        assert _spatial.frame_zone(_bb(0, 0, 20, 20)) == "top-left"

    def test_mid_center(self) -> None:
        """A box centered in the frame maps to 'mid-center'."""
        cx, cy = _spatial.FRAME_W / 2, _spatial.FRAME_H / 2
        assert _spatial.frame_zone(_bb(cx - 5, cy - 5, cx + 5, cy + 5)) == "mid-center"

    def test_bottom_right(self) -> None:
        """A box centered near the bottom-right maps to 'bottom-right'."""
        w, h = _spatial.FRAME_W, _spatial.FRAME_H
        assert _spatial.frame_zone(_bb(w - 20, h - 20, w, h)) == "bottom-right"


@pytest.mark.unit
class TestDepth:
    """Tests for depth()."""

    def test_foreground_for_large_box(self) -> None:
        """A box covering most of the frame is foreground."""
        w, h = _spatial.FRAME_W, _spatial.FRAME_H
        assert _spatial.depth(_bb(0, 0, w, h)) == "foreground"

    def test_midground_for_medium_box(self) -> None:
        """A box covering a moderate fraction of the frame is midground."""
        assert _spatial.depth(_bb(0, 0, 200, 200)) == "midground"

    def test_background_for_small_box(self) -> None:
        """A tiny box is background."""
        assert _spatial.depth(_bb(0, 0, 5, 5)) == "background"


@pytest.mark.unit
class TestIsHorizontal:
    """Tests for is_horizontal()."""

    def test_wider_than_tall_is_horizontal(self) -> None:
        """A box wider than it is tall is horizontal."""
        assert _spatial.is_horizontal(_bb(0, 0, 100, 10)) is True

    def test_taller_than_wide_is_not_horizontal(self) -> None:
        """A box taller than it is wide is not horizontal."""
        assert _spatial.is_horizontal(_bb(0, 0, 10, 100)) is False
