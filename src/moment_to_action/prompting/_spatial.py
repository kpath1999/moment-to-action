"""Spatial helpers — derive natural-language context from raw bounding box output.

Pure functions with no I/O, used by prompt builders to turn detector bounding
boxes into descriptive text (zone, depth, orientation, overlap) without
inventing any language that could not be computed from the boxes themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moment_to_action.models.image.detection._types import BoundingBox

# Standard frame dimensions assumed for bbox context derivation.
FRAME_W = 640
FRAME_H = 480

# Thresholds for spatial context derivation.
DEPTH_FG_THRESH = 0.25
"""Bbox area fraction above which a detection is considered foreground."""

DEPTH_MG_THRESH = 0.08
"""Bbox area fraction above which a detection is considered midground (else background)."""

OVERLAP_THRESH = 0.05
"""IoU above this value is described as "overlapping"."""

MIN_PAIR = 2
"""Minimum number of detections of a class required to compute pairwise IoU."""


def area(b: BoundingBox) -> float:
    """Compute bounding box area in pixels.

    Args:
        b: Bounding box.

    Returns:
        Area in pixels.
    """
    return (b.x2 - b.x1) * (b.y2 - b.y1)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Compute intersection-over-union between two bounding boxes.

    Args:
        a: First bounding box.
        b: Second bounding box.

    Returns:
        IoU in [0, 1].
    """
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def frame_zone(b: BoundingBox) -> str:
    """Return a natural-language frame zone for a bounding box centroid.

    Args:
        b: Bounding box.

    Returns:
        String like "bottom-left", "mid-center", etc.
    """
    cx = (b.x1 + b.x2) / 2
    cy = (b.y1 + b.y2) / 2
    h = "left" if cx < FRAME_W / 3 else ("right" if cx > 2 * FRAME_W / 3 else "center")
    v = "top" if cy < FRAME_H / 3 else ("bottom" if cy > 2 * FRAME_H / 3 else "mid")
    return f"{v}-{h}"


def depth(b: BoundingBox) -> str:
    """Return foreground/midground/background based on bbox area fraction.

    Args:
        b: Bounding box.

    Returns:
        "foreground", "midground", or "background".
    """
    frac = area(b) / (FRAME_W * FRAME_H)
    if frac > DEPTH_FG_THRESH:
        return "foreground"
    if frac > DEPTH_MG_THRESH:
        return "midground"
    return "background"


def is_horizontal(b: BoundingBox) -> bool:
    """Return True when the bounding box is wider than it is tall.

    Args:
        b: Bounding box.

    Returns:
        True if width > height.
    """
    return (b.x2 - b.x1) > (b.y2 - b.y1)
