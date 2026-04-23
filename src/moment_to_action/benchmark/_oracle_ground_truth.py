"""Detection ground-truth data types used by COCO benchmark code paths."""

from __future__ import annotations

import attrs


@attrs.frozen
class OracleBox:
    """A single detection box represented in xyxy pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float

    def iou(self, other: OracleBox) -> float:
        """Compute intersection-over-union against another box."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_self = max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)
        area_other = max(0.0, other.x2 - other.x1) * max(0.0, other.y2 - other.y1)
        union_area = area_self + area_other - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


@attrs.frozen
class OracleDetection:
    """All detections associated with a single image."""

    image_name: str
    boxes: list[OracleBox]
