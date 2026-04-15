"""Oracle ground truth store for accuracy benchmarking.

On MPS hardware the oracle models (GroundingDINO, SigLIP) record their
predictions for a fixed set of sample images.  On QCS hardware the edge
models (YOLOv8, MobileCLIP) load the stored ground truth and compare their
own predictions to compute accuracy metrics.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003
from typing import ClassVar

import attrs
import platformdirs


@attrs.frozen
class OracleBox:
    """A single bounding-box prediction from GroundingDINO."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float

    def iou(self, other: OracleBox) -> float:
        """Compute Intersection-over-Union against another box."""
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
    """GroundingDINO ground truth for a single image."""

    image_name: str
    boxes: list[OracleBox]


@attrs.frozen
class OracleClassification:
    """SigLIP ground truth for a single image."""

    image_name: str
    top_label: str
    scores: dict[str, float]


@attrs.frozen
class OracleGroundTruth:
    """Full oracle ground truth recorded on reference hardware."""

    detections: list[OracleDetection]
    classifications: list[OracleClassification]
    text_queries: list[str]
    text_prompts: list[str]
    hardware_target: str
    recorded_at: str
    dataset_name: str = "project"


class OracleStore:
    """Persist and load oracle ground truth records."""

    DEFAULT_PATH: ClassVar[Path] = (
        platformdirs.user_cache_path("moment_to_action", "GATech") / "oracle_ground_truth.json"
    )

    def __init__(self, path: Path | None = None, dataset_name: str = "project") -> None:
        self._path = path or self.path_for(dataset_name)

    @property
    def path(self) -> Path:
        """Default path used by this store instance."""
        return self._path

    @classmethod
    def path_for(cls, dataset_name: str) -> Path:
        """Return the default ground-truth path for a dataset."""
        if dataset_name == "project":
            return cls.DEFAULT_PATH
        base_dir = platformdirs.user_cache_path("moment_to_action", "GATech")
        return base_dir / f"oracle_{dataset_name}.json"

    def save(self, gt: OracleGroundTruth, path: Path | None = None, *, merge: bool = False) -> None:
        """Serialize ground truth to JSON."""
        out_path = path or self._path

        if merge:
            existing = self.load(path=out_path)
            if existing is not None:
                gt = self._merge(existing=existing, current=gt)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, object] = {
            "text_queries": gt.text_queries,
            "text_prompts": gt.text_prompts,
            "hardware_target": gt.hardware_target,
            "recorded_at": gt.recorded_at,
            "dataset_name": gt.dataset_name,
            "detections": [
                {
                    "image_name": d.image_name,
                    "boxes": [
                        {
                            "x1": b.x1,
                            "y1": b.y1,
                            "x2": b.x2,
                            "y2": b.y2,
                            "label": b.label,
                            "confidence": b.confidence,
                        }
                        for b in d.boxes
                    ],
                }
                for d in gt.detections
            ],
            "classifications": [
                {
                    "image_name": c.image_name,
                    "top_label": c.top_label,
                    "scores": c.scores,
                }
                for c in gt.classifications
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path | None = None) -> OracleGroundTruth | None:
        """Deserialize ground truth from JSON, returning None if not found."""
        in_path = path or self._path
        if not in_path.exists():
            return None

        raw = json.loads(in_path.read_text(encoding="utf-8"))

        detections = [
            OracleDetection(
                image_name=d["image_name"],
                boxes=[OracleBox(**b) for b in d["boxes"]],
            )
            for d in raw.get("detections", [])
        ]
        classifications = [
            OracleClassification(
                image_name=c["image_name"],
                top_label=c["top_label"],
                scores=c["scores"],
            )
            for c in raw.get("classifications", [])
        ]
        return OracleGroundTruth(
            detections=detections,
            classifications=classifications,
            text_queries=raw.get("text_queries", []),
            text_prompts=raw.get("text_prompts", []),
            hardware_target=raw.get("hardware_target", ""),
            recorded_at=raw.get("recorded_at", ""),
            dataset_name=raw.get("dataset_name", "project"),
        )

    @staticmethod
    def _merge(existing: OracleGroundTruth, current: OracleGroundTruth) -> OracleGroundTruth:
        existing_detections = {det.image_name: det for det in existing.detections}
        for det in current.detections:
            existing_detections[det.image_name] = det

        existing_classifications = {cls.image_name: cls for cls in existing.classifications}
        for cls in current.classifications:
            existing_classifications[cls.image_name] = cls

        return OracleGroundTruth(
            detections=list(existing_detections.values()),
            classifications=list(existing_classifications.values()),
            text_queries=current.text_queries or existing.text_queries,
            text_prompts=current.text_prompts or existing.text_prompts,
            hardware_target=current.hardware_target or existing.hardware_target,
            recorded_at=current.recorded_at,
            dataset_name=current.dataset_name or existing.dataset_name,
        )

    @staticmethod
    def now_iso() -> str:
        """Current UTC time as ISO-8601 string."""
        return datetime.now(tz=UTC).isoformat()
