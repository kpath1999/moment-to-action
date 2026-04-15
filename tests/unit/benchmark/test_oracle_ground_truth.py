"""Unit tests for oracle ground-truth persistence and merging behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from moment_to_action.benchmark._oracle_ground_truth import (
    OracleBox,
    OracleClassification,
    OracleDetection,
    OracleGroundTruth,
    OracleStore,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
def test_oracle_store_path_for_dataset() -> None:
    """Dataset-specific oracle files should map to expected filenames."""
    assert OracleStore.path_for("project") == OracleStore.DEFAULT_PATH
    assert OracleStore.path_for("coco_val2017").name == "oracle_coco_val2017.json"


@pytest.mark.unit
def test_oracle_store_merge_save(tmp_path: Path) -> None:
    """Merging saves should preserve records by image name across writes."""
    path = tmp_path / "oracle.json"
    store = OracleStore(path=path)

    first = OracleGroundTruth(
        detections=[
            OracleDetection(
                image_name="a.jpg",
                boxes=[OracleBox(0.0, 0.0, 10.0, 10.0, "person", 0.9)],
            )
        ],
        classifications=[
            OracleClassification(image_name="a.jpg", top_label="a person", scores={"a person": 0.8})
        ],
        text_queries=["person"],
        text_prompts=["a person"],
        hardware_target="x86_64",
        recorded_at="2026-01-01T00:00:00+00:00",
        dataset_name="project",
    )
    store.save(first)

    second = OracleGroundTruth(
        detections=[
            OracleDetection(
                image_name="b.jpg",
                boxes=[OracleBox(1.0, 1.0, 6.0, 6.0, "car", 0.7)],
            )
        ],
        classifications=[
            OracleClassification(image_name="b.jpg", top_label="a car", scores={"a car": 0.9})
        ],
        text_queries=["car"],
        text_prompts=["a car"],
        hardware_target="x86_64",
        recorded_at="2026-01-02T00:00:00+00:00",
        dataset_name="project",
    )
    store.save(second, merge=True)

    loaded = store.load()
    assert loaded is not None
    assert sorted(item.image_name for item in loaded.detections) == ["a.jpg", "b.jpg"]
    assert sorted(item.image_name for item in loaded.classifications) == ["a.jpg", "b.jpg"]
    assert loaded.text_queries == ["car"]
    assert loaded.text_prompts == ["a car"]
