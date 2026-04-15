"""Unit tests for COCO dataset sampling and caption loading."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from moment_to_action.benchmark._coco_dataset import CocoDataset

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
def test_coco_dataset_selects_deterministic_subset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Subset selection should be deterministic for a fixed random seed."""
    cache_dir = tmp_path / "coco"
    images_dir = cache_dir / "val2017"
    annotations_dir = cache_dir / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    for idx in range(1, 6):
        (images_dir / f"{idx:012d}.jpg").write_bytes(b"jpg")

    captions_payload = {
        "images": [{"id": idx, "file_name": f"{idx:012d}.jpg"} for idx in range(1, 6)],
        "annotations": [{"image_id": idx, "caption": f"caption-{idx}-a"} for idx in range(1, 6)]
        + [{"image_id": idx, "caption": f"caption-{idx}-b"} for idx in range(1, 6)],
    }
    (annotations_dir / "captions_val2017.json").write_text(
        json.dumps(captions_payload), encoding="utf-8"
    )

    def _skip_download(this: CocoDataset) -> None:
        del this

    monkeypatch.setattr(CocoDataset, "_ensure_dataset_files", _skip_download)

    dataset_a = CocoDataset(n_images=3, cache_dir=cache_dir, seed=7)
    dataset_b = CocoDataset(n_images=3, cache_dir=cache_dir, seed=7)

    assert [item.name for item in dataset_a.images()] == [item.name for item in dataset_b.images()]
    image_name = dataset_a.images()[0].name
    assert dataset_a.captions(image_name) == [
        f"caption-{int(image_name[:-4])}-a",
        f"caption-{int(image_name[:-4])}-b",
    ]


@pytest.mark.unit
def test_coco_dataset_rejects_non_positive_n_images() -> None:
    """Constructor should reject non-positive subset sizes."""
    with pytest.raises(ValueError, match="n_images must be greater than 0"):
        CocoDataset(n_images=0)
