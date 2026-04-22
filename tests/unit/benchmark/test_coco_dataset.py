"""Unit tests for COCO dataset sampling and caption loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

import pytest

from moment_to_action.benchmark._datasets._coco_dataset import CocoDataset
from moment_to_action.benchmark._oracle_ground_truth import OracleBox


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


@pytest.mark.unit
def test_coco_dataset_instance_detections_from_native_annotations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Native COCO instance annotations should map to OracleDetection records."""
    cache_dir = tmp_path / "coco"
    images_dir = cache_dir / "val2017"
    annotations_dir = cache_dir / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    for idx in range(1, 4):
        (images_dir / f"{idx:012d}.jpg").write_bytes(b"jpg")

    captions_payload = {
        "images": [{"id": idx, "file_name": f"{idx:012d}.jpg"} for idx in range(1, 4)],
        "annotations": [{"image_id": idx, "caption": f"caption-{idx}"} for idx in range(1, 4)],
    }
    (annotations_dir / "captions_val2017.json").write_text(
        json.dumps(captions_payload), encoding="utf-8"
    )

    instances_payload = {
        "images": [{"id": idx, "file_name": f"{idx:012d}.jpg"} for idx in range(1, 4)],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "bicycle"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0]},
            {"image_id": 2, "category_id": 2, "bbox": [1.0, 2.0, 3.0, 4.0]},
        ],
    }
    (annotations_dir / "instances_val2017.json").write_text(
        json.dumps(instances_payload), encoding="utf-8"
    )

    def _skip_download(this: CocoDataset) -> None:
        del this

    monkeypatch.setattr(CocoDataset, "_ensure_dataset_files", _skip_download)

    dataset = CocoDataset(n_images=3, cache_dir=cache_dir, seed=9)
    detections = dataset.instance_detections()

    assert len(detections) == 3
    by_name = {item.image_name: item for item in detections}
    assert by_name["000000000001.jpg"].boxes == [
        OracleBox(
            x1=10.0,
            y1=20.0,
            x2=40.0,
            y2=60.0,
            label="person",
            confidence=1.0,
        )
    ]
    assert by_name["000000000002.jpg"].boxes[0].label == "bicycle"
    assert by_name["000000000003.jpg"].boxes == []


@pytest.mark.unit
def test_coco_download_file_rejects_non_https(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only HTTPS"):
        CocoDataset._download_file("http://example.com/file.zip", tmp_path / "file.zip")


@pytest.mark.unit
def test_coco_download_file_skips_existing(tmp_path: Path) -> None:
    destination = tmp_path / "existing.zip"
    destination.write_bytes(b"ok")
    CocoDataset._download_file("https://example.com/file.zip", destination)
    assert destination.read_bytes() == b"ok"


@pytest.mark.unit
def test_coco_ensure_dataset_files_downloads_and_extracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = object.__new__(CocoDataset)
    dataset.cache_dir = tmp_path

    extracted: list[str] = []

    class _FakeZip:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb

        def extractall(self, path: Path) -> None:
            (path / "val2017").mkdir(parents=True, exist_ok=True)
            (path / "val2017" / "000000000001.jpg").write_bytes(b"jpg")

        def extract(self, member: str, path: Path) -> None:
            extracted.append(member)
            out = path / member
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        CocoDataset,
        "_download_file",
        lambda _self, _url, dest: dest.write_bytes(b"zip"),
    )
    monkeypatch.setattr("zipfile.ZipFile", lambda _path, **_kwargs: _FakeZip())

    CocoDataset._ensure_dataset_files(dataset)

    assert (tmp_path / "val2017").is_dir()
    assert "annotations/captions_val2017.json" in extracted
    assert "annotations/instances_val2017.json" in extracted


@pytest.mark.unit
def test_coco_load_captions_and_instances_skip_invalid(tmp_path: Path) -> None:
    dataset = object.__new__(CocoDataset)
    dataset.cache_dir = tmp_path

    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(parents=True)

    (ann_dir / "captions_val2017.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "1.jpg"}],
                "annotations": [
                    {"image_id": 1, "caption": "ok"},
                    {"image_id": 2, "caption": "skip"},
                ],
            }
        ),
        encoding="utf-8",
    )
    captions = CocoDataset._load_captions_map(dataset)
    assert captions == {"1.jpg": ["ok"]}

    (ann_dir / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "1.jpg"}],
                "categories": [{"id": 3, "name": "cat"}],
                "annotations": [
                    {"image_id": 1, "category_id": 3, "bbox": [1, 2, 3, 4]},
                    {"image_id": 1, "category_id": 3, "bbox": [1, 2, -1, 4]},
                    {"image_id": 99, "category_id": 3, "bbox": [1, 2, 3, 4]},
                    {"image_id": 1, "category_id": 3, "bbox": [1, 2, 3]},
                ],
            }
        ),
        encoding="utf-8",
    )
    instances = CocoDataset._load_instances_map(dataset)
    assert len(instances["1.jpg"]) == 1
    assert instances["1.jpg"][0].label == "cat"


@pytest.mark.unit
def test_coco_select_subset_images_raises_for_empty_or_no_eligible(tmp_path: Path) -> None:
    dataset = object.__new__(CocoDataset)
    dataset.cache_dir = tmp_path
    dataset.n_images = 1
    dataset.seed = 1
    dataset._captions_by_image = {}

    (tmp_path / "val2017").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="image directory is empty"):
        CocoDataset._select_subset_images(dataset)

    (tmp_path / "val2017" / "000000000001.jpg").write_bytes(b"jpg")
    with pytest.raises(RuntimeError, match="No COCO images with captions"):
        CocoDataset._select_subset_images(dataset)


@pytest.mark.unit
def test_coco_items_all_captions_and_dataset_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "coco"
    images_dir = cache_dir / "val2017"
    annotations_dir = cache_dir / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    (images_dir / "000000000001.jpg").write_bytes(b"jpg")

    (annotations_dir / "captions_val2017.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "000000000001.jpg"}],
                "annotations": [{"image_id": 1, "caption": "cap"}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(CocoDataset, "_ensure_dataset_files", lambda _self: None)
    dataset = CocoDataset(n_images=1, cache_dir=cache_dir)
    assert dataset.items() == dataset.images()
    assert dataset.all_captions() == {"000000000001.jpg": ["cap"]}
    assert dataset.dataset_name == "coco_val2017"


@pytest.mark.unit
def test_coco_download_file_https_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "file.bin"

    class _Resp:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb

        @staticmethod
        def read() -> bytes:
            return b"payload"

    monkeypatch.setattr("urllib.request.urlopen", lambda _url, **_kwargs: _Resp())

    CocoDataset._download_file("https://example.com/file.bin", destination)
    assert destination.read_bytes() == b"payload"
