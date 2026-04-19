from __future__ import annotations

import json
import random
import ssl
import urllib.request
import zipfile
from pathlib import Path

import attrs
import platformdirs

from moment_to_action.benchmark._datasets._base import BaseDataset
from moment_to_action.benchmark._oracle_ground_truth import OracleBox, OracleDetection

_VAL2017_URL = "https://images.cocodataset.org/zips/val2017.zip"
_ANNOTATIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
_VAL2017_ZIP = "val2017.zip"
_ANNOTATIONS_ZIP = "annotations_trainval2017.zip"
_CAPTIONS_MEMBER = "annotations/captions_val2017.json"
_INSTANCES_MEMBER = "annotations/instances_val2017.json"
_COCO_BBOX_LEN = 4


def _default_cache_dir() -> Path:
    return platformdirs.user_cache_path("moment_to_action", "GATech") / "coco_val2017"


@attrs.define
class CocoDataset(BaseDataset[Path]):
    """COCO val2017 image and caption loader with local cache.

    Downloads COCO val2017 images and the captions annotation file on first use,
    then reuses the local cache for subsequent runs.
    """

    n_images: int = 500
    cache_dir: Path = attrs.Factory(_default_cache_dir)
    seed: int = 42
    _subset_images: list[Path] = attrs.field(factory=list, init=False)
    _captions_by_image: dict[str, list[str]] = attrs.field(factory=dict, init=False)
    _instances_by_image: dict[str, list[OracleBox]] | None = attrs.field(default=None, init=False)

    def __attrs_post_init__(self) -> None:
        if self.n_images <= 0:
            msg = "n_images must be greater than 0"
            raise ValueError(msg)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_dataset_files()
        self._captions_by_image = self._load_captions_map()
        self._subset_images = self._select_subset_images()

    def images(self) -> list[Path]:
        """Return the cached subset of COCO image paths."""
        return list(self._subset_images)

    def items(self) -> list[Path]:
        """Return sampled image paths for evaluation."""
        return self.images()

    def captions(self, image_name: str) -> list[str]:
        """Return COCO captions for an image file name."""
        return list(self._captions_by_image.get(image_name, []))

    def all_captions(self) -> dict[str, list[str]]:
        """Return captions for all selected images keyed by image file name."""
        return {img.name: self.captions(img.name) for img in self._subset_images}

    def instance_detections(self) -> list[OracleDetection]:
        """Return native COCO instance detections for the selected image subset."""
        if self._instances_by_image is None:
            self._instances_by_image = self._load_instances_map()

        return [
            OracleDetection(
                image_name=image_path.name,
                boxes=list(self._instances_by_image.get(image_path.name, [])),
            )
            for image_path in self._subset_images
        ]

    @property
    def dataset_name(self) -> str:
        """Dataset identifier used for oracle store file naming."""
        return "coco_val2017"

    def _ensure_dataset_files(self) -> None:
        val_zip = self.cache_dir / _VAL2017_ZIP
        ann_zip = self.cache_dir / _ANNOTATIONS_ZIP
        images_dir = self.cache_dir / "val2017"
        captions_json = self.cache_dir / _CAPTIONS_MEMBER
        instances_json = self.cache_dir / _INSTANCES_MEMBER

        if not images_dir.is_dir() or not any(images_dir.glob("*.jpg")):
            self._download_file(_VAL2017_URL, val_zip)
            with zipfile.ZipFile(val_zip, mode="r") as archive:
                archive.extractall(self.cache_dir)

        if not captions_json.is_file() or not instances_json.is_file():
            self._download_file(_ANNOTATIONS_URL, ann_zip)
            with zipfile.ZipFile(ann_zip, mode="r") as archive:
                if not captions_json.is_file():
                    archive.extract(_CAPTIONS_MEMBER, path=self.cache_dir)
                if not instances_json.is_file():
                    archive.extract(_INSTANCES_MEMBER, path=self.cache_dir)

    @staticmethod
    def _download_file(url: str, destination: Path) -> None:
        if destination.exists():
            return

        if not url.startswith("https://"):
            msg = f"Only HTTPS download URLs are allowed: {url}"
            raise ValueError(msg)

        # images.cocodataset.org is served directly from an S3 bucket whose TLS
        # certificate is issued for s3.amazonaws.com, causing a hostname mismatch.
        # The certificate chain itself is valid (Amazon-issued).  We keep chain
        # verification (CERT_REQUIRED) but disable hostname checking for this
        # well-known public-dataset endpoint only.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_REQUIRED

        with urllib.request.urlopen(url, context=ctx) as response:  # noqa: S310
            destination.write_bytes(response.read())

    def _load_captions_map(self) -> dict[str, list[str]]:
        captions_path = self.cache_dir / _CAPTIONS_MEMBER
        raw = json.loads(captions_path.read_text(encoding="utf-8"))

        image_id_to_name: dict[int, str] = {
            int(image["id"]): str(image["file_name"]) for image in raw.get("images", [])
        }

        result: dict[str, list[str]] = {}
        for annotation in raw.get("annotations", []):
            image_id = int(annotation["image_id"])
            image_name = image_id_to_name.get(image_id)
            if image_name is None:
                continue
            result.setdefault(image_name, []).append(str(annotation["caption"]))
        return result

    def _load_instances_map(self) -> dict[str, list[OracleBox]]:
        instances_path = self.cache_dir / _INSTANCES_MEMBER
        raw = json.loads(instances_path.read_text(encoding="utf-8"))

        image_id_to_name: dict[int, str] = {
            int(image["id"]): str(image["file_name"]) for image in raw.get("images", [])
        }
        category_id_to_name: dict[int, str] = {
            int(category["id"]): str(category["name"]) for category in raw.get("categories", [])
        }

        result: dict[str, list[OracleBox]] = {}
        for annotation in raw.get("annotations", []):
            image_id = int(annotation.get("image_id", -1))
            image_name = image_id_to_name.get(image_id)
            if image_name is None:
                continue

            bbox = annotation.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != _COCO_BBOX_LEN:
                continue

            x1 = float(bbox[0])
            y1 = float(bbox[1])
            width = float(bbox[2])
            height = float(bbox[3])
            if width <= 0.0 or height <= 0.0:
                continue

            category_id = int(annotation.get("category_id", -1))
            label = category_id_to_name.get(category_id, str(category_id))
            result.setdefault(image_name, []).append(
                OracleBox(
                    x1=x1,
                    y1=y1,
                    x2=x1 + width,
                    y2=y1 + height,
                    label=label,
                    confidence=1.0,
                )
            )
        return result

    def _select_subset_images(self) -> list[Path]:
        images_dir = self.cache_dir / "val2017"
        all_images = sorted(images_dir.glob("*.jpg"))
        if not all_images:
            msg = "COCO val2017 image directory is empty after extraction"
            raise RuntimeError(msg)

        eligible = [img for img in all_images if img.name in self._captions_by_image]
        if not eligible:
            msg = "No COCO images with captions were found"
            raise RuntimeError(msg)

        sample_size = min(self.n_images, len(eligible))
        rng = random.Random(self.seed)  # noqa: S311
        selected = rng.sample(eligible, sample_size)
        return sorted(selected)
