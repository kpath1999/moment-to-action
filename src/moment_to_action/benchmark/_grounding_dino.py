"""Grounding DINO oracle benchmark.

Runs Grounding DINO on the sample images bundled with the project and records
its bounding-box predictions as ground truth in the OracleStore.  This oracle
is later used by YOLOBenchmark to compute detection accuracy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from moment_to_action.benchmark._base import ModelBenchmark
from moment_to_action.benchmark._oracle_ground_truth import (
    OracleBox,
    OracleDetection,
    OracleGroundTruth,
    OracleStore,
)
from moment_to_action.hardware._platforms._detection import detect_platform
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from moment_to_action.benchmark._coco_dataset import CocoDataset
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager

logger = logging.getLogger(__name__)

# Default text queries used when none are supplied.
_DEFAULT_QUERIES: list[str] = ["person", "weapon", "phone", "car", "smoke"]


@attrs.frozen
class _DinoHandle:
    """Internal handle carrying the GroundingDINO model and processor."""

    model: object
    processor: object
    device: str


class GroundingDINOBenchmark(ModelBenchmark):
    """Oracle benchmark for Grounding DINO.

    Records ground truth bounding boxes for the project sample images via
    GroundingDINO and persists them to the OracleStore.  The recorded
    detections are consumed by YOLOBenchmark._evaluate_accuracy.

    Parameters
    ----------
    text_queries:
        Object classes to detect (e.g. ``["person", "weapon"]``).  Defaults
        to ``_DEFAULT_QUERIES`` when omitted.
    sample_images:
        Paths to images to run inference on.  Defaults to every ``*.jpg``
        inside the project ``images/`` directory when omitted.
    oracle_store:
        OracleStore instance used to persist results.  A default store is
        created when omitted.
    box_threshold:
        Minimum score for GroundingDINO to emit a box (default 0.3).
    text_threshold:
        Token-matching threshold for GroundingDINO (default 0.3).
    """

    def __init__(
        self,
        text_queries: list[str] | None = None,
        sample_images: list[Path] | None = None,
        oracle_store: OracleStore | None = None,
        coco_dataset: CocoDataset | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
    ) -> None:
        self._coco_dataset = coco_dataset

        if coco_dataset is not None and text_queries is None:
            # Keep labels aligned with YOLO's closed-vocabulary COCO classes.
            from moment_to_action.stages.video._yolo import YOLOStage

            self._text_queries = list(YOLOStage.COCO_LABELS)
        else:
            self._text_queries = text_queries or _DEFAULT_QUERIES

        self._sample_images = sample_images or (
            coco_dataset.images() if coco_dataset is not None else _default_sample_images()
        )
        self._oracle_store = oracle_store or OracleStore(
            dataset_name=(coco_dataset.dataset_name if coco_dataset is not None else "project")
        )
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold

    @property
    def model_id(self) -> ModelID:
        return ModelID.GROUNDING_DINO_BASE

    # ── ModelBenchmark protocol ───────────────────────────────────────────────

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        policy = backend.resolve_torch_policy("auto")
        model_path = manager.get_path(self.model_id)
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(policy.device)
        model.train(mode=False)
        logger.info("GroundingDINOBenchmark: loaded from %s on %s", model_path, policy.device)
        return _DinoHandle(model=model, processor=processor, device=str(policy.device))

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        del batch_size
        h = self._cast_handle(handle)
        image = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        text = ". ".join(self._text_queries) + "."
        inputs = h.processor(images=image, text=text, return_tensors="pt")  # type: ignore[operator]
        return {k: v.to(h.device) for k, v in inputs.items()}

    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None:
        del backend
        h = self._cast_handle(handle)
        if not isinstance(inputs, dict):
            msg = "GroundingDINOBenchmark expects dict inputs"
            raise TypeError(msg)
        with torch.inference_mode():
            h.model(**inputs)  # type: ignore[operator]

    # ── Oracle-specific: record ground truth after profiling ─────────────────

    def _evaluate_accuracy(
        self,
        handle: object,
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> float | None:
        """Run GroundingDINO on all sample images and persist ground truth."""
        del backend, manager
        h = self._cast_handle(handle)
        text = ". ".join(self._text_queries) + "."
        oracle_detections: list[OracleDetection] = []

        for img_path in self._sample_images:
            image = Image.open(img_path).convert("RGB")
            inputs = h.processor(images=image, text=text, return_tensors="pt")  # type: ignore[operator]
            inputs = {k: v.to(h.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = h.model(**inputs)  # type: ignore[operator]

            results = h.processor.post_process_grounded_object_detection(  # type: ignore[union-attr,attr-defined]
                outputs,
                inputs["input_ids"],
                threshold=self._box_threshold,
                text_threshold=self._text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]

            boxes = [
                OracleBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    label=str(label),
                    confidence=float(score),
                )
                for box, score, label in zip(
                    results["boxes"],
                    results["scores"],
                    results["labels"],
                    strict=False,
                )
            ]
            oracle_detections.append(OracleDetection(image_name=img_path.name, boxes=boxes))
            logger.info("GroundingDINOBenchmark: %s → %d boxes", img_path.name, len(boxes))

        # Merge with any existing oracle ground truth (preserve classifications).
        existing = self._oracle_store.load()
        gt = OracleGroundTruth(
            detections=oracle_detections,
            classifications=existing.classifications if existing is not None else [],
            text_queries=self._text_queries,
            text_prompts=existing.text_prompts if existing is not None else [],
            hardware_target=detect_platform().name.lower(),
            recorded_at=OracleStore.now_iso(),
            dataset_name=(
                self._coco_dataset.dataset_name if self._coco_dataset is not None else "project"
            ),
        )
        self._oracle_store.save(gt, merge=True)
        logger.info("GroundingDINOBenchmark: ground truth saved to %s", self._oracle_store.path)
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cast_handle(handle: object) -> _DinoHandle:
        if not isinstance(handle, _DinoHandle):
            msg = f"Expected _DinoHandle, got {type(handle).__name__}"
            raise TypeError(msg)
        return handle


def _default_sample_images() -> list[Path]:
    """Locate the project images/ directory relative to this file."""
    candidate = Path(__file__).parents[4] / "images"
    if candidate.is_dir():
        return sorted(candidate.glob("*.jpg"))
    return []
