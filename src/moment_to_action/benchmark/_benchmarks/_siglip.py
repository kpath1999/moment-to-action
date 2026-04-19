"""SigLIP oracle benchmark.

Runs SigLIP on the project sample images and records per-image classification
scores as ground truth in the OracleStore.  The recorded scores are consumed
by MobileCLIPBenchmark._evaluate_accuracy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from moment_to_action.benchmark._benchmarks._base import ModelBenchmark
from moment_to_action.benchmark._oracle_ground_truth import (
    OracleClassification,
    OracleGroundTruth,
    OracleStore,
)
from moment_to_action.hardware._platforms._detection import detect_platform
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from moment_to_action.benchmark._datasets._coco_dataset import CocoDataset
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager

logger = logging.getLogger(__name__)

# Default text prompts used when none are supplied.
_DEFAULT_PROMPTS: list[str] = [
    "a person",
    "a weapon",
    "a happy person",
    "a pedestrian walking",
    "smoke or fire",
]


@attrs.frozen
class _SigLIPHandle:
    """Internal handle carrying the SigLIP model and processor."""

    model: object
    processor: object
    device: str


class SigLIPBenchmark(ModelBenchmark):
    """Oracle benchmark for SigLIP.

    Records ground truth classification scores for the project sample images
    via SigLIP and persists them to the OracleStore.  The recorded scores are
    consumed by MobileCLIPBenchmark._evaluate_accuracy.

    Parameters
    ----------
    text_prompts:
        Labels to score (e.g. ``["a person", "a weapon"]``).  Defaults to
        ``_DEFAULT_PROMPTS`` when omitted.
    sample_images:
        Paths to images to run inference on.  Defaults to every ``*.jpg``
        inside the project ``images/`` directory when omitted.
    oracle_store:
        OracleStore instance used to persist results.  A default store is
        created when omitted.
    """

    def __init__(
        self,
        text_prompts: list[str] | None = None,
        sample_images: list[Path] | None = None,
        oracle_store: OracleStore | None = None,
        coco_dataset: CocoDataset | None = None,
    ) -> None:
        self._coco_dataset = coco_dataset
        self._text_prompts = text_prompts or _DEFAULT_PROMPTS
        self._sample_images = sample_images or (
            coco_dataset.images() if coco_dataset is not None else _default_sample_images()
        )
        self._oracle_store = oracle_store or OracleStore(
            dataset_name=(coco_dataset.dataset_name if coco_dataset is not None else "project")
        )

    @property
    def model_id(self) -> ModelID:
        return ModelID.SIGLIP_SO400M

    # ── ModelBenchmark protocol ───────────────────────────────────────────────

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        policy = backend.resolve_torch_policy("auto")
        model_path = manager.get_path(self.model_id)
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(policy.device)
        model.train(mode=False)
        logger.info("SigLIPBenchmark: loaded from %s on %s", model_path, policy.device)
        return _SigLIPHandle(model=model, processor=processor, device=str(policy.device))

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        del batch_size
        h = self._cast_handle(handle)
        image = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        inputs = h.processor(  # type: ignore[operator]
            text=self._text_prompts,
            images=image,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.to(h.device) for k, v in inputs.items()}

    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None:
        del backend
        h = self._cast_handle(handle)
        if not isinstance(inputs, dict):
            msg = "SigLIPBenchmark expects dict inputs"
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
        """Run SigLIP on all sample images and persist classification ground truth."""
        del backend, manager
        h = self._cast_handle(handle)
        oracle_classifications: list[OracleClassification] = []

        for img_path in self._sample_images:
            image = Image.open(img_path).convert("RGB")
            prompts = self._text_prompts
            if self._coco_dataset is not None:
                prompts = self._coco_dataset.captions(img_path.name)
            if not prompts:
                continue

            inputs = h.processor(  # type: ignore[operator]
                text=prompts,
                images=image,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(h.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = h.model(**inputs)  # type: ignore[operator]

            logits = outputs.logits_per_image  # type: ignore[union-attr]
            probs: list[float] = torch.sigmoid(logits).cpu().numpy().tolist()[0]
            best_idx = int(np.argmax(probs))

            oracle_classifications.append(
                OracleClassification(
                    image_name=img_path.name,
                    top_label=prompts[best_idx],
                    scores={p: float(s) for p, s in zip(prompts, probs, strict=False)},
                )
            )
            logger.info(
                "SigLIPBenchmark: %s → '%s' (%.3f)",
                img_path.name,
                prompts[best_idx],
                probs[best_idx],
            )

        # Merge with any existing oracle ground truth (preserve detections).
        existing = self._oracle_store.load()
        gt = OracleGroundTruth(
            detections=existing.detections if existing is not None else [],
            classifications=oracle_classifications,
            text_queries=existing.text_queries if existing is not None else [],
            text_prompts=self._text_prompts if self._coco_dataset is None else [],
            hardware_target=detect_platform().name.lower(),
            recorded_at=OracleStore.now_iso(),
            dataset_name=(
                self._coco_dataset.dataset_name if self._coco_dataset is not None else "project"
            ),
        )
        self._oracle_store.save(gt, merge=True)
        logger.info("SigLIPBenchmark: ground truth saved to %s", self._oracle_store.path)
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cast_handle(handle: object) -> _SigLIPHandle:
        if not isinstance(handle, _SigLIPHandle):
            msg = f"Expected _SigLIPHandle, got {type(handle).__name__}"
            raise TypeError(msg)
        return handle


def _default_sample_images() -> list[Path]:
    """Locate the project images/ directory relative to this file."""
    candidate = Path(__file__).parents[4] / "images"
    if candidate.is_dir():
        return sorted(candidate.glob("*.jpg"))
    return []
