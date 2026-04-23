"""SigLIP benchmark for direct COCO text-to-image retrieval evaluation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import attrs
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from moment_to_action.benchmark._benchmarks._base import ModelBenchmark
from moment_to_action.benchmark._retrieval_metrics import compute_retrieval_metrics
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.benchmark._datasets._coco_dataset import CocoDataset
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager

logger = logging.getLogger(__name__)
_MAX_RETRIEVAL_ITEMS = 64


@attrs.frozen
class _SigLIPHandle:
    """Internal handle carrying the SigLIP model and processor."""

    model: object
    processor: object
    device: str


class SigLIPBenchmark(ModelBenchmark):
    """Benchmark implementation for SigLIP text-to-image retrieval on COCO."""

    def __init__(
        self,
        coco_dataset: CocoDataset | None = None,
    ) -> None:
        self._coco_dataset = coco_dataset

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
            text=["a photo"],
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

    def _evaluate_accuracy(
        self,
        handle: object,
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> float | None:
        del backend, manager
        if self._coco_dataset is None:
            return None

        return self._evaluate_coco_accuracy(handle)

    def _evaluate_coco_accuracy(self, handle: object) -> float | None:
        """Evaluate SigLIP text-to-image retrieval on COCO using recall@1."""
        h = self._cast_handle(handle)
        dataset = self._coco_dataset
        if dataset is None:
            return None

        paired_items: list[tuple[Path, str]] = []
        for image_path in dataset.images():
            captions = dataset.captions(image_path.name)
            if not captions:
                continue
            paired_items.append((image_path, captions[0]))

        if not paired_items:
            return None

        paired_items = paired_items[: min(len(paired_items), _MAX_RETRIEVAL_ITEMS)]
        prompt_bank = [caption for _, caption in paired_items]

        predicted_scores: dict[str, list[float]] = {}
        target_scores: dict[str, list[float]] = {}

        for image_idx, (img_path, _caption) in enumerate(paired_items):
            image = Image.open(img_path).convert("RGB")
            inputs = h.processor(  # type: ignore[operator]
                text=prompt_bank,
                images=image,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(h.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = h.model(**inputs)  # type: ignore[operator]

            logits = outputs.logits_per_image  # type: ignore[union-attr]
            probs: list[float] = torch.sigmoid(logits).cpu().numpy().tolist()[0]  # type: ignore[index]

            image_name = img_path.name
            predicted_scores[image_name] = [float(score) for score in probs]
            target = [0.0] * len(prompt_bank)
            target[image_idx] = 1.0
            target_scores[image_name] = target

        metrics = compute_retrieval_metrics(
            predictions=predicted_scores,
            ground_truth=target_scores,
        )
        self._set_accuracy_details({"recall_at_1": metrics.recall_at_1})
        logger.info("SigLIPBenchmark COCO retrieval: R@1=%.3f", metrics.recall_at_1)
        return metrics.recall_at_1

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cast_handle(handle: object) -> _SigLIPHandle:
        if not isinstance(handle, _SigLIPHandle):
            msg = f"Expected _SigLIPHandle, got {type(handle).__name__}"
            raise TypeError(msg)
        return handle
