from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import open_clip
from PIL import Image

from moment_to_action.benchmark._benchmarks._base import ModelBenchmark
from moment_to_action.benchmark._retrieval_metrics import compute_retrieval_metrics
from moment_to_action.models import ModelID
from moment_to_action.utils.ml import cosine_similarity

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.benchmark._datasets._coco_dataset import CocoDataset
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager

logger = logging.getLogger(__name__)
_MAX_RETRIEVAL_ITEMS = 64


class MobileCLIPBenchmark(ModelBenchmark):
    """Benchmark implementation for MobileCLIP-S2."""

    def __init__(
        self,
        *,
        coco_dataset: CocoDataset | None = None,
    ) -> None:
        super().__init__()
        self._coco_dataset = coco_dataset

    @property
    def model_id(self) -> ModelID:
        return ModelID.MOBILECLIP_S2

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        return backend.load_model(manager.get_path(self.model_id))

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        del handle
        return {
            "serving_default_args_0:0": np.zeros((batch_size, 3, 256, 256), dtype=np.float32),
            "serving_default_args_1:0": np.zeros((batch_size, 77), dtype=np.int64),
        }

    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None:
        if not isinstance(inputs, dict):
            msg = "MobileCLIP benchmark expects dict inputs"
            raise TypeError(msg)
        backend.run(handle, inputs)

    def _evaluate_accuracy(
        self,
        handle: object,
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> float | None:
        del manager
        if self._coco_dataset is None:
            return None
        return self._evaluate_coco_accuracy(handle=handle, backend=backend)

    def _evaluate_coco_accuracy(self, handle: object, backend: ComputeBackend) -> float | None:
        """Evaluate MobileCLIP text-to-image retrieval on COCO using recall@1."""
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
            logger.debug("MobileCLIPBenchmark: no captioned COCO items found -- skipping.")
            return None

        paired_items = paired_items[: min(len(paired_items), _MAX_RETRIEVAL_ITEMS)]
        prompt_bank = [caption for _, caption in paired_items]

        tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
        predicted_scores: dict[str, list[float]] = {}
        target_scores: dict[str, list[float]] = {}

        for image_idx, (img_path, _caption) in enumerate(paired_items):
            tokens: np.ndarray = np.asarray(tokenizer(prompt_bank)).astype(np.int64)
            img_tensor = _load_mobileclip_tensor(img_path)
            scores: list[float] = []

            for token_row in tokens:
                token_tensor = token_row[np.newaxis].astype(np.int64)
                outputs = backend.run(
                    handle,
                    {
                        "serving_default_args_0:0": img_tensor,
                        "serving_default_args_1:0": token_tensor,
                    },
                )
                image_emb = outputs[1][0]
                text_emb = outputs[0][0]
                scores.append(cosine_similarity(image_emb, text_emb))

            image_name = img_path.name
            predicted_scores[image_name] = scores
            target = [0.0] * len(prompt_bank)
            target[image_idx] = 1.0
            target_scores[image_name] = target

        if not predicted_scores:
            return None

        metrics = compute_retrieval_metrics(
            predictions=predicted_scores,
            ground_truth=target_scores,
        )
        self._set_accuracy_details({"recall_at_1": metrics.recall_at_1})
        logger.info(
            "MobileCLIPBenchmark COCO retrieval: R@1=%.3f",
            metrics.recall_at_1,
        )
        return metrics.recall_at_1


def _load_mobileclip_tensor(img_path: Path) -> np.ndarray:
    """Load a PIL image and convert to a float32 NCHW tensor for MobileCLIP (256x256)."""
    image = Image.open(img_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis]
