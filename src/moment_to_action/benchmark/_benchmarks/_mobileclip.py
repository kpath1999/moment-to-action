from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import open_clip
from PIL import Image

from moment_to_action.benchmark._accuracy import mean_embedding_similarity
from moment_to_action.benchmark._base import ModelBenchmark
from moment_to_action.benchmark._oracle_ground_truth import OracleStore
from moment_to_action.benchmark._retrieval_metrics import compute_retrieval_metrics
from moment_to_action.models import ModelID
from moment_to_action.utils.ml import cosine_similarity, softmax

if TYPE_CHECKING:
    from moment_to_action.benchmark._coco_dataset import CocoDataset
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager

logger = logging.getLogger(__name__)


class MobileCLIPBenchmark(ModelBenchmark):
    """Benchmark implementation for MobileCLIP-S2."""

    def __init__(
        self,
        eval_image_paths: list[Path] | None = None,
        *,
        coco_dataset: CocoDataset | None = None,
        oracle_store: OracleStore | None = None,
    ) -> None:
        super().__init__()
        self._eval_image_paths = eval_image_paths or []
        self._coco_dataset = coco_dataset
        self._oracle_store = oracle_store

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
        """Evaluate MobileCLIP against project oracle or COCO pseudo-ground-truth."""
        if self._coco_dataset is not None:
            return self._evaluate_coco_accuracy(handle=handle, backend=backend)

        if self._eval_image_paths:
            return self._evaluate_embedding_consistency(
                handle=handle, backend=backend, manager=manager
            )

        return self._evaluate_project_oracle_accuracy(handle=handle, backend=backend)

    def _evaluate_project_oracle_accuracy(
        self,
        handle: object,
        backend: ComputeBackend,
    ) -> float | None:
        """Evaluate MobileCLIP against oracle labels stored for project sample images."""
        gt = OracleStore().load()
        if gt is None or not gt.classifications:
            logger.debug(
                "MobileCLIPBenchmark: no oracle classifications found -- skipping accuracy."
            )
            return None

        text_prompts = gt.text_prompts
        if not text_prompts:
            return None

        tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
        tokens: np.ndarray = np.asarray(tokenizer(text_prompts)).astype(np.int64)

        sample_images = _default_sample_images()
        if not sample_images:
            return None

        image_map = {path.name: path for path in sample_images}
        correct = 0
        total = 0

        for oracle_cls in gt.classifications:
            img_path = image_map.get(oracle_cls.image_name)
            if img_path is None:
                continue

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

            best_idx = int(np.argmax(softmax(np.array(scores, dtype=np.float32))))
            predicted_label = text_prompts[best_idx]

            total += 1
            if predicted_label == oracle_cls.top_label:
                correct += 1

        if total == 0:
            return None

        accuracy = correct / total
        self._set_accuracy_details({"recall_at_1": accuracy})
        logger.info("MobileCLIPBenchmark accuracy: top-1 = %.3f (%d/%d)", accuracy, correct, total)
        return accuracy

    def _evaluate_coco_accuracy(self, handle: object, backend: ComputeBackend) -> float | None:
        """Evaluate MobileCLIP retrieval quality against SigLIP COCO pseudo-ground-truth."""
        dataset = self._coco_dataset
        if dataset is None:
            return None

        store = self._oracle_store or OracleStore(dataset_name=dataset.dataset_name)
        gt = store.load()
        if gt is None or not gt.classifications:
            logger.debug("MobileCLIPBenchmark: no COCO oracle classifications found -- skipping.")
            return None

        tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
        image_map = {path.name: path for path in dataset.images()}
        predicted_scores: dict[str, list[float]] = {}
        oracle_scores: dict[str, list[float]] = {}

        for oracle_cls in gt.classifications:
            img_path = image_map.get(oracle_cls.image_name)
            if img_path is None or not oracle_cls.scores:
                continue

            prompts = list(oracle_cls.scores.keys())
            tokens: np.ndarray = np.asarray(tokenizer(prompts)).astype(np.int64)
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

            predicted_scores[oracle_cls.image_name] = scores
            oracle_scores[oracle_cls.image_name] = [oracle_cls.scores[prompt] for prompt in prompts]

        if not predicted_scores:
            return None

        metrics = compute_retrieval_metrics(
            predictions=predicted_scores,
            ground_truth=oracle_scores,
        )
        self._set_accuracy_details(
            {
                "recall_at_1": metrics.recall_at_1,
                "recall_at_5": metrics.recall_at_5,
                "recall_at_10": metrics.recall_at_10,
                "kendall_tau": metrics.kendall_tau,
                "mean_rank_delta": metrics.mean_rank_delta,
            }
        )
        logger.info(
            "MobileCLIPBenchmark COCO pseudo-GT: R@1=%.3f R@5=%.3f R@10=%.3f tau=%.3f",
            metrics.recall_at_1,
            metrics.recall_at_5,
            metrics.recall_at_10,
            metrics.kendall_tau,
        )
        return metrics.recall_at_1

    def _evaluate_embedding_consistency(
        self,
        handle: object,
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> float | None:
        """Return mean cosine similarity of image embeddings vs a CPU oracle."""
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("opencv-python not installed -- skipping MobileCLIP accuracy evaluation")
            return None

        from moment_to_action.hardware import ComputeBackend
        from moment_to_action.hardware._types import ComputeUnit

        cpu_backend = ComputeBackend(preferred_unit=ComputeUnit.CPU)
        oracle_handle = cpu_backend.load_model(manager.get_path(self.model_id))
        dummy_tokens = np.zeros((1, 77), dtype=np.int64)

        oracle_embeddings: list[np.ndarray] = []
        eval_embeddings: list[np.ndarray] = []
        nan_count = 0

        for img_path in self._eval_image_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning("Could not load eval image %s -- skipping", img_path)
                continue

            img_tensor = _preprocess_mobileclip(img_bgr)
            oracle_inputs = {
                "serving_default_args_0:0": img_tensor,
                "serving_default_args_1:0": dummy_tokens,
            }
            oracle_out = cpu_backend.run(oracle_handle, oracle_inputs)

            eval_inputs = {
                "serving_default_args_0:0": img_tensor,
                "serving_default_args_1:0": dummy_tokens,
            }
            eval_out = backend.run(handle, eval_inputs)

            eval_emb = np.asarray(eval_out[0], dtype=np.float32)
            if bool(np.any(np.isnan(eval_emb))):
                nan_count += 1
                continue

            oracle_embeddings.append(np.asarray(oracle_out[0], dtype=np.float32))
            eval_embeddings.append(eval_emb)

        if nan_count > 0:
            logger.warning(
                "%d/%d eval image(s) produced NaN embeddings on %s -- accuracy unavailable",
                nan_count,
                len(self._eval_image_paths),
                backend.active_unit.name,
            )

        if not oracle_embeddings:
            return None

        return mean_embedding_similarity(oracle_embeddings, eval_embeddings)


def _load_mobileclip_tensor(img_path: Path) -> np.ndarray:
    """Load a PIL image and convert to a float32 NCHW tensor for MobileCLIP (256x256)."""
    image = Image.open(img_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis]


def _preprocess_mobileclip(img_bgr: np.ndarray) -> np.ndarray:
    """Return a float32 RGB NCHW tensor normalized to [0, 1]."""
    import cv2  # type: ignore[import-untyped]

    resized = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    return np.expand_dims(rgb.transpose(2, 0, 1), 0)


def _default_sample_images() -> list[Path]:
    """Locate the project images/ directory relative to this file."""
    candidate = Path(__file__).parents[4] / "images"
    if candidate.is_dir():
        return sorted(candidate.glob("*.jpg"))
    return []
