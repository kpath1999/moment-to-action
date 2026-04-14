from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from moment_to_action.benchmark._accuracy import mean_embedding_similarity
from moment_to_action.benchmark._base import ModelBenchmark
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager

logger = logging.getLogger(__name__)


class MobileCLIPBenchmark(ModelBenchmark):
    """Benchmark implementation for MobileCLIP-S2.

    Args:
        eval_image_paths: Optional list of image paths used for accuracy
            evaluation.  Each image is embedded by both the CPU/float32 oracle
            and the current variant; mean cosine similarity vs oracle is returned
            as the accuracy proxy.  Values near 1.0 indicate high quantization
            fidelity.
    """

    def __init__(self, eval_image_paths: list[Path] | None = None) -> None:
        self._eval_image_paths: list[Path] = eval_image_paths or []

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
        """Return mean cosine similarity of image embeddings vs CPU/float32 oracle.

        The CPU model is used as the oracle.  For each eval image both models
        produce an image embedding; the cosine similarity between oracle and
        eval embeddings is averaged across images.  A score of 1.0 means
        perfect agreement; values near 1.0 indicate low quantisation error.
        If no eval images are configured the method returns ``None``.
        """
        if not self._eval_image_paths:
            return None

        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("opencv-python not installed — skipping MobileCLIP accuracy evaluation")
            return None

        from moment_to_action.hardware import ComputeBackend
        from moment_to_action.hardware._types import ComputeUnit

        cpu_backend = ComputeBackend(preferred_unit=ComputeUnit.CPU)
        oracle_handle = cpu_backend.load_model(manager.get_path(self.model_id))

        # Dummy text tokens: zeros represent a neutral / unknown text prompt.
        # The key metric is image embedding consistency, not zero-shot accuracy.
        dummy_tokens = np.zeros((1, 77), dtype=np.int64)

        oracle_embeddings: list[np.ndarray] = []
        eval_embeddings: list[np.ndarray] = []

        nan_count = 0
        for img_path in self._eval_image_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning("Could not load eval image %s — skipping", img_path)
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
                # QNN GPU delegate uses FP16 internally; some models produce NaN
                # embeddings due to FP16 overflow.  Skip this image.
                nan_count += 1
                continue

            # The first output tensor carries the image embedding
            oracle_embeddings.append(np.asarray(oracle_out[0], dtype=np.float32))
            eval_embeddings.append(eval_emb)

        if nan_count > 0:
            logger.warning(
                "%d/%d eval image(s) produced NaN embeddings on %s "
                "(likely FP16 overflow in the GPU delegate) — accuracy unavailable",
                nan_count,
                len(self._eval_image_paths),
                backend.active_unit.name,
            )

        if not oracle_embeddings:
            return None

        return mean_embedding_similarity(oracle_embeddings, eval_embeddings)


# ---------------------------------------------------------------------------
# Image preprocessing helper for MobileCLIP (256x256, NCHW, [0,1])
# ---------------------------------------------------------------------------


def _preprocess_mobileclip(img_bgr: np.ndarray) -> np.ndarray:
    """Return float32 RGB NCHW tensor ``[1, 3, 256, 256]`` normalised to [0, 1]."""
    import cv2  # type: ignore[import-untyped]

    resized = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, normalise
    return np.expand_dims(rgb.transpose(2, 0, 1), 0)  # HWC→CHW→NCHW
