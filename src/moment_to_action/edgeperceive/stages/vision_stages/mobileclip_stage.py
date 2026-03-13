"""stages/vision.py

Concrete stages for vision pipelines.

SensorStage        loads an image from disk → RawFrameMessage
PreprocessorStage  resizes + normalizes    → TensorMessage
YOLOStage          runs YOLO               → DetectionMessage
MobileCLIPStage    runs MobileCLIP         → ClassificationMessage
ReasoningStage     runs an LLM             → ReasoningMessage

Each stage is independent. Swap or reorder them freely.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from moment_to_action.edgeperceive.core.messages import (
    ClassificationMessage,
    TensorMessage,
)
from moment_to_action.edgeperceive.stages.base import Stage

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# MobileCLIPStage
# ──────────────────────────────────────────────────────────────────


class MobileCLIPStage(Stage):
    """Runs MobileCLIP-S2 zero-shot classification on a preprocessed tensor.

    Input:  TensorMessage  (expects [1, 3, 256, 256] float32, channels-first)
    Output: ClassificationMessage

    Use PreprocessorStage with MobileCLIP config upstream:
        PreprocessorStage(target_size=(256, 256), mean=(0,0,0), std=(1,1,1))

    Text prompts define what the model looks for — swap them to change
    the application without reloading the model.
    """

    def __init__(
        self,
        model_path: str,
        text_prompts: list[str],
        compute_unit=None,
    ):
        from moment_to_action.edgeperceive.hardware.compute_backend import ComputeBackend
        from moment_to_action.edgeperceive.hardware.types import ComputeUnit

        if compute_unit is None:
            compute_unit = ComputeUnit.CPU

        self._backend = ComputeBackend(preferred_unit=compute_unit)
        self._handle = self._backend.load_model(model_path)
        self.text_prompts = text_prompts
        self._text_tokens = self._tokenize(text_prompts)
        logger.info(f"MobileCLIPStage: loaded {model_path} with {len(text_prompts)} prompts")

    def process(self, msg: TensorMessage) -> ClassificationMessage | None:
        t = time.perf_counter()

        scores = []
        for tokens in self._text_tokens:
            token_tensor = tokens[np.newaxis, ...].astype(np.int64)  # [1, 77]
            outputs = self._backend.run(
                self._handle,
                {
                    "serving_default_args_0:0": msg.tensor,  # [1, 3, 256, 256]
                    "serving_default_args_1:0": token_tensor,  # [1, 77]
                },
            )
            image_emb = outputs[1][0]  # [512]
            text_emb = outputs[0][0]  # [512]
            scores.append(self._cosine_similarity(image_emb, text_emb))

        scores_arr = np.array(scores, dtype=np.float32)
        scores_softmax = self._softmax(scores_arr)
        best_idx = int(np.argmax(scores_softmax))
        latency_ms = (time.perf_counter() - t) * 1000

        label = self.text_prompts[best_idx]
        confidence = float(scores_softmax[best_idx])
        logger.info(f"MobileCLIPStage: '{label}'  conf={confidence:.3f}  {latency_ms:.1f}ms")

        return ClassificationMessage(
            label=label,
            confidence=confidence,
            all_scores={p: float(s) for p, s in zip(self.text_prompts, scores_softmax)},
            latency_ms=latency_ms,
            timestamp=msg.timestamp,
        )

    def update_prompts(self, prompts: list[str]) -> None:
        """Swap prompts at runtime without reloading the model."""
        self.text_prompts = prompts
        self._text_tokens = self._tokenize(prompts)

    def _tokenize(self, prompts: list[str]) -> np.ndarray:
        try:
            import open_clip

            tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
            return tokenizer(prompts).numpy().astype(np.int64)
        except ImportError as err:
            raise RuntimeError("open_clip required: pip install open-clip-torch") from err

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
