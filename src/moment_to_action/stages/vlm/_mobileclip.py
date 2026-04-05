"""MobileCLIP-S2 zero-shot classification stage.

MobileCLIPStage runs MobileCLIP on a preprocessed FrameTensorMessage
and emits a ClassificationMessage with label + confidence scores.

Input:  FrameTensorMessage  (was TensorMessage — renamed to FrameTensorMessage)
Output: ClassificationMessage
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import open_clip

from moment_to_action.messages import ClassificationMessage, FrameTensorMessage
from moment_to_action.models import ModelID, ModelManager
from moment_to_action.stages._base import Stage
from moment_to_action.utils.ml import softmax

if TYPE_CHECKING:
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.messages import Message

logger = logging.getLogger(__name__)


class MobileCLIPStage(Stage):
    """Runs MobileCLIP-S2 zero-shot classification on a preprocessed tensor.

    Input:  FrameTensorMessage  (was TensorMessage — renamed to FrameTensorMessage)
            Expects [1, 3, 256, 256] float32, channels-first.
    Output: ClassificationMessage

    Use PreprocessorStage with MobileCLIP config upstream:
        PreprocessorStage(target_size=(256, 256), mean=(0,0,0), std=(1,1,1))

    Text prompts define what the model looks for — swap them to change
    the application without reloading the model.
    """

    def __init__(
        self,
        text_prompts: list[str],
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> None:
        super().__init__()
        self._backend = backend
        # Resolve the MobileCLIP model path through the manager — downloads/caches as needed.
        model_path = manager.get_path(ModelID.MOBILECLIP_S2)
        self._handle = self._backend.load_model(model_path)
        self._text_prompts = text_prompts
        self._text_tokens = self._tokenize(text_prompts)
        # NEW: Pre-compute text embeddings
        logger.info("Pre-computing text embeddings...")
        self._text_embeddings = self._precompute_text_embeddings()
        logger.info("MobileCLIPStage: loaded %s with %d prompts", model_path, len(text_prompts))

    def _precompute_text_embeddings(self) -> np.ndarray:
        """Encode all prompts once at startup."""
        dummy_image = np.zeros((1, 3, 256, 256), dtype=np.float32)

        text_embeddings = []
        for tokens in self._text_tokens:
            outputs = self._backend.run(
                self._handle,
                {
                    "serving_default_args_0:0": dummy_image,
                    "serving_default_args_1:0": tokens[np.newaxis, ...].astype(np.int64),
                },
            )
            text_emb = outputs[0][0]  # [512]
            text_embeddings.append(text_emb)

        text_embeddings = np.stack(text_embeddings)
        # Pre-normalize
        norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        return text_embeddings / (norms + 1e-8)

    def _process(self, msg: Message) -> ClassificationMessage | None:
        """Run zero-shot classification against all text prompts."""
        # NOTE: input type check uses FrameTensorMessage (renamed from TensorMessage)
        if not isinstance(msg, FrameTensorMessage):
            err = f"MobileCLIPStage expects FrameTensorMessage, got {type(msg).__name__}"
            raise TypeError(err)

        # Run model ONCE (only image encoder matters)
        dummy_tokens = self._text_tokens[0][np.newaxis, ...].astype(np.int64)

        outputs = self._backend.run(
            self._handle,
            {
                "serving_default_args_0:0": msg.tensor,
                "serving_default_args_1:0": dummy_tokens,
            },
        )

        """
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
        num_prompts = len(self._text_tokens)
        image_batch = np.repeat(msg.tensor, num_prompts, axis=0)
        text_batch = self._text_tokens.astype(np.int64)
        outputs = self._backend.run(
            self._handle,
            {
                "serving_default_args_0:0": image_batch,  # [N, 3, 256, 256]
                "serving_default_args_1:0": text_batch,   # [N, 77]
            },
        )
        image_embs = outputs[1]  # [N, 512]
        text_embs = outputs[0]   # [N, 512]
        scores = self._cosine_similarity_batch(image_embs, text_embs)
        """

        image_emb = outputs[1][0]  # [512]
        image_emb = image_emb / (np.linalg.norm(image_emb) + 1e-8)

        # Compare with ALL pre-computed text embeddings (vectorized!)
        scores = np.dot(self._text_embeddings, image_emb)  # [5] scores

        scores_arr = np.array(scores, dtype=np.float32)
        scores_softmax = softmax(scores_arr)
        best_idx = int(np.argmax(scores_softmax))

        label = self._text_prompts[best_idx]
        confidence = float(scores_softmax[best_idx])
        logger.info("MobileCLIPStage: '%s'  conf=%.3f", label, confidence)

        # latency_ms is stamped by Stage.process() via model_copy
        return ClassificationMessage(
            label=label,
            confidence=confidence,
            all_scores={
                p: float(s) for p, s in zip(self._text_prompts, scores_softmax, strict=False)
            },
            timestamp=msg.timestamp,
        )

    def update_prompts(self, prompts: list[str]) -> None:
        """Swap prompts at runtime without reloading the model."""
        self._text_prompts = prompts
        self._text_tokens = self._tokenize(prompts)

    def _tokenize(self, prompts: list[str]) -> np.ndarray:
        tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
        # Use np.asarray to handle both torch tensors and arrays uniformly.
        return np.asarray(tokenizer(prompts)).astype(np.int64)
