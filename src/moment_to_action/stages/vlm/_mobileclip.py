"""MobileCLIP-S2 zero-shot classification stage.

MobileCLIPStage runs MobileCLIP on a preprocessed FrameTensorMessage
and emits a ClassificationMessage with label + confidence scores.

Input:  FrameTensorMessage  (was TensorMessage — renamed to FrameTensorMessage)
Output: ClassificationMessage

Model expects batch=4 inputs:
  image: [4, 256, 256, 3]  float32  NHWC  (channels-last)
  text:  [4, 77]           int64
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import open_clip
from concurrent.futures import ThreadPoolExecutor

from moment_to_action.messages import ClassificationMessage, FrameTensorMessage
from moment_to_action.models import ModelID, ModelManager
from moment_to_action.stages._base import Stage
from moment_to_action.utils.ml import softmax

if TYPE_CHECKING:
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)

BATCH_SIZE = 4
IMG_H      = 256
IMG_W      = 256
IMG_C      = 3
SEQ_LEN    = 77


class MobileCLIPStage(Stage):
    """Runs MobileCLIP-S2 zero-shot classification on a preprocessed tensor.

    Input:  FrameTensorMessage
            Expects [1, 3, 256, 256] float32 channels-first (NCHW) per frame.
            Internally transposed to NHWC and batched to groups of 4.
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
        num_workers: int = 2,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._text_prompts = text_prompts
        self._text_tokens = self._tokenize(text_prompts)
        model_path = manager.get_path(ModelID.MOBILECLIP_S2_BATCHED)
        self._handle = self._backend.load_model(model_path)

        logger.info("Pre-computing text embeddings...")
        self._text_embeddings = self._precompute_text_embeddings()
        logger.info("MobileCLIPStage: loaded %s with %d prompts", model_path, len(text_prompts))

        self._executor = ThreadPoolExecutor(max_workers=num_workers)

    # ── Text embedding pre-computation ───────────────────────────────────────

    def _precompute_text_embeddings(self) -> np.ndarray:
        """Encode all prompts once at startup.

        The model requires exactly BATCH_SIZE inputs, so we pad with repeats
        and only keep the real embeddings.
        """
        # Dummy image batch — all zeros, NHWC
        dummy_images = np.zeros((BATCH_SIZE, IMG_H, IMG_W, IMG_C), dtype=np.float32)

        n_prompts = len(self._text_tokens)
        text_embeddings_list: list[np.ndarray] = []

        # Process prompts in groups of BATCH_SIZE, padding the last group if needed
        for start in range(0, n_prompts, BATCH_SIZE):
            batch_tokens = self._text_tokens[start : start + BATCH_SIZE]

            # Pad to BATCH_SIZE if this is the last (incomplete) group
            if len(batch_tokens) < BATCH_SIZE:
                pad = np.repeat(
                    batch_tokens[:1], BATCH_SIZE - len(batch_tokens), axis=0
                )
                batch_tokens = np.concatenate([batch_tokens, pad], axis=0)

            outputs = self._backend.run(
                self._handle,
                {
                    "image": dummy_images,
                    "text":  batch_tokens.astype(np.int64),
                },
            )
            # text embeddings output shape: [4, 512]
            text_embs = self._get_text_embeddings(outputs)  # [4, 512]

            # Only keep the real (non-padded) ones
            real_count = min(BATCH_SIZE, n_prompts - start)
            text_embeddings_list.append(text_embs[:real_count])

        text_embeddings = np.concatenate(text_embeddings_list, axis=0)  # [n_prompts, 512]

        # Pre-normalize
        norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        return text_embeddings / (norms + 1e-8)

    # ── Processing ───────────────────────────────────────────────────────────

    def _process(self, msg, _metrics):
        """Handle single frame or a list of frames."""
        if isinstance(msg, list):
            return self._process_batch(msg)
        return self._process_single(msg)

    def _process_batch(self, frame_msgs: list[FrameTensorMessage]) -> list[ClassificationMessage]:
        """Process a list of frames in groups of BATCH_SIZE."""
        results = []
        for start in range(0, len(frame_msgs), BATCH_SIZE):
            chunk = frame_msgs[start : start + BATCH_SIZE]
            chunk_results = self._run_image_batch(chunk)
            results.extend(chunk_results)
        return results

    def _process_single(self, msg: FrameTensorMessage) -> ClassificationMessage:
        """Process a single frame by padding it to a batch of BATCH_SIZE."""
        results = self._run_image_batch([msg])
        return results[0]

    def _run_image_batch(self, frame_msgs: list[FrameTensorMessage]) -> list[ClassificationMessage]:
        """Run inference on up to BATCH_SIZE frames, padding if needed."""
        real_count = len(frame_msgs)

        # Build image batch [BATCH_SIZE, H, W, C] in NHWC
        images = []
        for fm in frame_msgs:
            img = self._to_nhwc(fm.tensor)  # [1, H, W, C]
            images.append(img)

        # Pad to BATCH_SIZE if needed
        while len(images) < BATCH_SIZE:
            images.append(images[0])

        image_batch = np.concatenate(images, axis=0)  # [4, H, W, C]

        # Dummy text tokens (text embeddings are pre-computed; we still need
        # to feed something valid for the text input)
        dummy_tokens = np.tile(
            self._text_tokens[0][np.newaxis, :], (BATCH_SIZE, 1)
        ).astype(np.int64)  # [4, 77]

>>>>>>> 07e12ae (clip changes)
        outputs = self._backend.run(
            self._handle,
            {
                "image": image_batch,
                "text":  dummy_tokens,
            },
        )

        # image embeddings: [4, 512]
        image_embs = self._get_image_embeddings(outputs)

        results = []
        for i in range(real_count):
            image_emb = image_embs[i]  # [512]
            image_emb = image_emb / (np.linalg.norm(image_emb) + 1e-8)

            scores = np.dot(self._text_embeddings, image_emb)  # [n_prompts]
            scores_softmax = softmax(np.array(scores, dtype=np.float32))
            best_idx = int(np.argmax(scores_softmax))

            label      = self._text_prompts[best_idx]
            confidence = float(scores_softmax[best_idx])

            print(f"  frame ts={frame_msgs[i].timestamp:>4}  →  '{label}'  ({confidence:.3f})")

            results.append(ClassificationMessage(
                label=label,
                confidence=confidence,
                all_scores={p: float(s) for p, s in zip(self._text_prompts, scores_softmax)},
                timestamp=frame_msgs[i].timestamp,
            ))

        return results

    # ── Output parsing ────────────────────────────────────────────────────────

    def _get_image_embeddings(self, outputs) -> np.ndarray:
        """Extract the [4, 512] image embeddings from model outputs.

        onnx2tf may reorder outputs — find by shape rather than fixed index.
        """
        return self._find_output_by_shape(outputs, (BATCH_SIZE, 512))

    def _get_text_embeddings(self, outputs) -> np.ndarray:
        """Extract the [4, 512] text embeddings from model outputs.

        Both image and text embeddings have the same shape [4, 512].
        onnx2tf typically outputs image first, text second — but we use
        index 1 here as a fallback. If results look wrong, swap to index 0.
        """
        candidates = self._find_all_outputs_by_shape(outputs, (BATCH_SIZE, 512))
        # Return the second [4, 512] output (text), fallback to first if only one
        return candidates[1] if len(candidates) > 1 else candidates[0]

    def _find_output_by_shape(self, outputs, shape: tuple) -> np.ndarray:
        for out in outputs:
            arr = np.asarray(out)
            if arr.shape == shape:
                return arr
        raise RuntimeError(f"No output with shape {shape} found. Got: {[np.asarray(o).shape for o in outputs]}")

    def _find_all_outputs_by_shape(self, outputs, shape: tuple) -> list[np.ndarray]:
        found = [np.asarray(o) for o in outputs if np.asarray(o).shape == shape]
        if not found:
            raise RuntimeError(f"No output with shape {shape} found. Got: {[np.asarray(o).shape for o in outputs]}")
        return found

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_nhwc(self, tensor: np.ndarray) -> np.ndarray:
        """Convert [1, 3, H, W] NCHW tensor to [1, H, W, 3] NHWC."""
        if tensor.ndim == 4 and tensor.shape[1] == IMG_C:
            # NCHW -> NHWC
            return tensor.transpose(0, 2, 3, 1)
        elif tensor.ndim == 4 and tensor.shape[3] == IMG_C:
            # Already NHWC
            return tensor
        else:
            raise ValueError(f"Unexpected image tensor shape: {tensor.shape}")

    def update_prompts(self, prompts: list[str]) -> None:
        """Swap prompts at runtime without reloading the model."""
        self._text_prompts = prompts
        self._text_tokens = self._tokenize(prompts)
        logger.info("Re-computing text embeddings for %d new prompts...", len(prompts))
        self._text_embeddings = self._precompute_text_embeddings()

    def _tokenize(self, prompts: list[str]) -> np.ndarray:
        tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
        return np.asarray(tokenizer(prompts)).astype(np.int64)
