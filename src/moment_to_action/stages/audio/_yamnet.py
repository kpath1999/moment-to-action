"""YAMNet audio classification stage."""

from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING

import numpy as np

from moment_to_action.messages.audio import AudioClassificationMessage, AudioTensorMessage
from moment_to_action.metrics._types import SpanType
from moment_to_action.models import AssetID, ModelID, ModelManager
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)
_EXPECTED_RANK_TWO = 2
_EXPECTED_RANK_THREE = 3


class YAMNetStage(Stage):
    """Run YAMNet on an audio tensor and emit an audio classification."""

    def __init__(
        self,
        backend: ComputeBackend,
        manager: ModelManager,
        *,
        class_names: Sequence[str] | None = None,
        confidence_threshold: float = 0.0,
        aggregation: str = "mean",
        model_id: ModelID = ModelID.YAMNET_TFLITE,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._class_names: tuple[str, ...] = tuple(class_names) if class_names is not None else ()
        self._confidence_threshold = confidence_threshold
        self._aggregation = aggregation
        model_path = manager.get_path(model_id)
        labels_path = manager.get_asset_path(AssetID.YAMNET_CLASS_MAP)
        self._class_names = self._load_yamnet_labels(labels_path)
        self._handle = self._backend.load_model(model_path)
        logger.info("YAMNetStage: loaded %s", model_path)

    def _process(
        self,
        msg: Message,
        metrics: MetricsCollector,
    ) -> AudioClassificationMessage | None:
        if not isinstance(msg, AudioTensorMessage):
            err = f"YAMNetStage expects AudioTensorMessage, got {type(msg).__name__}"
            raise TypeError(err)

        model_input = self._prepare_input(msg.data)

        with metrics.start_span(SpanType.MODEL_INFERENCE, "YAMNet inference"):
            outputs = self._backend.run(self._handle, model_input)

        frame_scores = self._extract_score_matrix(outputs)
        if frame_scores.size == 0:
            logger.debug("YAMNetStage: model produced no frame scores")
            return None

        clip_scores = self._aggregate_scores(frame_scores)

        if len(self._class_names) != len(clip_scores):
            msg_text = (
                "YAMNet class map length does not match model outputs: "
                f"{len(self._class_names)} != {len(clip_scores)}"
            )
            raise ValueError(msg_text)

        top_k = min(5, len(clip_scores))
        top_indices = np.argsort(clip_scores)[::-1][:top_k]

        top_predictions = {
            self._class_names[int(idx)]: float(clip_scores[int(idx)]) for idx in top_indices
        }

        best_score = next(iter(top_predictions.values()))
        if best_score < self._confidence_threshold:
            logger.debug(
                "YAMNetStage: best score %.3f below threshold %.3f",
                best_score,
                self._confidence_threshold,
            )
            return None

        logger.info(
            "YAMNetStage top-5: %s",
            ", ".join(f"{label}={score:.3f}" for label, score in top_predictions.items()),
        )

        return AudioClassificationMessage(
            top_predictions=top_predictions,
            sample_rate=msg.sample_rate,
            source=msg.source,
            timestamp=msg.timestamp,
        )

    def _prepare_input(self, tensor: np.ndarray) -> np.ndarray:
        return np.asarray(tensor, dtype=np.uint8)

    def _load_yamnet_labels(self, labels_path: Path) -> tuple[str, ...]:
        labels: list[str] = []
        with labels_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            labels.extend(row["display_name"] for row in reader)
        return tuple(labels)

    #def _extract_score_matrix(self, outputs: list[np.ndarray]) -> np.ndarray:
    #    for output in outputs:
    #        array = np.asarray(output, dtype=np.float32)
    #        if array.ndim == _EXPECTED_RANK_TWO:
    #            return array
    #        if array.ndim == _EXPECTED_RANK_THREE and array.shape[0] == 1:
    #            return array[0]
    #    return np.empty((0, 0), dtype=np.float32)

    def _extract_score_matrix(self, outputs: list[np.ndarray]) -> np.ndarray:
        array = np.asarray(outputs[0], dtype=np.float32)
    
        # handle [1, 521] → [1, 521] (already 2D, treat as single frame)
        if array.ndim == _EXPECTED_RANK_THREE and array.shape[0] == 1:
            array = array[0]

        # apply softmax to convert logits to probabilities
        array = array - array.max(axis=-1, keepdims=True)   # numerical stability
        exp = np.exp(array)
        array = exp / exp.sum(axis=-1, keepdims=True)

        return array

    def _aggregate_scores(self, frame_scores: np.ndarray) -> np.ndarray:
        if self._aggregation == "max":
            return frame_scores.max(axis=0)
        if self._aggregation == "mean":
            return frame_scores.mean(axis=0)

        msg = f"Unsupported aggregation '{self._aggregation}'. Use 'mean' or 'max'."
        raise ValueError(msg)

    def _resolve_class_names(self, num_classes: int) -> tuple[str, ...]:
        if self._class_names is None:
            return tuple(f"class_{idx}" for idx in range(num_classes))
        if len(self._class_names) != num_classes:
            msg = (
                "YAMNet class_names length does not match model outputs: "
                f"{len(self._class_names)} != {num_classes}"
            )
            raise ValueError(msg)
        return self._class_names
