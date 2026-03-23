"""Temporal action-recognition stage.

TemporalActionStage consumes a
:class:`~moment_to_action.messages.video.VideoClipMessage` (a fixed-length
window of raw frames) and runs a temporal model over the whole clip to
produce an :class:`~moment_to_action.messages.video.ActionMessage`.

Supported model back-ends
--------------------------
The stage is **back-end agnostic**: it delegates inference to the
:class:`~moment_to_action.hardware.ComputeBackend` HAL, same as YOLO and
MobileCLIP.  Two clip preprocessors are provided out of the box:

- ``"movinet"``  — MoViNet / SlowFast-style: resize to 172x172, ImageNet
  normalisation, stack frames into an (N, H, W, C) float32 tensor.
- ``"videomae"`` — VideoMAE-style: resize to 224x224, ImageNet normalisation
  with patch-level mean, stack to (N, H, W, C) float32 tensor.
- ``"custom"``   — caller supplies a ``clip_preprocess_fn`` callable.

Label vocabulary
----------------
Pass a ``labels`` list to map integer class IDs to human-readable strings.
If not provided, class IDs are formatted as ``"action_{id}"``.

Usage example
-------------
::

    stage = TemporalActionStage(
        model_path="models/movinet_a0.tflite",
        clip_profile="movinet",
        labels=KINETICS_400_LABELS,
        top_k=3,
    )
    action_msg = stage._process(clip_msg)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable  # noqa: TC003
from typing import TYPE_CHECKING

import numpy as np

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.messages.video import ActionMessage, ActionPrediction, VideoClipMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.messages import Message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in clip pre-processors
# ---------------------------------------------------------------------------
# Each returns an (N, H, W, C) float32 array ready for the inference back-end.

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_movinet(frames: list[np.ndarray]) -> np.ndarray:
    """Resize to 172x172 and apply ImageNet normalisation, returning (N,172,172,3) float32."""
    import cv2

    target_h, target_w = 172, 172
    processed = []
    for frame in frames:
        resized = cv2.resize(frame, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        processed.append(norm)
    return np.stack(processed, axis=0)  # (N, H, W, C)


def _preprocess_videomae(frames: list[np.ndarray]) -> np.ndarray:
    """Resize to 224x224 and apply ImageNet normalisation, returning (N,224,224,3) float32."""
    import cv2

    target_h, target_w = 224, 224
    processed = []
    for frame in frames:
        resized = cv2.resize(frame, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        processed.append(norm)
    return np.stack(processed, axis=0)  # (N, H, W, C)


_BUILTIN_PROFILES: dict[str, Callable[[list[np.ndarray]], np.ndarray]] = {
    "movinet": _preprocess_movinet,
    "videomae": _preprocess_videomae,
}


# ---------------------------------------------------------------------------
# TemporalActionStage
# ---------------------------------------------------------------------------


class TemporalActionStage(Stage):
    """Run a temporal action-recognition model over a video clip.

    The model is loaded once (in ``__init__``) via :class:`ComputeBackend`
    and reused across all calls.  The stage is intentionally thin — all
    model-specific logic (temporal convolutions, attention, etc.) lives
    inside the model file itself.

    Args:
        model_path: Path to the model file (TFLite, ONNX, etc.).
        clip_profile: One of ``"movinet"``, ``"videomae"``, or ``"custom"``.
            When ``"custom"`` is specified, supply ``clip_preprocess_fn``.
        clip_preprocess_fn: Callable ``(list[NDArray]) → NDArray`` that
            converts raw BGR frames to a model-ready tensor.  Only used
            when ``clip_profile="custom"``.
        labels: Ordered list of action label strings.  Index *i* maps to
            class ID *i*.  Pass ``None`` to use ``"action_{id}"`` placeholders.
        top_k: Number of top predictions to include in :class:`ActionMessage`.
        confidence_threshold: Minimum confidence for a prediction to appear
            in the output.  ``0.0`` keeps all *top_k* results.
        compute_unit: Target compute unit (CPU, NPU, GPU, DSP).
        model_name: Optional human-readable model identifier stored in the
            output message for logging / debugging.

    Input:  :class:`~moment_to_action.messages.video.VideoClipMessage`
    Output: :class:`~moment_to_action.messages.video.ActionMessage`
    """

    def __init__(
        self,
        model_path: str,
        clip_profile: str = "movinet",
        clip_preprocess_fn: Callable[[list[np.ndarray]], np.ndarray] | None = None,
        labels: list[str] | None = None,
        top_k: int = 5,
        confidence_threshold: float = 0.0,
        compute_unit: ComputeUnit = ComputeUnit.CPU,
        model_name: str = "",
    ) -> None:
        if clip_profile == "custom" and clip_preprocess_fn is None:
            msg = "clip_preprocess_fn must be provided when clip_profile='custom'"
            raise ValueError(msg)
        if clip_profile not in _BUILTIN_PROFILES and clip_profile != "custom":
            valid = [*_BUILTIN_PROFILES.keys(), "custom"]
            msg = f"Unknown clip_profile {clip_profile!r}. Valid: {valid}"
            raise ValueError(msg)

        if clip_profile == "custom":
            # clip_preprocess_fn is guaranteed non-None by the guard above.
            if clip_preprocess_fn is None:  # pragma: no cover
                msg = "clip_preprocess_fn is None despite earlier guard"
                raise RuntimeError(msg)
            self._preprocess_fn: Callable[[list[np.ndarray]], np.ndarray] = clip_preprocess_fn
        else:
            self._preprocess_fn = _BUILTIN_PROFILES[clip_profile]
        self._labels = labels
        self._top_k = max(1, top_k)
        self._confidence_threshold = confidence_threshold
        self._model_name = model_name or model_path

        self._backend = ComputeBackend(preferred_unit=compute_unit)
        self._handle = self._backend.load_model(model_path)
        logger.info("TemporalActionStage: loaded %s (profile=%s)", model_path, clip_profile)

    # ------------------------------------------------------------------
    # Stage interface
    # ------------------------------------------------------------------

    def _process(self, msg: Message) -> ActionMessage | None:
        """Run the temporal model on*msg* and return action predictions.

        Returns:
            An :class:`~moment_to_action.messages.video.ActionMessage` with up
            to ``top_k`` predictions, or ``None`` if no prediction exceeds
            ``confidence_threshold``.
        """
        if not isinstance(msg, VideoClipMessage):
            type_name = type(msg).__name__
            err = f"TemporalActionStage expects VideoClipMessage, got {type_name}"
            raise TypeError(err)

        # Preprocess clip: list[NDArray] → (N, H, W, C) float32
        clip_tensor = self._preprocess_fn(msg.frames)  # type: ignore[arg-type]

        # Add batch dimension if needed → (1, N, H, W, C)
        if clip_tensor.ndim == 4:  # noqa: PLR2004
            clip_tensor = clip_tensor[np.newaxis]

        t = time.perf_counter()
        outputs = self._backend.run(self._handle, clip_tensor)
        latency_ms = (time.perf_counter() - t) * 1000

        predictions = self._parse_outputs(outputs)
        if not predictions:
            logger.debug("TemporalActionStage: no predictions above threshold")
            return None

        logger.info(
            "TemporalActionStage: top action = %s (%.2f)",
            predictions[0].label,
            predictions[0].confidence,
        )
        return ActionMessage(
            predictions=predictions,
            clip_num_frames=msg.num_frames,
            model_name=self._model_name,
            timestamp=msg.timestamp,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_outputs(
        self,
        outputs: list[np.ndarray],
    ) -> list[ActionPrediction]:
        """Convert raw model output into sorted :class:`ActionPrediction` list.

        Expects the standard classification output layout:
        - ``outputs[0]``: shape ``(1, num_classes)`` float32 logits or probabilities.

        Softmax is applied when values do not already sum to ~1 (i.e. logits).
        """
        if not outputs:
            return []

        scores_raw: np.ndarray = outputs[0]
        # Remove batch dimension if present.
        if scores_raw.ndim == 2 and scores_raw.shape[0] == 1:  # noqa: PLR2004
            scores_raw = scores_raw[0]

        # Apply softmax if it looks like raw logits.
        score_sum = float(scores_raw.sum())
        _softmax_sum_threshold = 1.5
        if abs(score_sum - 1.0) > _softmax_sum_threshold:
            exp_s = np.exp(scores_raw - scores_raw.max())
            scores = exp_s / exp_s.sum()
        else:
            scores = scores_raw.astype(np.float32)

        # Pick top-k.
        top_indices = np.argsort(scores)[::-1][: self._top_k]
        predictions: list[ActionPrediction] = []
        for idx in top_indices:
            conf = float(scores[idx])
            if conf < self._confidence_threshold:
                continue
            label = (
                self._labels[int(idx)]
                if self._labels and int(idx) < len(self._labels)
                else f"action_{idx}"
            )
            predictions.append(ActionPrediction(label=label, confidence=conf, class_id=int(idx)))

        return predictions
