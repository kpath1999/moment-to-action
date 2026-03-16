"""YOLO detection and LLM reasoning stages.

Concrete stages for the YOLO → LLM baseline pipeline.

SensorStage        loads an image from disk → RawFrameMessage
PreprocessorStage  resizes + normalizes    → TensorMessage
YOLOStage          runs YOLO               → DetectionMessage
ReasoningStage     runs an LLM             → ReasoningMessage

Each stage is independent. Swap or reorder them freely.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from moment_to_action.edgeperceive.core.messages import (
    BoundingBox,
    DetectionMessage,
    ReasoningMessage,
    TensorMessage,
)
from moment_to_action.edgeperceive.hardware.types import ComputeUnit
from moment_to_action.edgeperceive.stages.base import Stage

if TYPE_CHECKING:
    from moment_to_action.edgeperceive.core.messages import Message

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# 3. YOLOStage
# ──────────────────────────────────────────────────────────────────


class YOLOStage(Stage):
    """Runs YOLO on a preprocessed tensor and emits detections.

    Returns None if no detections pass the confidence threshold
    — downstream stages (LLM) won't run.

    Input:  TensorMessage
    Output: DetectionMessage, or None if nothing detected
    """

    # COCO class labels (80 classes)
    COCO_LABELS: ClassVar[tuple[str, ...]] = (
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    )

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        compute_unit: ComputeUnit | None = None,
    ) -> None:
        from moment_to_action.edgeperceive.hardware.compute_backend import ComputeBackend

        if compute_unit is None:
            compute_unit = ComputeUnit.CPU
        self._backend = ComputeBackend(preferred_unit=compute_unit)
        self._handle = self._backend.load_model(model_path)
        self.confidence_threshold = confidence_threshold
        logger.info("YOLOStage: loaded %s", model_path)

    def process(self, msg: Message) -> DetectionMessage | None:
        """Run YOLO inference and return detections above threshold."""
        if not isinstance(msg, TensorMessage):
            err = f"YOLOStage expects TensorMessage, got {type(msg).__name__}"
            raise TypeError(err)
        t = time.perf_counter()
        outputs = self._backend.run(self._handle, msg.tensor)
        latency_ms = (time.perf_counter() - t) * 1000

        boxes = self._parse_outputs(outputs, msg.original_size)
        boxes = [b for b in boxes if b.confidence >= self.confidence_threshold]

        if not boxes:
            logger.debug("YOLOStage: no detections above %s", self.confidence_threshold)
            return None

        logger.info("YOLOStage: %d detection(s)", len(boxes))
        for b in sorted(boxes, key=lambda x: -x.confidence):
            logger.info(
                "  %-20s  conf=%.2f  box=[%.0f,%.0f,%.0f,%.0f]",
                b.label,
                b.confidence,
                b.x1,
                b.y1,
                b.x2,
                b.y2,
            )

        return DetectionMessage(
            boxes=boxes,
            latency_ms=latency_ms,
            timestamp=msg.timestamp,
        )

    def _parse_outputs(
        self,
        outputs: list[np.ndarray],
        original_size: tuple,
    ) -> list[BoundingBox]:
        """Parse YOLOv8 3-output format (float32, NMS not baked in).

        outputs[0]: [1, N, 4]  float32 — boxes (x1,y1,x2,y2) in 640x640 space
        outputs[1]: [1, N]     float32 — confidence scores 0..1
        outputs[2]: [1, N]     uint8   — class IDs
        """
        _expected_output_count = 3
        if len(outputs) < _expected_output_count:
            return []

        boxes_raw = outputs[0][0].astype(np.float32)  # [N, 4]
        scores = outputs[1][0].astype(np.float32)  # [N]
        class_ids = outputs[2][0]  # [N]

        # Filter by confidence first
        mask = scores >= self.confidence_threshold
        boxes_raw = boxes_raw[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_raw) == 0:
            return []

        # NMS
        keep = self._nms(boxes_raw, scores, iou_threshold=0.45)
        boxes_raw = boxes_raw[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Scale from 640x640 input space to original image pixels
        orig_h, orig_w = original_size
        sx = orig_w / 640.0
        sy = orig_h / 640.0

        boxes = []
        for box, score, cid in zip(boxes_raw, scores, class_ids, strict=False):
            x1 = max(0, float(box[0]) * sx)
            y1 = max(0, float(box[1]) * sy)
            x2 = min(orig_w, float(box[2]) * sx)
            y2 = min(orig_h, float(box[3]) * sy)
            class_id = int(cid)
            label = (
                self.COCO_LABELS[class_id] if class_id < len(self.COCO_LABELS) else str(class_id)
            )
            boxes.append(
                BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(score),
                    class_id=class_id,
                    label=label,
                )
            )

        return boxes

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        """Pure numpy NMS."""
        indices = np.argsort(scores)[::-1]
        keep = []
        while len(indices) > 0:
            cur = indices[0]
            keep.append(int(cur))
            if len(indices) == 1:
                break
            cb = boxes[cur]
            rb = boxes[indices[1:]]
            x1 = np.maximum(cb[0], rb[:, 0])
            y1 = np.maximum(cb[1], rb[:, 1])
            x2 = np.minimum(cb[2], rb[:, 2])
            y2 = np.minimum(cb[3], rb[:, 3])
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            cur_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
            rem_areas = (rb[:, 2] - rb[:, 0]) * (rb[:, 3] - rb[:, 1])
            iou = inter / (cur_area + rem_areas - inter + 1e-6)
            indices = indices[1:][iou < iou_threshold]
        return keep


# ──────────────────────────────────────────────────────────────────
# 4. ReasoningStage
# ──────────────────────────────────────────────────────────────────


class ReasoningStage(Stage):
    """Formats YOLO detections into a prompt and runs an LLM.

    Input:  DetectionMessage
    Output: ReasoningMessage
    """

    def __init__(self, model_path: str | None = None, system_prompt: str = "") -> None:
        self._handle = None
        if model_path:
            from moment_to_action.edgeperceive.hardware.compute_backend import ComputeBackend
            from moment_to_action.edgeperceive.hardware.types import ComputeUnit

            self._backend = ComputeBackend(preferred_unit=ComputeUnit.CPU)
            self._handle = self._backend.load_model(model_path)
            logger.info("ReasoningStage: loaded %s", model_path)
        else:
            logger.info("ReasoningStage: running in stub mode (no model loaded)")
        self._system_prompt = system_prompt or (
            "You are analyzing detections from a wearable device. "
            "Based on the detected objects and their positions, assess the scene briefly."
        )

    def process(self, msg: Message) -> ReasoningMessage | None:
        """Format detections into a prompt and run the LLM."""
        if not isinstance(msg, DetectionMessage):
            err = f"ReasoningStage expects DetectionMessage, got {type(msg).__name__}"
            raise TypeError(err)
        prompt = self._build_prompt(msg)
        t = time.perf_counter()
        # LLM inference — tokenize, run, decode
        # Placeholder until Qwen is wired in
        response = self._run_llm(prompt)
        latency_ms = (time.perf_counter() - t) * 1000
        return ReasoningMessage(
            response=response,
            prompt=prompt,
            latency_ms=latency_ms,
            timestamp=msg.timestamp,
        )

    def _build_prompt(self, msg: DetectionMessage) -> str:
        lines = [self._system_prompt, "", "Detections:"]
        lines.extend(
            f"  - {box.label} (confidence: {box.confidence:.2f}, "
            f"position: [{box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f}])"
            for box in msg.top(5)
        )
        lines.append("\nWhat is happening in this scene?")
        return "\n".join(lines)

    def _run_llm(self, prompt: str) -> str:
        # NOTE(kausar): integrate with Kausar's LLM arch. LLM is a stage that
        # ingests the message, performs inference dispatched via ComputeBackend.
        # For now return the prompt so the pipeline is runnable end-to-end.
        return f"[LLM stub] Received prompt with {len(prompt)} chars."
