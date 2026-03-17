"""YOLO detection stage.

YOLOStage runs YOLO on a preprocessed FrameTensorMessage and emits
DetectionMessage with bounding boxes and class labels.

Input:  FrameTensorMessage  (was TensorMessage — renamed to FrameTensorMessage)
Output: DetectionMessage, or None if nothing detected
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from moment_to_action.messages.video import BoundingBox, DetectionMessage, FrameTensorMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.messages import Message

logger = logging.getLogger(__name__)


class YOLOStage(Stage):
    """Runs YOLO on a preprocessed tensor and emits detections.

    Returns None if no detections pass the confidence threshold
    — downstream stages (LLM) won't run.

    Input:  FrameTensorMessage  (was TensorMessage — renamed to FrameTensorMessage)
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
        backend: ComputeBackend,
        confidence_threshold: float = 0.5,
        stage_idx: int = 2,
    ) -> None:
        super().__init__(stage_idx)
        self._backend = backend
        self._handle = self._backend.load_model(model_path)
        self._confidence_threshold = confidence_threshold
        logger.info("YOLOStage: loaded %s", model_path)

    @property
    def confidence_threshold(self) -> float:
        """Detection confidence threshold in [0, 1]."""
        return self._confidence_threshold

    def _process(self, msg: Message) -> DetectionMessage | None:
        """Run YOLO inference and return detections above threshold."""
        # NOTE: input type check uses FrameTensorMessage (renamed from TensorMessage)
        if not isinstance(msg, FrameTensorMessage):
            err = f"YOLOStage expects FrameTensorMessage, got {type(msg).__name__}"
            raise TypeError(err)
        outputs = self._backend.run(self._handle, msg.tensor)

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

        # latency_ms is stamped by Stage.process() via model_copy
        return DetectionMessage(
            boxes=boxes,
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
