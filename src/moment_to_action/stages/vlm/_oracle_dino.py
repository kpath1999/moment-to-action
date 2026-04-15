"""Oracle Grounding DINO detection stage.

OracleGroundingDinoStage runs Grounding DINO on a VideoClipMessage or Frame
and emits a DetectionMessage with bounding boxes and class labels to serve as ground truth.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from moment_to_action.messages import BoundingBox, DetectionMessage, FrameTensorMessage
from moment_to_action.metrics._types import SpanType
from moment_to_action.models import ModelID, ModelManager
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class OracleGroundingDinoStage(Stage):
    """Runs Grounding DINO for oracle ground truth object detection."""

    def __init__(self, text_queries: list[str], manager: ModelManager) -> None:
        super().__init__()
        model_path = manager.get_path(ModelID.GROUNDING_DINO_BASE)
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to("mps")
        self._text_queries = text_queries
        logger.info("OracleGroundingDinoStage: loaded %s", model_path)

    def _process(self, msg: Message, metrics: MetricsCollector) -> DetectionMessage | None:
        if not isinstance(msg, FrameTensorMessage):
            return None

        # Denormalize and reshape tensor (CHW -> HWC)
        import numpy as np
        from PIL import Image

        img_tensor = msg.tensor.squeeze(0).transpose(1, 2, 0)
        if img_tensor.max() <= 1.0:
            img_tensor = img_tensor * 255

        image = Image.fromarray(img_tensor.astype(np.uint8))
        text = ". ".join(self._text_queries) + "."

        with metrics.start_span(SpanType.MODEL_INFERENCE, "GroundingDINO inference"):
            inputs = self._processor(images=image, text=text, return_tensors="pt").to("mps")
            with torch.no_grad():
                outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"], strict=False
        ):
            boxes.append(
                BoundingBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    confidence=float(score),
                    class_id=0,
                    label=str(label),
                )
            )

        return DetectionMessage(boxes=boxes, timestamp=msg.timestamp)
