"""Oracle SigLIP classification stage.

OracleSigLipStage runs SigLIP to extract ground truth embeddings
and output classification scores.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModel, AutoProcessor

from moment_to_action.messages import ClassificationMessage, FrameTensorMessage
from moment_to_action.metrics._types import SpanType
from moment_to_action.models import ModelID, ModelManager
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class OracleSigLipStage(Stage):
    """Runs SigLIP for oracle ground truth classification."""

    def __init__(self, text_prompts: list[str], manager: ModelManager) -> None:
        super().__init__()
        model_path = manager.get_path(ModelID.SIGLIP_SO400M)
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path).to("mps")
        self._text_prompts = text_prompts
        logger.info("OracleSigLipStage: loaded %s", model_path)

    def _process(self, msg: Message, metrics: MetricsCollector) -> ClassificationMessage | None:
        if not isinstance(msg, FrameTensorMessage):
            return None

        # Denormalize and reshape tensor (CHW -> HWC)
        import numpy as np
        from PIL import Image

        img_tensor = msg.tensor.squeeze(0).transpose(1, 2, 0)
        if img_tensor.max() <= 1.0:
            img_tensor = img_tensor * 255

        image = Image.fromarray(img_tensor.astype(np.uint8))

        with metrics.start_span(SpanType.MODEL_INFERENCE, "SigLIP inference"):
            inputs = self._processor(
                text=self._text_prompts, images=image, padding="max_length", return_tensors="pt"
            ).to("mps")
            with torch.no_grad():
                outputs = self._model(**inputs)

            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image).cpu().numpy().tolist()[0]

        best_idx = np.argmax(probs)
        label = self._text_prompts[best_idx]
        confidence = float(probs[best_idx])

        return ClassificationMessage(
            label=label,
            confidence=confidence,
            all_scores={p: float(s) for p, s in zip(self._text_prompts, probs, strict=False)},
            timestamp=msg.timestamp,
        )
