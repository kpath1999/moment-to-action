"""Initial stages for the YOLO → LLM baseline pipeline.

SensorStage        loads an image from disk → RawFrameMessage
PreprocessorStage  resizes + normalizes    → TensorMessage
YOLOStage          runs YOLO               → DetectionMessage
ReasoningStage     runs an LLM             → ReasoningMessage

Each stage is independent. Intent: swap or reorder them freely.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from moment_to_action.edgeperceive.core.messages import (
    RawFrameMessage,
    TensorMessage,
)
from moment_to_action.edgeperceive.preprocessors import (
    ImagePreprocessConfig,
    ImagePreprocessor,
)
from moment_to_action.edgeperceive.stages.base import Stage

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# 1. SensorStage
# ──────────────────────────────────────────────────────────────────


class SensorStage(Stage):
    """Loads an image from disk and emits a RawFrameMessage.

    In production this will read from a camera buffer instead.

    Input:  path string (passed via a bootstrap RawFrameMessage)
    Output: RawFrameMessage with the loaded frame
    """

    def process(self, msg: RawFrameMessage) -> RawFrameMessage | None:
        """Load an image from disk and return a RawFrameMessage."""
        try:
            import cv2

            frame = cv2.imread(msg.source)
            if frame is None:
                logger.error("SensorStage: could not read %s", msg.source)
                return None
            h, w = frame.shape[:2]
            logger.debug("SensorStage: loaded %s (%dx%d)", msg.source, w, h)
            return RawFrameMessage(
                frame=frame,
                timestamp=msg.timestamp or time.time(),
                source=msg.source,
            )
        except ImportError as err:
            msg = "opencv-python required: pip install opencv-python"
            raise RuntimeError(msg) from err


# ──────────────────────────────────────────────────────────────────
# 2. PreprocessorStage
# ──────────────────────────────────────────────────────────────────


class PreprocessorStage(Stage):
    """Resizes and normalizes a raw frame for a specific model.

    Wraps the existing ImagePreprocessor — config drives behaviour.

    Input:  RawFrameMessage
    Output: TensorMessage  (tensor shape depends on target model)
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (640, 640),
        *,
        letterbox: bool = True,
        channels_first: bool = True,
        mean: tuple = (0.0, 0.0, 0.0),
        std: tuple = (1.0, 1.0, 1.0),
    ) -> None:
        self._preprocessor = ImagePreprocessor(
            config=ImagePreprocessConfig(
                target_size=target_size,
                letterbox=letterbox,
                mean=mean,
                std=std,
            )
        )
        self._channels_first = channels_first

    def process(self, msg: RawFrameMessage) -> TensorMessage | None:
        """Preprocess a raw frame into a model-ready tensor."""
        img = RawFrameMessage(
            frame=msg.frame,
            timestamp=msg.timestamp,
            width=msg.frame.shape[1],
            height=msg.frame.shape[0],
        )
        processed = self._preprocessor.process(img)
        data = processed.data  # [H, W, C]
        if self._channels_first:
            data = np.transpose(data, (2, 0, 1))  # [H,W,C] → [C,H,W]
        tensor = data[np.newaxis, ...].astype(np.float32)  # → [1,C,H,W] or [1,H,W,C]
        return TensorMessage(
            tensor=tensor,
            original_size=processed.original_size,
            timestamp=msg.timestamp,
        )
