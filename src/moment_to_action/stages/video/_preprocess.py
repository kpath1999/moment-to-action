"""Vision preprocessors — extend BasePreprocessor.

Merged from video_preprocessing.py (configs + ImagePreprocessor) and
vision_preprocess_stage.py (PreprocessorStage).

ImagePreprocessor(BasePreprocessor[RawFrameMessage, ProcessedFrame])

Modality-level operations (live here, shared by MobileCLIP and YOLO):
  - Color space conversion
  - Resize / letterbox
  - Pixel normalization

Model-specific operations (stay inside model classes):
  - MobileCLIP's text-image embedding preparation
  - YOLO's anchor-relative coordinate decoding
  - MoViNet's temporal frame stacking
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import attrs
import numpy as np

from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.messages.video import FrameTensorMessage  # renamed from TensorMessage
from moment_to_action.stages._base import Stage
from moment_to_action.stages._preprocess import BasePreprocessor
from moment_to_action.utils import BufferSpec

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics._collector import MetricsCollector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@attrs.define
class ProcessedFrame:
    """Single preprocessed frame ready for model-specific tensor conversion.

    Handed off from ImagePreprocessor to a model's _preprocess().
    """

    data: np.ndarray  # HxWxC float32, normalized
    original_size: tuple  # (H, W) before resize
    timestamp: float


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@attrs.define
class ImagePreprocessConfig:
    """Tunable parameters for image preprocessing.

    MobileCLIP-S2: target_size=(256,256), crop_size=(224,224), mean/std=(0,0,0)/(1,1,1)
    YOLO:          target_size=(640,640), letterbox=True
    MoViNet:       target_size=(172,172), standard ImageNet mean/std
    """

    target_size: tuple[int, int] = (256, 256)  # (H, W) — resize to this first
    crop_size: tuple[int, int] | None = None  # then center-crop to this; None = no crop
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    to_rgb: bool = True
    letterbox: bool = False  # preserve aspect ratio + pad


# ---------------------------------------------------------------------------
# ImagePreprocessor
# ---------------------------------------------------------------------------


class ImagePreprocessor(BasePreprocessor[RawFrameMessage, ProcessedFrame]):
    """Preprocesses a single RawFrameMessage into a ProcessedFrame.

    Extends BasePreprocessor - inherits:
      - BufferPool (pre-allocated frame buffer)
      - ComputeDispatcher (_dispatch routes to GPU/DSP or CPU)
      - process() timing + metrics wrapper

    Usage:
        prep = ImagePreprocessor(
            config=ImagePreprocessConfig(target_size=(256, 256)),
        )
        frame = prep.process(image_input)   # returns ProcessedFrame
    """

    def __init__(
        self,
        config: ImagePreprocessConfig | None = None,
        compute_unit: ComputeUnit = ComputeUnit.CPU,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._config = config or ImagePreprocessConfig()
        super().__init__(compute_unit=compute_unit, metrics=metrics)

    # ------------------------------------------------------------------
    # BasePreprocessor interface
    # ------------------------------------------------------------------

    def _validate(self, data: RawFrameMessage) -> None:
        if data.frame is None:
            msg = "ImageInput has no frame data"
            raise ValueError(msg)
        if data.frame.ndim not in (2, 3):
            msg = f"Expected 2D or 3D frame, got shape {data.frame.shape}"
            raise ValueError(msg)
        if data.width <= 0 or data.height <= 0:
            msg = f"Invalid dimensions: {data.width}x{data.height}"
            raise ValueError(msg)

    def _allocate_buffers(self) -> None:
        if not hasattr(self, "_config"):
            return
        # Intermediate buffer for the resize step (always target_size)
        rh, rw = self._config.target_size
        self._buffers.register("resize", BufferSpec((rh, rw, 3), np.float32))
        # Final output buffer — crop_size if set, otherwise same as resize
        fh, fw = self._config.crop_size or self._config.target_size
        self._buffers.register("frame", BufferSpec((fh, fw, 3), np.float32))

    def _process(self, data: RawFrameMessage) -> ProcessedFrame:
        """Rgb → resize → [center crop] → normalize, each step dispatched."""
        if self._config.to_rgb:
            data = self._dispatch(self._to_rgb, data)

        frame = self._dispatch(
            self._resize,
            data,
            self._config.target_size,
            self._config.letterbox,
        )

        if self._config.crop_size is not None:
            frame = self._dispatch(self._center_crop, frame, self._config.crop_size)

        frame = self._dispatch(
            self._normalize,
            frame,
            self._config.mean,
            self._config.std,
        )

        return frame  # noqa: RET504

    # ------------------------------------------------------------------
    # Vision-specific operations
    # Called via self._dispatch() - routable to GPU when available
    # ------------------------------------------------------------------

    def _to_rgb(self, image: RawFrameMessage) -> RawFrameMessage:
        """Convert BGR (OpenCV default) to RGB."""
        if image.frame is None:
            return image
        try:
            import cv2

            rgb = cv2.cvtColor(image.frame, cv2.COLOR_BGR2RGB)
        except ImportError:
            rgb = image.frame[..., ::-1].copy()
        return RawFrameMessage(
            frame=rgb, timestamp=image.timestamp, width=image.width, height=image.height
        )

    def _center_crop(self, frame: ProcessedFrame, crop_size: tuple[int, int]) -> ProcessedFrame:
        """Crop the center crop_size (H, W) pixels from a frame.

        Assumes frame.data is already resized to >= crop_size in both dims.
        Matches torchvision.transforms.CenterCrop behaviour.
        """
        ch, cw = crop_size
        fh, fw = frame.data.shape[:2]
        if fh < ch or fw < cw:
            msg = (
                f"Frame {fh}x{fw} is smaller than crop size {ch}x{cw}. "
                f"Increase target_size so it is >= crop_size."
            )
            raise ValueError(msg)
        top = (fh - ch) // 2
        left = (fw - cw) // 2
        cropped = frame.data[top : top + ch, left : left + cw].copy()
        return ProcessedFrame(
            data=cropped,
            original_size=frame.original_size,
            timestamp=frame.timestamp,
        )

    def _resize(
        self,
        image: RawFrameMessage,
        target_size: tuple[int, int],
        letterbox: bool,  # noqa: FBT001
    ) -> ProcessedFrame:
        """Resize to target_size. Uses intermediate buffer separate from crop buffer."""
        if image.frame is None:
            msg = "Cannot resize a RawFrameMessage with frame=None"
            raise ValueError(msg)
        try:
            import cv2

            if letterbox:
                frame = self._letterbox_cv2(image.frame, target_size)
            else:
                frame = cv2.resize(image.frame, (target_size[1], target_size[0]))
        except ImportError:
            frame = self._resize_numpy(image.frame, target_size)

        # Use a dedicated resize buffer (not the final output buffer which may be crop_size)
        resize_buf = self._buffers.get_or_register(
            "resize", BufferSpec((target_size[0], target_size[1], 3), np.float32)
        )
        np.copyto(resize_buf, frame.astype(np.float32))

        return ProcessedFrame(
            data=resize_buf.copy(),
            original_size=(image.height, image.width),
            timestamp=image.timestamp,
        )

    def _normalize(
        self,
        frame: ProcessedFrame,
        mean: tuple,
        std: tuple,
    ) -> ProcessedFrame:
        """Normalize pixel values with mean/std. Scale [0,255]→[0,1] first."""
        data = frame.data.copy()
        if data.max() > 1.0:
            data /= 255.0
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr = np.array(std, dtype=np.float32)
        data = (data - mean_arr) / std_arr
        return ProcessedFrame(
            data=data, original_size=frame.original_size, timestamp=frame.timestamp
        )

    def _letterbox_cv2(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize preserving aspect ratio, pad remainder. Used by YOLO."""
        import cv2

        h, w = image.shape[:2]
        th, tw = target_size
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((th, tw, 3), 114, dtype=np.uint8)
        pad_top = (th - new_h) // 2
        pad_left = (tw - new_w) // 2
        padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
        return padded

    def _resize_numpy(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Numpy-only nearest-neighbor fallback."""
        th, tw = target_size
        h, w = image.shape[:2]
        row_idx = (np.arange(th) * h / th).astype(int)
        col_idx = (np.arange(tw) * w / tw).astype(int)
        return image[row_idx][:, col_idx]

    def reconfigure(self, config: ImagePreprocessConfig) -> None:
        """Swap config and re-allocate buffers if sizes changed."""
        self._config = config
        rh, rw = config.target_size
        self._buffers.register("resize", BufferSpec((rh, rw, 3), np.float32), overwrite=True)
        fh, fw = config.crop_size or config.target_size
        self._buffers.register("frame", BufferSpec((fh, fw, 3), np.float32), overwrite=True)
        logger.info(
            "ImagePreprocessor reconfigured: resize=%s crop=%s",
            config.target_size,
            config.crop_size,
        )


# ---------------------------------------------------------------------------
# PreprocessorStage
# ---------------------------------------------------------------------------


class PreprocessorStage(Stage):
    """Resizes and normalizes a raw frame for a specific model.

    Wraps ImagePreprocessor — config drives behaviour.

    Input:  RawFrameMessage
    Output: FrameTensorMessage  (was TensorMessage — renamed to FrameTensorMessage)
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

    def _process(self, msg: Message) -> FrameTensorMessage | None:
        """Preprocess a raw frame into a model-ready tensor.

        Returns a FrameTensorMessage (previously called TensorMessage).
        """
        if not isinstance(msg, RawFrameMessage):
            err = f"PreprocessorStage expects RawFrameMessage, got {type(msg).__name__}"
            raise TypeError(err)
        if msg.frame is None:
            return None
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
        # NOTE: output type is FrameTensorMessage (renamed from TensorMessage)
        return FrameTensorMessage(
            tensor=tensor,
            original_size=processed.original_size,
            timestamp=msg.timestamp,
        )
