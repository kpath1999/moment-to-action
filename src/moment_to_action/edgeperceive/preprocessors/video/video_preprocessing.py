"""preprocessors/vision.py

Vision preprocessors - extend BasePreprocessor.

ImagePreprocessor(BasePreprocessor[ImageInput, ProcessedFrame])
VideoPreprocessor(BasePreprocessor[VideoInput, list[ProcessedFrame]])

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
from dataclasses import dataclass

import numpy as np

from moment_to_action.edgeperceive.core.messages import RawFrameMessage
from moment_to_action.edgeperceive.hardware.types import ComputeUnit
from moment_to_action.edgeperceive.preprocessors.base import BasePreprocessor, BufferSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
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


@dataclass
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


@dataclass
class VideoPreprocessConfig:
    """Tunable parameters for video clip preprocessing."""

    target_size: tuple[int, int] = (256, 256)
    target_fps: float | None = None  # None = keep original fps
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    to_rgb: bool = True
    max_frames: int | None = None  # clip to N frames


# ---------------------------------------------------------------------------
# ImagePreprocessor
# ---------------------------------------------------------------------------


class ImagePreprocessor(BasePreprocessor[RawFrameMessage, ProcessedFrame]):
    """Preprocesses a single ImageInput into a ProcessedFrame.

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
        metrics=None,
    ):
        self.config = config or ImagePreprocessConfig()
        super().__init__(compute_unit=compute_unit, metrics=metrics)

    # ------------------------------------------------------------------
    # BasePreprocessor interface
    # ------------------------------------------------------------------

    def _validate(self, input: RawFrameMessage) -> None:
        if input.frame is None:
            raise ValueError("ImageInput has no frame data")
        if input.frame.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D frame, got shape {input.frame.shape}")
        if input.width <= 0 or input.height <= 0:
            raise ValueError(f"Invalid dimensions: {input.width}x{input.height}")

    def _allocate_buffers(self) -> None:
        if not hasattr(self, "config"):
            return
        # Intermediate buffer for the resize step (always target_size)
        rh, rw = self.config.target_size
        self._buffers.register("resize", BufferSpec((rh, rw, 3), np.float32))
        # Final output buffer — crop_size if set, otherwise same as resize
        fh, fw = self.config.crop_size or self.config.target_size
        self._buffers.register("frame", BufferSpec((fh, fw, 3), np.float32))

    def _process(self, input: RawFrameMessage) -> ProcessedFrame:
        """Rgb → resize → [center crop] → normalize, each step dispatched."""
        if self.config.to_rgb:
            input = self._dispatch(self._to_rgb, input)

        frame = self._dispatch(
            self._resize,
            input,
            self.config.target_size,
            self.config.letterbox,
        )

        if self.config.crop_size is not None:
            frame = self._dispatch(self._center_crop, frame, self.config.crop_size)

        frame = self._dispatch(
            self._normalize,
            frame,
            self.config.mean,
            self.config.std,
        )

        return frame

    # ------------------------------------------------------------------
    # Vision-specific operations
    # Called via self._dispatch() - routable to GPU when available
    # ------------------------------------------------------------------

    def _to_rgb(self, image: RawFrameMessage) -> RawFrameMessage:
        """Convert BGR (OpenCV default) to RGB."""
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
            raise ValueError(
                f"Frame {fh}x{fw} is smaller than crop size {ch}x{cw}. "
                f"Increase target_size so it is >= crop_size."
            )
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
        letterbox: bool,
    ) -> ProcessedFrame:
        """Resize to target_size. Uses intermediate buffer separate from crop buffer."""
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
        self.config = config
        rh, rw = config.target_size
        self._buffers.register("resize", BufferSpec((rh, rw, 3), np.float32), overwrite=True)
        fh, fw = config.crop_size or config.target_size
        self._buffers.register("frame", BufferSpec((fh, fw, 3), np.float32), overwrite=True)
        logger.info(
            f"ImagePreprocessor reconfigured: resize={config.target_size} crop={config.crop_size}"
        )
