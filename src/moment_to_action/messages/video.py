"""Video-pipeline messages: tensors and clips."""

from __future__ import annotations

from numpy.typing import NDArray  # noqa: TC002

from ._base import BaseMessage


class FrameTensorMessage(BaseMessage):
    """Preprocessed video frame ready for model inference."""

    tensor: NDArray
    """Preprocessed image tensor as a NumPy array (CxHxW or HxWxC)."""

    original_size: tuple[int, int]
    """``(width, height)`` of the source frame before preprocessing."""


class VideoClipMessage(BaseMessage):
    """A temporal window of raw frames captured from a live stream or file.

    Assembled by a windowed stage (``Stage(window=clip_len, stride=...)``)
    consuming raw frames. Consumed by vision-language model stages.

    All frames share the same spatial dimensions (width x height) and are
    stored in capture order (oldest first).
    """

    frames: list[NDArray]
    """Ordered list of raw BGR frames (HxWxC, uint8).  Oldest frame first."""

    source: str = ""
    """Identifier of the originating sensor (camera index, device path, etc.)."""

    width: int = 0
    """Frame width in pixels; ``0`` when unknown."""

    height: int = 0
    """Frame height in pixels; ``0`` when unknown."""

    fps: float = 0.0
    """Capture frame-rate reported by the sensor; ``0.0`` when unknown."""

    question: str = ""
    """Task question for a downstream VLM stage. Lets one loaded model serve any question."""

    @property
    def num_frames(self) -> int:
        """Number of frames in the clip."""
        return len(self.frames)
