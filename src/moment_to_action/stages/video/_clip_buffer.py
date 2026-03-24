"""Clip-buffer stage — accumulates raw frames into a temporal window.

ClipBufferStage sits immediately after the sensor/SensorStage in the pipeline.
It collects individual :class:`~moment_to_action.messages.sensor.RawFrameMessage`
frames into a sliding window and emits a
:class:`~moment_to_action.messages.video.VideoClipMessage` once the buffer is
filled.

Design decisions
----------------
- **Stateful stage**: unlike most stages, ClipBufferStage owns mutable state
  (the frame ring-buffer).  A single instance should only be used by one
  pipeline / thread at a time.
- **Sliding vs. fixed window**: controlled by ``stride``.
  - ``stride == clip_len`` → non-overlapping (tumbling) windows: the buffer is
    cleared after each emission.
  - ``stride == 1`` → maximally-overlapping sliding window: every new frame
    triggers an emission.
  - Any value in between is valid.
- **Dropped frames**: ``RawFrameMessage.frame is None`` signals a dropped
  capture.  The stage discards these silently rather than inserting gaps.
- **Return None to short-circuit**: while the buffer is filling up (< ``clip_len``
  frames accumulated) the stage returns ``None`` so the rest of the pipeline
  is skipped for that tick.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.messages.video import VideoClipMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.messages import Message

logger = logging.getLogger(__name__)


class ClipBufferStage(Stage):
    """Collects raw frames into a temporal clip window.

    Args:
        clip_len: Number of frames in each emitted clip.  Must be >= 1.
        stride:   How many *new* frames must arrive between emissions.
            Defaults to ``clip_len`` (non-overlapping windows).
            Set to ``1`` for a maximally-sliding window.
        min_fps:  If the clip's effective frame-rate (``clip_len / duration``)
            falls below this value the clip is discarded (late-arriving bursts
            can otherwise skew temporal models).  ``0.0`` disables the check.

    Input:  :class:`~moment_to_action.messages.sensor.RawFrameMessage`
    Output: :class:`~moment_to_action.messages.video.VideoClipMessage` (once
            the buffer has ``clip_len`` valid frames), or ``None`` while filling.

    Example:
        >>> stage = ClipBufferStage(clip_len=16, stride=8)
        >>> # After 16 frames arrive, emits a VideoClipMessage.
        >>> # After every subsequent 8 new frames, emits another.
    """

    def __init__(
        self,
        clip_len: int = 16,
        stride: int | None = None,
        min_fps: float = 0.0,
    ) -> None:
        if clip_len < 1:
            msg = f"clip_len must be >= 1, got {clip_len}"
            raise ValueError(msg)
        self._clip_len = clip_len
        self._stride = stride if stride is not None else clip_len
        if self._stride < 1:
            msg = f"stride must be >= 1, got {self._stride}"
            raise ValueError(msg)
        self._min_fps = min_fps

        # Ring-buffer: always holds the most recent `clip_len` frames.
        self._buffer: deque[RawFrameMessage] = deque(maxlen=clip_len)
        # How many new frames have arrived since the last emission.
        self._new_since_emit: int = 0
        # First clip should emit as soon as the buffer is full.
        self._has_emitted: bool = False

    # ------------------------------------------------------------------
    # Stage interface
    # ------------------------------------------------------------------

    def _process(self, msg: Message) -> VideoClipMessage | None:
        """Accumulate *msg* and emit a clip when stride is reached.

        Returns:
            A :class:`~moment_to_action.messages.video.VideoClipMessage` once
            ``clip_len`` frames have been collected and every ``stride`` new
            frames thereafter.  Returns ``None`` while the buffer is filling
            or when the clip is rejected by the fps guard.
        """
        if not isinstance(msg, RawFrameMessage):
            type_name = type(msg).__name__
            err = f"ClipBufferStage expects RawFrameMessage, got {type_name}"
            raise TypeError(err)

        # Discard dropped frames.
        if msg.frame is None:
            logger.debug("ClipBufferStage: discarded dropped frame from %s", msg.source)
            return None

        self._buffer.append(msg)
        self._new_since_emit += 1

        # Not enough frames yet — keep accumulating.
        if len(self._buffer) < self._clip_len:
            logger.debug(
                "ClipBufferStage: buffering %d/%d frames",
                len(self._buffer),
                self._clip_len,
            )
            return None

        # Emit first clip immediately once the buffer is full. Apply stride only
        # for subsequent clip emissions.
        if self._has_emitted and self._new_since_emit < self._stride:
            return None

        # --- ready to emit ---
        self._new_since_emit = 0
        self._has_emitted = True
        frames = [f.frame for f in self._buffer]  # oldest first (deque order)

        # FPS guard: reject abnormally slow clips.
        _min_buffer_for_fps_check = 2
        if self._min_fps > 0.0 and len(self._buffer) >= _min_buffer_for_fps_check:
            oldest_ts = self._buffer[0].timestamp
            newest_ts = self._buffer[-1].timestamp
            duration = newest_ts - oldest_ts
            if duration > 0:
                effective_fps = (self._clip_len - 1) / duration
                if effective_fps < self._min_fps:
                    logger.warning(
                        "ClipBufferStage: clip fps %.1f < min_fps %.1f — discarding",
                        effective_fps,
                        self._min_fps,
                    )
                    return None

        # Use the most recent frame's metadata as clip provenance.
        latest: RawFrameMessage = self._buffer[-1]
        logger.debug(
            "ClipBufferStage: emitting clip of %d frames from %s",
            self._clip_len,
            latest.source,
        )
        return VideoClipMessage(
            frames=frames,  # type: ignore[arg-type]  # frames are non-None (checked above)
            timestamp=latest.timestamp,
            source=latest.source,
            width=latest.width,
            height=latest.height,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the internal frame buffer and stride counter.

        Call this between clips when you want a hard cut (e.g. between
        scenes or camera switches).
        """
        self._buffer.clear()
        self._new_since_emit = 0
        self._has_emitted = False
        logger.debug("ClipBufferStage: buffer reset")
