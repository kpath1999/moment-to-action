"""Unit tests for ClipBufferStage."""

from __future__ import annotations

import time

import numpy as np
import pytest

from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.messages.video import VideoClipMessage
from moment_to_action.stages.video._clip_buffer import ClipBufferStage


def _make_frame_msg(*, frame: np.ndarray | None = None, ts: float | None = None) -> RawFrameMessage:
    """Create a RawFrameMessage with sensible defaults."""
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return RawFrameMessage(
        frame=frame,
        timestamp=ts if ts is not None else time.time(),
        source="test",
        width=640,
        height=480,
    )


@pytest.mark.unit
class TestClipBufferStageInit:
    """Tests for ClipBufferStage initialization."""

    def test_default_construction(self) -> None:
        """Stage constructs with default clip_len=16, stride=clip_len."""
        stage = ClipBufferStage()
        assert stage._clip_len == 16
        assert stage._stride == 16
        assert stage._min_fps == 0.0

    def test_custom_clip_len_and_stride(self) -> None:
        """Stage accepts custom clip_len and stride."""
        stage = ClipBufferStage(clip_len=8, stride=4)
        assert stage._clip_len == 8
        assert stage._stride == 4

    def test_stride_defaults_to_clip_len(self) -> None:
        """Stride defaults to clip_len when not specified."""
        stage = ClipBufferStage(clip_len=10)
        assert stage._stride == 10

    def test_clip_len_less_than_one_raises(self) -> None:
        """clip_len < 1 raises ValueError."""
        with pytest.raises(ValueError, match="clip_len must be >= 1"):
            ClipBufferStage(clip_len=0)

    def test_stride_less_than_one_raises(self) -> None:
        """Stride < 1 raises ValueError."""
        with pytest.raises(ValueError, match="stride must be >= 1"):
            ClipBufferStage(clip_len=4, stride=0)


@pytest.mark.unit
class TestClipBufferStageProcess:
    """Tests for ClipBufferStage._process()."""

    def test_returns_none_while_buffering(self) -> None:
        """Returns None until clip_len frames have accumulated."""
        stage = ClipBufferStage(clip_len=4)
        for _ in range(3):
            result = stage._process(_make_frame_msg())
            assert result is None

    def test_emits_clip_when_buffer_full(self) -> None:
        """Emits a VideoClipMessage once clip_len frames are accumulated."""
        stage = ClipBufferStage(clip_len=4)
        for _ in range(3):
            stage._process(_make_frame_msg())
        result = stage._process(_make_frame_msg())
        assert isinstance(result, VideoClipMessage)
        assert len(result.frames) == 4

    def test_clip_contains_correct_frames(self) -> None:
        """Emitted clip frames are the same frames that were buffered."""
        stage = ClipBufferStage(clip_len=3)
        frames = []
        for i in range(3):
            frame = np.full((480, 640, 3), i, dtype=np.uint8)
            msg = _make_frame_msg(frame=frame)
            frames.append(frame)
            result = stage._process(msg)
        assert result is not None
        for i in range(3):
            np.testing.assert_array_equal(result.frames[i], frames[i])

    def test_stride_controls_subsequent_emissions(self) -> None:
        """After the first emission, next emission requires stride new frames."""
        stage = ClipBufferStage(clip_len=4, stride=2)
        # Fill buffer (4 frames -> emit)
        for _ in range(4):
            result = stage._process(_make_frame_msg())
        assert isinstance(result, VideoClipMessage)

        # Next frame: 1 new, need 2 -> no emit
        result = stage._process(_make_frame_msg())
        assert result is None

        # 2nd new frame -> emit
        result = stage._process(_make_frame_msg())
        assert isinstance(result, VideoClipMessage)

    def test_discards_dropped_frames(self) -> None:
        """Dropped frames (frame=None) are discarded, not buffered."""
        stage = ClipBufferStage(clip_len=2)
        dropped = RawFrameMessage(frame=None, timestamp=time.time(), source="test")
        result = stage._process(dropped)
        assert result is None
        assert len(stage._buffer) == 0

    def test_wrong_message_type_raises(self) -> None:
        """Passing a non-RawFrameMessage raises TypeError."""
        from moment_to_action.messages.vlm import ClassificationMessage

        stage = ClipBufferStage(clip_len=2)
        wrong_msg = ClassificationMessage(
            timestamp=time.time(), label="x", confidence=1.0, all_scores={"x": 1.0}
        )
        with pytest.raises(TypeError, match="expects RawFrameMessage"):
            stage._process(wrong_msg)

    def test_unexpected_none_frame_in_buffer_raises(self) -> None:
        """RuntimeError if a None frame is found in the buffer at emit time."""
        stage = ClipBufferStage(clip_len=3)
        # Manually inject messages including one with frame=None to trigger
        # the defensive check that normally can't be hit via the public API.
        stage._buffer.append(
            RawFrameMessage(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=1.0,
                source="test",
                width=640,
                height=480,
            )
        )
        stage._buffer.append(RawFrameMessage(frame=None, timestamp=2.0, source="test"))
        # Buffer now has 2 items. Sending one more valid frame brings it to 3
        # (clip_len) and triggers the emit. The None frame stays in the deque.
        stage._new_since_emit = 2
        stage._frames_seen = 2
        with pytest.raises(RuntimeError, match="Unexpected None frame in buffer"):
            stage._process(_make_frame_msg(ts=3.0))

    def test_fps_guard_discards_slow_clip(self) -> None:
        """Clip is discarded when effective fps < min_fps."""
        stage = ClipBufferStage(clip_len=4, min_fps=30.0)
        base_ts = 1000.0
        for i in range(4):
            # 1 second apart -> effective fps = 3/3 = 1 fps, far below 30
            result = stage._process(_make_frame_msg(ts=base_ts + i))
        assert result is None

    def test_fps_guard_passes_fast_clip(self) -> None:
        """Clip is emitted when effective fps >= min_fps."""
        stage = ClipBufferStage(clip_len=4, min_fps=10.0)
        base_ts = 1000.0
        for i in range(4):
            # 0.01s apart -> effective fps = 3/0.03 = 100 fps
            result = stage._process(_make_frame_msg(ts=base_ts + i * 0.01))
        assert isinstance(result, VideoClipMessage)

    def test_clip_metadata_from_latest_frame(self) -> None:
        """Emitted clip uses metadata from the most recent frame."""
        stage = ClipBufferStage(clip_len=2)
        stage._process(
            RawFrameMessage(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=1.0,
                source="cam0",
                width=640,
                height=480,
            )
        )
        result = stage._process(
            RawFrameMessage(
                frame=np.zeros((720, 1280, 3), dtype=np.uint8),
                timestamp=2.0,
                source="cam1",
                width=1280,
                height=720,
            )
        )
        assert result is not None
        assert result.source == "cam1"
        assert result.width == 1280
        assert result.height == 720


@pytest.mark.unit
class TestClipBufferStageReset:
    """Tests for ClipBufferStage.reset()."""

    def test_reset_clears_buffer(self) -> None:
        """Reset empties the buffer and resets counters."""
        stage = ClipBufferStage(clip_len=4)
        for _ in range(3):
            stage._process(_make_frame_msg())
        assert len(stage._buffer) == 3

        stage.reset()
        assert len(stage._buffer) == 0
        assert stage._new_since_emit == 0
        assert stage._has_emitted_once is False
        assert stage._frames_seen == 0

    def test_reset_allows_fresh_emission(self) -> None:
        """After reset, a new full buffer re-emits without stride delay."""
        stage = ClipBufferStage(clip_len=2, stride=2)
        # Fill and emit
        stage._process(_make_frame_msg())
        result = stage._process(_make_frame_msg())
        assert isinstance(result, VideoClipMessage)

        stage.reset()
        # Fill again — should emit immediately (first emission after reset)
        stage._process(_make_frame_msg())
        result = stage._process(_make_frame_msg())
        assert isinstance(result, VideoClipMessage)
