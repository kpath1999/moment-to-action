"""Integration tests for SmolVLM2 pipeline.

Tests the ClipBufferStage -> SmolVLM2Stage pipeline end-to-end using frames
read from images/smoke_test.mp4.  The model is mocked so these tests run
quickly without downloading model weights.

What is actually exercised:
- Real OpenCV frame extraction from smoke_test.mp4
- Real ClipBufferStage accumulation and clip emission
- Real SmolVLM2Stage._process routing (VideoClipMessage -> ClassificationMessage)
- Real utils.video.sample_frames and to_pil_rgb on real video frames
- Real Pipeline.run message plumbing and latency stamping
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import cv2
import pytest
import torch

from moment_to_action.messages import ClassificationMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.models import ModelManager
from moment_to_action.stages import Pipeline
from moment_to_action.stages.video import ClipBufferStage
from moment_to_action.stages.vlm import SmolVLM2Stage

_SMOKE_VIDEO = Path(__file__).parent.parent.parent / "images" / "smoke_test.mp4"


def _load_frames(video_path: Path, max_frames: int = 64) -> list[RawFrameMessage]:
    """Read up to *max_frames* frames from *video_path*."""
    if not video_path.exists():
        msg = f"Smoke-test video not found: {video_path}"
        raise FileNotFoundError(msg)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Cannot open video: {video_path}"
        raise OSError(msg)

    messages: list[RawFrameMessage] = []
    try:
        while len(messages) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            messages.append(
                RawFrameMessage(
                    frame=frame,
                    timestamp=time.time(),
                    source=str(video_path),
                    width=w,
                    height=h,
                )
            )
    finally:
        cap.release()

    return messages


def _make_pipeline(
    clip_len: int = 32,
    stride: int = 32,
    prompt: str = "Describe the scene.",
) -> Pipeline:
    """Return a Pipeline with a real ClipBufferStage and a mocked SmolVLM2Stage.

    The model weights are never downloaded: AutoProcessor and
    AutoModelForImageTextToText are replaced by mocks that return a fixed caption.
    """
    backend = mock.MagicMock()
    policy = mock.MagicMock()
    policy.device = torch.device("cpu")
    policy.dtype = torch.float32
    backend.resolve_torch_policy.return_value = policy

    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/smolvlm2_mock")

    mock_model = mock.MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.device = "cpu"
    mock_model.eval.return_value = mock_model
    mock_model.generate.return_value = torch.zeros(1, 20, dtype=torch.long)

    mock_processor = mock.MagicMock()
    mock_processor.apply_chat_template.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
    }
    mock_processor.batch_decode.return_value = ["A person is walking down the street."]

    with (
        mock.patch(
            "moment_to_action.stages.vlm._smolvlm2.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ),
        mock.patch(
            "moment_to_action.stages.vlm._smolvlm2.AutoModelForImageTextToText.from_pretrained",
            return_value=mock_model,
        ),
    ):
        stage = SmolVLM2Stage(backend=backend, manager=manager, prompt=prompt)

    stage._processor = mock_processor
    stage._model = mock_model

    return Pipeline(stages=[ClipBufferStage(clip_len=clip_len, stride=stride), stage])


@pytest.mark.integration
def test_smoke_video_is_readable() -> None:
    """Sanity check: smoke_test.mp4 is readable and has at least 32 frames."""
    assert _SMOKE_VIDEO.exists(), f"Smoke video missing: {_SMOKE_VIDEO}"
    frames = _load_frames(_SMOKE_VIDEO, max_frames=128)
    assert len(frames) >= 32, f"smoke_test.mp4 has only {len(frames)} frames; need >= 32."


@pytest.mark.integration
def test_smolvlm2_pipeline_produces_classification_message() -> None:
    """Pipeline returns at least one ClassificationMessage from smoke_test.mp4.

    Uses a mocked model so no model download is required.  Verifies that the
    real pipeline plumbing (ClipBufferStage, SmolVLM2Stage routing, Pipeline.run
    latency stamping) works end-to-end with real video frames.
    """
    frames = _load_frames(_SMOKE_VIDEO, max_frames=64)
    assert len(frames) >= 32, f"Smoke video has only {len(frames)} frames; need >= 32."

    pipeline = _make_pipeline(clip_len=32, stride=32)

    results: list[ClassificationMessage] = []
    for msg in frames:
        result = pipeline.run(msg)
        if isinstance(result, ClassificationMessage):
            results.append(result)

    assert len(results) >= 1, "Pipeline should produce at least one ClassificationMessage."
    first = results[0]
    assert isinstance(first.label, str)
    assert len(first.label) > 0, "Label must be non-empty."
    assert first.confidence == 1.0
    assert first.latency_ms > 0


@pytest.mark.integration
def test_smolvlm2_pipeline_with_custom_prompt() -> None:
    """Pipeline works with a custom prompt."""
    frames = _load_frames(_SMOKE_VIDEO, max_frames=64)
    pipeline = _make_pipeline(
        clip_len=32,
        stride=32,
        prompt="What is the main activity visible in these frames?",
    )

    results: list[ClassificationMessage] = []
    for msg in frames:
        result = pipeline.run(msg)
        if isinstance(result, ClassificationMessage):
            results.append(result)

    assert len(results) >= 1
    assert len(results[0].label) > 0


@pytest.mark.integration
def test_smolvlm2_pipeline_shorter_clip_len_produces_more_clips() -> None:
    """Shorter clip_len produces more clips from the same video.

    With clip_len=16 and stride=16, 64 frames should produce at least 2 clips.
    """
    frames = _load_frames(_SMOKE_VIDEO, max_frames=64)
    pipeline = _make_pipeline(clip_len=16, stride=16)

    results: list[ClassificationMessage] = []
    for msg in frames:
        result = pipeline.run(msg)
        if isinstance(result, ClassificationMessage):
            results.append(result)

    assert len(results) >= 2, f"Expected >= 2 clips at clip_len=16, got {len(results)}"
    for r in results:
        assert len(r.label) > 0


@pytest.mark.integration
def test_smolvlm2_pipeline_frames_flow_through_clip_buffer() -> None:
    """Frames accumulate in ClipBufferStage before reaching SmolVLM2Stage.

    The first 31 frames should produce None (clip not full yet).
    Frame 32 (index 31) triggers the first ClassificationMessage.
    """
    frames = _load_frames(_SMOKE_VIDEO, max_frames=64)
    pipeline = _make_pipeline(clip_len=32, stride=32)

    results_by_frame: list[tuple[int, object]] = []
    for i, msg in enumerate(frames):
        result = pipeline.run(msg)
        if result is not None:
            results_by_frame.append((i, result))

    assert results_by_frame, "No results produced"
    first_frame_idx, first_result = results_by_frame[0]
    assert first_frame_idx == 31, f"Expected first result at frame 31, got {first_frame_idx}"
    assert isinstance(first_result, ClassificationMessage)
