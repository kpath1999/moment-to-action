"""Unit tests for VLMDescriptionStage."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from moment_to_action.hardware import ComputeUnit, DataType, ModelType
from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.messages.llm import GenerationMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.messages.video import VideoClipMessage
from moment_to_action.metrics import MetricsCollector, SpanType
from moment_to_action.models.vlm._base import LlamaVLModel
from moment_to_action.stages.vlm._description import VLMDescriptionStage

if TYPE_CHECKING:
    from collections.abc import Generator

    from moment_to_action.messages import Message

_FAKE_BACKENDS = {ComputeUnit.CPU: {"model": "model.gguf", "mmproj": "mmproj.gguf"}}


class _FakeLlamaVLModel(LlamaVLModel):
    """Fake LlamaVLModel whose stream() yields scripted tokens as a real generator."""

    def __init__(self, tokens: list[str]) -> None:
        """Store the tokens to yield and initialize call-tracking state."""
        super().__init__(
            "default", Path("/x"), ModelType.LLAMA_CPP, DataType.FP32, backends=_FAKE_BACKENDS
        )
        self.tokens = tokens
        self.closed = False
        self.last_inputs: tuple[str, list[str]] | None = None
        self.last_grammar: str | None = None

    def stream(  # type: ignore[override]
        self, inputs: tuple[str, list[str]], *, grammar: str | None = None
    ) -> Generator[str, None, None]:
        """Yield scripted tokens, recording the inputs/grammar and close state."""
        self.last_inputs = inputs
        self.last_grammar = grammar
        try:
            yield from self.tokens
        finally:
            self.closed = True


def _frame() -> np.ndarray:
    """Return a small BGR frame for testing."""
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _gen(msg: Message) -> GenerationMessage:
    """Assert *msg* is a GenerationMessage and return it, narrowing the type for mypy."""
    assert isinstance(msg, GenerationMessage)
    return msg


@pytest.mark.unit
class TestVLMDescriptionStage:
    """Tests for VLMDescriptionStage."""

    def test_single_frame_message_yields_messages(self) -> None:
        """A RawFrameMessage with a valid frame streams a description."""
        model = _FakeLlamaVLModel(["A", " dog"])
        stage = VLMDescriptionStage(model)
        msg = RawFrameMessage(
            frame=_frame(), timestamp=time.time(), question="Describe this scene."
        )

        results = list(stage.process(iter([msg])))

        assert len(results) == 3  # 2 partials + final
        assert all(isinstance(r, GenerationMessage) for r in results)
        assert all(_gen(r).type == "response" for r in results)
        assert _gen(results[-1]).done is True
        assert _gen(results[-1]).text == "A dog"

    def test_dropped_raw_frame_yields_nothing(self) -> None:
        """A RawFrameMessage with frame=None is dropped."""
        model = _FakeLlamaVLModel(["tok"])
        stage = VLMDescriptionStage(model)
        msg = RawFrameMessage(frame=None, timestamp=time.time(), question="Q?")
        results = list(stage.process(iter([msg])))
        assert results == []
        assert model.last_inputs is None

    def test_video_clip_message_passes_all_frames(self) -> None:
        """A VideoClipMessage's frames are all base64-encoded and passed to the model."""
        model = _FakeLlamaVLModel(["desc"])
        stage = VLMDescriptionStage(model)
        clip = VideoClipMessage(
            frames=[_frame(), _frame(), _frame()], timestamp=time.time(), question="Describe."
        )

        list(stage.process(iter([clip])))

        assert model.last_inputs is not None
        task, b64_frames = model.last_inputs
        assert task == "Describe."
        assert len(b64_frames) == 3

    def test_non_frame_message_yields_nothing(self) -> None:
        """A DetectionMessage (wrong type) is dropped."""
        model = _FakeLlamaVLModel(["tok"])
        stage = VLMDescriptionStage(model)
        msg = DetectionMessage(timestamp=time.time(), detections=[])
        results = list(stage.process(iter([msg])))
        assert results == []

    def test_grammar_forwarded(self) -> None:
        """The configured grammar is forwarded to model.stream()."""
        model = _FakeLlamaVLModel(["tok"])
        stage = VLMDescriptionStage(model, grammar="root ::= .*")
        list(
            stage.process(
                iter([RawFrameMessage(frame=_frame(), timestamp=time.time(), question="Q?")])
            )
        )
        assert model.last_grammar == "root ::= .*"

    def test_metrics_span_recorded(self) -> None:
        """Processing records a STAGE span via the metrics collector."""
        model = _FakeLlamaVLModel(["a", "b"])
        metrics = MetricsCollector(session_id="test_vlm_stage")
        stage = VLMDescriptionStage(model, metrics=metrics)
        with metrics.start_trace():
            list(
                stage.process(
                    iter([RawFrameMessage(frame=_frame(), timestamp=time.time(), question="Q?")])
                )
            )

        stage_spans = [s for s in metrics.spans if s.type_ is SpanType.STAGE]
        assert len(stage_spans) == 1

    def test_early_close_closes_model_stream(self) -> None:
        """Breaking early closes the underlying VLM stream generator."""
        model = _FakeLlamaVLModel(["a", "b", "c"])
        stage = VLMDescriptionStage(model)
        gen = stage.process(
            iter([RawFrameMessage(frame=_frame(), timestamp=time.time(), question="Q?")])
        )
        next(gen)
        gen.close()
        assert model.closed is True

    def test_load_calls_model_load(self) -> None:
        """load() forwards platform and unit to the wrapped model."""
        model = MagicMock(spec=LlamaVLModel)
        stage = VLMDescriptionStage(model)
        platform = MagicMock()
        stage.load(platform, ComputeUnit.CPU)
        model.load.assert_called_once_with(platform, ComputeUnit.CPU)

    def test_load_without_unit_raises(self) -> None:
        """load() without a compute unit raises ValueError."""
        model = MagicMock(spec=LlamaVLModel)
        stage = VLMDescriptionStage(model)
        with pytest.raises(ValueError, match="compute unit"):
            stage.load(MagicMock())

    def test_unload_calls_model_unload(self) -> None:
        """unload() delegates to the wrapped model."""
        model = MagicMock(spec=LlamaVLModel)
        stage = VLMDescriptionStage(model)
        stage.unload()
        model.unload.assert_called_once()
