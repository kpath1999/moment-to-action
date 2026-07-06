"""Unit tests for LLMStage and its think/response router."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from moment_to_action.hardware import ComputeUnit, DataType, ModelType
from moment_to_action.messages.control import EndOfClipMessage
from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.messages.llm import GenerationMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.metrics import MetricsCollector, SpanType
from moment_to_action.models.llm._base import LlamaGGUFModel
from moment_to_action.stages.llm._llm import (
    LLMStage,
    _partial_tag_hold_len,
    _ThinkResponseRouter,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from moment_to_action.messages import Message

_FAKE_BACKENDS = {ComputeUnit.CPU: {"model": "model.gguf"}}


class _FakeLlamaGGUFModel(LlamaGGUFModel):
    """Fake LlamaGGUFModel whose stream() yields scripted tokens as a real generator."""

    def __init__(self, tokens: list[str]) -> None:
        """Store the tokens to yield and initialize call-tracking state."""
        super().__init__(
            "default", Path("/x"), ModelType.LLAMA_CPP, DataType.FP32, backends=_FAKE_BACKENDS
        )
        self.tokens = tokens
        self.closed = False
        self.last_prompt: str | None = None
        self.last_grammar: str | None = None

    def stream(self, prompt: str, *, grammar: str | None = None) -> Generator[str, None, None]:
        """Yield scripted tokens, recording the prompt/grammar and close state."""
        self.last_prompt = prompt
        self.last_grammar = grammar
        try:
            yield from self.tokens
        finally:
            self.closed = True


def _detection_msg(timestamp: float = 1.0, question: str = "") -> DetectionMessage:
    """Build an empty-detections DetectionMessage for testing."""
    return DetectionMessage(timestamp=timestamp, detections=[], question=question)


def _gen(msg: Message) -> GenerationMessage:
    """Assert *msg* is a GenerationMessage and return it, narrowing the type for mypy."""
    assert isinstance(msg, GenerationMessage)
    return msg


@pytest.mark.unit
class TestPartialTagHoldLen:
    """Tests for _partial_tag_hold_len()."""

    def test_no_overlap_returns_zero(self) -> None:
        """A buffer with no tag-prefix suffix holds nothing back."""
        assert _partial_tag_hold_len("hello", "<think>") == 0

    def test_partial_suffix_detected(self) -> None:
        """A buffer ending in a partial tag prefix holds back that suffix."""
        assert _partial_tag_hold_len("hello<thi", "<think>") == len("<thi")

    def test_full_tag_present_not_held(self) -> None:
        """A buffer containing the full tag doesn't need any hold (caller finds it directly)."""
        # _partial_tag_hold_len only checks the trailing suffix; a full tag in the
        # middle isn't a "partial" suffix candidate once it's already complete.
        assert _partial_tag_hold_len("<think>", "<think>") == 0


@pytest.mark.unit
class TestThinkResponseRouter:
    """Tests for _ThinkResponseRouter."""

    def test_starts_in_response_phase(self) -> None:
        """The router starts in the response phase."""
        router = _ThinkResponseRouter()
        assert router.phase == "response"

    def test_no_think_tag_stays_response(self) -> None:
        """Tokens with no <think> tag stay in the response phase throughout."""
        router = _ThinkResponseRouter()
        segments = router.feed("hello world")
        assert router.phase == "response"
        assert segments == [("response", "hello world")]

    def test_think_tag_switches_phase(self) -> None:
        """A <think> tag switches to the think phase and is swallowed."""
        router = _ThinkResponseRouter()
        segments = router.feed("<think>reasoning")
        assert router.phase == "think"
        assert segments == [("think", "reasoning")]

    def test_think_close_switches_back(self) -> None:
        """A </think> tag switches back to response and is swallowed."""
        router = _ThinkResponseRouter()
        router.feed("<think>reasoning")
        segments = router.feed("</think>answer")
        assert router.phase == "response"
        assert segments == [("response", "answer")]

    def test_full_roundtrip_across_many_small_tokens(self) -> None:
        """Feeding the full sequence one character at a time reconstructs think/response text."""
        router = _ThinkResponseRouter()
        raw = "<think>abc</think>xyz"
        think_text = ""
        resp_text = ""
        for ch in raw:
            for phase, seg in router.feed(ch):
                if phase == "think":
                    think_text += seg
                else:
                    resp_text += seg
        assert think_text == "abc"
        assert resp_text == "xyz"

    def test_tag_split_across_tokens(self) -> None:
        """A tag split across two feed() calls is still recognized correctly."""
        router = _ThinkResponseRouter()
        segments1 = router.feed("<thi")
        assert segments1 == []  # held back, could be a partial tag
        segments2 = router.feed("nk>reasoning")
        assert router.phase == "think"
        assert segments2 == [("think", "reasoning")]

    def test_multiple_transitions_in_one_token(self) -> None:
        """A single token containing both tags produces two correctly attributed segments."""
        router = _ThinkResponseRouter()
        segments = router.feed("<think>abc</think>xyz")
        assert segments == [("think", "abc"), ("response", "xyz")]


@pytest.mark.unit
class TestLLMStage:
    """Tests for LLMStage."""

    def test_yields_partial_and_final_messages(self) -> None:
        """Each token yields a partial GenerationMessage, ending with EndOfClipMessage."""
        model = _FakeLlamaGGUFModel(["Hello", " world"])
        stage = LLMStage(model)
        msg = _detection_msg(question="Is this safe?")

        results = list(stage.process(iter([msg])))

        assert len(results) == 4  # 2 partials + 1 final content message + end-of-generation
        gen_msgs = results[:3]
        assert all(isinstance(r, GenerationMessage) for r in gen_msgs)
        assert isinstance(results[3], EndOfClipMessage)
        assert _gen(results[0]).text == "Hello"
        assert _gen(results[1]).text == "Hello world"
        assert _gen(results[2]).text == "Hello world"
        assert all(_gen(r).type == "response" for r in gen_msgs)

    def test_no_think_block_stays_response_throughout(self) -> None:
        """A model with no <think> block emits type='response' for every message."""
        model = _FakeLlamaGGUFModel(["plain", " text"])
        stage = LLMStage(model)
        results = list(stage.process(iter([_detection_msg(question="Q?")])))
        gen_msgs = [r for r in results if isinstance(r, GenerationMessage)]
        assert gen_msgs
        assert all(_gen(r).type == "response" for r in gen_msgs)

    def test_think_block_split_into_think_then_response(self) -> None:
        """A <think>...</think> block is routed to type='think' then type='response'."""
        model = _FakeLlamaGGUFModel(["<think>", "reasoning", "</think>", "YES"])
        stage = LLMStage(model)
        results = list(stage.process(iter([_detection_msg(question="Q?")])))

        think_msgs = [r for r in results if isinstance(r, GenerationMessage) and r.type == "think"]
        resp_msgs = [
            r for r in results if isinstance(r, GenerationMessage) and r.type == "response"
        ]
        assert think_msgs
        assert resp_msgs
        assert think_msgs[-1].text == "reasoning"
        assert resp_msgs[-1].text == "YES"

    def test_prompt_includes_question(self) -> None:
        """The prompt built from the DetectionMessage includes the configured question."""
        model = _FakeLlamaGGUFModel(["tok"])
        stage = LLMStage(model)
        list(stage.process(iter([_detection_msg(question="Is this violent?")])))
        assert model.last_prompt is not None
        assert "Is this violent?" in model.last_prompt

    def test_grammar_forwarded_to_model_stream(self) -> None:
        """The configured grammar is forwarded to model.stream()."""
        model = _FakeLlamaGGUFModel(["YES"])
        stage = LLMStage(model, grammar='root ::= "YES" | "NO"')
        list(stage.process(iter([_detection_msg(question="Q?")])))
        assert model.last_grammar == 'root ::= "YES" | "NO"'

    def test_no_grammar_by_default(self) -> None:
        """Without an explicit grammar, None is forwarded to model.stream()."""
        model = _FakeLlamaGGUFModel(["tok"])
        stage = LLMStage(model)
        list(stage.process(iter([_detection_msg(question="Q?")])))
        assert model.last_grammar is None

    def test_non_detection_message_dropped(self) -> None:
        """A non-DetectionMessage input yields nothing and never touches the model."""
        model = _FakeLlamaGGUFModel(["tok"])
        stage = LLMStage(model)
        other = RawFrameMessage(frame=None, timestamp=time.time())
        results = list(stage.process(iter([other])))
        assert results == []
        assert model.last_prompt is None

    def test_ttft_recorded_on_metrics_span(self) -> None:
        """Streaming through the stage records ttft_ms via timed_stream on the model span."""
        model = _FakeLlamaGGUFModel(["a", "b"])
        metrics = MetricsCollector(session_id="test_llm_stage")
        stage = LLMStage(model, metrics=metrics)
        with metrics.start_trace():
            list(stage.process(iter([_detection_msg(question="Q?")])))

        stage_spans = [s for s in metrics.spans if s.type_ is SpanType.STAGE]
        assert len(stage_spans) == 1
        assert "ttft_ms" in stage_spans[0].metadata

    def test_model_stream_closed_on_early_break(self) -> None:
        """Breaking out of the consumer loop closes the model's underlying generator."""
        model = _FakeLlamaGGUFModel(["a", "b", "c"])
        stage = LLMStage(model)
        gen = stage.process(iter([_detection_msg(question="Q?")]))
        next(gen)
        gen.close()
        assert model.closed is True

    def test_load_calls_model_load(self) -> None:
        """load() forwards platform and unit to the wrapped model."""
        model = MagicMock(spec=LlamaGGUFModel)
        stage = LLMStage(model)
        platform = MagicMock()
        stage.load(platform, ComputeUnit.CPU)
        model.load.assert_called_once_with(platform, ComputeUnit.CPU)

    def test_load_without_unit_raises(self) -> None:
        """load() without a compute unit raises ValueError."""
        model = MagicMock(spec=LlamaGGUFModel)
        stage = LLMStage(model)
        with pytest.raises(ValueError, match="compute unit"):
            stage.load(MagicMock())

    def test_unload_calls_model_unload(self) -> None:
        """unload() delegates to the wrapped model."""
        model = MagicMock(spec=LlamaGGUFModel)
        stage = LLMStage(model)
        stage.unload()
        model.unload.assert_called_once()
