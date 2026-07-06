"""Unit tests for DecisionStage, including the early-abort GeneratorExit contract."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from moment_to_action.hardware import ComputeUnit, DataType, ModelType
from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.messages.llm import (
    DecisionMessage,
    DecisionReasoningMessage,
    EndOfGenerationMessage,
)
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models.llm._base import LlamaGGUFModel
from moment_to_action.pipeline import Pipeline
from moment_to_action.prompting import YES_NO_GRAMMAR
from moment_to_action.stages.llm._decision import DecisionStage, _strip_decision_prefix
from moment_to_action.stages.llm._llm import LLMStage

if TYPE_CHECKING:
    from collections.abc import Generator

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

    def stream(self, prompt: str, *, grammar: str | None = None) -> Generator[str, None, None]:
        """Yield scripted tokens, tracking whether the generator was closed early."""
        try:
            yield from self.tokens
        finally:
            self.closed = True


def _detection_msg(timestamp: float = 1.0) -> DetectionMessage:
    """Build an empty-detections DetectionMessage for testing."""
    return DetectionMessage(timestamp=timestamp, detections=[])


@pytest.mark.unit
class TestStripDecisionPrefix:
    """Tests for _strip_decision_prefix()."""

    def test_strips_leading_yes(self) -> None:
        """A leading YES token is stripped along with trailing punctuation."""
        assert _strip_decision_prefix("YES, because of X") == "because of X"

    def test_strips_leading_no(self) -> None:
        """A leading NO token is stripped."""
        assert _strip_decision_prefix("NO further action needed") == "further action needed"

    def test_no_prefix_returns_unchanged(self) -> None:
        """Text without a recognized prefix is returned unchanged (aside from lstrip)."""
        assert _strip_decision_prefix("unrelated text") == "unrelated text"


@pytest.mark.unit
class TestDecisionStage:
    """Tests for DecisionStage, driven directly against LLMStage output."""

    def test_emits_decision_once_response_is_unambiguous(self) -> None:
        """A DecisionMessage is emitted as soon as YES/NO is unambiguous."""
        model = _FakeLlamaGGUFModel(["YES", ", because of reasons"])
        llm_stage = LLMStage(model, grammar=YES_NO_GRAMMAR)
        decision_stage = DecisionStage()

        stream = decision_stage.process(llm_stage.process(iter([_detection_msg()])))
        results = list(stream)

        decisions = [r for r in results if isinstance(r, DecisionMessage)]
        assert len(decisions) == 1
        assert decisions[0].decision == "yes"

    def test_emits_reasoning_after_decision(self) -> None:
        """DecisionReasoningMessage partials follow the DecisionMessage."""
        model = _FakeLlamaGGUFModel(["YES", ", because of reasons"])
        llm_stage = LLMStage(model, grammar=YES_NO_GRAMMAR)
        decision_stage = DecisionStage()

        stream = decision_stage.process(llm_stage.process(iter([_detection_msg()])))
        results = list(stream)

        reasoning = [r for r in results if isinstance(r, DecisionReasoningMessage)]
        assert reasoning
        assert "because of reasons" in reasoning[-1].text
        assert isinstance(results[-1], EndOfGenerationMessage)

    def test_no_decision_yields_nothing(self) -> None:
        """No DecisionMessage is emitted while the decision is still ambiguous."""
        model = _FakeLlamaGGUFModel(["I", " am", " thinking"])
        llm_stage = LLMStage(model)
        decision_stage = DecisionStage()

        stream = decision_stage.process(llm_stage.process(iter([_detection_msg()])))
        results = list(stream)

        assert not any(isinstance(r, DecisionMessage) for r in results)

    def test_non_response_type_message_dropped(self) -> None:
        """A GenerationMessage in the 'think' phase is dropped, not misread as a decision."""
        model = _FakeLlamaGGUFModel(["<think>", "YES seems right", "</think>", "NO"])
        llm_stage = LLMStage(model)
        decision_stage = DecisionStage()

        stream = decision_stage.process(llm_stage.process(iter([_detection_msg()])))
        results = list(stream)

        decisions = [r for r in results if isinstance(r, DecisionMessage)]
        assert len(decisions) == 1
        assert decisions[0].decision == "no"

    def test_non_generation_message_dropped(self) -> None:
        """Non-GenerationMessage input yields nothing."""
        decision_stage = DecisionStage()
        other = RawFrameMessage(frame=None, timestamp=time.time())
        results = list(decision_stage.process(iter([other])))
        assert results == []

    def test_decision_state_reset_after_done(self) -> None:
        """Once a prompt's stream completes, its decided-state is cleared for reuse."""
        model = _FakeLlamaGGUFModel(["YES", " ok"])
        llm_stage = LLMStage(model)
        decision_stage = DecisionStage()

        list(decision_stage.process(llm_stage.process(iter([_detection_msg()]))))
        assert decision_stage._decided_prompts == set()


@pytest.mark.unit
class TestEarlyAbort:
    """The load-bearing early-abort contract: verdict-only sinks stop generation."""

    def test_generator_exit_propagates_to_model_stream_and_metrics_recorded(self) -> None:
        """Breaking right after the DecisionMessage closes the model's generator early.

        This is the key streaming-core feature: a sink that only needs the
        verdict stops pulling immediately after DecisionMessage, and the
        resulting GeneratorExit propagates all the way down to the model's
        stream() generator, closing it before the full response is generated.
        Metrics (ttft_ms) must still be recorded despite the early close.
        """
        model = _FakeLlamaGGUFModel(["YES", " because", " of", " many", " reasons", " here"])
        metrics = MetricsCollector(session_id="test_early_abort")
        llm_stage = LLMStage(model, grammar=YES_NO_GRAMMAR, metrics=metrics)
        decision_stage = DecisionStage(metrics=metrics)
        pipeline = Pipeline([llm_stage, decision_stage], metrics=metrics)

        with metrics.start_trace():
            gen = pipeline.run(iter([_detection_msg()]))
            for out in gen:
                if isinstance(out, DecisionMessage):
                    break
            gen.close()

        assert model.closed is True

        stage_spans = [s for s in metrics.spans if s.name == "LLMStage"]
        assert len(stage_spans) == 1
        assert "ttft_ms" in stage_spans[0].metadata
