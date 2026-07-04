"""Unit tests for Pipeline class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.metrics import MetricsCollector, NullMetricsCollector, SpanType
from moment_to_action.pipeline import Pipeline
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.messages import Message


def _make_frame(timestamp: float = 0.0) -> RawFrameMessage:
    """Build a RawFrameMessage for testing."""
    return RawFrameMessage(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        source="test_source",
        width=640,
        height=480,
        timestamp=timestamp,
    )


class _RecordingStage(Stage):
    """A stage that records the inputs it receives and passes them through."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize and track received inputs for assertions."""
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.received: list[Message] = []

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Record and pass through each buffered item."""
        self.received.extend(items)
        yield from items


class _DroppingStage(Stage):
    """A stage that drops every message, short-circuiting the pipeline."""

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Yield nothing."""
        yield from ()


@pytest.mark.unit
class TestPipeline:
    """Tests for Pipeline class."""

    @pytest.fixture
    def sample_messages(self) -> list[RawFrameMessage]:
        """Create sample RawFrameMessages for testing."""
        return [_make_frame(float(i)) for i in range(3)]

    def test_pipeline_run_with_stages_in_order(
        self, sample_messages: list[RawFrameMessage]
    ) -> None:
        """Pipeline.run() chains stages in order and passes messages through."""
        stage1 = _RecordingStage()
        stage2 = _RecordingStage()

        pipeline = Pipeline([stage1, stage2])
        results = list(pipeline.run(iter(sample_messages)))

        assert stage1.received == sample_messages
        assert stage2.received == sample_messages
        assert results == sample_messages

    def test_pipeline_drop_short_circuits_downstream_stages(
        self, sample_messages: list[RawFrameMessage]
    ) -> None:
        """A stage that yields nothing stops messages from reaching later stages."""
        stage1 = _RecordingStage()
        stage2 = _DroppingStage()
        stage3 = _RecordingStage()

        pipeline = Pipeline([stage1, stage2, stage3])
        results = list(pipeline.run(iter(sample_messages)))

        assert results == []
        assert stage1.received == sample_messages
        assert stage3.received == []

    def test_pipeline_metrics_span_is_recorded(
        self, sample_messages: list[RawFrameMessage]
    ) -> None:
        """Pipeline.run() records a SpanType.PIPELINE span on the provided collector."""
        metrics = MetricsCollector(session_id="test_pipeline_metrics")
        pipeline = Pipeline([_RecordingStage()], metrics=metrics)

        with metrics.start_trace():
            list(pipeline.run(iter(sample_messages)))

        pipeline_spans = [s for s in metrics.spans if s.type_ is SpanType.PIPELINE]
        assert len(pipeline_spans) == 1
        assert pipeline_spans[0].name == "Pipeline Run"

    def test_pipeline_empty_returns_input_unchanged(
        self, sample_messages: list[RawFrameMessage]
    ) -> None:
        """An empty pipeline (no stages) yields the input messages unchanged."""
        pipeline = Pipeline([])
        results = list(pipeline.run(iter(sample_messages)))
        assert results == sample_messages

    def test_pipeline_properties(self) -> None:
        """Pipeline.stages returns the constructed stage list."""
        stage1 = _RecordingStage()
        stage2 = _RecordingStage()
        pipeline = Pipeline([stage1, stage2])
        assert pipeline.stages == [stage1, stage2]

    def test_pipeline_none_metrics_defaults_to_null_collector(self) -> None:
        """Pipeline() without metrics defaults to a NullMetricsCollector."""
        pipeline = Pipeline([], metrics=None)
        assert isinstance(pipeline._metrics, NullMetricsCollector)

    def test_pipeline_message_flow_through_stages(
        self, sample_messages: list[RawFrameMessage]
    ) -> None:
        """Messages flow from stage1's output into stage2's input in order."""
        stage1 = _RecordingStage()
        stage2 = _RecordingStage()

        pipeline = Pipeline([stage1, stage2])
        list(pipeline.run(iter(sample_messages)))

        assert stage2.received == stage1.received

    def test_pipeline_run_is_lazy(self, sample_messages: list[RawFrameMessage]) -> None:
        """Pipeline.run() must not pull from the source until the consumer iterates."""
        stage = _RecordingStage()
        pipeline = Pipeline([stage])

        gen = pipeline.run(iter(sample_messages))
        assert stage.received == []
        next(gen)
        assert stage.received == [sample_messages[0]]

    def test_pipeline_generator_exit_propagates_to_stages(
        self, sample_messages: list[RawFrameMessage]
    ) -> None:
        """Breaking the consumer loop closes the pipeline generator via GeneratorExit."""
        closed = False

        class ExitTrackingStage(Stage):
            def _process(self, items: list[Message]) -> Iterator[Message]:
                nonlocal closed
                try:
                    for item in items:
                        yield item
                        yield item
                finally:
                    closed = True

        pipeline = Pipeline([ExitTrackingStage()])
        gen = pipeline.run(iter(sample_messages))
        next(gen)
        gen.close()

        assert closed is True
