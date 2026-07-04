"""Unit tests for Stage base class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.metrics import MetricsCollector, SpanType
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.messages import Message


def _make_frame(timestamp: float = 0.0) -> RawFrameMessage:
    """Build a RawFrameMessage with a fixed-size frame for testing."""
    return RawFrameMessage(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        source="test",
        width=640,
        height=480,
        timestamp=timestamp,
    )


class _PassthroughStage(Stage):
    """A concrete stage that passes each message through unchanged."""

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Yield each buffered item unchanged."""
        yield from items


class _CountingWindowStage(Stage):
    """A stage that yields the number of items in each buffered window."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize and track emitted windows for assertions."""
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.emitted_windows: list[list[Message]] = []

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Record the window and yield nothing."""
        self.emitted_windows.append(items)
        yield from ()


@pytest.mark.unit
class TestStage:
    """Tests for Stage base class."""

    def test_stage_name_property_returns_class_name(self) -> None:
        """Test that Stage.name property returns the class name."""
        stage = _PassthroughStage()
        assert stage.name == "_PassthroughStage"

    def test_process_is_lazy_generator(self) -> None:
        """process() must return a generator, not eagerly consume the input stream."""
        calls: list[int] = []

        def source() -> Iterator[Message]:
            for i in range(3):
                calls.append(i)
                yield _make_frame(float(i))

        stage = _PassthroughStage()
        gen = stage.process(source())
        assert calls == []  # nothing pulled yet
        next(gen)
        assert calls == [0]

    def test_window_one_emits_every_message(self) -> None:
        """Default window=1 emits once per input message."""
        stage = _PassthroughStage()
        frames = [_make_frame(float(i)) for i in range(3)]
        results = list(stage.process(iter(frames)))
        assert results == frames

    def test_window_buffers_before_emitting(self) -> None:
        """window=N withholds emission until N messages have been buffered."""
        stage = _CountingWindowStage(window=3)
        frames = [_make_frame(float(i)) for i in range(3)]
        list(stage.process(iter(frames[:2])))
        assert stage.emitted_windows == []

        stage2 = _CountingWindowStage(window=3)
        list(stage2.process(iter(frames)))
        assert len(stage2.emitted_windows) == 1
        assert stage2.emitted_windows[0] == frames

    def test_stride_gates_subsequent_emissions(self) -> None:
        """After the first full window, stride new messages are required per emission."""
        stage = _CountingWindowStage(window=2, stride=2)
        frames = [_make_frame(float(i)) for i in range(5)]
        list(stage.process(iter(frames)))
        # First emission once full (frames[0:2]); then every 2 new frames.
        assert len(stage.emitted_windows) == 2
        assert stage.emitted_windows[0] == frames[0:2]
        assert stage.emitted_windows[1] == frames[2:4]

    def test_stride_one_emits_every_new_frame_once_full(self) -> None:
        """stride=1 slides the window on every new frame once full."""
        stage = _CountingWindowStage(window=2, stride=1)
        frames = [_make_frame(float(i)) for i in range(4)]
        list(stage.process(iter(frames)))
        assert len(stage.emitted_windows) == 3
        assert stage.emitted_windows[0] == frames[0:2]
        assert stage.emitted_windows[1] == frames[1:3]
        assert stage.emitted_windows[2] == frames[2:4]

    def test_ready_predicate_overrides_count_based_emit(self) -> None:
        """A custom ready predicate fully controls emission, bypassing stride."""
        stage = _CountingWindowStage(window=5, ready=lambda items: len(items) >= 1)
        frames = [_make_frame(float(i)) for i in range(3)]
        list(stage.process(iter(frames)))
        assert len(stage.emitted_windows) == 3

    def test_ready_predicate_false_skips_emission(self) -> None:
        """A ready predicate returning False withholds emission for that message."""
        stage = _CountingWindowStage(window=5, ready=lambda items: len(items) >= 2)
        frames = [_make_frame(float(i)) for i in range(3)]
        list(stage.process(iter(frames)))
        # Only the 2nd and 3rd messages satisfy len(items) >= 2.
        assert len(stage.emitted_windows) == 2

    def test_drop_predicate_discards_before_buffering(self) -> None:
        """Messages matching the drop predicate never reach _process."""
        stage = _CountingWindowStage(window=1, drop=lambda m: m.timestamp < 0)
        frames = [_make_frame(-1.0), _make_frame(1.0)]
        list(stage.process(iter(frames)))
        assert len(stage.emitted_windows) == 1
        assert stage.emitted_windows[0][0].timestamp == 1.0

    def test_process_returns_nothing_when_process_yields_nothing(self) -> None:
        """A stage whose _process yields no messages produces an empty output stream."""

        class NoneStage(Stage):
            def _process(self, items: list[Message]) -> Iterator[Message]:
                yield from ()

        results = list(NoneStage().process(iter([_make_frame()])))
        assert results == []

    def test_window_less_than_one_raises(self) -> None:
        """Window < 1 raises ValueError."""
        with pytest.raises(ValueError, match="window"):
            _PassthroughStage(window=0)

    def test_stride_less_than_one_raises(self) -> None:
        """An explicit stride < 1 raises ValueError."""
        with pytest.raises(ValueError, match="stride"):
            _PassthroughStage(stride=0)

    def test_process_records_stage_span_in_metrics(self) -> None:
        """Each emission is wrapped in a SpanType.STAGE span on the provided collector."""
        metrics = MetricsCollector(session_id="test_stage_spans")
        stage = _PassthroughStage(metrics=metrics)
        with metrics.start_trace():
            list(stage.process(iter([_make_frame()])))

        stage_spans = [s for s in metrics.spans if s.type_ is SpanType.STAGE]
        assert len(stage_spans) == 1
        assert stage_spans[0].name == "_PassthroughStage"

    def test_process_works_without_metrics(self) -> None:
        """Stage is standalone-constructable and usable without a MetricsCollector."""
        stage = _PassthroughStage()
        results = list(stage.process(iter([_make_frame()])))
        assert len(results) == 1

    def test_generator_exit_propagates_on_early_break(self) -> None:
        """Breaking out of a consumer loop closes the stage's generator (GeneratorExit)."""
        closed = False

        class ExitTrackingStage(Stage):
            def _process(self, items: list[Message]) -> Iterator[Message]:
                nonlocal closed
                try:
                    for item in items:
                        yield item
                        yield item  # yield twice so we can break mid-emission
                finally:
                    closed = True

        stage = ExitTrackingStage()
        frames = [_make_frame(float(i)) for i in range(3)]
        gen = stage.process(iter(frames))
        next(gen)  # pull the first output only
        gen.close()
        assert closed is True

    def test_generator_exit_cascades_through_multiple_stages(self) -> None:
        """Closing the last stage's generator closes every upstream stage's generator too.

        `for msg in stream` (used to pull from the upstream stage) does not
        automatically forward close()/throw() the way `yield from` does, so
        Stage.process() must explicitly close its upstream `stream` on exit —
        this is what makes early-abort (e.g. stopping LLM generation once a
        decision fires) actually reach all the way back to the source.
        """
        upstream_closed = False
        downstream_closed = False

        class UpstreamStage(Stage):
            def _process(self, items: list[Message]) -> Iterator[Message]:
                nonlocal upstream_closed
                try:
                    for item in items:
                        yield item
                        yield item
                finally:
                    upstream_closed = True

        class DownstreamStage(Stage):
            def _process(self, items: list[Message]) -> Iterator[Message]:
                nonlocal downstream_closed
                try:
                    yield from items
                finally:
                    downstream_closed = True

        upstream = UpstreamStage()
        downstream = DownstreamStage()
        frames = [_make_frame(float(i)) for i in range(3)]

        gen = downstream.process(upstream.process(iter(frames)))
        next(gen)
        gen.close()

        assert downstream_closed is True
        assert upstream_closed is True

    def test_multi_stage_pipeline_flows_through(self) -> None:
        """Chaining process() calls the way Pipeline does still flows messages through."""
        from moment_to_action.pipeline import Pipeline

        stage1 = _PassthroughStage()
        stage2 = _PassthroughStage()

        metrics = MetricsCollector(session_id="test_pipeline_stage_order")
        pipeline = Pipeline([stage1, stage2], metrics=metrics)
        frames = [_make_frame(float(i)) for i in range(2)]
        with metrics.start_trace():
            results = list(pipeline.run(iter(frames)))

        assert results == frames
