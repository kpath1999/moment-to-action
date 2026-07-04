"""Sequential pipeline that composes a list of Stage objects into one lazy generator chain."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.metrics import NullMetricsCollector, SpanType

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector
    from moment_to_action.stages._base import Stage


class Pipeline:
    """Sequential pipeline of stages, composed into one lazy generator chain."""

    def __init__(self, stages: list[Stage], *, metrics: MetricsCollector | None = None) -> None:
        """Initialize the pipeline with an ordered list of stages.

        Args:
            stages: Ordered list of stages to run. Each stage's ``process()``
                is chained onto the previous stage's output stream.
            metrics: Metrics collector used to record the overall
                ``SpanType.PIPELINE`` span. Pass the same instance used to
                construct each stage so the pipeline span is the parent of the
                per-stage spans in the same trace. Defaults to a
                ``NullMetricsCollector``.
        """
        self._stages = stages
        self._metrics = metrics or NullMetricsCollector()

    @property
    def stages(self) -> list[Stage]:
        """Return the list of stages."""
        return self._stages

    def run(self, source: Iterator[Message]) -> Generator[Message, None, None]:
        """Run messages from *source* through all stages, lazily.

        The caller drives this generator (typically with ``with
        metrics.start_trace(): for out in pipeline.run(sensor.stream()): ...``).
        Breaking out of that loop closes this generator, propagating
        ``GeneratorExit`` up through every stage's ``process()`` — the
        mechanism by which a sink can abort in-flight upstream work (e.g. LLM
        token generation) simply by no longer pulling.

        Args:
            source: Iterator of input messages (e.g. a sensor's frame stream).

        Yields:
            Output messages from the final stage in the chain.
        """
        with self._metrics.start_span(SpanType.PIPELINE, "Pipeline Run"):
            stream = source
            for stage in self._stages:
                stream = stage.process(stream)
            yield from stream
