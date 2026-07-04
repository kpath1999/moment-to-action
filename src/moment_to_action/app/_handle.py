"""PipelineHandle — a named, loadable/unloadable wrapper around a real Pipeline."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from moment_to_action.hardware import ComputeUnit, Platform
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector, MetricsReport, Trace
    from moment_to_action.pipeline import Pipeline
    from moment_to_action.stages._base import Stage


@attrs.define
class PipelineHandle:
    """A named pipeline built by :class:`~moment_to_action.app._builder.PipelineBuilder`.

    Wraps a real :class:`~moment_to_action.pipeline.Pipeline` plus the
    ``(Stage, ComputeUnit)`` pairs needed to load/unload it, so
    :class:`~moment_to_action.app._app.Moment2Action` can cycle its device
    resources without reconstructing any stage or model.
    """

    name: str
    """Name this pipeline is registered under."""

    _pipeline: Pipeline
    _metrics: MetricsCollector
    _stage_units: list[tuple[Stage, ComputeUnit | None]]
    loaded: bool = False
    """Whether this pipeline's stages currently hold loaded device resources."""

    @property
    def stages(self) -> list[Stage]:
        """Return this pipeline's stages, in order.

        Exposed for cases where a single chained ``run()`` can't express the
        needed data flow (e.g. an out-of-band, per-invocation aggregation step
        between two stages) and the caller must drive individual stages directly
        while still benefiting from this pipeline's shared load/unload/metrics.
        """
        return self._pipeline.stages

    def load(self, platform: Platform) -> None:
        """Load every stage in this pipeline onto *platform*.

        Args:
            platform: The hardware platform to load onto.

        Raises:
            RuntimeError: If this pipeline is already loaded.
        """
        if self.loaded:
            msg = f"Pipeline {self.name!r} is already loaded."
            raise RuntimeError(msg)
        with self._metrics.start_trace():
            for stage, unit in self._stage_units:
                stage.load(platform, unit)
        self.loaded = True

    def unload(self) -> None:
        """Unload every stage in this pipeline."""
        with self._metrics.start_trace():
            for stage, _ in self._stage_units:
                stage.unload()
        self.loaded = False

    def metrics_report(self) -> MetricsReport:
        """Return this pipeline's own metrics report."""
        return self._metrics.report()

    @contextlib.contextmanager
    def trace(self) -> Generator[Trace, None, None]:
        """Open a trace for manually driving this pipeline's stages outside ``run()``.

        Use when a single chained ``run()`` can't express the needed data flow
        (e.g. an out-of-band aggregation step with a per-invocation window between
        two stages) and the caller must call ``stage.process()`` directly — wrap
        those calls in this so their ``STAGE``/``MODEL_*`` spans land on this
        pipeline's own trace/report, exactly as ``run()`` would.

        Yields:
            The newly opened :class:`~moment_to_action.metrics.Trace`.
        """
        with self._metrics.start_trace() as trace:
            yield trace

    def run(self, source: Iterator[Message]) -> Generator[Message, None, None]:
        """Run messages from *source* through this pipeline.

        Args:
            source: Iterator of input messages.

        Yields:
            Output messages from the final stage in the chain.

        Raises:
            RuntimeError: If this pipeline has not been loaded yet.
        """
        if not self.loaded:
            msg = f"Pipeline {self.name!r} is not loaded — call load_pipeline() first."
            raise RuntimeError(msg)
        with self._metrics.start_trace():
            yield from self._pipeline.run(source)
