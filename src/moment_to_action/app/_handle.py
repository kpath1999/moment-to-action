"""PipelineHandle — a named, loadable/unloadable wrapper around a real Pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from moment_to_action.hardware import ComputeUnit, Platform
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector, MetricsReport
    from moment_to_action.pipeline import Pipeline
    from moment_to_action.stages._base import Stage


class PipelineHandle:
    """A named pipeline built by :class:`~moment_to_action.app._builder.PipelineBuilder`.

    Wraps a real :class:`~moment_to_action.pipeline.Pipeline` plus the
    ``(Stage, ComputeUnit)`` pairs needed to load/unload it, so
    :class:`~moment_to_action.app._app.Moment2Action` can cycle its device
    resources without reconstructing any stage or model.
    """

    def __init__(
        self,
        name: str,
        pipeline: Pipeline,
        metrics: MetricsCollector,
        stage_units: list[tuple[Stage, ComputeUnit | None]],
    ) -> None:
        """Initialize the handle around an already-built, unloaded Pipeline.

        Args:
            name: Name this pipeline is registered under.
            pipeline: The wrapped :class:`~moment_to_action.pipeline.Pipeline`.
            metrics: This pipeline's own metrics collector (shared by every
                stage/model constructed for it).
            stage_units: ``(Stage, ComputeUnit)`` pairs, one per stage, used by
                :meth:`load`/:meth:`unload` to acquire/release device resources.
        """
        self.name = name
        self._pipeline = pipeline
        self._metrics = metrics
        self._stage_units = stage_units
        self.loaded = False

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
