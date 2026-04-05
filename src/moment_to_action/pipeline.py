"""Sequential pipeline that runs a list of Stage objects."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from moment_to_action.metrics import NullMetricsCollector, SpanType

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector
    from moment_to_action.stages._base import Stage


logger = logging.getLogger(__name__)


class Pipeline:
    """Sequential pipeline of stages."""

    def __init__(self, stages: list[Stage]) -> None:
        self._stages = stages

    @property
    def stages(self) -> list[Stage]:
        """Return the list of stages."""
        return self._stages

    def run(self, msg: Message, metrics: MetricsCollector | None = None) -> Message | None:
        """Run the message through all stages sequentially.

        Args:
            msg: The input message to process through the pipeline.
            metrics: MetricsCollector to use for collecting metrics during pipeline execution.
                If not provided, a NullMetricsCollector will be used that does nothing.
        """
        # If no metrics collector provided, use a null one that does nothing
        if metrics is None:
            metrics = NullMetricsCollector()

        # Start a trace for this pipeline execution
        current: Message = msg

        # Start a span for the entire pipeline execution
        with metrics.start_span(SpanType.PIPELINE, "Pipeline Run"):
            # Run through the stages sequentially
            for stage in self._stages:
                # run the stage and check if we should exit
                new = stage.process(current, metrics=metrics)
                if new is None:
                    return None

                # update for next stage
                current = new

        return current
