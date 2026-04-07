"""Abstract Stage base class.

MetricsCollector must be passed to every stage. Pipeline creates a default
NullMetricsCollector if none is provided to avoid null checks in stage code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from moment_to_action.metrics import NullMetricsCollector, SpanType

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class Stage(ABC):
    """Abstract base for all pipeline stages."""

    @property
    def name(self) -> str:
        """Return the class name as the stage identifier."""
        return self.__class__.__name__

    def process(
        self,
        msg: Message,
        metrics: MetricsCollector | None = None,
    ) -> Message | None:
        """Execute the stage, timing it, setting latency on the result, and logging to metrics.

        Args:
            msg:       Incoming message to process.
            metrics:   Metrics collector for recording stage latency.
                      If not provided, a default NullMetricsCollector is used.
        """
        # Ensure we always have a metrics collector to avoid null checks in stage code
        #
        # This is done here so that stages can be used standalone, outside of a pipeline, if desired
        # Could be useful for testing purposes
        if metrics is None:
            metrics = NullMetricsCollector()

        # Run the stage processing, timing it with the metrics collector
        with metrics.start_span(SpanType.STAGE, self.name) as span:
            span_id = span.id_  # save so we can get the latency later

            # Run the stage's processing logic, which may return None to stop the pipeline
            result = self._process(msg, metrics)

        # Stamp latency on the result so consumers don't need to measure it themselves
        elapsed_ms = metrics.get_span(span_id).latency_ms

        if result is not None:
            result = result.model_copy(update={"latency_ms": elapsed_ms})

        # Log the stage execution and latency
        status = "→ None (stopped)" if result is None else f"→ {type(result).__name__}"
        logger.debug("%s: %.1fms  %s", self.name, elapsed_ms, status)

        return result

    @abstractmethod
    def _process(
        self,
        msg: Message,
        metrics: MetricsCollector,
    ) -> Message | None:
        """Process a message and return the result or None to stop the pipeline.

        Args:
            msg:     Incoming message to process.
            metrics: Metrics collector for custom stage instrumentation.
                    Always provided (never None).
        """
        ...
