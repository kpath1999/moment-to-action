"""Abstract Stage base class.

MetricsCollector must be passed to every stage. Pipeline creates a default
NullMetricsCollector if none is provided to avoid null checks in stage code.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from moment_to_action.metrics import NullMetricsCollector, SpanType
# memory metrics library
from moment_to_action.metrics._collector import _rss_mb

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
        stage_idx: int = 0,
        metrics: MetricsCollector | None = None,
    ) -> Message | None:

        self._metrics = metrics
        self._stage_idx = stage_idx
        """Execute the stage, timing it, setting latency on the result, and logging to metrics.

        Args:
            msg:       Incoming message to process.
            metrics:   Metrics collector for recording stage latency.
                      If not provided, a default NullMetricsCollector is used.
        """
<<<<<<< HEAD
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

        ##DIFF
        # Log the stage execution and latency
        #calculate additional memory used by stage (memory used by stage)
=======
        # calculate additional memory used by stage (memory used by stage)
>>>>>>> ec2df30 (style: fix formatting with ruff2)
        mem_before = _rss_mb()

        t = time.perf_counter()
        result = self._process(msg, metrics)
        elapsed_ms = (time.perf_counter() - t) * 1000

        mem_after = _rss_mb()
        mem_delta = mem_after - mem_before

        # Stamp latency on the result so consumers don't need to measure it.
        if result is not None:
            result = result.model_copy(update={"latency_ms": elapsed_ms})

        if metrics is not None:
            # If LLMStage is the current stage, then it will have to use log_llm
            llm_metrics = self._llm_metrics()
            if llm_metrics:
                metrics.log_llm(
                    stage_name=self.name,
                    stage_idx=stage_idx,
                    latency_ms=elapsed_ms,
                    init_memory_bytes=0,
                    runtime_memory_bytes=round(mem_delta, 2),
                    **llm_metrics,
                )
            else:
                metrics.log_stage(
                    stage_name=self.name,
                    stage_idx=stage_idx,
                    latency_ms=elapsed_ms,
                    init_memory_bytes=0,
                    runtime_memory_bytes=round(mem_delta, 2),
                )

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

    def _llm_metrics(self) -> dict:
        #LLM metrics require HTTP communication with the server, hence separating it.
        return {}
