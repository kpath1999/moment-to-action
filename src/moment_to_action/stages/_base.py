"""Abstract Stage base class.

MetricsCollector is optional — pass one to Pipeline and every stage
reports its latency automatically. No metrics code inside stages.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# memory metrics library
from moment_to_action.metrics._collector import _rss_mb

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics._collector import MetricsCollector

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
            stage_idx: Zero-based position of this stage in the pipeline, assigned
                       by the pipeline itself — not stored on the stage.
            metrics:   Optional collector; receives a ``log_stage`` call when provided.
        """
        # calculate additional memory used by stage (memory used by stage)
        mem_before = _rss_mb()

        t = time.perf_counter()
        result = self._process(msg)
        elapsed_ms = (time.perf_counter() - t) * 1000

        mem_after = _rss_mb()
        mem_delta = mem_after - mem_before

        # Stamp latency on the result so consumers don't need to measure it.
        if result is not None:
            result = result.model_copy(update={"latency_ms": elapsed_ms})

        if metrics is not None:
            # If LLMStage is the current stage, then it will have to use log_llm to log LLM related data
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
    def _process(self, msg: Message) -> Message | None:
        """Process a message and return the result or None to stop the pipeline."""
        ...

    def _llm_metrics(self) -> dict:
        """The LLM has extra metrics which requires HTTP comm. with the server, hence separating it"""
        return {}
