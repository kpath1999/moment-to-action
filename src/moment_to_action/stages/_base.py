"""Abstract Stage base class.

MetricsCollector is optional — pass one to Pipeline and every stage
reports its latency automatically. No metrics code inside stages.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics._collector import MetricsCollector

logger = logging.getLogger(__name__)


class Stage(ABC):
    """Abstract base for all pipeline stages."""

    def __init__(self, stage_idx: int) -> None:
        """Initialise the stage.

        Args:
            stage_idx: Pipeline stage index passed to the metrics collector
                       on every call (1 = trigger/sensor, 2 = vision/LLM, …).
        """
        self._stage_idx = stage_idx

    @property
    def name(self) -> str:
        """Return the class name as the stage identifier."""
        return self.__class__.__name__

    @property
    def stage_idx(self) -> int:
        """Return the pipeline stage index."""
        return self._stage_idx

    def process(
        self,
        msg: Message,
        metrics: MetricsCollector | None = None,
    ) -> Message | None:
        """Execute the stage, timing it, setting latency on the result, and logging to metrics."""
        t = time.perf_counter()
        result = self._process(msg)
        elapsed_ms = (time.perf_counter() - t) * 1000

        # Stamp latency on the result so consumers don't need to measure it.
        if result is not None:
            result = result.model_copy(update={"latency_ms": elapsed_ms})

        if metrics is not None:
            metrics.log_stage(self.name, self._stage_idx, elapsed_ms)

        status = "→ None (stopped)" if result is None else f"→ {type(result).__name__}"
        logger.debug("%s: %.1fms  %s", self.name, elapsed_ms, status)
        return result

    @abstractmethod
    def _process(self, msg: Message) -> Message | None:
        """Process a message and return the result or None to stop the pipeline."""
        ...
