"""Base stage and Pipeline.

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
    from moment_to_action.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)


class Stage(ABC):
    """Abstract base for all pipeline stages."""

    @property
    def name(self) -> str:
        """Return the class name as the stage identifier."""
        return self.__class__.__name__

    def run(
        self,
        msg: Message,
        metrics: MetricsCollector | None = None,
    ) -> Message | None:
        """Execute the stage, timing it and logging to metrics if provided."""
        t = time.perf_counter()
        result = self.process(msg)
        elapsed_ms = (time.perf_counter() - t) * 1000

        if metrics is not None:
            metrics.log_stage(self.name, elapsed_ms)

        status = "→ None (stopped)" if result is None else f"→ {type(result).__name__}"
        logger.debug("%s: %.1fms  %s", self.name, elapsed_ms, status)
        return result

    @abstractmethod
    def process(self, msg: Message) -> Message | None:
        """Process a message and return the result or None to stop."""
        ...


class Pipeline:
    """Sequential pipeline of stages."""

    def __init__(
        self,
        stages: list[Stage],
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.stages = stages
        self.metrics = metrics

    def run(self, msg: Message) -> Message | None:
        """Run the message through all stages sequentially."""
        current: Message | None = msg
        for stage in self.stages:
            current = stage.run(current, metrics=self.metrics)  # type: ignore[arg-type]
            if current is None:
                return None
        return current
