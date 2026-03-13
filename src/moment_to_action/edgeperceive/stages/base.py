"""stages/base.py

Base stage and Pipeline.

MetricsCollector is optional — pass one to Pipeline and every stage
reports its latency automatically. No metrics code inside stages.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from moment_to_action.edgeperceive.core.messages import Message

if TYPE_CHECKING:
    from moment_to_action.edgeperceive.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)


class Stage(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def run(
        self,
        msg: Message,
        metrics: MetricsCollector | None = None,
    ) -> Message | None:
        t = time.perf_counter()
        result = self.process(msg)
        elapsed_ms = (time.perf_counter() - t) * 1000

        if metrics is not None:
            metrics.log_stage(self.name, elapsed_ms)

        status = "→ None (stopped)" if result is None else f"→ {type(result).__name__}"
        logger.debug(f"{self.name}: {elapsed_ms:.1f}ms  {status}")
        return result

    @abstractmethod
    def process(self, msg: Message) -> Message | None: ...


class Pipeline:
    def __init__(
        self,
        stages: list[Stage],
        metrics: MetricsCollector | None = None,
    ):
        self.stages = stages
        self.metrics = metrics

    def run(self, msg: Message) -> Message | None:
        for stage in self.stages:
            msg = stage.run(msg, metrics=self.metrics)
            if msg is None:
                return None
        return msg
