"""Sequential pipeline that runs a list of Stage objects."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics._collector import MetricsCollector
    from moment_to_action.stages._base import Stage

from moment_to_action.metrics._collector import _rss_mb


logger = logging.getLogger(__name__)

class Pipeline:
    """Sequential pipeline of stages."""

    def __init__(
        self,
        stages: list[Stage],
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._stages = stages
        self._metrics = metrics

    @property
    def stages(self) -> list[Stage]:
        """Return the list of stages."""
        return self._stages

    @property
    def metrics(self) -> MetricsCollector | None:
        """Return the optional metrics collector."""
        return self._metrics

    def run(self, msg: Message) -> Message | None:
        """Run the message through all stages sequentially."""
        current: Message = msg

        for idx, stage in enumerate(self._stages):
            new = stage.process(current, stage_idx=idx, metrics=self._metrics)
            if new is None:
                return None

            current = new

        return current
