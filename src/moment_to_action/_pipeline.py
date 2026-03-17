"""Sequential pipeline that runs a list of Stage objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics._collector import MetricsCollector
    from moment_to_action.stages._base import Stage


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
        current: Message | None = msg
        for stage in self._stages:
            current = stage.process(current, metrics=self._metrics)  # type: ignore[arg-type]
            if current is None:
                return None
        return current
