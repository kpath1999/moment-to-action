"""Trigger stage for collecting upstream model outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from moment_to_action.messages import Message, TriggerMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.metrics import MetricsCollector


class TriggerStage(Stage):
    """Dummy trigger stage that always fires and accumulates outputs."""

    def _process(
        self,
        msg: Message,
        _metrics: MetricsCollector,
    ) -> TriggerMessage:
        if isinstance(msg, TriggerMessage):
            payload = msg.payload
            accumulated = [*msg.accumulated, payload]
        else:
            payload = msg
            accumulated = [payload]

        return TriggerMessage(
            payload=payload,
            accumulated=accumulated,
            fired=True,
            trigger_source=self._trigger_source(payload),
            timestamp=payload.timestamp,
        )

    @staticmethod
    def _trigger_source(msg: Message) -> str:
        return cast("str", getattr(msg, "source_stage", type(msg).__name__))
