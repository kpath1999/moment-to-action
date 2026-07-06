"""DetectionAggregationStage — merges per-frame detections into one representative set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.messages.control import EndOfClipMessage
from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector
    from moment_to_action.models.image.detection._types import Detection


class DetectionAggregationStage(Stage):
    """Aggregates one clip's per-frame detections into one representative set.

    Consumes a run of :class:`~moment_to_action.messages.detection.DetectionMessage`
    messages (typically one per frame, as produced by
    :class:`~moment_to_action.stages.image.ImageDetectionStage`) terminated by an
    :class:`~moment_to_action.messages.control.EndOfClipMessage`, and emits a single
    aggregated ``DetectionMessage`` keeping, per unique
    :attr:`~moment_to_action.models.image.detection._types.Detection.label`, the
    highest-confidence instance seen across the clip.

    Processes one frame at a time (``window=1``) and keeps the running
    highest-confidence-per-label accumulation as instance state, so it emits in
    real time without buffering the clip's frames or needing to know the clip's
    length in advance. One stage instance can be reused across many sequential
    clips: the accumulation resets after each ``EndOfClipMessage``.
    """

    def __init__(self, *, metrics: MetricsCollector | None = None) -> None:
        """Initialize the stage with an empty running accumulation.

        Args:
            metrics: Metrics collector used to time this stage's execution.
        """
        super().__init__(metrics=metrics)
        self._best: dict[str, Detection] = {}
        self._question = ""

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Accumulate one detection message, or flush on end-of-clip.

        Args:
            items: Single-element window containing the incoming
                ``DetectionMessage`` or ``EndOfClipMessage``.

        Yields:
            A single aggregated ``DetectionMessage`` when *items* holds an
            ``EndOfClipMessage``; nothing otherwise.
        """
        msg = items[0]
        if isinstance(msg, EndOfClipMessage):
            yield DetectionMessage(
                timestamp=msg.timestamp,
                detections=list(self._best.values()),
                question=self._question,
            )
            self._best = {}
            self._question = ""
            return

        assert isinstance(msg, DetectionMessage)  # noqa: S101  # only input type expected
        self._question = msg.question
        for det in msg.detections:
            if det.label not in self._best or det.confidence > self._best[det.label].confidence:
                self._best[det.label] = det
