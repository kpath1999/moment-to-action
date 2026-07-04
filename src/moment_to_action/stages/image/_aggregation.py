"""DetectionAggregationStage — merges per-frame detections into one representative set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector
    from moment_to_action.models.image.detection._types import Detection


class DetectionAggregationStage(Stage):
    """Aggregates a window of per-frame detections into one representative set.

    Consumes ``window`` consecutive :class:`~moment_to_action.messages.detection.DetectionMessage`
    messages (typically one per frame, as produced by
    :class:`~moment_to_action.stages.image.ImageDetectionStage`) and emits a single
    :class:`~moment_to_action.messages.detection.DetectionMessage` keeping, per unique
    :attr:`~moment_to_action.models.image.detection._types.Detection.label`, the
    highest-confidence instance seen across the window.
    """

    def __init__(self, *, window: int, metrics: MetricsCollector | None = None) -> None:
        """Initialize the stage with the number of frames to aggregate over.

        Args:
            window: Number of consecutive detection messages to buffer before
                emitting one aggregated message (e.g. a clip's frame count).
            metrics: Metrics collector used to time this stage's execution.
        """
        super().__init__(window=window, metrics=metrics)

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Merge the buffered window into one representative detection set.

        Args:
            items: The buffered window of incoming detection messages.

        Yields:
            A single :class:`~moment_to_action.messages.detection.DetectionMessage`
            with the highest-confidence detection per unique label, timestamped
            with the last buffered message's timestamp and carrying its ``question``.
        """
        best: dict[str, Detection] = {}
        for msg in items:
            assert isinstance(msg, DetectionMessage)  # noqa: S101  # only input type expected
            for det in msg.detections:
                if det.label not in best or det.confidence > best[det.label].confidence:
                    best[det.label] = det

        last = items[-1]
        assert isinstance(last, DetectionMessage)  # noqa: S101  # only input type expected
        yield DetectionMessage(
            timestamp=last.timestamp, detections=list(best.values()), question=last.question
        )
