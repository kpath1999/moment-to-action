"""ImageDetectionStage — wraps an ImageDetectionModel in a pipeline stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.messages.control import EndOfClipMessage
from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.models.image.detection._base import ImageDetectionModel
from moment_to_action.stages.image._base import ImageStage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector


def _is_undroppable_frame(msg: Message) -> bool:
    """Drop predicate: discard anything that is not a usable frame or ``EndOfClipMessage``."""
    if isinstance(msg, EndOfClipMessage):
        return False
    return not isinstance(msg, RawFrameMessage) or msg.frame is None


class ImageDetectionStage(ImageStage[ImageDetectionModel]):
    """Pipeline stage that runs object detection on a raw frame.

    Wraps any :class:`~moment_to_action.models.image.detection._base.ImageDetectionModel`
    in the standard :class:`~moment_to_action.stages._base.Stage` interface.  Accepts a
    :class:`~moment_to_action.messages.sensor.RawFrameMessage` and yields a
    :class:`~moment_to_action.messages.detection.DetectionMessage`.

    Drops any other message type or a frame that is ``None`` (dropped frame)
    before it reaches ``_process``, except
    :class:`~moment_to_action.messages.control.EndOfClipMessage`, which is passed
    through unchanged so a downstream aggregation stage can see clip boundaries.
    """

    def __init__(
        self, model: ImageDetectionModel, *, metrics: MetricsCollector | None = None
    ) -> None:
        """Initialize the stage with an unloaded detection model.

        Args:
            model: An unloaded
                :class:`~moment_to_action.models.image.detection._base.ImageDetectionModel`
                — call :meth:`load` (or ``model.load()``) before processing.
            metrics: Metrics collector used to time this stage's execution.

        Raises:
            ValueError: If *model* is already loaded.
        """
        super().__init__(model, window=1, drop=_is_undroppable_frame, metrics=metrics)

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Run detection on the buffered frame, or pass an EndOfClipMessage through.

        Args:
            items: Single-element window containing the incoming
                :class:`~moment_to_action.messages.sensor.RawFrameMessage` or
                :class:`~moment_to_action.messages.control.EndOfClipMessage`.

        Yields:
            A :class:`~moment_to_action.messages.detection.DetectionMessage` with
            detection results, or the incoming ``EndOfClipMessage`` unchanged.
        """
        msg = items[0]
        if isinstance(msg, EndOfClipMessage):
            yield msg
            return

        assert isinstance(msg, RawFrameMessage)  # noqa: S101  # guaranteed by drop predicate
        assert msg.frame is not None  # noqa: S101  # guaranteed by drop predicate

        prepared = self._model.prepare(msg.frame)
        raw = self._model.run(prepared)
        detections = self._model.post_proc(raw)

        yield DetectionMessage(
            timestamp=msg.timestamp, detections=detections, question=msg.question
        )
