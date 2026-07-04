"""ImageDetectionStage — wraps an ImageDetectionModel in a pipeline stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.stages.image._base import ImageStage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector
    from moment_to_action.models.image.detection._base import ImageDetectionModel


def _is_undroppable_frame(msg: Message) -> bool:
    """Drop predicate: discard anything that is not a usable ``RawFrameMessage``."""
    return not isinstance(msg, RawFrameMessage) or msg.frame is None


class ImageDetectionStage(ImageStage):
    """Pipeline stage that runs object detection on a raw frame.

    Wraps any :class:`~moment_to_action.models.image.detection._base.ImageDetectionModel`
    in the standard :class:`~moment_to_action.stages._base.Stage` interface.  Accepts a
    :class:`~moment_to_action.messages.sensor.RawFrameMessage` and yields a
    :class:`~moment_to_action.messages.detection.DetectionMessage`.

    Drops any other message type or a frame that is ``None`` (dropped frame)
    before it reaches ``_process``.
    """

    def __init__(
        self, model: ImageDetectionModel, *, metrics: MetricsCollector | None = None
    ) -> None:
        """Initialize the stage with a detection model.

        Args:
            model: An :class:`~moment_to_action.models.image.detection._base.ImageDetectionModel`
                instance.  The caller must call ``model.load(backend)`` before passing
                it here.
            metrics: Metrics collector used to time this stage's execution.
        """
        super().__init__(window=1, drop=_is_undroppable_frame, metrics=metrics)
        self._model = model

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Run detection on the buffered frame.

        Args:
            items: Single-element window containing the incoming
                :class:`~moment_to_action.messages.sensor.RawFrameMessage`.

        Yields:
            A :class:`~moment_to_action.messages.detection.DetectionMessage` with
            detection results.
        """
        msg = items[0]
        assert isinstance(msg, RawFrameMessage)  # noqa: S101  # guaranteed by drop predicate
        assert msg.frame is not None  # noqa: S101  # guaranteed by drop predicate

        prepared = self._model.prepare(msg.frame)
        raw = self._model.run(prepared)
        detections = self._model.post_proc(raw)

        yield DetectionMessage(timestamp=msg.timestamp, detections=detections)
