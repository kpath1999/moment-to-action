"""ImageDetectionStage — wraps an ImageDetectionModel in a pipeline stage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.stages.image._base import ImageStage

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector
    from moment_to_action.models.image.detection._base import ImageDetectionModel

logger = logging.getLogger(__name__)


class ImageDetectionStage(ImageStage):
    """Pipeline stage that runs object detection on a raw frame.

    Wraps any :class:`~moment_to_action.models.image.detection._base.ImageDetectionModel`
    in the standard :class:`~moment_to_action.stages._base.Stage` interface.  Accepts a
    :class:`~moment_to_action.messages.sensor.RawFrameMessage` and returns a
    :class:`~moment_to_action.messages.detection.DetectionMessage`.

    Short-circuits (returns ``None``) for any other message type or when the
    incoming frame is ``None`` (dropped frame).
    """

    def __init__(self, model: ImageDetectionModel) -> None:
        """Initialize the stage with a detection model.

        Args:
            model: An :class:`~moment_to_action.models.image.detection._base.ImageDetectionModel`
                instance.  The caller must call ``model.load(backend)`` before passing
                it here.
        """
        self._model = model

    def _process(self, msg: Message, metrics: MetricsCollector) -> Message | None:
        """Run detection on a raw frame, timing each model call with the metrics collector.

        Args:
            msg: Incoming pipeline message.  Must be a
                :class:`~moment_to_action.messages.sensor.RawFrameMessage` with a
                non-``None`` ``frame`` field.
            metrics: Metrics collector used to record per-call timing spans for
                ``prepare``, ``run``, and ``post_proc``.

        Returns:
            A :class:`~moment_to_action.messages.detection.DetectionMessage` with
            detection results, or ``None`` if ``msg`` is not a
            :class:`~moment_to_action.messages.sensor.RawFrameMessage` or if
            ``msg.frame`` is ``None``.
        """
        if not isinstance(msg, RawFrameMessage):
            logger.warning(
                "%s: expected RawFrameMessage, got %s — skipping",
                self.name,
                type(msg).__name__,
            )
            return None
        if msg.frame is None:
            return None

        prepared = self._model.prepare(msg.frame, metrics=metrics)
        raw = self._model.run(prepared, metrics=metrics)
        detections = self._model.post_proc(raw, metrics=metrics)

        return DetectionMessage(timestamp=msg.timestamp, detections=detections)
