"""ImageClassificationStage — wraps an ImageClassificationModel in a pipeline stage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from moment_to_action.messages._image_classification import ImageClassificationMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.metrics import SpanType
from moment_to_action.stages.image._base import ImageStage

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector
    from moment_to_action.models.image.classification._base import ImageClassificationModel

logger = logging.getLogger(__name__)


class ImageClassificationStage(ImageStage):
    """Pipeline stage that runs image classification on a raw frame.

    Wraps any :class:`~moment_to_action.models.image.classification._base.ImageClassificationModel`
    in the standard :class:`~moment_to_action.stages._base.Stage` interface.  Accepts a
    :class:`~moment_to_action.messages.sensor.RawFrameMessage` and returns an
    :class:`~moment_to_action.messages._image_classification.ImageClassificationMessage`.

    Short-circuits (returns ``None``) for any other message type or when the
    incoming frame is ``None`` (dropped frame).
    """

    def __init__(self, model: ImageClassificationModel) -> None:
        """Initialize the stage with a classification model.

        Args:
            model: An
                :class:`~moment_to_action.models.image.classification._base.ImageClassificationModel`
                instance.  The caller must call ``model.load(backend)`` before passing
                it here.
        """
        self._model = model

    def _process(self, msg: Message, metrics: MetricsCollector) -> Message | None:
        """Run classification on a raw frame, timing each model call.

        Args:
            msg: Incoming pipeline message.  Must be a
                :class:`~moment_to_action.messages.sensor.RawFrameMessage` with a
                non-``None`` ``frame`` field.
            metrics: Metrics collector for per-call timing spans.

        Returns:
            An :class:`~moment_to_action.messages._image_classification.ImageClassificationMessage`
            with top-k classification results, or ``None`` if ``msg`` is not a
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

        with metrics.start_span(SpanType.MODEL_PREPROCESS, "prepare"):
            prepared = self._model.prepare(msg.frame)

        with metrics.start_span(SpanType.MODEL_INFERENCE, "run"):
            raw = self._model.run(prepared)

        with metrics.start_span(SpanType.MODEL_POST_PROCESS, "post_process"):
            classifications = self._model.post_proc(raw)

        return ImageClassificationMessage(timestamp=msg.timestamp, classifications=classifications)
