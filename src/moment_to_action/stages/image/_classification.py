"""ImageClassificationStage — wraps an ImageClassificationModel in a pipeline stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.messages._image_classification import ImageClassificationMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.models.image.classification._base import ImageClassificationModel
from moment_to_action.stages.image._base import ImageStage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector


def _is_undroppable_frame(msg: Message) -> bool:
    """Drop predicate: discard anything that is not a usable ``RawFrameMessage``."""
    return not isinstance(msg, RawFrameMessage) or msg.frame is None


class ImageClassificationStage(ImageStage[ImageClassificationModel]):
    """Pipeline stage that runs image classification on a raw frame.

    Wraps any :class:`~moment_to_action.models.image.classification._base.ImageClassificationModel`
    in the standard :class:`~moment_to_action.stages._base.Stage` interface.  Accepts a
    :class:`~moment_to_action.messages.sensor.RawFrameMessage` and yields an
    :class:`~moment_to_action.messages._image_classification.ImageClassificationMessage`.

    Drops any other message type or a frame that is ``None`` (dropped frame)
    before it reaches ``_process``.
    """

    def __init__(
        self, model: ImageClassificationModel, *, metrics: MetricsCollector | None = None
    ) -> None:
        """Initialize the stage with an unloaded classification model.

        Args:
            model: An unloaded
                :class:`~moment_to_action.models.image.classification._base.ImageClassificationModel`
                — call :meth:`load` (or ``model.load()``) before processing.
            metrics: Metrics collector used to time this stage's execution.

        Raises:
            ValueError: If *model* is already loaded.
        """
        super().__init__(model, window=1, drop=_is_undroppable_frame, metrics=metrics)

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Run classification on the buffered frame.

        Args:
            items: Single-element window containing the incoming
                :class:`~moment_to_action.messages.sensor.RawFrameMessage`.

        Yields:
            An :class:`~moment_to_action.messages._image_classification.ImageClassificationMessage`
            with top-k classification results.
        """
        msg = items[0]
        assert isinstance(msg, RawFrameMessage)  # noqa: S101  # guaranteed by drop predicate
        assert msg.frame is not None  # noqa: S101  # guaranteed by drop predicate

        prepared = self._model.prepare(msg.frame)
        raw = self._model.run(prepared)
        classifications = self._model.post_proc(raw)

        yield ImageClassificationMessage(timestamp=msg.timestamp, classifications=classifications)
