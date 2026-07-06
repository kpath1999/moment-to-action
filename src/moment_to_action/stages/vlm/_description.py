"""VLMDescriptionStage — streams a vision-language model's response over raw frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.messages.control import EndOfClipMessage
from moment_to_action.messages.llm import GenerationMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.messages.video import VideoClipMessage
from moment_to_action.models.vlm._base import LlamaVLModel
from moment_to_action.stages._base import ModelStage
from moment_to_action.stages.vlm._encode import bgr_to_b64

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector


def _frames_and_question_from(msg: Message) -> tuple[list[np.ndarray], str] | None:
    """Extract the raw BGR frames and question carried by *msg*, or ``None`` if unusable.

    Args:
        msg: Incoming message — a single-frame
            :class:`~moment_to_action.messages.sensor.RawFrameMessage` or a
            multi-frame :class:`~moment_to_action.messages.video.VideoClipMessage`.

    Returns:
        A ``(frames, question)`` pair, or ``None`` if *msg* is not a usable
        frame-carrying message (wrong type, or a dropped ``RawFrameMessage``).
    """
    if isinstance(msg, RawFrameMessage):
        return None if msg.frame is None else ([msg.frame], msg.question)
    if isinstance(msg, VideoClipMessage):
        return msg.frames, msg.question
    return None


class VLMDescriptionStage(ModelStage[LlamaVLModel]):
    """Streams a vision-language model's response to a task over raw frames.

    Consumes a :class:`~moment_to_action.messages.sensor.RawFrameMessage` (single
    frame) or :class:`~moment_to_action.messages.video.VideoClipMessage` (multiple
    frames — use ``window=N, stride=...`` on an upstream stage to assemble one),
    base64-JPEG-encodes the frames, and streams the model's response token by
    token as :class:`~moment_to_action.messages.llm.GenerationMessage` partials.

    VLMs have no ``<think>`` phase, so every message has ``type="response"``.

    *system_prompt* and *max_tokens* are configured on *model* at construction
    (``ModelManager.get_model(..., system_prompt=..., max_tokens=...)``); the
    per-message task comes from the incoming message's ``question`` field, so one
    loaded model/stage instance can serve any question — it isn't fixed at
    construction.
    """

    def __init__(
        self,
        model: LlamaVLModel,
        *,
        grammar: str | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize the stage with an unloaded vision-language model.

        Args:
            model: An unloaded :class:`~moment_to_action.models.vlm._base.LlamaVLModel`
                — call :meth:`load` (or ``model.load()``) before processing.
            grammar: Optional GBNF grammar constraining generation.
            metrics: Metrics collector used to time this stage and record
                per-token ttft/itl via ``MetricsCollector.timed_stream``.

        Raises:
            ValueError: If *model* is already loaded.
        """
        super().__init__(model, window=1, metrics=metrics)
        self._grammar = grammar

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Stream the model's response to the incoming frames' question.

        Args:
            items: Single-element window containing the incoming frame-carrying
                message.

        Yields:
            Partial :class:`~moment_to_action.messages.llm.GenerationMessage`
            objects, one per token, a final one with the complete response text,
            then an :class:`~moment_to_action.messages.control.EndOfClipMessage`.
        """
        msg = items[0]
        extracted = _frames_and_question_from(msg)
        if extracted is None:
            return
        frames, question = extracted

        b64_frames = [bgr_to_b64(frame) for frame in frames]
        text = ""
        for token in self._metrics.timed_stream(
            self._model.stream((question, b64_frames), grammar=self._grammar)
        ):
            text += token
            yield GenerationMessage(
                timestamp=msg.timestamp,
                prompt=question,
                text=text,
                type="response",
            )

        yield GenerationMessage(
            timestamp=msg.timestamp,
            prompt=question,
            text=text,
            type="response",
        )
        yield EndOfClipMessage(timestamp=msg.timestamp)
