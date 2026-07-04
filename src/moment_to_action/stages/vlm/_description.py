"""VLMDescriptionStage — streams a vision-language model's response over raw frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.messages.llm import GenerationMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.messages.video import VideoClipMessage
from moment_to_action.stages._base import Stage
from moment_to_action.stages.vlm._encode import bgr_to_b64

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector
    from moment_to_action.models.vlm._base import LlamaVLModel


def _frames_from(msg: Message) -> list[np.ndarray] | None:
    """Extract the raw BGR frame list carried by *msg*, or ``None`` if unusable.

    Args:
        msg: Incoming message — a single-frame
            :class:`~moment_to_action.messages.sensor.RawFrameMessage` or a
            multi-frame :class:`~moment_to_action.messages.video.VideoClipMessage`.

    Returns:
        A list of BGR uint8 frames, or ``None`` if *msg* is not a usable
        frame-carrying message (wrong type, or a dropped ``RawFrameMessage``).
    """
    if isinstance(msg, RawFrameMessage):
        return None if msg.frame is None else [msg.frame]
    if isinstance(msg, VideoClipMessage):
        return msg.frames
    return None


class VLMDescriptionStage(Stage):
    """Streams a vision-language model's response to a fixed task over raw frames.

    Consumes a :class:`~moment_to_action.messages.sensor.RawFrameMessage` (single
    frame) or :class:`~moment_to_action.messages.video.VideoClipMessage` (multiple
    frames — use ``window=N, stride=...`` on an upstream stage to assemble one),
    base64-JPEG-encodes the frames, and streams the model's response token by
    token as :class:`~moment_to_action.messages.llm.GenerationMessage` partials.

    VLMs have no ``<think>`` phase, so every message has ``type="response"``.

    *system_prompt* and *max_tokens* are configured on *model* at construction
    (``ModelManager.get_model(..., system_prompt=..., max_tokens=...)``); this
    stage only adds the per-instance *task* and optional *grammar*.
    """

    def __init__(
        self,
        model: LlamaVLModel,
        *,
        task: str,
        grammar: str | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize the stage with a vision-language model and its fixed task.

        Args:
            model: A loaded :class:`~moment_to_action.models.vlm._base.LlamaVLModel`.
            task: The question/instruction posed to the model for every incoming
                clip (this stage always asks the same task; build one
                ``VLMDescriptionStage`` per application/question).
            grammar: Optional GBNF grammar constraining generation.
            metrics: Metrics collector used to time this stage and record
                per-token ttft/itl via ``MetricsCollector.timed_stream``.
        """
        super().__init__(window=1, metrics=metrics)
        self._model = model
        self._task = task
        self._grammar = grammar

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Stream the model's response to the fixed task over the incoming frames.

        Args:
            items: Single-element window containing the incoming frame-carrying
                message.

        Yields:
            Partial :class:`~moment_to_action.messages.llm.GenerationMessage`
            objects, one per token, followed by a final ``done=True`` message.
        """
        msg = items[0]
        frames = _frames_from(msg)
        if frames is None:
            return

        b64_frames = [bgr_to_b64(frame) for frame in frames]
        text = ""
        for token in self._metrics.timed_stream(
            self._model.stream((self._task, b64_frames), grammar=self._grammar)
        ):
            text += token
            yield GenerationMessage(
                timestamp=msg.timestamp,
                prompt=self._task,
                text=text,
                type="response",
                done=False,
            )

        yield GenerationMessage(
            timestamp=msg.timestamp,
            prompt=self._task,
            text=text,
            type="response",
            done=True,
        )
