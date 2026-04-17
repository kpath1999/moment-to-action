from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from faster_whisper import WhisperModel

from moment_to_action.messages.audio import AudioTensorMessage, AudioTranscriptionMessage
from moment_to_action.metrics._types import SpanType
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class WhisperStage(Stage):
    """Run speech transcription using faster-whisper."""

    def __init__(
        self,
        model_size_or_path: str = "small",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        language: str | None = None,
        vad_filter: bool = False,
        task: str = "transcribe",
    ) -> None:
        super().__init__()
        self._beam_size = beam_size
        self._language = language
        self._vad_filter = vad_filter
        self._task = task

        self._model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
        )

        logger.info(
            "WhisperStage: loaded model=%s device=%s compute_type=%s",
            model_size_or_path,
            device,
            compute_type,
        )

    def _process(
        self,
        msg: Message,
        metrics: MetricsCollector,
    ) -> AudioTranscriptionMessage:
        if not isinstance(msg, AudioTensorMessage):
            err = f"WhisperStage expects AudioTensorMessage, got {type(msg).__name__}"
            raise TypeError(err)

        waveform = np.asarray(msg.data, dtype=np.float32)

        with metrics.start_span(SpanType.MODEL_INFERENCE, "Whisper inference"):
            segments, info = self._model.transcribe(
                waveform,
                beam_size=self._beam_size,
                language=self._language,
                vad_filter=self._vad_filter,
                task=self._task,
            )
            segments = list(segments)

        text = " ".join(segment.text.strip() for segment in segments).strip()

        segment_dicts: list[dict[str, Any]] = [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            for segment in segments
        ]

        logger.info(
            "WhisperStage: language=%s segments=%d chars=%d",
            getattr(info, "language", None),
            len(segment_dicts),
            len(text),
        )

        return AudioTranscriptionMessage(
            text=text,
            language=getattr(info, "language", None),
            confidence=getattr(info, "language_probability", None),
            segments=segment_dicts,
            sample_rate=msg.sample_rate,
            source=msg.source,
            timestamp=msg.timestamp,
        )
