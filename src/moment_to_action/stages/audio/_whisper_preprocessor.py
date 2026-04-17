from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages.sensor import AudioInput
from moment_to_action.stages._base import Stage

from ._base_audio_preprocessor import BaseAudioPreprocessor

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.messages.audio import AudioTensorMessage
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)

_TARGET_SR = 16000


class WhisperPreprocessor(BaseAudioPreprocessor):
    """Prepare mono float32 audio for faster-whisper."""

    def __init__(
        self,
        compute_unit: ComputeUnit = ComputeUnit.CPU,
        *,
        normalise: bool = True,
    ) -> None:
        super().__init__(
            compute_unit=compute_unit,
            target_sample_rate=_TARGET_SR,
            normalise=normalise,
        )

    def _allocate_buffers(self) -> None:
        """No fixed buffer needed for faster-whisper."""
        return

    def _process(self, data: AudioInput) -> AudioTensorMessage:
        waveform, _ = self._load_waveform(data)

        logger.debug(
            "WhisperPreprocessor: samples=%d duration=%.2fs sr=%d",
            len(waveform),
            len(waveform) / self._target_sample_rate,
            self._target_sample_rate,
        )

        return self._to_message(waveform, data)


class WhisperPreprocessorStage(Stage):
    """Pipeline stage wrapper around WhisperPreprocessor."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._preprocessor = WhisperPreprocessor(**kwargs)

    def _process(
        self,
        msg: Message,
        metrics: MetricsCollector,
    ) -> AudioTensorMessage:
        if not isinstance(msg, AudioInput):
            message = f"WhisperPreprocessorStage expects AudioInput, got {type(msg).__name__}"
            raise TypeError(message)
        return self._preprocessor.process(msg, metrics=metrics)
