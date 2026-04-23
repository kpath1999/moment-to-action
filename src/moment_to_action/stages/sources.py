"""Source stages that produce raw messages from files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

from moment_to_action.messages.sensor import AudioInput, RawFrameMessage
from moment_to_action.sensors import FileAudioSensor, FileImageSensor
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class AudioSourceStage(Stage):
    """Load raw audio from a file path."""

    def __init__(self, source_path: str) -> None:
        super().__init__()
        self._source_path = Path(source_path)

    def process(
        self,
        msg: Message | None = None,
        stage_idx: int = 0,
        metrics: MetricsCollector | None = None,
    ) -> AudioInput | None:
        """Load the configured source path, ignoring the incoming message."""
        _ = msg
        return cast(
            "AudioInput | None",
            super().process(cast("Message", None), stage_idx=stage_idx, metrics=metrics),
        )

    def _process(
        self,
        _msg: Message,
        _metrics: MetricsCollector,
    ) -> AudioInput | None:
        if not self._source_path.is_file():
            logger.warning("AudioSourceStage: file not found: %s", self._source_path)
            return None
        with FileAudioSensor(self._source_path) as sensor:
            return sensor.read()


class ImageSourceStage(Stage):
    """Load a raw image from a file path."""

    def __init__(self, source_path: str) -> None:
        super().__init__()
        self._source_path = Path(source_path)

    def process(
        self,
        msg: Message | None = None,
        stage_idx: int = 0,
        metrics: MetricsCollector | None = None,
    ) -> RawFrameMessage | None:
        """Load the configured source path, ignoring the incoming message."""
        _ = msg
        return cast(
            "RawFrameMessage | None",
            super().process(cast("Message", None), stage_idx=stage_idx, metrics=metrics),
        )

    def _process(
        self,
        _msg: Message,
        _metrics: MetricsCollector,
    ) -> RawFrameMessage | None:
        if not self._source_path.is_file():
            logger.warning("ImageSourceStage: file not found: %s", self._source_path)
            return None

        with FileImageSensor(self._source_path) as sensor:
            return sensor.read()
