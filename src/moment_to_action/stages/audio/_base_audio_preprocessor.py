from __future__ import annotations

import logging
import time
from math import gcd
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import resample_poly

from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages.audio import AudioTensorMessage
from moment_to_action.stages._preprocess import BasePreprocessor

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from moment_to_action.messages.sensor import AudioInput

logger = logging.getLogger(__name__)
_WAVEFORM_PEAK_EPSILON = 1e-6


class BaseAudioPreprocessor(BasePreprocessor["AudioInput", AudioTensorMessage]):
    """Shared audio preprocessing utilities for model-specific preprocessors."""

    def __init__(
        self,
        compute_unit: ComputeUnit = ComputeUnit.CPU,
        *,
        target_sample_rate: int = 16_000,
        normalise: bool = True,
    ) -> None:
        self._target_sample_rate = target_sample_rate
        self._normalise = normalise
        super().__init__(compute_unit)

    def _validate(self, data: AudioInput) -> None:
        if data.waveform is None:
            message = "AudioInput.waveform cannot be None."
            raise ValueError(message)
        if data.waveform.ndim != 1:
            message = f"Waveform must be 1-D mono, got shape {data.waveform.shape}."
            raise ValueError(message)
        if data.sample_rate <= 0:
            message = f"Invalid sample_rate: {data.sample_rate}"
            raise ValueError(message)

    def _load_waveform(self, data: AudioInput) -> tuple[NDArray[np.float32], int]:
        waveform = np.asarray(data.waveform, dtype=np.float32)
        waveform = self._dispatch(
            self._resample,
            waveform,
            data.sample_rate,
            self._target_sample_rate,
        )

        if self._normalise:
            waveform = self._dispatch(self._normalise_waveform, waveform)

        return waveform, self._target_sample_rate

    def _to_message(self, waveform: NDArray[np.float32], data: AudioInput) -> AudioTensorMessage:
        return AudioTensorMessage(
            data=waveform,
            sample_rate=self._target_sample_rate,
            source=data.source,
            timestamp=time.time(),
        )

    @staticmethod
    def _resample(
        waveform: NDArray[np.float32],
        src_sr: int,
        dst_sr: int,
    ) -> NDArray[np.float32]:
        if src_sr == dst_sr:
            return waveform.astype(np.float32)

        g = gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g
        return resample_poly(waveform, up, down).astype(np.float32)

    @staticmethod
    def _normalise_waveform(waveform: NDArray[np.float32]) -> NDArray[np.float32]:
        peak = float(np.abs(waveform).max(initial=0.0))
        if peak > _WAVEFORM_PEAK_EPSILON:
            return (waveform / peak).astype(np.float32)
        return waveform.astype(np.float32)
