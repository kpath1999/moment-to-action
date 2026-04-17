from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorflow as tf

from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages.audio import AudioTensorMessage
from moment_to_action.stages._base import Stage
from moment_to_action.utils.buffer import BufferSpec

from ._base_audio_preprocessor import BaseAudioPreprocessor

if TYPE_CHECKING:
    from moment_to_action.messages.sensor import AudioInput
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)

_TARGET_SR = 16000
_NUM_FRAMES = 96
_NUM_MEL_BINS = 64

_STFT_WINDOW_SECONDS = 0.025
_STFT_HOP_SECONDS = 0.010

_FRAME_LENGTH = int(_TARGET_SR * _STFT_WINDOW_SECONDS)  # 400
_FRAME_STEP = int(_TARGET_SR * _STFT_HOP_SECONDS)  # 160
_FFT_LENGTH = 512

# Quantization params from your interpreter metadata
_INPUT_SCALE = 0.04793788492679596
_INPUT_ZERO_POINT = 144
_LOG_MEL_EPSILON = 1e-6


class YAMNetPreprocessor(BaseAudioPreprocessor):
    """Prepare audio for YAMNet's quantized log-mel input."""

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
        self._buffers.register(
            "yamnet_input",
            BufferSpec((1, 1, _NUM_FRAMES, _NUM_MEL_BINS), np.uint8),
        )

    def _process(self, data: AudioInput) -> AudioTensorMessage:
        waveform, _ = self._load_waveform(data)

        features = self._waveform_to_log_mel(waveform)  # [96, 64]
        quantized = self._quantize(features)  # [96, 64] uint8

        buf = self._buffers.get("yamnet_input")
        buf[0, 0, :, :] = quantized

        return AudioTensorMessage(
            data=buf.copy(),
            sample_rate=_TARGET_SR,
            source=data.source,
            timestamp=time.time(),
        )

    def _waveform_to_log_mel(self, waveform: np.ndarray) -> np.ndarray:
        waveform = waveform.astype(np.float32)

        # YAMNet-style patch wants enough samples for 96 frames at 10ms hop
        min_samples = (_NUM_FRAMES - 1) * _FRAME_STEP + _FRAME_LENGTH
        if len(waveform) < min_samples:
            waveform = np.pad(waveform, (0, min_samples - len(waveform)))
        else:
            waveform = waveform[:min_samples]

        stft = tf.signal.stft(
            waveform,
            frame_length=_FRAME_LENGTH,
            frame_step=_FRAME_STEP,
            fft_length=_FFT_LENGTH,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )

        spectrogram = tf.abs(stft)

        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=_NUM_MEL_BINS,
            num_spectrogram_bins=(_FFT_LENGTH // 2) + 1,
            sample_rate=_TARGET_SR,
            lower_edge_hertz=125.0,
            upper_edge_hertz=7500.0,
        )

        mel_spectrogram = tf.matmul(tf.square(spectrogram), mel_matrix)
        log_mel = tf.math.log(mel_spectrogram + _LOG_MEL_EPSILON).numpy().astype(np.float32)

        # Ensure exact [96, 64]
        if log_mel.shape[0] < _NUM_FRAMES:
            pad = _NUM_FRAMES - log_mel.shape[0]
            log_mel = np.pad(log_mel, ((0, pad), (0, 0)))
        else:
            log_mel = log_mel[:_NUM_FRAMES, :]

        return log_mel

    def _quantize(self, features: np.ndarray) -> np.ndarray:
        quantized = np.round(features / _INPUT_SCALE + _INPUT_ZERO_POINT)
        return np.clip(quantized, 0, 255).astype(np.uint8)


class YAMNetPreprocessorStage(Stage):
    """Pipeline stage wrapper around YAMNetPreprocessor."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._preprocessor = YAMNetPreprocessor(**kwargs)

    def _process(self, msg: AudioInput, metrics: MetricsCollector) -> AudioTensorMessage:
        return self._preprocessor.process(msg, metrics=metrics)
