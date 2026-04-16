"""Whisper audio preprocessor.

Converts raw audio input (file path or mic waveform) into an AudioMessage
formatted exactly as whisper.cpp expects:

  * Mono (downmix if stereo)
  * float32 in [-1.0, 1.0]
  * 16 000 Hz sample rate
  * Padded or trimmed to exactly 30 seconds (480 000 samples)

Why 30 seconds?
---------------
Whisper's encoder was trained on exactly 30 s log-mel spectrograms.
whisper.cpp handles the mel conversion internally, so we only need to
deliver a correctly-sized float32 waveform — the stage does zero
additional signal processing.

For clips shorter than 30 s the waveform is zero-padded (silence).
For clips longer than 30 s the first 30 s are used by default; set
``truncate=False`` to raise instead of silently trimming.

Input:  AudioInput
Output: AudioMessage  (waveform is exactly 480 000 samples)
"""

from __future__ import annotations

import logging
import time
from math import gcd
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages.audio import AudioMessage
from moment_to_action.preprocessors._audio_input import AudioInput
from moment_to_action.preprocessors._preprocess import BasePreprocessor
from moment_to_action.utils.buffer import BufferPool, BufferSpec

# Re-use the shared loading and resampling helpers from the YAMNet preprocessor
from moment_to_action.preprocessors._yamnet_preprocessor import (
    _load_audio,
    _normalise,
    _resample,
)

logger = logging.getLogger(__name__)

# ── Whisper constants ─────────────────────────────────────────────────────────
_TARGET_SR = 16_000
_WINDOW_SECONDS = 30
_WINDOW_SAMPLES = _TARGET_SR * _WINDOW_SECONDS   # 480 000


class WhisperPreprocessor(BasePreprocessor[AudioInput, AudioMessage]):
    """Prepare raw audio for whisper.cpp inference.

    Produces a fixed-length 480 000-sample float32 waveform (30 s at
    16 kHz) that whisper.cpp can accept directly without any further
    signal processing.

    Parameters
    ----------
    compute_unit:
        CPU or DSP routing for dispatch.  Defaults to CPU.
    truncate:
        If ``True`` (default), clips longer than 30 s are silently
        trimmed to the first 30 s.  Set to ``False`` to raise a
        ``ValueError`` for long clips — useful when you want to
        guarantee you're not losing speech.
    normalise:
        If ``True`` (default), peak-normalise to [-1, 1] before padding.
        Disable if your capture stage guarantees this.

    Examples
    --------
    From a file (testing)::

        pre = WhisperPreprocessor()
        msg = pre.process(AudioInput(path="interview.mp4"))
        # msg.waveform.shape == (480000,) — ready for WhisperStage

    From a mic waveform (deployment)::

        pre = WhisperPreprocessor()
        msg = pre.process(AudioInput(waveform=pcm_chunk, sample_rate=44100))

    Raise on long clips instead of truncating::

        pre = WhisperPreprocessor(truncate=False)
    """

    def __init__(
        self,
        compute_unit: ComputeUnit = ComputeUnit.CPU,
        truncate: bool = True,
        normalise: bool = True,
    ) -> None:
        self._truncate = truncate
        self._normalise = normalise
        super().__init__(compute_unit)

    # ------------------------------------------------------------------
    # BasePreprocessor interface
    # ------------------------------------------------------------------

    def _allocate_buffers(self) -> None:
        """Pre-allocate the fixed 30 s output buffer (always the same size)."""
        self._buffers.register(
            "window",
            BufferSpec((_WINDOW_SAMPLES,), np.float32),
        )

    def _validate(self, data: AudioInput) -> None:
        if data.path is not None and not data.path.exists():
            raise ValueError(f"Audio file not found: {data.path}")
        if data.waveform is not None and data.waveform.ndim != 1:
            raise ValueError(
                f"Waveform must be 1-D mono, got shape {data.waveform.shape}. "
                "Downmix to mono before passing to WhisperPreprocessor."
            )
        if data.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {data.sample_rate}")

    def _process(self, data: AudioInput) -> AudioMessage:
        # 1. Load from file or use provided waveform
        if data.path is not None:
            waveform, src_sr = _load_audio(data.path)
        else:
            waveform = data.waveform.astype(np.float32)
            src_sr = data.sample_rate

        # 2. Resample to 16 kHz
        waveform = self._dispatch(_resample, waveform, src_sr, _TARGET_SR)

        # 3. Normalise
        if self._normalise:
            waveform = self._dispatch(_normalise, waveform)

        # 4. Trim or pad to exactly 30 s (480 000 samples)
        n = len(waveform)

        if n > _WINDOW_SAMPLES:
            if not self._truncate:
                raise ValueError(
                    f"Audio clip is {n / _TARGET_SR:.1f} s, which exceeds the "
                    f"Whisper 30 s window. Set truncate=True to trim silently."
                )
            logger.debug(
                "WhisperPreprocessor: trimming %.1f s clip to 30 s", n / _TARGET_SR
            )
            waveform = waveform[:_WINDOW_SAMPLES]

        # 5. Write into the fixed pre-allocated buffer (always 480 000 samples)
        buf: NDArray[np.float32] = self._buffers.get("window")
        buf[:] = 0.0            # zero the whole window (handles padding implicitly)
        samples_to_copy = min(n, _WINDOW_SAMPLES)
        buf[:samples_to_copy] = waveform[:samples_to_copy]

        if n < _WINDOW_SAMPLES:
            pad_s = (_WINDOW_SAMPLES - n) / _TARGET_SR
            logger.debug(
                "WhisperPreprocessor: padded %.2f s of silence (clip was %.2f s)",
                pad_s, n / _TARGET_SR,
            )

        out = buf.copy()   # hand off a copy — the buffer is reused next call

        logger.debug(
            "WhisperPreprocessor: ready  samples=%d  duration=%.1fs  sr=%d",
            len(out), len(out) / _TARGET_SR, _TARGET_SR,
        )

        return AudioMessage(
            waveform=out,
            sample_rate=_TARGET_SR,
            source_path=data.source_label,
            timestamp=time.time(),
        )
